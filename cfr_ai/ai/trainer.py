"""
Deep CFR Trainer for Fafnir (v2).

Major improvements over v1:
- Correct Outcome Sampling MCCFR regret estimation (all legal actions)
- DCFR (Discounted CFR) weighting scheme
- Score randomization for game-context learning
- Adaptive exploration (epsilon decay)
- Observation space: 42 dimensions (was 34)
- Dueling network architecture

Implements Deep CFR with Outcome Sampling MCCFR:
- Instead of iterating ALL legal actions per node (exponential),
  we SAMPLE a single action for the traverser too, then use
  importance sampling to correct the regret estimates.
- This makes each traversal O(depth) instead of O(branching^depth).
- Many more traversals are needed, but each is very fast.

Architecture:
- Regret Network: predicts counterfactual regrets per action (Dueling)
- Strategy Network: predicts average strategy (Dueling)
- Value Network: evaluates non-terminal leaf nodes
- Reservoir buffers for training data
- Color symmetry data augmentation
"""
import os
import sys
import time
import random
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any

from .game_engine import (
    FafnirState, new_game, step_auction, NUM_COLORS,
    compute_hand_score, clamp_score, is_trash_limit_reached,
    should_force_round_end_by_bag, setup_offer, do_round_end,
    resolve_auction, check_game_end, SCORE_TO_WIN,
)
from .action_space import (
    NUM_ACTIONS, ACTION_TABLE, get_legal_mask, action_id_to_counts, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker, OBS_DIM
from .networks import (
    RegretNetwork, StrategyNetwork, ValueNetwork,
    regret_matching, masked_softmax,
)
from .symmetry import augment_sample, augment_sample_sparse


# ============================================================
# Reservoir Buffer (with DCFR weighting)
# ============================================================
class ReservoirBuffer:
    """Reservoir sampling buffer for Deep CFR.

    DCFR (Discounted CFR): イテレーション番号に基づく重み付けで、
    古いデータを割り引き、新しいデータを重視する。

    Weight scheme:
    - Positive regrets: weight = t^alpha / (t^alpha + 1)
    - Negative regrets: weight = t^beta / (t^beta + 1)
    - Strategy: weight = (t/T)^gamma
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.total_seen = 0
        self._needs_mask = True  # whether to reconstruct masks on sample
        self._sparse_target = False  # if True, target is (action_id, value) pair
        self._sparse_strategy = False  # if True, target is list of [action_id, prob]
        self._sparse_regret_multi = False  # if True, target is list of [action_id, regret_value]

    def add(self, obs: np.ndarray, target: np.ndarray, iteration: int, legal_mask: np.ndarray = None):
        """Add a sample. legal_mask arg is accepted but ignored (for API compat)."""
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, target, iteration))
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (obs, target, iteration)

    def sample_batches(self, batch_size: int, num_batches: int, num_actions: int = 0, current_iteration: int = 0):
        total_samples = batch_size * num_batches
        n_avail = len(self.buffer)
        if n_avail == 0 or total_samples == 0:
            return

        # DCFR weighting: newer iterations get higher weight
        if current_iteration > 1:
            iters = np.fromiter((b[2] for b in self.buffer), dtype=np.float64, count=n_avail)
            np.clip(iters, 1, None, out=iters)
            # DCFR gamma=2 weighting: (t/T)^2
            ratios = iters / current_iteration
            weights = ratios ** 2.0
            weights /= weights.sum()
            all_indices = np.random.choice(n_avail, size=total_samples, replace=True, p=weights)
        else:
            all_indices = np.random.randint(0, n_avail, size=total_samples)

        for i in range(num_batches):
            indices = all_indices[i * batch_size : (i + 1) * batch_size]
            obs = np.stack([self.buffer[idx][0] for idx in indices])

            # Reconstruct dense targets from sparse
            if self._sparse_regret_multi and num_actions > 0:
                # Multi-action sparse regret: list of [action_id, regret_value]
                targets = np.zeros((len(indices), num_actions), dtype=np.float32)
                for j, idx in enumerate(indices):
                    sparse = self.buffer[idx][1]
                    if len(sparse) > 0:
                        act_ids = sparse[:, 0].astype(int)
                        values = sparse[:, 1]
                        targets[j, act_ids] = values
            elif self._sparse_target and num_actions > 0:
                targets = np.zeros((len(indices), num_actions), dtype=np.float32)
                for j, idx in enumerate(indices):
                    sparse = self.buffer[idx][1]
                    targets[j, int(sparse[0])] = sparse[1]
            elif getattr(self, '_sparse_strategy', False) and num_actions > 0:
                targets = np.zeros((len(indices), num_actions), dtype=np.float32)
                for j, idx in enumerate(indices):
                    sparse = self.buffer[idx][1]
                    if len(sparse) > 0:
                        act_ids = sparse[:, 0].astype(int)
                        probs = sparse[:, 1]
                        targets[j, act_ids] = probs
            else:
                targets = np.stack([self.buffer[idx][1] for idx in indices])

            # Reconstruct legal masks
            if self._needs_mask and num_actions > 1:
                masks = np.stack([
                    get_legal_mask(
                        self.buffer[idx][0][:6].astype(int).tolist(),
                        self.buffer[idx][0][6:12].astype(int).tolist(),
                    ) for idx in indices
                ])
            else:
                masks = np.ones((len(indices), 1), dtype=np.float32)

            yield obs, targets, masks

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Deep CFR Trainer (v2)
# ============================================================
class DeepCFRTrainer:

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        buffer_capacity: int = 2_000_000,
        batch_size: int = 1024,
        train_steps_per_iter: int = 200,
        max_depth: int = 50,
        num_augments: int = 2,
        explore_epsilon: float = 0.4,
        device: str = "auto",
        save_dir: str = "cfr_ai/ai/checkpoints",
        num_workers: int = 1,
        # DCFR parameters
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.5,
        dcfr_gamma: float = 2.0,
        # Score randomization
        score_randomize: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_workers = num_workers
        print(f"[DeepCFR v2] Device: {self.device}, Actions: {NUM_ACTIONS}, Workers: {self.num_workers}")
        print(f"[DeepCFR v2] Obs dim: {OBS_DIM}, Hidden: {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.train_steps_per_iter = train_steps_per_iter
        self.max_depth = max_depth
        self.num_augments = num_augments
        self.explore_epsilon = explore_epsilon
        self.save_dir = save_dir

        # DCFR parameters
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma

        # Score randomization
        self.score_randomize = score_randomize

        # Networks (v2: Dueling architecture, OBS_DIM input)
        self.regret_net = RegretNetwork(OBS_DIM, NUM_ACTIONS, hidden_dim).to(self.device)
        self.strategy_net = StrategyNetwork(OBS_DIM, NUM_ACTIONS, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(OBS_DIM, hidden_dim).to(self.device)

        # Optimizers
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Buffers
        self.regret_buffer = ReservoirBuffer(buffer_capacity)
        self.regret_buffer._sparse_regret_multi = True  # v2: multi-action sparse regret
        self.strategy_buffer = ReservoirBuffer(buffer_capacity)
        self.strategy_buffer._sparse_strategy = True
        self.value_buffer = ReservoirBuffer(buffer_capacity)

        # Stats
        self.iteration = 0
        self.total_traversals = 0

        # Baseline EMA for variance reduction
        self._baseline = 0.0
        self._baseline_alpha = 0.01  # EMA smoothing factor

        # Multiprocessing pool (lazy-init)
        self._pool = None

    # ============================================================
    # Adaptive Exploration
    # ============================================================
    def _get_epsilon(self) -> float:
        """Adaptive exploration: high initial exploration, decaying over time."""
        base = self.explore_epsilon  # default 0.4
        return max(0.05, base * (0.997 ** self.iteration))

    # ============================================================
    # Score Randomization
    # ============================================================
    def _randomize_scores(self, state: FafnirState):
        """Inject random starting scores to learn score-dependent strategy.

        This replaces multi-round traversal at zero cost:
        - 50% of games: scores are (0, 0) (normal)
        - 50% of games: random scores to simulate mid-game situations
        """
        if not self.score_randomize:
            return

        if random.random() < 0.5:
            return  # Keep default (0, 0)

        # Random scores: both players get independent random scores (0~990)
        state.scores[0] = random.randint(0, 990)
        state.scores[1] = random.randint(0, 990)

    # ============================================================
    # Outcome Sampling CFR Traversal (v2: correct regret estimation)
    # ============================================================
    def traverse_game(self, traverser: int) -> float:
        """
        Run a single complete game using outcome sampling.

        v2 improvements:
        - Correct regret estimation for ALL legal actions (not just the chosen one)
        - Adaptive epsilon exploration
        - Score randomization for game-context learning

        At each decision point:
        - Compute strategy from regret network (regret matching)
        - For traverser: sample one action, compute regret estimate for ALL legal actions
        - For opponent: sample one action from their strategy
        - Store samples in buffers

        Returns terminal value for traverser.
        """
        state = new_game()
        self._randomize_scores(state)

        tracker = BidTracker()
        depth = 0
        initial_round = state.round_num
        initial_scores = state.scores[:]

        # Collect all decision points for this traversal
        decision_points = []

        epsilon = self._get_epsilon()

        while state.phase == "BIDDING" and depth < self.max_depth and state.round_num == initial_round:
            # Get obs and masks for both players
            obs = [None, None]
            masks = [None, None]
            strategies = [None, None]

            for p in range(2):
                obs[p] = build_observation(state, p, tracker)
                masks[p] = get_legal_mask(state.hand[p], state.offer)

                with torch.inference_mode():
                    obs_t = torch.tensor(obs[p], dtype=torch.float32, device=self.device).unsqueeze(0)
                    regrets = self.regret_net(obs_t).cpu().numpy()[0]
                strategies[p] = regret_matching(regrets, masks[p])

            # Sample actions for both players
            actions = [0, 0]
            sample_probs = [1.0, 1.0]

            for p in range(2):
                legal = np.where(masks[p])[0]
                if len(legal) == 0:
                    actions[p] = PASS_ACTION_ID
                    continue

                if p == traverser:
                    # Regret-based pruning: skip clearly bad actions
                    if self.iteration > 100 and len(legal) > 3:
                        regret_vals = regrets[legal]
                        max_r = regret_vals.max()
                        threshold = max(0.0, max_r * 0.1)  # relative threshold
                        keep = regret_vals >= threshold
                        if keep.sum() >= 2:
                            legal = legal[keep]

                    # Exploration: epsilon-greedy with uniform over legal actions
                    explore_probs = masks[p].astype(np.float32) / max(1, masks[p].sum())
                    mixed = (1 - epsilon) * strategies[p] + epsilon * explore_probs
                    # Renormalize
                    mixed_legal = mixed[legal]
                    mixed_legal = mixed_legal / (mixed_legal.sum() + 1e-10)
                    chosen_idx = np.random.choice(len(legal), p=mixed_legal)
                    actions[p] = legal[chosen_idx]
                    sample_probs[p] = mixed[actions[p]]
                else:
                    # Opponent: sample from strategy
                    strat_legal = strategies[p][legal]
                    strat_legal = strat_legal / (strat_legal.sum() + 1e-10)
                    chosen_idx = np.random.choice(len(legal), p=strat_legal)
                    actions[p] = legal[chosen_idx]
                    sample_probs[p] = strategies[p][actions[p]]

            # Record decision point (with scores snapshot for per-point values)
            decision_points.append({
                'obs': obs,
                'masks': masks,
                'strategies': strategies,
                'actions': actions,
                'sample_probs': sample_probs,
                'offer_snapshot': state.offer[:],
                'scores_before': state.scores[:],
            })

            # Execute auction
            bid0 = action_id_to_counts(actions[0])
            bid1 = action_id_to_counts(actions[1])

            # Update tracker before step
            old_offer = state.offer[:]
            step_auction(state, bid0, bid1)

            # Update bid tracker
            total0, total1 = sum(bid0), sum(bid1)
            if max(total0, total1) > 0:
                if total0 > total1:
                    winner = 0
                elif total1 > total0:
                    winner = 1
                else:
                    winner = 1 - (state.caretaker if state.caretaker != (1 - traverser) else traverser)
                    # Actually, caretaker was already updated by step_auction
                    # Use bid comparison directly
                    if total0 == total1:
                        # Before step_auction, caretaker was at the old value
                        # We need to determine winner before state mutation
                        pass  # winner is already set correctly by step_auction logic

                loser = 1 - winner
                bid_w = bid0 if winner == 0 else bid1
                bid_l = bid0 if loser == 0 else bid1
                tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

            depth += 1

        # === Dense Reward Shaping: per-decision-point effective values ===
        opp = 1 - traverser
        if state.round_num > initial_round:
            # Round completed — do_round_end() already added hand scores
            final_t = state.scores[traverser]
            final_o = state.scores[opp]
        else:
            # GAME_END or depth limit — manually add hand scores
            final_t = state.scores[traverser] + compute_hand_score(state, traverser)
            final_o = state.scores[opp] + compute_hand_score(state, opp)

        # Each decision point sees "how much score was gained from here onward"
        effective_values = []
        for dp in decision_points:
            sb = dp['scores_before']
            gained_from_here = (final_t - sb[traverser]) - (final_o - sb[opp])
            effective_values.append(max(-1.0, min(1.0, gained_from_here / 50.0)))

        # Process with per-point values (replaces flat terminal_value)
        self._process_decision_points(
            decision_points, traverser, effective_values
        )

        # Stats: first point's value = total round value
        total_value = effective_values[0] if effective_values else 0.0
        self._baseline += self._baseline_alpha * (total_value - self._baseline)

        return total_value

    # ============================================================
    # Win Probability Estimation
    # ============================================================
    @staticmethod
    def _win_probability(my_score: float, opp_score: float) -> float:
        """ゲーム勝利確率の推定（残りポイント比率ベース）。

        「1000点に先に到達した方が勝ち」のゲームにおいて、
        各プレイヤーの残りポイント数の逆比で勝率を推定する。

        例:
          (0, 0)     → 0.500 (五分五分)
          (500, 500)  → 0.500
          (900, 100)  → 0.989 (ほぼ勝ち確定)
          (990, 990)  → 0.500
          (995, 990)  → 0.667
        """
        if my_score >= SCORE_TO_WIN:
            return 1.0
        if opp_score >= SCORE_TO_WIN:
            return 0.0
        my_remaining = max(1.0, SCORE_TO_WIN - my_score)
        opp_remaining = max(1.0, SCORE_TO_WIN - opp_score)
        return opp_remaining / (my_remaining + opp_remaining)

    def _compute_terminal_value(
        self, state: FafnirState, traverser: int,
        initial_round: int, initial_scores: list,
    ) -> float:
        """1ラウンドでの獲得スコア差を報酬として返す。

        報酬 = (自分のラウンド獲得点 - 相手のラウンド獲得点) / 正規化係数

        このゲームではオークション勝ち(+1)よりも石の得点(手札スコア)の
        方がはるかに大きいため、手札スコアを含めた総合スコア差を使う。
        正規化係数50: 典型的なラウンドスコア差(±40)を[-1, 1]に収める。
        """
        opp = 1 - traverser

        # 最終的なスコアを決定
        if state.round_num > initial_round:
            # ラウンド完了 — do_round_end()で手札スコア加算済み
            final_t = state.scores[traverser]
            final_o = state.scores[opp]
        else:
            # GAME_END or 深度制限 — 手札スコアを手動加算
            final_t = state.scores[traverser] + compute_hand_score(state, traverser)
            final_o = state.scores[opp] + compute_hand_score(state, opp)

        # ラウンドスコア差
        gained = (final_t - initial_scores[traverser]) - \
                 (final_o - initial_scores[opp])

        return max(-1.0, min(1.0, gained / 50.0))

    def _process_decision_points(
        self,
        decision_points: List[Dict],
        traverser: int,
        effective_values: List[float],
    ):
        """
        Process decision points to compute regret estimates (v3: Dense Reward).

        v3 changes over v2:
        - Per-decision-point effective values instead of flat terminal_value.
          Earlier decisions see "how much was gained from that point onward".
          This solves the Credit Assignment Problem.
        - Value Network baseline for per-point variance reduction.
        - Increased importance weight clamping (10 → 50).
        - Value buffer is now populated for value network training.

        For Outcome Sampling MCCFR, the counterfactual regret for action a at
        info set I is estimated as:
          r(I, a) = w * (u_a - v(I))

        Where:
          - w = importance weight = 1 / q(z)
          - u_a = counterfactual value of action a (now per-point)
          - v(I) = baseline from value network
        """
        for i, dp in enumerate(decision_points):
            obs = dp['obs'][traverser]
            mask = dp['masks'][traverser]
            strategy = dp['strategies'][traverser]
            chosen_action = dp['actions'][traverser]
            sample_prob = dp['sample_probs'][traverser]

            # Per-point effective value (Dense Reward Shaping)
            point_value = effective_values[i]

            # Value Network baseline (improved variance reduction)
            with torch.inference_mode():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                baseline = self.value_net(obs_t).cpu().item()

            # Store value training sample
            self.value_buffer.add(
                obs,
                np.array([point_value], dtype=np.float32),
                self.iteration,
            )

            # Importance weight with relaxed clamping (10 → 50)
            weight = min(50.0, 1.0 / max(sample_prob, 1e-6))

            # Baseline-subtracted per-point value
            adjusted_value = (point_value - baseline) * weight

            # === Compute regret for ALL legal actions ===
            legal_actions = np.where(mask)[0]

            # Sparse multi-action regret: list of [action_id, regret_value]
            regret_pairs = []
            for a in legal_actions:
                if a == chosen_action:
                    # Chosen action: regret = (1 - σ(a)) * adjusted_value
                    regret_a = (1.0 - strategy[a]) * adjusted_value
                else:
                    # Unchosen action: regret = -σ(a) * adjusted_value
                    regret_a = -strategy[a] * adjusted_value
                regret_pairs.append([a, regret_a])

            sparse_regret = np.array(regret_pairs, dtype=np.float32)
            self.regret_buffer.add(obs, sparse_regret, self.iteration)

            # Store strategy sample as sparse format
            nonzero_strats = np.nonzero(strategy)[0]
            sparse_strat = np.zeros((len(nonzero_strats), 2), dtype=np.float32)
            sparse_strat[:, 0] = nonzero_strats
            sparse_strat[:, 1] = strategy[nonzero_strats]
            self.strategy_buffer.add(obs, sparse_strat, self.iteration)

            # Color symmetry augmentation
            if self.num_augments > 0 and random.random() < 0.5:
                chosen_regret = (1.0 - strategy[chosen_action]) * adjusted_value
                aug_triples = augment_sample_sparse(
                    obs, chosen_action, chosen_regret, ACTION_TABLE, self.num_augments
                )
                for aug_obs, aug_aid, aug_val in aug_triples:
                    aug_sparse = np.array([[aug_aid, aug_val]], dtype=np.float32)
                    self.regret_buffer.add(aug_obs, aug_sparse, self.iteration)

    # ============================================================
    # Network Training
    # ============================================================
    def train_regret_network(self) -> float:
        if len(self.regret_buffer) < self.batch_size:
            return 0.0
        self.regret_net.train()
        total_loss = 0.0
        steps = min(self.train_steps_per_iter, len(self.regret_buffer) // self.batch_size)
        if steps == 0:
            return 0.0

        batch_generator = self.regret_buffer.sample_batches(self.batch_size, steps, num_actions=NUM_ACTIONS, current_iteration=self.iteration)
        for obs, targets, masks in batch_generator:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
            masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device)
            pred = self.regret_net(obs_t)
            loss = ((pred - targets_t) ** 2 * masks_t).sum() / (masks_t.sum() + 1e-8)
            self.regret_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
            self.regret_optimizer.step()
            total_loss += loss.item()
        self.regret_net.eval()
        return total_loss / max(steps, 1)

    def train_strategy_network(self) -> float:
        if len(self.strategy_buffer) < self.batch_size:
            return 0.0
        self.strategy_net.train()
        total_loss = 0.0
        steps = min(self.train_steps_per_iter, len(self.strategy_buffer) // self.batch_size)
        if steps == 0:
            return 0.0

        batch_generator = self.strategy_buffer.sample_batches(self.batch_size, steps, num_actions=NUM_ACTIONS, current_iteration=self.iteration)
        for obs, targets, masks in batch_generator:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
            masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device)
            logits = self.strategy_net(obs_t)
            logits_masked = logits + (1 - masks_t) * (-1e9)
            log_probs = torch.log_softmax(logits_masked, dim=-1)
            loss = -(targets_t * log_probs * masks_t).sum() / (obs_t.shape[0] + 1e-8)
            self.strategy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), 1.0)
            self.strategy_optimizer.step()
            total_loss += loss.item()
        self.strategy_net.eval()
        return total_loss / max(steps, 1)

    def train_value_network(self) -> float:
        if len(self.value_buffer) < self.batch_size:
            return 0.0
        self.value_net.train()
        total_loss = 0.0
        steps = min(self.train_steps_per_iter // 2, len(self.value_buffer) // self.batch_size)
        if steps == 0:
            return 0.0

        batch_generator = self.value_buffer.sample_batches(self.batch_size, steps, current_iteration=self.iteration)
        for obs, targets, _ in batch_generator:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            values_t = torch.tensor(targets[:, 0], dtype=torch.float32, device=self.device)
            pred = self.value_net(obs_t)
            loss = nn.MSELoss()(pred, values_t)
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
            total_loss += loss.item()
        self.value_net.eval()
        return total_loss / max(steps, 1)

    # ============================================================
    # Main Training Loop
    # ============================================================
    def run_iteration(self, num_traversals: int = 500):
        if self.num_workers > 1:
            return self._run_iteration_parallel(num_traversals)
        return self._run_iteration_serial(num_traversals)

    def _run_iteration_serial(self, num_traversals: int):
        """Original single-process iteration."""
        self.iteration += 1
        self.regret_net.eval()
        self.strategy_net.eval()
        self.value_net.eval()

        t0 = time.time()
        total_value = 0.0

        for _ in range(num_traversals):
            # Alternate traverser
            traverser = self.total_traversals % 2
            value = self.traverse_game(traverser)
            total_value += value
            self.total_traversals += 1

        traverse_time = time.time() - t0
        avg_value = total_value / num_traversals

        # Train networks (use all threads for training)
        t1 = time.time()
        rl = self.train_regret_network()
        sl = self.train_strategy_network()
        vl = self.train_value_network()
        train_time = time.time() - t1

        eps = self._get_epsilon()
        mem_str = ""
        if self.iteration % 10 == 0:
            mem_mb = self._estimate_memory_mb()
            mem_str = f" Mem~{mem_mb:.0f}MB"
        print(
            f"[DeepCFR v2] Iter={self.iteration} | "
            f"Trav={self.total_traversals} AvgV={avg_value:.3f} | "
            f"Buf: R={len(self.regret_buffer)} S={len(self.strategy_buffer)} V={len(self.value_buffer)} | "
            f"Loss: R={rl:.4f} S={sl:.4f} V={vl:.4f} | "
            f"ε={eps:.3f} | "
            f"T={traverse_time:.1f}s Tr={train_time:.1f}s{mem_str}"
        )
        return {'regret_loss': rl, 'strategy_loss': sl, 'value_loss': vl}

    def _run_iteration_parallel(self, num_traversals: int):
        """Parallel traversal iteration using a persistent multiprocessing pool."""
        from .parallel import _worker_init, _worker_traverse_batch

        self.iteration += 1
        self.regret_net.eval()
        self.strategy_net.eval()
        self.value_net.eval()

        t0 = time.time()

        # Create persistent pool on first call
        if self._pool is None:
            print(f"[DeepCFR v2] Creating worker pool ({self.num_workers} workers)...")
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.hidden_dim,),
            )

        # Prepare model state dicts (CPU tensors for serialization)
        regret_sd = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}
        value_sd = {k: v.cpu() for k, v in self.value_net.state_dict().items()}

        epsilon = self._get_epsilon()

        # Distribute traversals across workers
        per_worker = num_traversals // self.num_workers
        remainder = num_traversals % self.num_workers

        work_items = []
        offset = self.total_traversals
        for w in range(self.num_workers):
            n = per_worker + (1 if w < remainder else 0)
            work_items.append((
                n, offset, self.iteration,
                self.max_depth, self.num_augments, epsilon,
                self._baseline,
                self.score_randomize,
                regret_sd, value_sd,
            ))
            offset += n

        try:
            results = self._pool.map(_worker_traverse_batch, work_items)
        except Exception as e:
            print(f"[DeepCFR v2] Pool error: {e}, recreating pool...")
            self._pool.terminate()
            self._pool = None
            # Fall back to serial for this iteration
            return self._run_iteration_serial(num_traversals)

        # Aggregate results
        total_value = 0.0
        total_trav = 0
        for r in results:
            total_value += r['total_value']
            total_trav += r['num_traversals']
            for sample in r['regret_samples']:
                self.regret_buffer.add(*sample)
            for sample in r['strategy_samples']:
                self.strategy_buffer.add(*sample)
            for sample in r['value_samples']:
                self.value_buffer.add(*sample)

        self.total_traversals += total_trav
        traverse_time = time.time() - t0
        avg_value = total_value / max(total_trav, 1)

        # Train networks (use multiple threads for matrix ops)
        torch.set_num_threads(self.num_workers)
        t1 = time.time()
        rl = self.train_regret_network()
        sl = self.train_strategy_network()
        vl = self.train_value_network()
        train_time = time.time() - t1

        mem_str = ""
        if self.iteration % 10 == 0:
            mem_mb = self._estimate_memory_mb()
            mem_str = f" Mem~{mem_mb:.0f}MB"
        print(
            f"[DeepCFR v2] Iter={self.iteration} | "
            f"Trav={self.total_traversals} AvgV={avg_value:.3f} | "
            f"Buf: R={len(self.regret_buffer)} S={len(self.strategy_buffer)} V={len(self.value_buffer)} | "
            f"Loss: R={rl:.4f} S={sl:.4f} V={vl:.4f} | "
            f"ε={epsilon:.3f} | "
            f"T={traverse_time:.1f}s Tr={train_time:.1f}s [{self.num_workers}w]{mem_str}"
        )
        return {'regret_loss': rl, 'strategy_loss': sl, 'value_loss': vl}

    def shutdown_pool(self):
        """Cleanly shut down the worker pool."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def _estimate_memory_mb(self) -> float:
        """バッファのおおよそのメモリ使用量をMB単位で推定。"""
        total_bytes = 0
        for buf in [self.regret_buffer, self.strategy_buffer, self.value_buffer]:
            for item in buf.buffer[:100]:  # サンプリングで推定
                total_bytes += item[0].nbytes  # obs
                total_bytes += item[1].nbytes if hasattr(item[1], 'nbytes') else 8  # target
                total_bytes += 8  # iteration int
                total_bytes += 120  # Python object overhead
            if len(buf.buffer) > 100:
                avg = total_bytes / 100
                total_bytes = int(avg * len(buf.buffer))
        return total_bytes / (1024 * 1024)

    # ============================================================
    # Save / Load
    # ============================================================
    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        os.makedirs(path, exist_ok=True)
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'iteration': self.iteration,
            'total_traversals': self.total_traversals,
            'hidden_dim': self.hidden_dim,
            'obs_dim': OBS_DIM,
            'version': 2,
        }, os.path.join(path, 'deep_cfr_checkpoint.pt'))
        print(f"[DeepCFR v2] Saved to {path}")

    def load(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        ckpt_path = os.path.join(path, 'deep_cfr_checkpoint.pt')
        if not os.path.exists(ckpt_path):
            print(f"[DeepCFR v2] No checkpoint at {ckpt_path}")
            return False
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Version check
        version = ckpt.get('version', 1)
        ckpt_obs_dim = ckpt.get('obs_dim', 34)
        if version < 2 or ckpt_obs_dim != OBS_DIM:
            print(f"[DeepCFR v2] WARNING: Checkpoint is v{version} (obs_dim={ckpt_obs_dim}), "
                  f"current is v2 (obs_dim={OBS_DIM}). Incompatible, starting fresh.")
            return False

        self.regret_net.load_state_dict(ckpt['regret_net'])
        self.strategy_net.load_state_dict(ckpt['strategy_net'])
        self.value_net.load_state_dict(ckpt['value_net'])
        self.regret_optimizer.load_state_dict(ckpt['regret_optimizer'])
        self.strategy_optimizer.load_state_dict(ckpt['strategy_optimizer'])
        self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
        self.iteration = ckpt['iteration']
        self.total_traversals = ckpt['total_traversals']
        self.regret_net.eval()
        self.strategy_net.eval()
        self.value_net.eval()
        print(f"[DeepCFR v2] Loaded from {path} (iter={self.iteration})")
        return True

    # ============================================================
    # Inference
    # ============================================================
    def get_action(
        self, obs: np.ndarray, legal_mask: np.ndarray,
        temperature: float = 0.5,
    ) -> int:
        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(obs_t).cpu().numpy()[0]
        probs = masked_softmax(logits, legal_mask, temperature)
        legal = np.where(legal_mask)[0]
        if len(legal) == 0:
            return PASS_ACTION_ID
        lp = probs[legal]
        lp = lp / (lp.sum() + 1e-10)
        return int(np.random.choice(legal, p=lp))

