"""
Deep CFR Trainer for Fafnir (v2).

Major improvements over v1:
- Correct Outcome Sampling MCCFR regret estimation (all legal actions)
- DCFR (Discounted CFR) weighting scheme
- Optional score randomization, disabled by default because scores are not observed
- Adaptive exploration (epsilon decay)
- Observation space: 33 dimensions, with game progress and score context removed
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
import glob
import csv
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
    resolve_auction, check_game_end, SCORE_TO_WIN, determine_auction_winner,
)
from .action_space import (
    NUM_ACTIONS, ACTION_TABLE, get_legal_mask, action_id_to_counts_np, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker, OBS_DIM
from .networks import (
    RegretNetwork, StrategyNetwork, ValueNetwork,
    regret_matching, masked_softmax,
)
from .symmetry import augment_sample, augment_sample_sparse


PROGRAM_VERSION = 2
LATEST_CHECKPOINT_NAME = "deep_cfr_checkpoint.pt"


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
        self.buffer: List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]] = []
        self._iterations = np.empty(capacity, dtype=np.int32)
        self.total_seen = 0
        self._needs_mask = True  # whether to reconstruct masks on sample
        self._sparse_target = False  # if True, target is (action_id, value) pair
        self._sparse_strategy = False  # if True, target is list of [action_id, prob]
        self._sparse_regret_multi = False  # if True, target is list of [action_id, regret_value]

    def add(self, obs: np.ndarray, target: np.ndarray, iteration: int, legal_mask: np.ndarray = None):
        """Add a sample, optionally caching its legal-action mask."""
        self.total_seen += 1
        cached_mask = None
        if legal_mask is not None:
            cached_mask = np.packbits(legal_mask.astype(np.bool_, copy=False))
        item = (obs, target, iteration, cached_mask)
        if len(self.buffer) < self.capacity:
            write_idx = len(self.buffer)
            self.buffer.append(item)
            self._iterations[write_idx] = iteration
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = item
                self._iterations[idx] = iteration

    def sample_batches(self, batch_size: int, num_batches: int, num_actions: int = 0, current_iteration: int = 0):
        total_samples = batch_size * num_batches
        n_avail = len(self.buffer)
        if n_avail == 0 or total_samples == 0:
            return

        # DCFR weighting: newer iterations get higher weight
        if current_iteration > 1:
            iters = self._iterations[:n_avail].astype(np.float64, copy=True)
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
                cached_masks = [
                    self.buffer[idx][3] if len(self.buffer[idx]) > 3 else None
                    for idx in indices
                ]
                if all(mask is not None for mask in cached_masks):
                    masks = np.unpackbits(
                        np.stack(cached_masks), axis=1, count=num_actions
                    ).astype(np.float32, copy=False)
                else:
                    masks = []
                    for idx, cached_mask in zip(indices, cached_masks):
                        item = self.buffer[idx]
                        if cached_mask is not None:
                            masks.append(np.unpackbits(cached_mask, count=num_actions).astype(np.float32))
                        else:
                            masks.append(get_legal_mask(
                                item[0][:6].astype(int).tolist(),
                                item[0][6:12].astype(int).tolist(),
                            ).astype(np.float32, copy=False))
                    masks = np.stack(masks)
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
        lr: float = 5e-4,
        buffer_capacity: int = 2_000_000,
        batch_size: int = 2048,
        train_steps_per_iter: int = 150,
        max_depth: int = 50,
        num_augments: int = 3,
        explore_epsilon: float = 0.3,
        device: str = "auto",
        save_dir: str = "cfr_ai/ai/checkpoints",
        num_workers: int = 1,
        # DCFR parameters
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.5,
        dcfr_gamma: float = 2.0,
        # Score randomization
        score_randomize: bool = False,
        target_mode: str = "terminal",
        # Past-self opponent mixing
        program_version: int = PROGRAM_VERSION,
        past_opponent_prob: float = 0.0,
        max_past_opponents: int = 8,
        past_opponent_selection: str = "recent",
        past_opponent_manifest: Optional[str] = None,
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
        self.program_version = program_version
        self.past_opponent_prob = max(0.0, min(1.0, past_opponent_prob))
        self.max_past_opponents = max(0, max_past_opponents)
        self.past_opponent_selection = past_opponent_selection
        self.past_opponent_manifest = past_opponent_manifest

        # DCFR parameters
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma

        # Score randomization
        self.score_randomize = score_randomize
        if target_mode not in ("terminal", "dense"):
            raise ValueError(f"target_mode must be 'terminal' or 'dense', got {target_mode!r}")
        self.target_mode = target_mode

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

        # Baseline EMA for stats tracking (value net does the real baseline)
        self._baseline = 0.0
        self._baseline_alpha = 0.005

        # Multiprocessing pool (lazy-init)
        self._pool = None

        # Past-self opponent snapshots. These are frozen regret networks used
        # only for non-traverser action sampling.
        self.past_opponent_states: List[Dict[str, torch.Tensor]] = []
        self._past_opponent_net: Optional[RegretNetwork] = None
        self._past_opponent_loaded_idx: Optional[int] = None
        self._archive_paths_by_iteration: Dict[int, str] = {}

    # ============================================================
    # Adaptive Exploration
    # ============================================================
    def _get_epsilon(self) -> float:
        """Adaptive exploration: high initial exploration, decaying over time."""
        base = self.explore_epsilon  # default 0.3
        return max(0.05, base * (0.998 ** self.iteration))

    # ============================================================
    # Score Randomization
    # ============================================================
    def _randomize_scores(self, state: FafnirState):
        """Optionally inject random starting scores.

        This is disabled by default because current observations intentionally
        omit scores. Enabling it makes scores a hidden variable that can affect
        rewards and game end.
        """
        if not self.score_randomize:
            return

        if random.random() < 0.5:
            return  # Keep default (0, 0)

        # Random scores: both players get independent random scores (0~990)
        state.scores[0] = random.randint(0, 990)
        state.scores[1] = random.randint(0, 990)

    # ============================================================
    # Past-self opponent pool
    # ============================================================
    def _archive_stem(self, iteration: Optional[int] = None) -> str:
        it = self.iteration if iteration is None else iteration
        return f"deep_cfr_checkpoint_v{self.program_version}_iter{it:06d}"

    def _archive_filename(self, iteration: Optional[int] = None, duplicate_index: int = 0) -> str:
        stem = self._archive_stem(iteration)
        run_index = duplicate_index if duplicate_index > 0 else 1
        return f"{stem}_run{run_index:02d}.pt"

    def _next_archive_path(self, path: str, iteration: Optional[int] = None) -> str:
        legacy_path = os.path.join(path, f"{self._archive_stem(iteration)}.pt")
        archive_path = os.path.join(path, self._archive_filename(iteration))
        if not os.path.exists(archive_path) and not os.path.exists(legacy_path):
            return archive_path
        for duplicate_index in range(2, 10000):
            archive_path = os.path.join(path, self._archive_filename(iteration, duplicate_index))
            if not os.path.exists(archive_path):
                return archive_path
        raise RuntimeError(f"Could not find an unused archive name for {self._archive_stem(iteration)}")

    def _cpu_regret_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.regret_net.state_dict().items()}

    def add_current_to_past_opponents(self):
        """Add the current regret network to the frozen past-opponent pool."""
        if self.max_past_opponents <= 0:
            return
        self.past_opponent_states.append(self._cpu_regret_state())
        if len(self.past_opponent_states) > self.max_past_opponents:
            self.past_opponent_states = self.past_opponent_states[-self.max_past_opponents:]
        self._past_opponent_loaded_idx = None
        print(
            f"[DeepCFR v2] Past-opponent pool size: "
            f"{len(self.past_opponent_states)}/{self.max_past_opponents}"
        )

    def load_past_opponents_from_dir(self, path: Optional[str] = None):
        """Load versioned archived checkpoints as frozen opponents."""
        if self.max_past_opponents <= 0:
            return
        if path is None:
            path = self.save_dir
        candidates = self._select_past_opponent_paths(path)
        source = self.past_opponent_manifest if self.past_opponent_manifest else path
        loaded = 0
        for ckpt_path in candidates:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ckpt.get("hidden_dim", self.hidden_dim) != self.hidden_dim:
                    continue
                if ckpt.get("obs_dim", OBS_DIM) != OBS_DIM:
                    continue
                if "regret_net" not in ckpt:
                    continue
                state = {k: v.detach().cpu().clone() for k, v in ckpt["regret_net"].items()}
                self.past_opponent_states.append(state)
                loaded += 1
            except Exception as e:
                print(f"[DeepCFR v2] Skip past opponent {ckpt_path}: {e}")
        if len(self.past_opponent_states) > self.max_past_opponents:
            self.past_opponent_states = self.past_opponent_states[-self.max_past_opponents:]
        if loaded:
            print(f"[DeepCFR v2] Loaded {loaded} past opponents from {source}")

    def _select_past_opponent_paths(self, path: str) -> List[str]:
        if self.past_opponent_selection == "manifest" or self.past_opponent_manifest:
            candidates = self._read_past_opponent_manifest()
        else:
            pattern = os.path.join(path, f"deep_cfr_checkpoint_v{self.program_version}_iter*.pt")
            candidates = sorted(glob.glob(pattern))

        candidates = [p for p in candidates if p and os.path.exists(p)]
        if not candidates:
            return []

        limit = self.max_past_opponents
        mode = self.past_opponent_selection
        if mode == "manifest":
            return candidates[:limit]
        if mode == "spread":
            if len(candidates) <= limit:
                return candidates
            idxs = np.linspace(0, len(candidates) - 1, num=limit, dtype=int)
            return [candidates[int(i)] for i in idxs]
        if mode == "random":
            if len(candidates) <= limit:
                return candidates
            return sorted(random.sample(candidates, limit))
        return candidates[-limit:]

    def _read_past_opponent_manifest(self) -> List[str]:
        if not self.past_opponent_manifest:
            return []
        if not os.path.exists(self.past_opponent_manifest):
            print(f"[DeepCFR v2] Past-opponent manifest not found: {self.past_opponent_manifest}")
            return []

        base_dir = os.path.dirname(os.path.abspath(self.past_opponent_manifest))
        paths: List[str] = []
        with open(self.past_opponent_manifest, "r", encoding="utf-8") as f:
            first = f.readline()
            f.seek(0)
            if "," in first and "checkpoint" in first:
                for row in csv.DictReader(f):
                    value = row.get("checkpoint") or row.get("path") or ""
                    paths.append(value.strip())
            else:
                for line in f:
                    value = line.strip()
                    if not value or value.startswith("#"):
                        continue
                    paths.append(value.split(",", 1)[0].strip())

        resolved = []
        for p in paths:
            if os.path.isabs(p):
                resolved.append(p)
            else:
                resolved.append(os.path.normpath(os.path.join(base_dir, p)))
        return resolved

    def _sample_past_opponent_net(self) -> Optional[RegretNetwork]:
        if not self.past_opponent_states:
            return None
        if self.past_opponent_prob <= 0.0 or random.random() >= self.past_opponent_prob:
            return None

        idx = random.randrange(len(self.past_opponent_states))
        if self._past_opponent_net is None:
            self._past_opponent_net = RegretNetwork(OBS_DIM, NUM_ACTIONS, self.hidden_dim).to(self.device)
            self._past_opponent_net.eval()
        if self._past_opponent_loaded_idx != idx:
            self._past_opponent_net.load_state_dict(self.past_opponent_states[idx])
            self._past_opponent_net.to(self.device)
            self._past_opponent_net.eval()
            self._past_opponent_loaded_idx = idx
        return self._past_opponent_net

    def _sample_past_opponent_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.past_opponent_states:
            return None
        if self.past_opponent_prob <= 0.0 or random.random() >= self.past_opponent_prob:
            return None
        return random.choice(self.past_opponent_states)

    # ============================================================
    # Outcome Sampling CFR Traversal (v2: correct regret estimation)
    # ============================================================
    def traverse_game(self, traverser: int) -> float:
        """
        Run a single complete game using outcome sampling.

        v2 improvements:
        - Correct regret estimation for ALL legal actions (not just the chosen one)
        - Adaptive epsilon exploration
        - Optional score randomization support

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
        initial_scores = state.scores.copy()

        # Collect all decision points for this traversal
        decision_points = []

        epsilon = self._get_epsilon()
        past_opponent_net = self._sample_past_opponent_net()

        while state.phase == "BIDDING" and depth < self.max_depth and state.round_num == initial_round:
            # Get obs and masks for both players
            obs = [None, None]
            masks = [None, None]
            strategies = [None, None]
            regrets_per_player = [None, None]

            for p in range(2):
                obs[p] = build_observation(state, p, tracker)
                masks[p] = get_legal_mask(state.hand[p], state.offer)

            if past_opponent_net is None:
                with torch.inference_mode():
                    obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
                    regrets_batch = self.regret_net(obs_t).cpu().numpy()
                regrets_per_player[0] = regrets_batch[0]
                regrets_per_player[1] = regrets_batch[1]
            else:
                for p in range(2):
                    with torch.inference_mode():
                        obs_t = torch.as_tensor(obs[p], dtype=torch.float32, device=self.device).unsqueeze(0)
                        net = past_opponent_net if p != traverser else self.regret_net
                        regrets_per_player[p] = net(obs_t).cpu().numpy()[0]

            for p in range(2):
                regrets = regrets_per_player[p]
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
                        regret_vals = regrets_per_player[traverser][legal]
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
                'offer_snapshot': state.offer.copy(),
                'scores_before': state.scores.copy(),
            })

            # Execute auction
            bid0 = action_id_to_counts_np(actions[0])
            bid1 = action_id_to_counts_np(actions[1])

            # Update tracker before step
            old_offer = state.offer.copy()
            old_caretaker = state.caretaker
            winner = determine_auction_winner(bid0, bid1, old_caretaker)
            step_auction(state, bid0, bid1)

            # Update bid tracker
            if winner is not None:
                loser = 1 - winner
                bid_w = bid0 if winner == 0 else bid1
                bid_l = bid0 if loser == 0 else bid1
                tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

            depth += 1

        # === Round target values ===
        opp = 1 - traverser
        if state.round_num > initial_round:
            # Round completed — do_round_end() already added hand scores
            final_t = state.scores[traverser]
            final_o = state.scores[opp]
        else:
            # GAME_END or depth limit — manually add hand scores
            final_t = state.scores[traverser] + compute_hand_score(state, traverser)
            final_o = state.scores[opp] + compute_hand_score(state, opp)

        terminal_gained = (final_t - initial_scores[traverser]) - (final_o - initial_scores[opp])
        terminal_value = max(-1.0, min(1.0, terminal_gained / 50.0))
        if self.target_mode == "dense":
            effective_values = []
            for dp in decision_points:
                sb = dp['scores_before']
                gained_from_here = (final_t - sb[traverser]) - (final_o - sb[opp])
                effective_values.append(max(-1.0, min(1.0, gained_from_here / 50.0)))
        else:
            effective_values = [terminal_value] * len(decision_points)

        # Process with either pure terminal targets or per-point dense targets.
        self._process_decision_points(
            decision_points, traverser, effective_values
        )

        total_value = terminal_value if decision_points else 0.0
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
        if not decision_points:
            return

        baseline_obs = np.stack([dp['obs'][traverser] for dp in decision_points])
        with torch.inference_mode():
            obs_t = torch.as_tensor(baseline_obs, dtype=torch.float32, device=self.device)
            baselines = self.value_net(obs_t).detach().cpu().numpy()

        for i, dp in enumerate(decision_points):
            obs = dp['obs'][traverser]
            mask = dp['masks'][traverser]
            strategy = dp['strategies'][traverser]
            chosen_action = dp['actions'][traverser]
            sample_prob = dp['sample_probs'][traverser]

            # Per-point effective value (Dense Reward Shaping)
            point_value = effective_values[i]

            # Value Network baseline (improved variance reduction)
            baseline = float(baselines[i])

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
            self.regret_buffer.add(obs, sparse_regret, self.iteration, mask)

            # Store strategy sample as sparse format
            nonzero_strats = np.nonzero(strategy)[0]
            sparse_strat = np.zeros((len(nonzero_strats), 2), dtype=np.float32)
            sparse_strat[:, 0] = nonzero_strats
            sparse_strat[:, 1] = strategy[nonzero_strats]
            self.strategy_buffer.add(obs, sparse_strat, self.iteration, mask)

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
            opponent_sd = self._sample_past_opponent_state_dict()
            work_items.append((
                n, offset, self.iteration,
                self.max_depth, self.num_augments, epsilon,
                self._baseline,
                self.score_randomize,
                self.target_mode,
                regret_sd, value_sd,
                opponent_sd,
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
    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
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
            'program_version': self.program_version,
        }

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self._checkpoint_payload(), os.path.join(path, LATEST_CHECKPOINT_NAME))
        print(f"[DeepCFR v2] Saved to {path}")

    def save_archive(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        os.makedirs(path, exist_ok=True)
        cached_path = self._archive_paths_by_iteration.get(self.iteration)
        if cached_path and os.path.exists(cached_path):
            print(f"[DeepCFR v2] Archive already saved for iter={self.iteration}: {cached_path}")
            return cached_path
        archive_path = self._next_archive_path(path)
        torch.save(self._checkpoint_payload(), archive_path)
        self._archive_paths_by_iteration[self.iteration] = archive_path
        print(f"[DeepCFR v2] Archived checkpoint: {archive_path}")
        return archive_path

    def load(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        ckpt_path = os.path.join(path, LATEST_CHECKPOINT_NAME)
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

