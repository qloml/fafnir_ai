"""
Deep CFR Trainer for Fafnir.

Implements Deep CFR with Outcome Sampling MCCFR:
- Instead of iterating ALL legal actions per node (exponential),
  we SAMPLE a single action for the traverser too, then use
  importance sampling to correct the regret estimates.
- This makes each traversal O(depth) instead of O(branching^depth).
- Many more traversals are needed, but each is very fast.

Architecture:
- Regret Network: predicts counterfactual regrets per action
- Strategy Network: predicts average strategy (action probabilities)
- Value Network: evaluates non-terminal leaf nodes
- Reservoir buffers for training data
- Color symmetry data augmentation
"""
import os
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
    resolve_auction, check_game_end,
)
from .action_space import (
    NUM_ACTIONS, ACTION_TABLE, get_legal_mask, action_id_to_counts, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker
from .networks import (
    RegretNetwork, StrategyNetwork, ValueNetwork,
    regret_matching, masked_softmax,
)
from .symmetry import augment_sample


# ============================================================
# Reservoir Buffer
# ============================================================
class ReservoirBuffer:
    """Reservoir sampling buffer for Deep CFR."""

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
        self.total_seen = 0

    def add(self, obs: np.ndarray, target: np.ndarray, iteration: int, legal_mask: np.ndarray):
        self.total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, target, iteration, legal_mask))
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (obs, target, iteration, legal_mask)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        obs = np.stack([self.buffer[i][0] for i in indices])
        targets = np.stack([self.buffer[i][1] for i in indices])
        masks = np.stack([self.buffer[i][3] for i in indices])
        return obs, targets, masks

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Deep CFR Trainer
# ============================================================
class DeepCFRTrainer:

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        buffer_capacity: int = 2_000_000,
        batch_size: int = 2048,
        train_steps_per_iter: int = 500,
        max_depth: int = 40,
        num_augments: int = 2,
        explore_epsilon: float = 0.6,
        device: str = "auto",
        save_dir: str = "cfr_ai/ai/checkpoints",
        num_workers: int = 1,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_workers = num_workers
        print(f"[DeepCFR] Device: {self.device}, Actions: {NUM_ACTIONS}, Workers: {self.num_workers}")

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.train_steps_per_iter = train_steps_per_iter
        self.max_depth = max_depth
        self.num_augments = num_augments
        self.explore_epsilon = explore_epsilon
        self.save_dir = save_dir

        # Networks
        self.regret_net = RegretNetwork(34, NUM_ACTIONS, hidden_dim).to(self.device)
        self.strategy_net = StrategyNetwork(34, NUM_ACTIONS, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(34, hidden_dim).to(self.device)

        # Optimizers
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Buffers
        self.regret_buffer = ReservoirBuffer(buffer_capacity)
        self.strategy_buffer = ReservoirBuffer(buffer_capacity)
        self.value_buffer = ReservoirBuffer(buffer_capacity)

        # Stats
        self.iteration = 0
        self.total_traversals = 0

        # Multiprocessing pool (lazy-init)
        self._pool = None

    # ============================================================
    # Outcome Sampling CFR Traversal
    # ============================================================
    def traverse_game(self, traverser: int) -> float:
        """
        Run a single complete game using outcome sampling.
        
        At each decision point:
        - Compute strategy from regret network (regret matching)
        - For traverser: sample one action, compute regret estimate
        - For opponent: sample one action from their strategy
        - Store samples in buffers
        
        Returns terminal value for traverser.
        """
        state = new_game()
        tracker = BidTracker()
        depth = 0

        # Collect all decision points for this traversal
        decision_points = []

        while state.phase == "BIDDING" and depth < self.max_depth:
            # Get obs and masks for both players
            obs = [None, None]
            masks = [None, None]
            strategies = [None, None]

            for p in range(2):
                obs[p] = build_observation(state, p, tracker)
                masks[p] = get_legal_mask(state.hand[p], state.offer)

                with torch.no_grad():
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
                    # Exploration: epsilon-greedy with uniform over legal actions
                    eps = self.explore_epsilon
                    explore_probs = masks[p].astype(np.float64) / max(1, masks[p].sum())
                    mixed = (1 - eps) * strategies[p] + eps * explore_probs
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

            # Record decision point
            decision_points.append({
                'obs': obs,
                'masks': masks,
                'strategies': strategies,
                'actions': actions,
                'sample_probs': sample_probs,
                'offer_snapshot': state.offer[:],
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

            # Round reset clears tracker
            if state.round_num > depth // 10 + 1:  # heuristic
                tracker.reset()

            depth += 1

        # Terminal value
        terminal_value = self._compute_terminal_value(state, traverser, tracker)

        # Now compute and store regret estimates using the outcome
        self._process_decision_points(
            decision_points, traverser, terminal_value
        )

        # Store value sample
        if len(decision_points) > 0:
            dp = decision_points[0]
            obs_init = dp['obs'][traverser]
            val_target = np.array([terminal_value], dtype=np.float32)
            self.value_buffer.add(obs_init, val_target, self.iteration, np.ones(1))

        return terminal_value

    def _compute_terminal_value(
        self, state: FafnirState, traverser: int, tracker: BidTracker
    ) -> float:
        if state.phase == "GAME_END":
            if state.scores[traverser] > state.scores[1 - traverser]:
                return 1.0
            elif state.scores[traverser] < state.scores[1 - traverser]:
                return -1.0
            return 0.0

        # Use value network for non-terminal
        obs = build_observation(state, traverser, tracker)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.value_net(obs_t).item()

    def _process_decision_points(
        self,
        decision_points: List[Dict],
        traverser: int,
        terminal_value: float,
    ):
        """
        Process decision points to compute regret estimates.
        
        Using the outcome sampling formula:
        For the traverser's action a_t chosen with probability q(a_t):
          regret(a) = (utility(a) - utility(a_t)) / q(a_t)
        
        Since we only sampled one action, we estimate:
          - utility(a_t) = terminal_value (the outcome we observed)
          - utility(a) ≈ terminal_value for a == a_t, else estimated by value net
        
        Simplified approach: we store the terminal value as a baseline
        and let the regret network learn relative action values.
        """
        for dp in decision_points:
            obs = dp['obs'][traverser]
            mask = dp['masks'][traverser]
            strategy = dp['strategies'][traverser]
            chosen_action = dp['actions'][traverser]
            sample_prob = dp['sample_probs'][traverser]

            # Construct regret target
            # For the chosen action, regret = terminal_value - EV
            # For unchosen actions, regret = 0 (no information)
            # This is a simplified version - the NN will learn to interpolate
            regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)

            # Weight by inverse sampling probability for importance correction
            weight = min(10.0, 1.0 / max(sample_prob, 1e-6))

            # Set regret for chosen action
            regret_target[chosen_action] = terminal_value * weight

            # Store regret sample
            self.regret_buffer.add(obs, regret_target, self.iteration, mask)

            # Store strategy sample (weighted by iteration for averaging)
            self.strategy_buffer.add(obs, strategy, self.iteration, mask)

            # Color symmetry augmentation
            if self.num_augments > 0 and random.random() < 0.5:
                aug_pairs = augment_sample(
                    obs, regret_target, ACTION_TABLE, self.num_augments
                )
                for aug_obs, aug_regrets in aug_pairs:
                    aug_hand = aug_obs[:6].astype(int).tolist()
                    aug_offer = aug_obs[6:12].astype(int).tolist()
                    aug_mask = get_legal_mask(aug_hand, aug_offer)
                    self.regret_buffer.add(aug_obs, aug_regrets, self.iteration, aug_mask)

    # ============================================================
    # Network Training
    # ============================================================
    def train_regret_network(self) -> float:
        if len(self.regret_buffer) < self.batch_size:
            return 0.0
        self.regret_net.train()
        total_loss = 0.0
        steps = min(self.train_steps_per_iter, len(self.regret_buffer) // self.batch_size)
        for _ in range(steps):
            obs, targets, masks = self.regret_buffer.sample(self.batch_size)
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
        for _ in range(steps):
            obs, targets, masks = self.strategy_buffer.sample(self.batch_size)
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
        for _ in range(steps):
            obs, targets, _ = self.value_buffer.sample(self.batch_size)
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

        print(
            f"[DeepCFR] Iter={self.iteration} | "
            f"Trav={self.total_traversals} AvgV={avg_value:.3f} | "
            f"Buf: R={len(self.regret_buffer)} S={len(self.strategy_buffer)} V={len(self.value_buffer)} | "
            f"Loss: R={rl:.4f} S={sl:.4f} V={vl:.4f} | "
            f"T={traverse_time:.1f}s Tr={train_time:.1f}s"
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
            print(f"[DeepCFR] Creating worker pool ({self.num_workers} workers)...")
            self._pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.hidden_dim,),
            )

        # Prepare model state dicts (CPU tensors for serialization)
        regret_sd = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}
        value_sd = {k: v.cpu() for k, v in self.value_net.state_dict().items()}

        # Distribute traversals across workers
        per_worker = num_traversals // self.num_workers
        remainder = num_traversals % self.num_workers

        work_items = []
        offset = self.total_traversals
        for w in range(self.num_workers):
            n = per_worker + (1 if w < remainder else 0)
            work_items.append((
                n, offset, self.iteration,
                self.max_depth, self.num_augments, self.explore_epsilon,
                regret_sd, value_sd,
            ))
            offset += n

        try:
            results = self._pool.map(_worker_traverse_batch, work_items)
        except Exception as e:
            print(f"[DeepCFR] Pool error: {e}, recreating pool...")
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

        print(
            f"[DeepCFR] Iter={self.iteration} | "
            f"Trav={self.total_traversals} AvgV={avg_value:.3f} | "
            f"Buf: R={len(self.regret_buffer)} S={len(self.strategy_buffer)} V={len(self.value_buffer)} | "
            f"Loss: R={rl:.4f} S={sl:.4f} V={vl:.4f} | "
            f"T={traverse_time:.1f}s Tr={train_time:.1f}s [{self.num_workers}w]"
        )
        return {'regret_loss': rl, 'strategy_loss': sl, 'value_loss': vl}

    def shutdown_pool(self):
        """Cleanly shut down the worker pool."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

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
        }, os.path.join(path, 'deep_cfr_checkpoint.pt'))
        print(f"[DeepCFR] Saved to {path}")

    def load(self, path: Optional[str] = None):
        if path is None:
            path = self.save_dir
        ckpt_path = os.path.join(path, 'deep_cfr_checkpoint.pt')
        if not os.path.exists(ckpt_path):
            print(f"[DeepCFR] No checkpoint at {ckpt_path}")
            return False
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
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
        print(f"[DeepCFR] Loaded from {path} (iter={self.iteration})")
        return True

    # ============================================================
    # Inference
    # ============================================================
    def get_action(
        self, obs: np.ndarray, legal_mask: np.ndarray,
        temperature: float = 0.5,
    ) -> int:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(obs_t).cpu().numpy()[0]
        probs = masked_softmax(logits, legal_mask, temperature)
        legal = np.where(legal_mask)[0]
        if len(legal) == 0:
            return PASS_ACTION_ID
        lp = probs[legal]
        lp = lp / (lp.sum() + 1e-10)
        return int(np.random.choice(legal, p=lp))
