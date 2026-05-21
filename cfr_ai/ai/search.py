"""
Realtime search for Deep CFR inference.

Performs lightweight rollout-based search at inference time to improve
action quality beyond the raw strategy network output.

Uses the strategy network as a prior, then runs short Monte Carlo rollouts
to refine the action selection.

Usage:
    Used by cfr_bot.py with --search flag (optional).
"""
import numpy as np
import torch
import time
from typing import List, Optional

from .game_engine import (
    FafnirState, new_game, step_auction, NUM_COLORS,
    compute_hand_score, is_trash_limit_reached,
    should_force_round_end_by_bag,
)
from .action_space import (
    NUM_ACTIONS, get_legal_mask, action_id_to_counts, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker, OBS_DIM
from .networks import (
    StrategyNetwork, ValueNetwork, masked_softmax, regret_matching,
)


class RealtimeSearch:
    """
    Lightweight rollout-based search at inference time.

    For each legal action, performs N rollouts (using the strategy network
    for both players' remaining moves) and estimates the action's value.

    Final action selection combines:
    - Strategy network prior (baseline policy)
    - Rollout-based value estimates

    Parameters:
    - num_rollouts: total rollouts to distribute across actions
    - max_depth: max depth per rollout
    - time_limit_ms: maximum time for search (in milliseconds)
    """

    def __init__(
        self,
        strategy_net: torch.nn.Module,
        value_net: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        num_rollouts: int = 50,
        max_depth: int = 20,
        time_limit_ms: float = 200.0,
        temperature: float = 0.3,
    ):
        self.strategy_net = strategy_net
        self.value_net = value_net
        self.device = device
        self.num_rollouts = num_rollouts
        self.max_depth = max_depth
        self.time_limit_ms = time_limit_ms
        self.temperature = temperature

    def search_action(
        self,
        state: FafnirState,
        player: int,
        tracker: BidTracker,
    ) -> int:
        """
        Search for the best action from the given state.

        Returns the chosen action_id.
        """
        mask = get_legal_mask(state.hand[player], state.offer)
        legal = np.where(mask)[0]

        if len(legal) == 0:
            return PASS_ACTION_ID
        if len(legal) == 1:
            return int(legal[0])

        # Get strategy network prior
        obs = build_observation(state, player, tracker)
        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.strategy_net(obs_t).cpu().numpy()[0]
        prior_probs = masked_softmax(logits, mask, self.temperature)

        # Allocate rollouts proportionally to prior probability
        # But ensure at least 1 rollout per action (for top-K actions)
        top_k = min(len(legal), 10)  # Focus on top 10 actions
        top_actions = legal[np.argsort(-prior_probs[legal])[:top_k]]

        rollouts_per_action = {}
        remaining = self.num_rollouts
        for a in top_actions:
            n = max(1, int(prior_probs[a] * self.num_rollouts))
            rollouts_per_action[a] = n
            remaining -= n

        # Distribute remaining rollouts to top action
        if remaining > 0:
            rollouts_per_action[top_actions[0]] += remaining

        # Run rollouts with time limit
        start_time = time.perf_counter()
        action_values = {}

        for action_id, n_rollouts in rollouts_per_action.items():
            values = []
            for _ in range(n_rollouts):
                # Check time limit
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms > self.time_limit_ms:
                    break

                v = self._rollout(state, player, action_id, tracker)
                values.append(v)

            if values:
                action_values[action_id] = np.mean(values)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.time_limit_ms:
                break

        if not action_values:
            # Fallback to strategy network
            lp = prior_probs[legal]
            lp = lp / (lp.sum() + 1e-10)
            return int(np.random.choice(legal, p=lp))

        # Combine prior with rollout values
        # Score = prior_prob * α + rollout_value * (1 - α)
        alpha = 0.3  # Weight for prior
        combined_scores = {}
        for a in action_values:
            combined_scores[a] = alpha * prior_probs[a] + (1 - alpha) * (action_values[a] + 1) / 2

        # Select best action (with small temperature for diversity)
        best_action = max(combined_scores, key=combined_scores.get)
        return int(best_action)

    def _rollout(
        self,
        state: FafnirState,
        player: int,
        first_action_id: int,
        tracker: BidTracker,
    ) -> float:
        """Run a single rollout from state with first_action forced."""
        sim_state = state.clone()
        sim_tracker = BidTracker()
        # Copy tracker state
        sim_tracker.confirmed = [tracker.confirmed[0][:], tracker.confirmed[1][:]]

        opp = 1 - player
        initial_round = sim_state.round_num
        initial_scores = sim_state.scores[:]

        # First move: force the specified action for player
        opp_mask = get_legal_mask(sim_state.hand[opp], sim_state.offer)
        opp_legal = np.where(opp_mask)[0]
        if len(opp_legal) == 0:
            opp_action_id = PASS_ACTION_ID
        else:
            opp_obs = build_observation(sim_state, opp, sim_tracker)
            with torch.inference_mode():
                obs_t = torch.tensor(opp_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.strategy_net(obs_t).cpu().numpy()[0]
            probs = masked_softmax(logits, opp_mask, self.temperature)
            lp = probs[opp_legal]
            lp = lp / (lp.sum() + 1e-10)
            opp_action_id = int(np.random.choice(opp_legal, p=lp))

        # Execute first step
        if player == 0:
            bid0 = action_id_to_counts(first_action_id)
            bid1 = action_id_to_counts(opp_action_id)
        else:
            bid0 = action_id_to_counts(opp_action_id)
            bid1 = action_id_to_counts(first_action_id)

        step_auction(sim_state, bid0, bid1)

        # Continue rollout using strategy network for both players
        depth = 1
        while (sim_state.phase == "BIDDING" and
               depth < self.max_depth and
               sim_state.round_num == initial_round):

            action_ids = [0, 0]
            for p in range(2):
                mask = get_legal_mask(sim_state.hand[p], sim_state.offer)
                legal = np.where(mask)[0]
                if len(legal) == 0:
                    action_ids[p] = PASS_ACTION_ID
                    continue

                obs = build_observation(sim_state, p, sim_tracker)
                with torch.inference_mode():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits = self.strategy_net(obs_t).cpu().numpy()[0]
                probs = masked_softmax(logits, mask, self.temperature)
                lp = probs[legal]
                lp = lp / (lp.sum() + 1e-10)
                action_ids[p] = int(np.random.choice(legal, p=lp))

            bid0 = action_id_to_counts(action_ids[0])
            bid1 = action_id_to_counts(action_ids[1])
            step_auction(sim_state, bid0, bid1)
            depth += 1

        # Compute value
        if sim_state.round_num > initial_round or sim_state.phase == "GAME_END":
            gained = (sim_state.scores[player] - initial_scores[player]) - \
                     (sim_state.scores[opp] - initial_scores[opp])
        else:
            auction_diff = (sim_state.scores[player] - initial_scores[player]) - \
                           (sim_state.scores[opp] - initial_scores[opp])
            hand_diff = compute_hand_score(sim_state, player) - compute_hand_score(sim_state, opp)
            gained = auction_diff + hand_diff

        return max(-1.0, min(1.0, gained / 20.0))
