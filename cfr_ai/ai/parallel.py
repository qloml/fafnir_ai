"""
Parallel traversal worker for Deep CFR (v2).

Uses multiprocessing to distribute game traversals across CPU cores.
Each worker initializes its network architecture once (via pool initializer),
then receives updated weights each iteration via work arguments.

v2 changes:
- Correct regret estimation for all legal actions
- Score randomization support
- Adaptive exploration support
- OBS_DIM = 42

Designed for Windows (spawn-based multiprocessing).
"""
import numpy as np
import torch
import random
import signal
from typing import List, Tuple, Dict, Any, Optional

from .game_engine import (
    FafnirState, new_game, step_auction, NUM_COLORS,
    compute_hand_score, clamp_score, is_trash_limit_reached,
    should_force_round_end_by_bag, setup_offer, do_round_end,
    resolve_auction, check_game_end, SCORE_TO_WIN, determine_auction_winner,
)
from .action_space import (
    NUM_ACTIONS, ACTION_TABLE, get_legal_mask, action_id_to_counts, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker, OBS_DIM
from .networks import (
    RegretNetwork, StrategyNetwork, ValueNetwork,
    regret_matching,
)
from .symmetry import augment_sample_sparse

_w_regret_net = None
_w_value_net = None
_w_opp_regret_net = None
_w_hidden_dim = None


def _worker_init(hidden_dim: int):
    """
    Initialize worker-local network architecture (called once per worker process).
    Weights will be loaded per-batch from work arguments.
    """
    global _w_regret_net, _w_value_net, _w_opp_regret_net, _w_hidden_dim

    _w_hidden_dim = hidden_dim

    # Ignore Ctrl+C in workers. Let the main process handle the shutdown cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Use single thread per worker to avoid oversubscription
    torch.set_num_threads(1)

    _w_regret_net = RegretNetwork(OBS_DIM, NUM_ACTIONS, hidden_dim)
    _w_regret_net.eval()

    _w_value_net = ValueNetwork(OBS_DIM, hidden_dim)
    _w_value_net.eval()

    _w_opp_regret_net = RegretNetwork(OBS_DIM, NUM_ACTIONS, hidden_dim)
    _w_opp_regret_net.eval()


def _worker_traverse_batch(args: Tuple) -> Dict[str, Any]:
    """
    Run a batch of traversals in this worker process.

    Args:
        args: (num_traversals, start_traversal_id, iteration,
               max_depth, num_augments, explore_epsilon,
               baseline, score_randomize,
               regret_state_dict, value_state_dict, opponent_regret_state_dict)

    Returns:
        dict with collected samples and stats.
    """
    (num_traversals, start_traversal_id, iteration,
     max_depth, num_augments, explore_epsilon,
     baseline, score_randomize,
     regret_sd, value_sd, opponent_sd) = args

    # Load updated weights for this iteration
    _w_regret_net.load_state_dict(regret_sd)
    _w_regret_net.eval()
    _w_value_net.load_state_dict(value_sd)
    _w_value_net.eval()
    if opponent_sd is not None:
        _w_opp_regret_net.load_state_dict(opponent_sd)
        _w_opp_regret_net.eval()
        opponent_net = _w_opp_regret_net
    else:
        opponent_net = None

    regret_samples = []
    strategy_samples = []
    value_samples = []
    total_value = 0.0

    for i in range(num_traversals):
        traverser = (start_traversal_id + i) % 2
        value, r_samps, s_samps, v_samps = _single_traverse(
            traverser, iteration, max_depth, num_augments, explore_epsilon,
            baseline, score_randomize, opponent_net
        )
        total_value += value
        regret_samples.extend(r_samps)
        strategy_samples.extend(s_samps)
        value_samples.extend(v_samps)

    return {
        'total_value': total_value,
        'regret_samples': regret_samples,
        'strategy_samples': strategy_samples,
        'value_samples': value_samples,
        'num_traversals': num_traversals,
    }


def _single_traverse(
    traverser: int,
    iteration: int,
    max_depth: int,
    num_augments: int,
    explore_epsilon: float,
    baseline: float = 0.0,
    score_randomize: bool = True,
    opponent_net=None,
) -> Tuple[float, list, list, list]:
    """
    Run a single game traversal. Same logic as DeepCFRTrainer.traverse_game
    but uses worker-local models and returns samples instead of storing them.

    v2: Correct regret estimation + score randomization.
    """
    state = new_game()

    # Score randomization
    if score_randomize and random.random() < 0.5:
        state.scores[0] = random.randint(0, 990)
        state.scores[1] = random.randint(0, 990)

    tracker = BidTracker()
    depth = 0
    initial_round = state.round_num
    initial_scores = state.scores[:]

    decision_points = []

    while state.phase == "BIDDING" and depth < max_depth and state.round_num == initial_round:
        obs = [None, None]
        masks = [None, None]
        strategies = [None, None]

        regrets_per_player = [None, None]

        for p in range(2):
            obs[p] = build_observation(state, p, tracker)
            masks[p] = get_legal_mask(state.hand[p], state.offer)

            with torch.inference_mode():
                obs_t = torch.tensor(obs[p], dtype=torch.float32).unsqueeze(0)
                net = opponent_net if (p != traverser and opponent_net is not None) else _w_regret_net
                regrets_per_player[p] = net(obs_t).numpy()[0]
            strategies[p] = regret_matching(regrets_per_player[p], masks[p])

        actions = [0, 0]
        sample_probs = [1.0, 1.0]

        for p in range(2):
            legal = np.where(masks[p])[0]
            if len(legal) == 0:
                actions[p] = PASS_ACTION_ID
                continue

            if p == traverser:
                # Regret-based pruning: skip clearly bad actions
                if iteration > 100 and len(legal) > 3:
                    regret_vals = regrets_per_player[traverser][legal]
                    max_r = regret_vals.max()
                    threshold = max(0.0, max_r * 0.1)  # relative threshold
                    keep = regret_vals >= threshold
                    if keep.sum() >= 2:
                        legal = legal[keep]

                eps = explore_epsilon
                explore_probs = masks[p].astype(np.float32) / max(1, masks[p].sum())
                mixed = (1 - eps) * strategies[p] + eps * explore_probs
                mixed_legal = mixed[legal]
                mixed_legal = mixed_legal / (mixed_legal.sum() + 1e-10)
                chosen_idx = np.random.choice(len(legal), p=mixed_legal)
                actions[p] = legal[chosen_idx]
                sample_probs[p] = mixed[actions[p]]
            else:
                strat_legal = strategies[p][legal]
                strat_legal = strat_legal / (strat_legal.sum() + 1e-10)
                chosen_idx = np.random.choice(len(legal), p=strat_legal)
                actions[p] = legal[chosen_idx]
                sample_probs[p] = strategies[p][actions[p]]

        decision_points.append({
            'obs': obs,
            'masks': masks,
            'strategies': strategies,
            'actions': actions,
            'sample_probs': sample_probs,
            'offer_snapshot': state.offer[:],
            'scores_before': state.scores[:],
        })

        bid0 = action_id_to_counts(actions[0])
        bid1 = action_id_to_counts(actions[1])
        old_offer = state.offer[:]
        old_caretaker = state.caretaker
        winner = determine_auction_winner(bid0, bid1, old_caretaker)
        step_auction(state, bid0, bid1)

        if winner is not None:
            loser = 1 - winner
            bid_w = bid0 if winner == 0 else bid1
            bid_l = bid0 if loser == 0 else bid1
            tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

        depth += 1

    # === Dense Reward Shaping: per-decision-point effective values ===
    opp = 1 - traverser
    if state.round_num > initial_round:
        final_t = state.scores[traverser]
        final_o = state.scores[opp]
    else:
        final_t = state.scores[traverser] + compute_hand_score(state, traverser)
        final_o = state.scores[opp] + compute_hand_score(state, opp)

    effective_values = []
    for dp in decision_points:
        sb = dp['scores_before']
        gained_from_here = (final_t - sb[traverser]) - (final_o - sb[opp])
        effective_values.append(max(-1.0, min(1.0, gained_from_here / 50.0)))

    # Process decision points with per-point values
    regret_samples, strategy_samples, value_samples = _process_decision_points(
        decision_points, traverser, effective_values, iteration, num_augments
    )

    total_value = effective_values[0] if effective_values else 0.0
    return total_value, regret_samples, strategy_samples, value_samples


def _win_probability(my_score: float, opp_score: float) -> float:
    """ゲーム勝利確率の推定（残りポイント比率ベース）。"""
    if my_score >= SCORE_TO_WIN:
        return 1.0
    if opp_score >= SCORE_TO_WIN:
        return 0.0
    my_remaining = max(1.0, SCORE_TO_WIN - my_score)
    opp_remaining = max(1.0, SCORE_TO_WIN - opp_score)
    return opp_remaining / (my_remaining + opp_remaining)


def _compute_terminal_value(
    state: FafnirState, traverser: int,
    initial_round: int, initial_scores: list,
) -> float:
    """1ラウンドでの獲得スコア差を報酬として返す。"""
    opp = 1 - traverser

    if state.round_num > initial_round:
        final_t = state.scores[traverser]
        final_o = state.scores[opp]
    else:
        final_t = state.scores[traverser] + compute_hand_score(state, traverser)
        final_o = state.scores[opp] + compute_hand_score(state, opp)

    gained = (final_t - initial_scores[traverser]) - \
             (final_o - initial_scores[opp])

    return max(-1.0, min(1.0, gained / 50.0))


def _process_decision_points(
    decision_points: list,
    traverser: int,
    effective_values: list,
    iteration: int,
    num_augments: int,
) -> Tuple[list, list, list]:
    """Returns (regret_samples, strategy_samples, value_samples).

    v3: Dense Reward Shaping with per-decision-point values.
    Uses Value Network baseline for variance reduction.
    """
    regret_samples = []
    strategy_samples = []
    value_samples = []

    for i, dp in enumerate(decision_points):
        obs = dp['obs'][traverser]
        mask = dp['masks'][traverser]
        strategy = dp['strategies'][traverser]
        chosen_action = dp['actions'][traverser]
        sample_prob = dp['sample_probs'][traverser]

        # Per-point effective value
        point_value = effective_values[i]

        # Value Network baseline
        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            baseline = _w_value_net(obs_t).item()

        # Store value training sample
        value_samples.append((obs, np.array([point_value], dtype=np.float32), iteration))

        # Importance weight with relaxed clamping (10 → 50)
        weight = min(50.0, 1.0 / max(sample_prob, 1e-6))
        adjusted_value = (point_value - baseline) * weight

        # Compute regret for ALL legal actions
        legal_actions = np.where(mask)[0]
        regret_pairs = []
        for a in legal_actions:
            if a == chosen_action:
                regret_a = (1.0 - strategy[a]) * adjusted_value
            else:
                regret_a = -strategy[a] * adjusted_value
            regret_pairs.append([a, regret_a])

        sparse_regret = np.array(regret_pairs, dtype=np.float32)
        regret_samples.append((obs, sparse_regret, iteration))

        # Store sparse strategy
        nonzero_strats = np.nonzero(strategy)[0]
        sparse_strat = np.zeros((len(nonzero_strats), 2), dtype=np.float32)
        sparse_strat[:, 0] = nonzero_strats
        sparse_strat[:, 1] = strategy[nonzero_strats]
        strategy_samples.append((obs, sparse_strat, iteration))

        # Augmentation
        if num_augments > 0 and random.random() < 0.5:
            chosen_regret = (1.0 - strategy[chosen_action]) * adjusted_value
            aug_triples = augment_sample_sparse(
                obs, chosen_action, chosen_regret, ACTION_TABLE, num_augments
            )
            for aug_obs, aug_aid, aug_val in aug_triples:
                aug_sparse = np.array([[aug_aid, aug_val]], dtype=np.float32)
                regret_samples.append((aug_obs, aug_sparse, iteration))

    return regret_samples, strategy_samples, value_samples
