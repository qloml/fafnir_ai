"""
Parallel traversal worker for Deep CFR.

Uses multiprocessing to distribute game traversals across CPU cores.
Each worker initializes its network architecture once (via pool initializer),
then receives updated weights each iteration via work arguments.

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
    resolve_auction, check_game_end,
)
from .action_space import (
    NUM_ACTIONS, ACTION_TABLE, get_legal_mask, action_id_to_counts, PASS_ACTION_ID,
)
from .observation import build_observation, BidTracker
from .networks import (
    RegretNetwork, StrategyNetwork, ValueNetwork,
    regret_matching,
)
from .symmetry import augment_sample_sparse

_w_regret_net = None
_w_value_net = None
_w_hidden_dim = None


def _worker_init(hidden_dim: int):
    """
    Initialize worker-local network architecture (called once per worker process).
    Weights will be loaded per-batch from work arguments.
    """
    global _w_regret_net, _w_value_net, _w_hidden_dim

    _w_hidden_dim = hidden_dim

    # Ignore Ctrl+C in workers. Let the main process handle the shutdown cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Use single thread per worker to avoid oversubscription
    torch.set_num_threads(1)

    _w_regret_net = RegretNetwork(34, NUM_ACTIONS, hidden_dim)
    _w_regret_net.eval()

    _w_value_net = ValueNetwork(34, hidden_dim)
    _w_value_net.eval()


def _worker_traverse_batch(args: Tuple) -> Dict[str, Any]:
    """
    Run a batch of traversals in this worker process.

    Args:
        args: (num_traversals, start_traversal_id, iteration,
               max_depth, num_augments, explore_epsilon,
               regret_state_dict, value_state_dict)

    Returns:
        dict with collected samples and stats.
    """
    (num_traversals, start_traversal_id, iteration,
     max_depth, num_augments, explore_epsilon,
     baseline,
     regret_sd, value_sd) = args

    # Load updated weights for this iteration
    _w_regret_net.load_state_dict(regret_sd)
    _w_regret_net.eval()
    _w_value_net.load_state_dict(value_sd)
    _w_value_net.eval()

    regret_samples = []
    strategy_samples = []
    value_samples = []
    total_value = 0.0

    for i in range(num_traversals):
        traverser = (start_traversal_id + i) % 2
        value, r_samps, s_samps, v_samps = _single_traverse(
            traverser, iteration, max_depth, num_augments, explore_epsilon, baseline
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
) -> Tuple[float, list, list, list]:
    """
    Run a single game traversal. Same logic as DeepCFRTrainer.traverse_game
    but uses worker-local models and returns samples instead of storing them.
    """
    state = new_game()
    tracker = BidTracker()
    depth = 0
    initial_round = state.round_num
    initial_scores = state.scores[:]

    decision_points = []

    while state.phase == "BIDDING" and depth < max_depth and state.round_num == initial_round:
        obs = [None, None]
        masks = [None, None]
        strategies = [None, None]

        for p in range(2):
            obs[p] = build_observation(state, p, tracker)
            masks[p] = get_legal_mask(state.hand[p], state.offer)

            with torch.inference_mode():
                obs_t = torch.tensor(obs[p], dtype=torch.float32).unsqueeze(0)
                regrets = _w_regret_net(obs_t).numpy()[0]
            strategies[p] = regret_matching(regrets, masks[p])

        actions = [0, 0]
        sample_probs = [1.0, 1.0]

        for p in range(2):
            legal = np.where(masks[p])[0]
            if len(legal) == 0:
                actions[p] = PASS_ACTION_ID
                continue

            if p == traverser:
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
        })

        bid0 = action_id_to_counts(actions[0])
        bid1 = action_id_to_counts(actions[1])
        old_offer = state.offer[:]
        step_auction(state, bid0, bid1)

        total0, total1 = sum(bid0), sum(bid1)
        if max(total0, total1) > 0:
            if total0 > total1:
                winner = 0
            elif total1 > total0:
                winner = 1
            else:
                winner = 1 - (state.caretaker if state.caretaker != (1 - traverser) else traverser)
                if total0 == total1:
                    pass

            loser = 1 - winner
            bid_w = bid0 if winner == 0 else bid1
            bid_l = bid0 if loser == 0 else bid1
            tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

        if state.round_num > depth // 10 + 1:
            tracker.reset()

        depth += 1

    # Terminal value
    terminal_value = _compute_terminal_value(state, traverser, initial_round, initial_scores)

    # Process decision points
    regret_samples, strategy_samples = _process_decision_points(
        decision_points, traverser, terminal_value, iteration, num_augments, baseline
    )

    value_samples = []

    return terminal_value, regret_samples, strategy_samples, value_samples


def _compute_terminal_value(
    state: FafnirState, traverser: int,
    initial_round: int, initial_scores: list,
) -> float:
    """1ラウンド分のスコア差を報酬として返す。"""
    if state.round_num > initial_round or state.phase == "GAME_END":
        gained = (state.scores[traverser] - initial_scores[traverser]) - \
                 (state.scores[1 - traverser] - initial_scores[1 - traverser])
    else:
        auction_diff = (state.scores[traverser] - initial_scores[traverser]) - \
                       (state.scores[1 - traverser] - initial_scores[1 - traverser])
        hand_diff = compute_hand_score(state, traverser) - compute_hand_score(state, 1 - traverser)
        gained = auction_diff + hand_diff
    return max(-1.0, min(1.0, gained / 40.0))


def _process_decision_points(
    decision_points: list,
    traverser: int,
    terminal_value: float,
    iteration: int,
    num_augments: int,
    baseline: float = 0.0,
) -> Tuple[list, list]:
    """Returns (regret_samples, strategy_samples). Baseline subtraction で分散低減。"""
    regret_samples = []
    strategy_samples = []

    for dp in decision_points:
        obs = dp['obs'][traverser]
        mask = dp['masks'][traverser]
        strategy = dp['strategies'][traverser]
        chosen_action = dp['actions'][traverser]
        sample_prob = dp['sample_probs'][traverser]

        weight = min(10.0, 1.0 / max(sample_prob, 1e-6))
        regret_value = (terminal_value - baseline) * weight

        # Store sparse regret (action_id, value)
        sparse_regret = np.array([chosen_action, regret_value], dtype=np.float32)
        regret_samples.append((obs, sparse_regret, iteration))
        
        # Store sparse strategy
        nonzero_strats = np.nonzero(strategy)[0]
        sparse_strat = np.zeros((len(nonzero_strats), 2), dtype=np.float32)
        sparse_strat[:, 0] = nonzero_strats
        sparse_strat[:, 1] = strategy[nonzero_strats]
        strategy_samples.append((obs, sparse_strat, iteration))

        # Augmentation (sparse-optimized: O(1) per perm)
        if num_augments > 0 and random.random() < 0.5:
            aug_triples = augment_sample_sparse(
                obs, chosen_action, regret_value, ACTION_TABLE, num_augments
            )
            for aug_obs, aug_aid, aug_val in aug_triples:
                aug_sparse = np.array([aug_aid, aug_val], dtype=np.float32)
                regret_samples.append((aug_obs, aug_sparse, iteration))

    return regret_samples, strategy_samples
