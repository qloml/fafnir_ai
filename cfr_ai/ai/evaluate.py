"""
Evaluation pipeline for Deep CFR Fafnir AI.

Provides tools to measure AI strength:
1. vs Random: Win rate against a random player
2. vs Checkpoint: Win rate against a previous model version

Usage:
    python -m cfr_ai.ai.evaluate [options]
"""
import numpy as np
import torch
import random
from typing import Optional

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
    StrategyNetwork, masked_softmax, regret_matching,
)


def _play_one_round(
    strategy_net: torch.nn.Module,
    device: torch.device,
    ai_player: int,
    temperature: float = 0.3,
) -> float:
    """Play one round: AI vs Random. Returns score difference (AI - Random)."""
    state = new_game()
    tracker = BidTracker()
    opp = 1 - ai_player
    depth = 0
    initial_round = state.round_num
    initial_scores = state.scores[:]

    while state.phase == "BIDDING" and depth < 60 and state.round_num == initial_round:
        actions = [[0]*NUM_COLORS, [0]*NUM_COLORS]
        action_ids = [0, 0]

        for p in range(2):
            mask = get_legal_mask(state.hand[p], state.offer)
            legal = np.where(mask)[0]

            if len(legal) == 0:
                action_ids[p] = PASS_ACTION_ID
                continue

            if p == ai_player:
                # AI: use strategy network
                obs = build_observation(state, p, tracker)
                with torch.inference_mode():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = strategy_net(obs_t).cpu().numpy()[0]
                probs = masked_softmax(logits, mask, temperature)
                lp = probs[legal]
                lp = lp / (lp.sum() + 1e-10)
                action_ids[p] = int(np.random.choice(legal, p=lp))
            else:
                # Random: uniform over legal actions
                action_ids[p] = int(np.random.choice(legal))

        bid0 = action_id_to_counts(action_ids[0])
        bid1 = action_id_to_counts(action_ids[1])

        old_offer = state.offer[:]
        step_auction(state, bid0, bid1)

        # Update tracker
        total0, total1 = sum(bid0), sum(bid1)
        if max(total0, total1) > 0:
            if total0 > total1:
                winner = 0
            elif total1 > total0:
                winner = 1
            else:
                winner = 1 - state.caretaker
            loser = 1 - winner
            bid_w = bid0 if winner == 0 else bid1
            bid_l = bid0 if loser == 0 else bid1
            tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

        depth += 1

    # Score difference
    ai_gained = state.scores[ai_player] - initial_scores[ai_player]
    opp_gained = state.scores[opp] - initial_scores[opp]

    # Also add hand score if round completed
    if state.round_num > initial_round or state.phase == "GAME_END":
        return ai_gained - opp_gained
    else:
        # Estimate from hand scores
        ai_hand = compute_hand_score(state, ai_player)
        opp_hand = compute_hand_score(state, opp)
        return (ai_gained + ai_hand) - (opp_gained + opp_hand)


def evaluate_vs_random(
    strategy_net: torch.nn.Module,
    device: torch.device,
    num_games: int = 500,
    temperature: float = 0.3,
) -> float:
    """
    Evaluate strategy network against a random player.

    Returns win rate (fraction of rounds where AI scored higher).
    """
    strategy_net.eval()
    wins = 0
    draws = 0

    for i in range(num_games):
        ai_player = i % 2  # Alternate sides
        score_diff = _play_one_round(strategy_net, device, ai_player, temperature)

        if score_diff > 0:
            wins += 1
        elif score_diff == 0:
            draws += 1

    return wins / num_games


def evaluate_vs_checkpoint(
    strategy_net: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    num_games: int = 500,
    temperature: float = 0.3,
) -> float:
    """
    Evaluate current strategy network against a previous checkpoint.

    Returns win rate of current model vs checkpoint model.
    """
    import os
    if not os.path.exists(checkpoint_path):
        print(f"[EVAL] Checkpoint not found: {checkpoint_path}")
        return 0.5

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get('hidden_dim', 192)

    opponent_net = StrategyNetwork(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden=hidden_dim,
    ).to(device)

    try:
        opponent_net.load_state_dict(ckpt['strategy_net'])
    except RuntimeError:
        print(f"[EVAL] Incompatible checkpoint architecture")
        return 0.5

    opponent_net.eval()
    strategy_net.eval()

    wins = 0

    for i in range(num_games):
        state = new_game()
        tracker = BidTracker()
        current_player = i % 2  # current model plays as this player
        opp_player = 1 - current_player
        depth = 0
        initial_round = state.round_num
        initial_scores = state.scores[:]

        while state.phase == "BIDDING" and depth < 60 and state.round_num == initial_round:
            action_ids = [0, 0]

            for p in range(2):
                mask = get_legal_mask(state.hand[p], state.offer)
                legal = np.where(mask)[0]

                if len(legal) == 0:
                    action_ids[p] = PASS_ACTION_ID
                    continue

                obs = build_observation(state, p, tracker)
                net = strategy_net if p == current_player else opponent_net

                with torch.inference_mode():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits = net(obs_t).cpu().numpy()[0]

                probs = masked_softmax(logits, mask, temperature)
                lp = probs[legal]
                lp = lp / (lp.sum() + 1e-10)
                action_ids[p] = int(np.random.choice(legal, p=lp))

            bid0 = action_id_to_counts(action_ids[0])
            bid1 = action_id_to_counts(action_ids[1])
            old_offer = state.offer[:]
            step_auction(state, bid0, bid1)

            total0, total1 = sum(bid0), sum(bid1)
            if max(total0, total1) > 0:
                if total0 > total1:
                    winner = 0
                elif total1 > total0:
                    winner = 1
                else:
                    winner = 1 - state.caretaker
                loser = 1 - winner
                bid_w = bid0 if winner == 0 else bid1
                bid_l = bid0 if loser == 0 else bid1
                tracker.update_from_auction(winner, bid_w, bid_l, old_offer)

            depth += 1

        # Determine winner
        cur_gained = state.scores[current_player] - initial_scores[current_player]
        opp_gained = state.scores[opp_player] - initial_scores[opp_player]

        if state.round_num > initial_round or state.phase == "GAME_END":
            score_diff = cur_gained - opp_gained
        else:
            cur_hand = compute_hand_score(state, current_player)
            opp_hand = compute_hand_score(state, opp_player)
            score_diff = (cur_gained + cur_hand) - (opp_gained + opp_hand)

        if score_diff > 0:
            wins += 1

    return wins / num_games


# ============================================================
# CLI
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Deep CFR AI")
    parser.add_argument("--checkpoint", default="cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vs-checkpoint", default=None,
                        help="Path to opponent checkpoint (for model vs model)")
    args = parser.parse_args()

    import os
    device = torch.device(args.device)

    if not os.path.exists(args.checkpoint):
        print(f"[EVAL] No checkpoint: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden_dim = ckpt.get('hidden_dim', 192)

    net = StrategyNetwork(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden=hidden_dim,
    ).to(device)
    net.load_state_dict(ckpt['strategy_net'])
    net.eval()

    print(f"[EVAL] Loaded model (iter={ckpt.get('iteration', '?')}, hidden={hidden_dim})")

    if args.vs_checkpoint:
        print(f"[EVAL] vs Checkpoint: {args.vs_checkpoint}")
        wr = evaluate_vs_checkpoint(net, args.vs_checkpoint, device,
                                    args.games, args.temperature)
        print(f"[EVAL] Win rate: {wr:.1%} ({args.games} games)")
    else:
        print(f"[EVAL] vs Random ({args.games} games)...")
        wr = evaluate_vs_random(net, device, args.games, args.temperature)
        print(f"[EVAL] Win rate: {wr:.1%}")


if __name__ == "__main__":
    main()
