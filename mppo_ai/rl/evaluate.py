# mppo_ai/rl/evaluate.py
"""
Evaluate a trained FAFNIR RL agent against various opponents.

Usage:
    python mppo_ai/rl/evaluate.py --model mppo_ai/rl/output/fafnir_final
    python mppo_ai/rl/evaluate.py --model mppo_ai/rl/output/fafnir_final --games 500 --score-to-win 1000
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from rl.game_env import (
    FafnirEnv, RandomOpponent, ModelOpponent,
    N_COLORS, MAX_BID_PER_COLOR,
)


def mask_fn(env):
    return env.valid_action_mask()


class GreedyOpponent:
    """Always bids 1 stone (cheapest available), never overbids."""
    def choose_bid(self, hand, offer, rng):
        bid = np.zeros(N_COLORS, dtype=np.int32)
        for c in range(N_COLORS):
            if offer[c] == 0 and hand[c] > 0:
                bid[c] = 1
                break
        return bid


class AggressiveOpponent:
    """Bids 2-4 stones aggressively."""
    def choose_bid(self, hand, offer, rng):
        bid = np.zeros(N_COLORS, dtype=np.int32)
        biddable = []
        for c in range(N_COLORS):
            if offer[c] == 0 and hand[c] > 0:
                for _ in range(hand[c]):
                    biddable.append(c)
        if not biddable:
            return bid
        n = min(rng.integers(2, 5), len(biddable))
        chosen = list(rng.choice(len(biddable), size=n, replace=False))
        for idx in chosen:
            bid[biddable[idx]] += 1
        return bid


def evaluate_model(model_path: str, opponent, opponent_name: str,
                   n_games: int = 200, score_to_win: int = 40,
                   max_turns: int = 500, deterministic: bool = True):
    """Run n_games and return statistics."""
    env = FafnirEnv(score_to_win=score_to_win, max_turns=max_turns, opponent=opponent)
    env_masked = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path)
    wins = 0
    losses = 0
    draws = 0
    total_score_agent = 0
    total_score_opp = 0
    total_turns = 0

    for ep in range(n_games):
        obs, info = env_masked.reset()
        done = False
        while not done:
            mask = env.valid_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env_masked.step(action)
            done = terminated or truncated

        s = info["scores"]
        total_score_agent += s[0]
        total_score_opp += s[1]
        total_turns += info["total_turns"]

        if s[0] > s[1]:
            wins += 1
        elif s[1] > s[0]:
            losses += 1
        else:
            draws += 1

    avg_score_agent = total_score_agent / n_games
    avg_score_opp = total_score_opp / n_games
    avg_turns = total_turns / n_games
    win_rate = wins / n_games * 100

    print(f"\n{'='*60}")
    print(f"vs {opponent_name} ({n_games} games, score_to_win={score_to_win})")
    print(f"{'='*60}")
    print(f"  Win rate:    {win_rate:.1f}%  ({wins}W / {losses}L / {draws}D)")
    print(f"  Avg score:   Agent={avg_score_agent:.1f}  Opp={avg_score_opp:.1f}")
    print(f"  Avg turns:   {avg_turns:.1f}")
    print(f"{'='*60}")

    return {
        "opponent": opponent_name,
        "games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_score_agent": avg_score_agent,
        "avg_score_opp": avg_score_opp,
        "avg_turns": avg_turns,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate FAFNIR RL agent")
    ap.add_argument("--model", type=str, required=True, help="Path to trained model")
    ap.add_argument("--games", type=int, default=200, help="Number of games per opponent")
    ap.add_argument("--score-to-win", type=int, default=40, help="Score to win")
    ap.add_argument("--max-turns", type=int, default=500, help="Max turns per game")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    args = ap.parse_args()

    if not os.path.exists(args.model + ".zip"):
        if not os.path.exists(args.model):
            print(f"Model not found: {args.model}")
            sys.exit(1)

    print(f"Evaluating model: {args.model}")

    opponents = [
        (RandomOpponent(), "Random"),
        (GreedyOpponent(), "Greedy (1-stone)"),
        (AggressiveOpponent(), "Aggressive (2-4 stones)"),
    ]

    results = []
    for opp, name in opponents:
        r = evaluate_model(
            args.model, opp, name,
            n_games=args.games,
            score_to_win=args.score_to_win,
            max_turns=args.max_turns,
            deterministic=args.deterministic,
        )
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  vs {r['opponent']:<25} Win: {r['win_rate']:>5.1f}%  "
              f"({r['wins']}W/{r['losses']}L/{r['draws']}D)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
