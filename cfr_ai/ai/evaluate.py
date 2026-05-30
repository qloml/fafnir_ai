"""
Evaluation pipeline for Deep CFR Fafnir AI.

Quick-strength tools:
- vs random / heuristic / checkpoint opponents
- strategy_net vs regret_net policy comparison
- temperature sweep
- one-round and full-game evaluation
- lightweight action metrics
- best checkpoint selection from a glob
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .action_space import (
    NUM_ACTIONS,
    PASS_ACTION_ID,
    action_id_to_counts,
    get_legal_mask,
)
from .game_engine import (
    NUM_COLORS,
    SCORE_TO_WIN,
    compute_hand_score,
    determine_auction_winner,
    new_game,
    step_auction,
)
from .networks import RegretNetwork, StrategyNetwork, masked_softmax, regret_matching
from .observation import BidTracker, OBS_DIM, build_observation


@dataclass
class ModelPolicy:
    strategy_net: StrategyNetwork
    regret_net: Optional[RegretNetwork]
    mode: str = "strategy"
    name: str = "model"

    def eval(self) -> None:
        self.strategy_net.eval()
        if self.regret_net is not None:
            self.regret_net.eval()


@dataclass
class EvalStats:
    games: int = 0
    wins: int = 0
    draws: int = 0
    score_diff_sum: float = 0.0
    turns: int = 0
    ai_actions: int = 0
    ai_bid_total: int = 0
    ai_passes: int = 0
    ai_gold_bid: int = 0

    def record_action(self, bid: Sequence[int]) -> None:
        total = int(sum(bid))
        self.ai_actions += 1
        self.ai_bid_total += total
        if total == 0:
            self.ai_passes += 1
        if bid and bid[0] > 0:
            self.ai_gold_bid += 1

    def record_game(self, score_diff: float, turns: int) -> None:
        self.games += 1
        self.score_diff_sum += float(score_diff)
        self.turns += int(turns)
        if score_diff > 0:
            self.wins += 1
        elif score_diff == 0:
            self.draws += 1

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games)

    def summary(self) -> Dict[str, float]:
        return {
            "games": float(self.games),
            "win_rate": self.win_rate,
            "draw_rate": self.draws / max(1, self.games),
            "mean_score_diff": self.score_diff_sum / max(1, self.games),
            "avg_turns": self.turns / max(1, self.games),
            "avg_ai_bid": self.ai_bid_total / max(1, self.ai_actions),
            "ai_pass_rate": self.ai_passes / max(1, self.ai_actions),
            "ai_gold_bid_rate": self.ai_gold_bid / max(1, self.ai_actions),
        }


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_policy(
    checkpoint_path: str,
    device: torch.device,
    mode: str = "strategy",
    name: str = "model",
) -> Tuple[Optional[ModelPolicy], Dict]:
    if not os.path.exists(checkpoint_path):
        print(f"[EVAL] Checkpoint not found: {checkpoint_path}")
        return None, {}

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 192)

    strategy_net = StrategyNetwork(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        hidden=hidden_dim,
    ).to(device)
    strategy_net.load_state_dict(ckpt["strategy_net"])

    regret_net = None
    if "regret_net" in ckpt:
        regret_net = RegretNetwork(
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            hidden=hidden_dim,
        ).to(device)
        regret_net.load_state_dict(ckpt["regret_net"])

    policy = ModelPolicy(strategy_net, regret_net, mode=mode, name=name)
    policy.eval()
    return policy, ckpt


def select_model_action(
    policy: ModelPolicy,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    temperature: float,
    deterministic: bool,
) -> int:
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return PASS_ACTION_ID

    with torch.inference_mode():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if policy.mode == "regret":
            if policy.regret_net is None:
                raise ValueError("policy mode 'regret' requires regret_net in checkpoint")
            regrets = policy.regret_net(obs_t).cpu().numpy()[0]
            probs = regret_matching(regrets, mask)
        else:
            logits = policy.strategy_net(obs_t).cpu().numpy()[0]
            probs = masked_softmax(logits, mask, temperature)

    legal_probs = probs[legal]
    legal_probs = legal_probs / (legal_probs.sum() + 1e-10)
    if deterministic:
        return int(legal[np.argmax(legal_probs)])
    return int(np.random.choice(legal, p=legal_probs))


def select_random_action(mask: np.ndarray) -> int:
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return PASS_ACTION_ID
    return int(np.random.choice(legal))


def select_heuristic_action(hand: List[int], offer: List[int], mask: np.ndarray) -> int:
    """A simple legal-information baseline stronger than uniform random."""
    legal = np.where(mask)[0]
    if len(legal) == 0:
        return PASS_ACTION_ID

    offer_total = sum(offer)
    offer_gold = offer[0]
    offer_colors = sum(1 for x in offer if x > 0)
    hand_total = sum(hand)

    best_action = PASS_ACTION_ID
    best_score = -1e9
    for aid in legal:
        bid = action_id_to_counts(int(aid))
        bid_total = sum(bid)
        score = 0.0

        if bid_total == 0:
            score = -0.8 if offer_total >= 3 else -0.2
        else:
            score += offer_total * 1.4
            score += offer_gold * 1.5
            score += offer_colors * 0.25
            score -= bid_total * 1.05
            score -= bid[0] * 2.5

            # Prefer discarding colors that are already overcrowded in our hand.
            for c in range(1, NUM_COLORS):
                if hand[c] >= 5:
                    score += bid[c] * 1.2
                elif hand[c] >= 4:
                    score += bid[c] * 0.4

            # Avoid spending too much of a small hand.
            if hand_total > 0 and bid_total / hand_total > 0.4:
                score -= 1.2

        # Small deterministic tie-break toward fewer stones.
        score -= bid_total * 0.01
        if score > best_score:
            best_score = score
            best_action = int(aid)

    return best_action


def choose_action(
    player: int,
    state,
    tracker: BidTracker,
    ai_player: int,
    ai_policy: ModelPolicy,
    opponent_policy: Optional[ModelPolicy],
    opponent: str,
    device: torch.device,
    temperature: float,
    opponent_temperature: float,
    deterministic: bool,
) -> int:
    mask = get_legal_mask(state.hand[player], state.offer)

    if player == ai_player:
        obs = build_observation(state, player, tracker)
        return select_model_action(ai_policy, obs, mask, device, temperature, deterministic)

    if opponent == "random":
        return select_random_action(mask)

    if opponent == "heuristic":
        return select_heuristic_action(state.hand[player], state.offer, mask)

    if opponent == "checkpoint":
        if opponent_policy is None:
            raise ValueError("checkpoint opponent requires opponent_policy")
        obs = build_observation(state, player, tracker)
        return select_model_action(
            opponent_policy,
            obs,
            mask,
            device,
            opponent_temperature,
            deterministic,
        )

    raise ValueError(f"unknown opponent: {opponent}")


def update_tracker_after_step(
    tracker: BidTracker,
    old_offer: List[int],
    old_caretaker: int,
    bid0: List[int],
    bid1: List[int],
) -> None:
    winner = determine_auction_winner(bid0, bid1, old_caretaker)
    if winner is None:
        return
    loser = 1 - winner
    bid_w = bid0 if winner == 0 else bid1
    bid_l = bid0 if loser == 0 else bid1
    tracker.update_from_auction(winner, bid_w, bid_l, old_offer)


def score_diff_for_round(state, ai_player: int, initial_scores: List[int], initial_round: int) -> float:
    opp = 1 - ai_player
    ai_gained = state.scores[ai_player] - initial_scores[ai_player]
    opp_gained = state.scores[opp] - initial_scores[opp]
    if state.round_num > initial_round or state.phase == "GAME_END":
        return ai_gained - opp_gained
    return (
        ai_gained + compute_hand_score(state, ai_player)
        - opp_gained - compute_hand_score(state, opp)
    )


def play_one_eval_game(
    ai_policy: ModelPolicy,
    device: torch.device,
    ai_player: int,
    opponent: str,
    opponent_policy: Optional[ModelPolicy] = None,
    temperature: float = 0.3,
    opponent_temperature: float = 0.3,
    deterministic: bool = False,
    full_game: bool = False,
    target_score: int = SCORE_TO_WIN,
    max_turns: int = 2000,
    stats: Optional[EvalStats] = None,
) -> float:
    state = new_game()
    tracker = BidTracker()
    initial_round = state.round_num
    initial_scores = state.scores[:]
    last_round = state.round_num
    turns = 0

    while state.phase == "BIDDING" and turns < max_turns:
        if state.round_num != last_round:
            tracker.reset()
            last_round = state.round_num
        if not full_game and state.round_num != initial_round:
            break
        if full_game and max(state.scores) >= target_score:
            break

        action_ids = [0, 0]
        for p in range(2):
            action_ids[p] = choose_action(
                p,
                state,
                tracker,
                ai_player,
                ai_policy,
                opponent_policy,
                opponent,
                device,
                temperature,
                opponent_temperature,
                deterministic,
            )

        bid0 = action_id_to_counts(action_ids[0])
        bid1 = action_id_to_counts(action_ids[1])

        if stats is not None:
            stats.record_action(bid0 if ai_player == 0 else bid1)

        old_offer = state.offer[:]
        old_caretaker = state.caretaker
        step_auction(state, bid0, bid1)
        update_tracker_after_step(tracker, old_offer, old_caretaker, bid0, bid1)
        turns += 1

        if state.phase == "GAME_END":
            break
        if full_game and max(state.scores) >= target_score:
            break

    if full_game:
        opp = 1 - ai_player
        score_diff = state.scores[ai_player] - state.scores[opp]
        if max(state.scores) >= target_score and state.scores[ai_player] != state.scores[opp]:
            score_diff = 1.0 if state.scores[ai_player] > state.scores[opp] else -1.0
    else:
        score_diff = score_diff_for_round(state, ai_player, initial_scores, initial_round)

    if stats is not None:
        stats.record_game(score_diff, turns)
    return score_diff


def evaluate_policy(
    ai_policy: ModelPolicy,
    device: torch.device,
    num_games: int = 500,
    temperature: float = 0.3,
    opponent: str = "random",
    opponent_policy: Optional[ModelPolicy] = None,
    opponent_temperature: float = 0.3,
    deterministic: bool = False,
    full_game: bool = False,
    target_score: int = SCORE_TO_WIN,
    max_turns: int = 2000,
) -> EvalStats:
    ai_policy.eval()
    if opponent_policy is not None:
        opponent_policy.eval()

    stats = EvalStats()
    for i in range(num_games):
        ai_player = i % 2
        play_one_eval_game(
            ai_policy=ai_policy,
            device=device,
            ai_player=ai_player,
            opponent=opponent,
            opponent_policy=opponent_policy,
            temperature=temperature,
            opponent_temperature=opponent_temperature,
            deterministic=deterministic,
            full_game=full_game,
            target_score=target_score,
            max_turns=max_turns,
            stats=stats,
        )
    return stats


def evaluate_vs_random(
    strategy_net: torch.nn.Module,
    device: torch.device,
    num_games: int = 500,
    temperature: float = 0.3,
) -> float:
    """Backward-compatible API used by train.py."""
    policy = ModelPolicy(strategy_net=strategy_net, regret_net=None, mode="strategy", name="current")
    return evaluate_policy(policy, device, num_games, temperature, opponent="random").win_rate


def evaluate_vs_checkpoint(
    strategy_net: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    num_games: int = 500,
    temperature: float = 0.3,
) -> float:
    """Backward-compatible API."""
    opponent_policy, _ = load_model_policy(checkpoint_path, device, mode="strategy", name="opponent")
    if opponent_policy is None:
        return 0.5
    policy = ModelPolicy(strategy_net=strategy_net, regret_net=None, mode="strategy", name="current")
    return evaluate_policy(
        policy,
        device,
        num_games,
        temperature,
        opponent="checkpoint",
        opponent_policy=opponent_policy,
        opponent_temperature=temperature,
    ).win_rate


def format_summary(stats: EvalStats) -> str:
    s = stats.summary()
    return (
        f"win={s['win_rate']:.1%} draw={s['draw_rate']:.1%} "
        f"diff={s['mean_score_diff']:.2f} turns={s['avg_turns']:.1f} "
        f"bid={s['avg_ai_bid']:.2f} pass={s['ai_pass_rate']:.1%} "
        f"gold_bid={s['ai_gold_bid_rate']:.1%}"
    )


def parse_temperatures(text: Optional[str], fallback: float) -> List[float]:
    if not text:
        return [fallback]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def evaluate_checkpoint_path(
    checkpoint_path: str,
    args: argparse.Namespace,
    device: torch.device,
    temperature: float,
) -> Optional[Tuple[float, EvalStats, Dict]]:
    policy, ckpt = load_model_policy(checkpoint_path, device, mode=args.policy, name=checkpoint_path)
    if policy is None:
        return None

    opponent_policy = None
    opponent = args.opponent
    if args.vs_checkpoint:
        opponent = "checkpoint"
        opponent_policy, _ = load_model_policy(
            args.vs_checkpoint,
            device,
            mode=args.opponent_policy,
            name="opponent",
        )
        if opponent_policy is None:
            return None

    stats = evaluate_policy(
        policy,
        device,
        num_games=args.games,
        temperature=temperature,
        opponent=opponent,
        opponent_policy=opponent_policy,
        opponent_temperature=args.opponent_temperature,
        deterministic=args.deterministic,
        full_game=args.full_game,
        target_score=args.target_score,
        max_turns=args.max_turns,
    )
    return stats.win_rate, stats, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deep CFR AI")
    parser.add_argument("--checkpoint", default="cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt")
    parser.add_argument("--checkpoint-glob", default=None, help="Evaluate all matching checkpoints")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--temperatures", default=None, help="Comma-separated sweep, e.g. 0.05,0.1,0.2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", choices=["strategy", "regret"], default="strategy")
    parser.add_argument("--opponent-policy", choices=["strategy", "regret"], default="strategy")
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    parser.add_argument("--opponent-temperature", type=float, default=0.3)
    parser.add_argument("--vs-checkpoint", default=None, help="Path to opponent checkpoint")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--full-game", action="store_true", help="Evaluate until target score instead of one round")
    parser.add_argument("--target-score", type=int, default=SCORE_TO_WIN)
    parser.add_argument("--max-turns", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_paths = [args.checkpoint]
    if args.checkpoint_glob:
        checkpoint_paths = sorted(glob.glob(args.checkpoint_glob, recursive=True))
        if not checkpoint_paths:
            print(f"[EVAL] No checkpoints matched: {args.checkpoint_glob}")
            return

    temperatures = parse_temperatures(args.temperatures, args.temperature)
    best = None

    for path in checkpoint_paths:
        for temp in temperatures:
            result = evaluate_checkpoint_path(path, args, device, temp)
            if result is None:
                continue
            score, stats, ckpt = result
            iter_s = ckpt.get("iteration", "?")
            hidden = ckpt.get("hidden_dim", "?")
            opponent = "checkpoint" if args.vs_checkpoint else args.opponent
            print(
                f"[EVAL] {path} iter={iter_s} hidden={hidden} "
                f"policy={args.policy} opponent={opponent} temp={temp} | "
                f"{format_summary(stats)}"
            )
            key = (score, stats.summary()["mean_score_diff"])
            if best is None or key > best[0]:
                best = (key, path, temp, stats, iter_s)

    if best is not None and (len(checkpoint_paths) > 1 or len(temperatures) > 1):
        _, path, temp, stats, iter_s = best
        print(
            f"[EVAL] BEST path={path} iter={iter_s} temp={temp} | "
            f"{format_summary(stats)}"
        )


if __name__ == "__main__":
    main()
