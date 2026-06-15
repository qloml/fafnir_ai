"""
Checkpoint league evaluator for MaskablePPO Fafnir AI.

Ranks saved MPPO checkpoints against simple baseline opponents and, optionally,
against each other. Results are written as CSV files plus a past-opponent
manifest that can be used to choose stronger self-play opponents manually.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from mppo_ai.rl.evaluate import AggressiveOpponent, GreedyOpponent
from mppo_ai.rl.game_env import FafnirEnv, ModelOpponent, N_COLORS, RandomOpponent

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATH_FIELDS = {"checkpoint", "opponent_checkpoint"}


def mask_fn(env: FafnirEnv) -> np.ndarray:
    return env.valid_action_mask()


class DeterministicModelOpponent(ModelOpponent):
    """Checkpoint opponent with configurable deterministic/stochastic play."""

    def __init__(self, model: MaskablePPO, deterministic: bool):
        super().__init__(model)
        self.deterministic = deterministic

    def choose_bid_from_obs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        return np.asarray(action, dtype=np.int64)


@dataclass
class EvalStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_score_agent: float = 0.0
    total_score_opp: float = 0.0
    total_turns: float = 0.0
    total_bid: float = 0.0
    actions: int = 0
    passes: int = 0
    gold_actions: int = 0
    gold_games: int = 0

    def summary(self) -> Dict[str, float]:
        games = max(1, self.games)
        actions = max(1, self.actions)
        return {
            "games": float(self.games),
            "wins": float(self.wins),
            "losses": float(self.losses),
            "draws": float(self.draws),
            "win_rate": self.wins / games,
            "loss_rate": self.losses / games,
            "draw_rate": self.draws / games,
            "avg_score_agent": self.total_score_agent / games,
            "avg_score_opp": self.total_score_opp / games,
            "mean_score_diff": (self.total_score_agent - self.total_score_opp) / games,
            "avg_turns": self.total_turns / games,
            "avg_bid": self.total_bid / actions,
            "pass_rate": self.passes / actions,
            "gold_bid_rate": self.gold_actions / actions,
            "gold_game_rate": self.gold_games / games,
        }


@dataclass
class EvalRow:
    checkpoint: str
    steps: int
    opponent: str
    opponent_checkpoint: str
    games: int
    deterministic: bool
    win_rate: float
    loss_rate: float
    draw_rate: float
    mean_diff: float
    avg_score_agent: float
    avg_score_opp: float
    avg_turns: float
    avg_bid: float
    pass_rate: float
    gold_bid_rate: float
    gold_game_rate: float
    score: float


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def to_report_path(path: str) -> str:
    if not path:
        return ""
    return os.path.relpath(os.path.abspath(path), PROJECT_ROOT)


def row_for_csv(row: Dict[str, object]) -> Dict[str, object]:
    return {
        key: to_report_path(str(value)) if key in PATH_FIELDS else value
        for key, value in row.items()
    }


def checkpoint_steps(path: str) -> int:
    name = os.path.basename(path)
    patterns = [
        r"fafnir_rl_(\d+)_steps",
        r"selfplay_(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


def checkpoint_sort_key(path: str) -> Tuple[int, str]:
    return checkpoint_steps(path), path


def find_checkpoints(patterns: str, max_checkpoints: int) -> List[str]:
    paths: List[str] = []
    seen = set()
    for pattern in parse_csv_list(patterns):
        for path in glob.glob(pattern, recursive=True):
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            paths.append(path)

    numbered = sorted(
        [path for path in paths if checkpoint_steps(path) >= 0],
        key=checkpoint_sort_key,
    )
    unnumbered = sorted(
        [path for path in paths if checkpoint_steps(path) < 0],
        key=lambda path: path.lower(),
    )

    if max_checkpoints > 0:
        numbered = numbered[-max_checkpoints:]
    return sorted(unnumbered + numbered, key=checkpoint_sort_key)


def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if not path.endswith(".zip") and os.path.exists(path + ".zip"):
        return path + ".zip"
    raise FileNotFoundError(path)


def make_baseline(name: str):
    key = name.lower()
    if key == "random":
        return RandomOpponent()
    if key in ("greedy", "greedy1", "greedy-1"):
        return GreedyOpponent()
    if key in ("aggressive", "aggro"):
        return AggressiveOpponent()
    raise ValueError(f"Unknown opponent: {name}")


def load_checkpoint(path: str, device: str) -> MaskablePPO:
    return MaskablePPO.load(resolve_model_path(path), device=device)


def score_stats(stats: EvalStats, gold_penalty: float) -> float:
    s = stats.summary()
    return s["win_rate"] * 100.0 + s["mean_score_diff"] - gold_penalty * s["gold_bid_rate"]


def evaluate_model(
    model_path: str,
    opponent,
    opponent_name: str,
    *,
    opponent_checkpoint: str = "",
    games: int,
    score_to_win: int,
    max_turns: int,
    deterministic: bool,
    seed: Optional[int],
    device: str,
    gold_penalty: float,
    progress_every: int,
) -> EvalRow:
    model = load_checkpoint(model_path, device)
    env = FafnirEnv(score_to_win=score_to_win, max_turns=max_turns, opponent=opponent)
    env_masked = ActionMasker(env, mask_fn)
    stats = EvalStats()

    for ep in range(games):
        reset_seed = None if seed is None else seed + ep
        obs, info = env_masked.reset(seed=reset_seed)
        done = False
        game_used_gold = False

        while not done:
            mask = env.valid_action_mask()
            action, _ = model.predict(
                obs,
                action_masks=mask,
                deterministic=deterministic,
            )
            action = np.asarray(action, dtype=np.int64)
            bid_sum = int(action.sum())

            stats.actions += 1
            stats.total_bid += bid_sum
            if bid_sum == 0:
                stats.passes += 1
            if int(action[0]) > 0:
                stats.gold_actions += 1
                game_used_gold = True

            obs, reward, terminated, truncated, info = env_masked.step(action)
            done = bool(terminated or truncated)

        scores = info["scores"]
        agent_score = int(scores[0])
        opp_score = int(scores[1])
        stats.games += 1
        stats.total_score_agent += agent_score
        stats.total_score_opp += opp_score
        stats.total_turns += int(info["total_turns"])
        if game_used_gold:
            stats.gold_games += 1

        if agent_score > opp_score:
            stats.wins += 1
        elif opp_score > agent_score:
            stats.losses += 1
        else:
            stats.draws += 1

        if progress_every > 0 and (stats.games % progress_every == 0 or stats.games == games):
            diff = (stats.total_score_agent - stats.total_score_opp) / max(1, stats.games)
            print(
                f"[MPPO-LEAGUE]   progress {stats.games}/{games} "
                f"opponent={opponent_name} win={stats.wins}/{stats.games} "
                f"mean_diff={diff:.2f}",
                flush=True,
            )

    env.close()
    summary = stats.summary()
    score = score_stats(stats, gold_penalty)
    return EvalRow(
        checkpoint=os.path.abspath(resolve_model_path(model_path)),
        steps=checkpoint_steps(model_path),
        opponent=opponent_name,
        opponent_checkpoint=os.path.abspath(resolve_model_path(opponent_checkpoint)) if opponent_checkpoint else "",
        games=int(summary["games"]),
        deterministic=deterministic,
        win_rate=summary["win_rate"],
        loss_rate=summary["loss_rate"],
        draw_rate=summary["draw_rate"],
        mean_diff=summary["mean_score_diff"],
        avg_score_agent=summary["avg_score_agent"],
        avg_score_opp=summary["avg_score_opp"],
        avg_turns=summary["avg_turns"],
        avg_bid=summary["avg_bid"],
        pass_rate=summary["pass_rate"],
        gold_bid_rate=summary["gold_bid_rate"],
        gold_game_rate=summary["gold_game_rate"],
        score=score,
    )


def write_rows(path: str, rows: Iterable[EvalRow]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row_for_csv(row.__dict__))


def summarize_baselines(rows: List[EvalRow]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[EvalRow]] = {}
    for row in rows:
        grouped.setdefault(row.checkpoint, []).append(row)

    ranking: List[Dict[str, object]] = []
    for checkpoint, items in grouped.items():
        n = len(items)
        ranking.append({
            "checkpoint": checkpoint,
            "steps": max(row.steps for row in items),
            "baseline_score": sum(row.score for row in items) / n,
            "baseline_win_rate": sum(row.win_rate for row in items) / n,
            "baseline_diff": sum(row.mean_diff for row in items) / n,
            "avg_bid": sum(row.avg_bid for row in items) / n,
            "pass_rate": sum(row.pass_rate for row in items) / n,
            "gold_bid_rate": sum(row.gold_bid_rate for row in items) / n,
            "league_evaluated": False,
            "league_score": "",
            "league_win_rate": "",
            "combined_score": sum(row.score for row in items) / n,
        })
    ranking.sort(key=lambda item: float(item["combined_score"]), reverse=True)
    return ranking


def summarize_checkpoints(checkpoints: List[str]) -> List[Dict[str, object]]:
    ranking: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        resolved = os.path.abspath(resolve_model_path(checkpoint))
        ranking.append({
            "checkpoint": resolved,
            "steps": checkpoint_steps(checkpoint),
            "baseline_score": "",
            "baseline_win_rate": "",
            "baseline_diff": "",
            "avg_bid": "",
            "pass_rate": "",
            "gold_bid_rate": "",
            "league_evaluated": False,
            "league_score": "",
            "league_win_rate": "",
            "combined_score": "",
        })
    ranking.sort(
        key=lambda item: (
            int(item["steps"]) < 0,
            int(item["steps"]),
            str(item["checkpoint"]),
        ),
        reverse=True,
    )
    return ranking


def apply_league_results(
    ranking: List[Dict[str, object]],
    league_rows: List[EvalRow],
    league_weight: float,
) -> None:
    grouped: Dict[str, List[EvalRow]] = {}
    for row in league_rows:
        grouped.setdefault(row.checkpoint, []).append(row)

    for item in ranking:
        checkpoint = str(item["checkpoint"])
        rows = grouped.get(checkpoint)
        if not rows:
            if item["combined_score"] == "":
                item["combined_score"] = 0.0
            continue
        league_score = sum(row.score for row in rows) / len(rows)
        item["league_evaluated"] = True
        item["league_score"] = league_score
        item["league_win_rate"] = sum(row.win_rate for row in rows) / len(rows)
        if item["baseline_score"] == "":
            item["combined_score"] = league_score
        else:
            baseline_score = float(item["baseline_score"])
            item["combined_score"] = baseline_score * (1.0 - league_weight) + league_score * league_weight

    if league_rows:
        ranking.sort(
            key=lambda item: (
                bool(item["league_evaluated"]),
                float(item["combined_score"]),
            ),
            reverse=True,
        )
    else:
        ranking.sort(key=lambda item: float(item["combined_score"]), reverse=True)


def write_ranking(path: str, ranking: List[Dict[str, object]]) -> None:
    if not ranking:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0].keys()))
        writer.writeheader()
        for row in ranking:
            writer.writerow(row_for_csv(row))


def write_past_opponents(path: str, ranking: List[Dict[str, object]], count: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ranked MPPO checkpoints generated by mppo_ai.rl.league\n")
        for item in ranking[:count]:
            checkpoint = str(item["checkpoint"])
            f.write(f"{os.path.relpath(checkpoint, manifest_dir)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MPPO checkpoint league evaluation")
    parser.add_argument(
        "--checkpoint-glob",
        default="mppo_ai/rl/output/checkpoints/fafnir_rl_*_steps.zip,"
                "mppo_ai/rl/output/completed/fafnir_v3_*.zip",
        help="Comma-separated checkpoint glob patterns",
    )
    parser.add_argument("--max-checkpoints", type=int, default=8,
                        help="Evaluate newest N step-numbered checkpoints; unnumbered completed models are also included (0=all)")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per baseline opponent/checkpoint")
    parser.add_argument("--league-games", type=int, default=100,
                        help="Games per checkpoint-vs-checkpoint match (0=disable)")
    parser.add_argument("--league-top-k", type=int, default=6,
                        help="Top baseline checkpoints included in checkpoint league")
    parser.add_argument("--ai-only", action="store_true",
                        help="Skip baseline opponents and rank only checkpoint-vs-checkpoint results")
    parser.add_argument("--opponents", default="random,greedy,aggressive")
    parser.add_argument("--score-to-win", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--opponent-deterministic", action="store_true",
                        help="Use deterministic play for checkpoint opponents")
    parser.add_argument("--gold-penalty", type=float, default=0.0,
                        help="Ranking penalty multiplied by gold_bid_rate")
    parser.add_argument("--league-weight", type=float, default=0.5)
    parser.add_argument("--past-opponents", type=int, default=8)
    parser.add_argument("--output-dir", default="mppo_ai/rl/reports")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress every N games within each matchup (0=off)")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoint_glob, args.max_checkpoints)
    if not checkpoints:
        print(f"[MPPO-LEAGUE] No checkpoints matched: {args.checkpoint_glob}")
        return
    if args.ai_only and args.league_games <= 0:
        print("[MPPO-LEAGUE] --ai-only requires --league-games > 0")
        return

    run_name = time.strftime("league_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    baseline_rows: List[EvalRow] = []
    if args.ai_only:
        print(f"[MPPO-LEAGUE] AI-only mode: {len(checkpoints)} checkpoints selected")
        ranking = summarize_checkpoints(checkpoints)
    else:
        opponents = parse_csv_list(args.opponents)
        total = len(checkpoints) * len(opponents)
        done = 0

        for checkpoint in checkpoints:
            for opponent_name in opponents:
                done += 1
                print(f"[MPPO-LEAGUE] baseline {done}/{total}: {checkpoint} vs {opponent_name}")
                opponent = make_baseline(opponent_name)
                row = evaluate_model(
                    checkpoint,
                    opponent,
                    opponent_name,
                    games=args.games,
                    score_to_win=args.score_to_win,
                    max_turns=args.max_turns,
                    deterministic=args.deterministic,
                    seed=args.seed,
                    device=args.device,
                    gold_penalty=args.gold_penalty,
                    progress_every=args.progress_every,
                )
                baseline_rows.append(row)

        ranking = summarize_baselines(baseline_rows)
    league_rows: List[EvalRow] = []

    if args.league_games > 0 and len(ranking) > 1:
        candidates = [str(item["checkpoint"]) for item in ranking[:args.league_top_k]]
        total_matches = len(candidates) * max(0, len(candidates) - 1)
        done = 0
        for checkpoint in candidates:
            for opponent_checkpoint in candidates:
                if checkpoint == opponent_checkpoint:
                    continue
                done += 1
                print(
                    f"[MPPO-LEAGUE] checkpoint {done}/{total_matches}: "
                    f"{checkpoint} vs {opponent_checkpoint}"
                )
                opponent_model = load_checkpoint(opponent_checkpoint, args.device)
                opponent = DeterministicModelOpponent(
                    opponent_model,
                    deterministic=args.opponent_deterministic,
                )
                row = evaluate_model(
                    checkpoint,
                    opponent,
                    "checkpoint",
                    opponent_checkpoint=opponent_checkpoint,
                    games=args.league_games,
                    score_to_win=args.score_to_win,
                    max_turns=args.max_turns,
                    deterministic=args.deterministic,
                    seed=args.seed,
                    device=args.device,
                    gold_penalty=args.gold_penalty,
                    progress_every=args.progress_every,
                )
                league_rows.append(row)

    apply_league_results(ranking, league_rows, args.league_weight)

    baseline_csv = os.path.join(output_dir, "baseline_results.csv")
    league_csv = os.path.join(output_dir, "league_results.csv")
    ranking_csv = os.path.join(output_dir, "ranking.csv")
    past_txt = os.path.join(output_dir, "past_opponents.txt")

    write_rows(baseline_csv, baseline_rows)
    write_rows(league_csv, league_rows)
    write_ranking(ranking_csv, ranking)
    write_past_opponents(past_txt, ranking, args.past_opponents)

    if baseline_rows:
        print(f"[MPPO-LEAGUE] Wrote: {baseline_csv}")
    if league_rows:
        print(f"[MPPO-LEAGUE] Wrote: {league_csv}")
    print(f"[MPPO-LEAGUE] Wrote: {ranking_csv}")
    print(f"[MPPO-LEAGUE] Wrote: {past_txt}")

    for i, item in enumerate(ranking[:5], start=1):
        win_value = item["league_win_rate"] if item["league_win_rate"] != "" else item["baseline_win_rate"]
        diff_value = item["baseline_diff"] if item["baseline_diff"] != "" else 0.0
        gold_value = item["gold_bid_rate"] if item["gold_bid_rate"] != "" else 0.0
        print(
            f"[MPPO-LEAGUE] TOP{i} score={float(item['combined_score']):.2f} "
            f"win={float(win_value):.1%} "
            f"diff={float(diff_value):.2f} "
            f"gold={float(gold_value):.1%} "
            f"path={item['checkpoint']}"
        )


if __name__ == "__main__":
    main()
