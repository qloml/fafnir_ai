"""
Checkpoint league evaluator for Deep CFR Fafnir AI.

This module ranks checkpoints across policy modes, temperatures, baseline
opponents, and optional checkpoint-vs-checkpoint matches. It also exports a
past-opponent manifest that train.py can use for stronger self-play mixing.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .evaluate import (
    EvalStats,
    evaluate_policy,
    load_model_policy,
    parse_temperatures,
    set_seed,
)


@dataclass
class EvalRow:
    checkpoint: str
    iteration: str
    policy: str
    temperature: float
    opponent: str
    opponent_checkpoint: str
    games: int
    win_rate: float
    draw_rate: float
    mean_diff: float
    avg_turns: float
    avg_bid: float
    pass_rate: float
    gold_bid_rate: float
    gold_game_rate: float
    diff_gold: float
    diff_no_gold: float
    gold_actions_per_gold_game: float
    score: float


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def checkpoint_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    marker = "_iter"
    if marker in name:
        tail = name.split(marker, 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits), path
    return -1, path


def find_checkpoints(pattern: str, max_checkpoints: int) -> List[str]:
    paths = sorted(glob.glob(pattern, recursive=True), key=checkpoint_sort_key)
    if max_checkpoints > 0:
        paths = paths[-max_checkpoints:]
    return paths


def score_stats(stats: EvalStats, gold_penalty: float, gold_gap_penalty: float) -> float:
    s = stats.summary()
    gold_gap = 0.0
    if stats.gold_games > 0 and stats.no_gold_games > 0:
        gold_gap = max(0.0, s["mean_diff_no_gold_games"] - s["mean_diff_gold_games"])
    return (
        s["win_rate"] * 100.0
        + s["mean_score_diff"]
        - gold_penalty * s["ai_gold_bid_rate"]
        - gold_gap_penalty * gold_gap
    )


def make_row(
    checkpoint: str,
    ckpt: Dict,
    policy: str,
    temperature: float,
    opponent: str,
    opponent_checkpoint: str,
    stats: EvalStats,
    score: float,
) -> EvalRow:
    s = stats.summary()
    return EvalRow(
        checkpoint=os.path.abspath(checkpoint),
        iteration=str(ckpt.get("iteration", "?")),
        policy=policy,
        temperature=temperature,
        opponent=opponent,
        opponent_checkpoint=os.path.abspath(opponent_checkpoint) if opponent_checkpoint else "",
        games=int(s["games"]),
        win_rate=s["win_rate"],
        draw_rate=s["draw_rate"],
        mean_diff=s["mean_score_diff"],
        avg_turns=s["avg_turns"],
        avg_bid=s["avg_ai_bid"],
        pass_rate=s["ai_pass_rate"],
        gold_bid_rate=s["ai_gold_bid_rate"],
        gold_game_rate=s["gold_game_rate"],
        diff_gold=s["mean_diff_gold_games"],
        diff_no_gold=s["mean_diff_no_gold_games"],
        gold_actions_per_gold_game=s["avg_gold_actions_in_gold_games"],
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
            writer.writerow(row.__dict__)


def evaluate_config(
    checkpoint: str,
    policy_mode: str,
    temperature: float,
    opponent: str,
    device: torch.device,
    games: int,
    deterministic: bool,
    gold_penalty: float,
    gold_gap_penalty: float,
    opponent_checkpoint: str = "",
    opponent_temperature: float = 0.3,
) -> Optional[EvalRow]:
    try:
        policy, ckpt = load_model_policy(checkpoint, device, mode=policy_mode, name=checkpoint)
        if policy is None:
            return None
        opponent_policy = None
        opponent_name = opponent
        if opponent_checkpoint:
            opponent_name = "checkpoint"
            opponent_policy, _ = load_model_policy(
                opponent_checkpoint,
                device,
                mode="strategy",
                name=opponent_checkpoint,
            )
            if opponent_policy is None:
                return None
        stats = evaluate_policy(
            policy,
            device,
            num_games=games,
            temperature=temperature,
            opponent=opponent_name,
            opponent_policy=opponent_policy,
            opponent_temperature=opponent_temperature,
            deterministic=deterministic,
        )
    except Exception as e:
        print(f"[LEAGUE] Skip {checkpoint}: {e}")
        return None

    score = score_stats(stats, gold_penalty, gold_gap_penalty)
    return make_row(
        checkpoint,
        ckpt,
        policy_mode,
        temperature,
        opponent_name,
        opponent_checkpoint,
        stats,
        score,
    )


def summarize_configs(rows: List[EvalRow]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, float], List[EvalRow]] = {}
    for row in rows:
        key = (row.checkpoint, row.policy, row.temperature)
        grouped.setdefault(key, []).append(row)

    ranking = []
    for (checkpoint, policy, temperature), items in grouped.items():
        ranking.append({
            "checkpoint": checkpoint,
            "iteration": items[0].iteration,
            "policy": policy,
            "temperature": temperature,
            "baseline_score": sum(r.score for r in items) / len(items),
            "baseline_win_rate": sum(r.win_rate for r in items) / len(items),
            "baseline_diff": sum(r.mean_diff for r in items) / len(items),
            "gold_bid_rate": sum(r.gold_bid_rate for r in items) / len(items),
            "gold_game_rate": sum(r.gold_game_rate for r in items) / len(items),
            "diff_gold": sum(r.diff_gold for r in items) / len(items),
            "diff_no_gold": sum(r.diff_no_gold for r in items) / len(items),
            "league_score": "",
            "league_win_rate": "",
            "combined_score": sum(r.score for r in items) / len(items),
        })
    ranking.sort(key=lambda x: float(x["combined_score"]), reverse=True)
    return ranking


def apply_league_results(ranking: List[Dict[str, object]], league_rows: List[EvalRow], league_weight: float) -> None:
    by_key: Dict[Tuple[str, str, float], List[EvalRow]] = {}
    for row in league_rows:
        key = (row.checkpoint, row.policy, row.temperature)
        by_key.setdefault(key, []).append(row)

    for item in ranking:
        key = (str(item["checkpoint"]), str(item["policy"]), float(item["temperature"]))
        rows = by_key.get(key)
        baseline_score = float(item["baseline_score"])
        if not rows:
            item["combined_score"] = baseline_score
            continue
        league_score = sum(r.score for r in rows) / len(rows)
        item["league_score"] = league_score
        item["league_win_rate"] = sum(r.win_rate for r in rows) / len(rows)
        item["combined_score"] = baseline_score * (1.0 - league_weight) + league_score * league_weight
    ranking.sort(key=lambda x: float(x["combined_score"]), reverse=True)


def write_ranking(path: str, ranking: List[Dict[str, object]]) -> None:
    if not ranking:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(ranking[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranking:
            writer.writerow(item)


def write_past_opponents(path: str, ranking: List[Dict[str, object]], count: int) -> None:
    seen = set()
    selected = []
    for item in ranking:
        checkpoint = str(item["checkpoint"])
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        selected.append(checkpoint)
        if len(selected) >= count:
            break

    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ranked past-opponent checkpoints generated by cfr_ai.ai.league\n")
        for checkpoint in selected:
            f.write(f"{os.path.relpath(checkpoint, manifest_dir)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint league evaluation")
    parser.add_argument("--checkpoint-glob", default="cfr_ai/ai/checkpoints/**/*.pt")
    parser.add_argument("--max-checkpoints", type=int, default=0,
                        help="Limit to newest N checkpoints after sorting by iter (0=all)")
    parser.add_argument("--games", type=int, default=500,
                        help="Games per baseline opponent/config")
    parser.add_argument("--league-games", type=int, default=200,
                        help="Games per checkpoint-vs-checkpoint match (0=disable)")
    parser.add_argument("--league-top-k", type=int, default=8,
                        help="Top configs included in checkpoint league")
    parser.add_argument("--temperatures", default="0.1,0.2,0.3")
    parser.add_argument("--policies", default="strategy",
                        help="Comma-separated: strategy,regret")
    parser.add_argument("--opponents", default="heuristic",
                        help="Comma-separated: random,heuristic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--gold-penalty", type=float, default=20.0,
                        help="Ranking penalty multiplied by gold_bid_rate")
    parser.add_argument("--gold-gap-penalty", type=float, default=1.0,
                        help="Penalty for diff_no_gold - diff_gold when gold games are worse")
    parser.add_argument("--league-weight", type=float, default=0.5,
                        help="Weight of checkpoint-vs-checkpoint score in combined ranking")
    parser.add_argument("--past-opponents", type=int, default=8,
                        help="Number of ranked checkpoints written to past_opponents.txt")
    parser.add_argument("--output-dir", default="cfr_ai/ai/reports")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoints = find_checkpoints(args.checkpoint_glob, args.max_checkpoints)
    if not checkpoints:
        print(f"[LEAGUE] No checkpoints matched: {args.checkpoint_glob}")
        return

    temperatures = parse_temperatures(args.temperatures, 0.3)
    policies = parse_csv_list(args.policies)
    opponents = parse_csv_list(args.opponents)
    run_name = time.strftime("league_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    baseline_rows: List[EvalRow] = []
    total = len(checkpoints) * len(policies) * len(temperatures) * len(opponents)
    done = 0
    for checkpoint in checkpoints:
        for policy in policies:
            for temperature in temperatures:
                for opponent in opponents:
                    done += 1
                    print(
                        f"[LEAGUE] baseline {done}/{total} "
                        f"{checkpoint} policy={policy} temp={temperature} opponent={opponent}"
                    )
                    row = evaluate_config(
                        checkpoint,
                        policy,
                        temperature,
                        opponent,
                        device,
                        args.games,
                        args.deterministic,
                        args.gold_penalty,
                        args.gold_gap_penalty,
                    )
                    if row is not None:
                        baseline_rows.append(row)

    ranking = summarize_configs(baseline_rows)
    league_rows: List[EvalRow] = []
    if args.league_games > 0 and len(ranking) > 1:
        league_candidates = ranking[:args.league_top_k]
        opponent_paths = []
        for item in league_candidates:
            checkpoint = str(item["checkpoint"])
            if checkpoint not in opponent_paths:
                opponent_paths.append(checkpoint)

        total_matches = 0
        for item in league_candidates:
            for opponent_checkpoint in opponent_paths:
                if str(item["checkpoint"]) != opponent_checkpoint:
                    total_matches += 1

        done = 0
        for item in league_candidates:
            checkpoint = str(item["checkpoint"])
            policy = str(item["policy"])
            temperature = float(item["temperature"])
            for opponent_checkpoint in opponent_paths:
                if checkpoint == opponent_checkpoint:
                    continue
                done += 1
                print(
                    f"[LEAGUE] checkpoint {done}/{total_matches} "
                    f"{checkpoint} vs {opponent_checkpoint} policy={policy} temp={temperature}"
                )
                row = evaluate_config(
                    checkpoint,
                    policy,
                    temperature,
                    "checkpoint",
                    device,
                    args.league_games,
                    args.deterministic,
                    args.gold_penalty,
                    args.gold_gap_penalty,
                    opponent_checkpoint=opponent_checkpoint,
                )
                if row is not None:
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

    print(f"[LEAGUE] Wrote: {baseline_csv}")
    if league_rows:
        print(f"[LEAGUE] Wrote: {league_csv}")
    print(f"[LEAGUE] Wrote: {ranking_csv}")
    print(f"[LEAGUE] Wrote: {past_txt}")

    for i, item in enumerate(ranking[:5], start=1):
        print(
            f"[LEAGUE] TOP{i} score={float(item['combined_score']):.2f} "
            f"win={float(item['baseline_win_rate']):.1%} "
            f"gold={float(item['gold_bid_rate']):.1%} "
            f"policy={item['policy']} temp={item['temperature']} "
            f"path={item['checkpoint']}"
        )


if __name__ == "__main__":
    main()
