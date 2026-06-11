"""
Training script for Deep CFR Fafnir AI (v2).

Usage:
    python -m cfr_ai.ai.train [options]

Options:
    --iterations N       Number of CFR iterations (default: 4000)
    --traversals N       Traversals per iteration (default: 2000)
    --hidden N           Hidden layer size (default: 256)
    --lr FLOAT           Learning rate (default: 5e-4)
    --batch-size N       Training batch size (default: 2048)
    --train-steps N      Training steps per iteration (default: 100)
    --max-depth N        Max traversal depth (default: 50)
    --augments N         Symmetry augmentations per sample (default: 3)
    --save-dir PATH      Checkpoint save directory
    --resume             Resume from checkpoint
    --device DEVICE      cpu/cuda/auto (default: auto)
    --workers N          Number of parallel workers (default: 7)
    --eval-every N       Evaluate vs random every N iterations (default: 0)
    --no-score-rand      Disable score randomization
"""
import argparse
import os
import time
import signal
import sys
from .trainer import DeepCFRTrainer, PROGRAM_VERSION
from .game_engine import warmup as warmup_game_engine
from .observation import OBS_DIM


def main():
    parser = argparse.ArgumentParser(description="Deep CFR Training for Fafnir (v2)")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--traversals", type=int, default=2000,
                        help="Traversals per iteration (1 traversal = 1 round)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=50,
                        help="Max depth per traversal (1 round is typically 10-25 turns)")
    parser.add_argument("--augments", type=int, default=3)
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000,
                        help="Reservoir buffer capacity per network")
    parser.add_argument("--save-dir", type=str, default="cfr_ai/ai/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--workers", type=int, default=7,
                        help="Parallel workers (0=auto, 1=single-process)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Evaluate vs random every N iterations (0=disable)")
    parser.add_argument("--final-eval-games", type=int, default=0,
                        help="Final evaluation games after training (0=disable)")
    parser.add_argument("--no-score-rand", action="store_true",
                        help="Disable score randomization")
    parser.add_argument("--target-mode", choices=["terminal", "dense"], default="terminal",
                        help="Training target: pure round terminal value or per-point dense value")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Initial exploration epsilon (decays over training)")
    parser.add_argument("--program-version", type=int, default=PROGRAM_VERSION,
                        help="Program/checkpoint compatibility version used in archive names")
    parser.add_argument("--archive-every", type=int, default=100,
                        help="Save versioned checkpoint archive every N iterations (0=disable)")
    parser.add_argument("--past-opponent-prob", type=float, default=0.15,
                        help="Probability that the non-traverser uses a frozen past self")
    parser.add_argument("--max-past-opponents", type=int, default=8,
                        help="Maximum archived past opponents kept in memory")
    parser.add_argument("--past-opponent-selection",
                        choices=["recent", "spread", "random", "manifest"],
                        default="recent",
                        help="How archived past opponents are selected")
    parser.add_argument("--past-opponent-manifest", default=None,
                        help="Text or CSV file listing checkpoint paths to use as past opponents")
    args = parser.parse_args()

    warmup_game_engine()

    # Auto-detect workers: cap at 4 to limit memory usage
    if args.workers <= 0:
        cpu_count = os.cpu_count() or 4
        args.workers = min(4, max(1, cpu_count - 1))
        print(f"[DeepCFR v2] Auto-detected {cpu_count} CPU cores, using {args.workers} workers")

    trainer = DeepCFRTrainer(
        hidden_dim=args.hidden,
        lr=args.lr,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        train_steps_per_iter=args.train_steps,
        max_depth=args.max_depth,
        num_augments=args.augments,
        explore_epsilon=args.epsilon,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.workers,
        score_randomize=not args.no_score_rand,
        target_mode=args.target_mode,
        program_version=args.program_version,
        past_opponent_prob=args.past_opponent_prob,
        max_past_opponents=args.max_past_opponents,
        past_opponent_selection=args.past_opponent_selection,
        past_opponent_manifest=args.past_opponent_manifest,
    )

    if args.resume:
        trainer.load()
    trainer.load_past_opponents_from_dir()

    # Graceful Ctrl+C handling: save before exit
    interrupted = False
    last_archived_iteration = None

    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force exit
            print("\n[DeepCFR v2] Force exit!")
            os._exit(1)
        interrupted = True
        print(f"\n[DeepCFR v2] Interrupt received! Saving checkpoint...")
        try:
            trainer.shutdown_pool()
            trainer.save()
            if args.archive_every > 0:
                trainer.save_archive()
        except Exception as e:
            print(f"[DeepCFR v2] Error during save: {e}")
        print(f"[DeepCFR v2] Saved. Exiting.")
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n{'='*60}")
    print(f"  Deep CFR Training for Fafnir (v2)")
    print(f"  Iterations: {args.iterations}")
    print(f"  Traversals/iter: {args.traversals}")
    print(f"  Hidden dim: {args.hidden}")
    print(f"  Obs dim: {OBS_DIM}")
    print(f"  Workers: {args.workers}")
    print(f"  Device: {trainer.device}")
    print(f"  Score randomization: {not args.no_score_rand}")
    print(f"  Target mode: {args.target_mode}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Program version: v{args.program_version}")
    print(f"  Archive every: {args.archive_every} iters")
    print(f"  Past opponent prob: {args.past_opponent_prob}")
    print(f"  Max past opponents: {args.max_past_opponents}")
    print(f"  Past opponent selection: {args.past_opponent_selection}")
    if args.past_opponent_manifest:
        print(f"  Past opponent manifest: {args.past_opponent_manifest}")
    print(f"  Eval every: {args.eval_every} iters")
    print(f"  Ctrl+C to stop (progress will be saved)")
    print(f"{'='*60}\n")

    total_start = time.time()

    for i in range(args.iterations):
        if interrupted:
            break
        stats = trainer.run_iteration(num_traversals=args.traversals)

        # Periodic evaluation
        if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            try:
                from .evaluate import evaluate_vs_random
                win_rate = evaluate_vs_random(trainer.strategy_net, trainer.device,
                                             num_games=200)
                print(f"[EVAL] vs Random: win_rate={win_rate:.1%} (200 games)")
            except Exception as e:
                print(f"[EVAL] Error: {e}")

        # Save periodically
        if (i + 1) % args.save_every == 0:
            trainer.save()
        if args.archive_every > 0 and (i + 1) % args.archive_every == 0:
            trainer.save_archive()
            trainer.add_current_to_past_opponents()
            last_archived_iteration = trainer.iteration

    # Final save and cleanup
    if not interrupted:
        trainer.save()
        if args.archive_every > 0 and last_archived_iteration != trainer.iteration:
            trainer.save_archive()
            trainer.add_current_to_past_opponents()
    trainer.shutdown_pool()

    total_time = time.time() - total_start
    print(f"\n[DeepCFR v2] Training complete! Total time: {total_time:.1f}s")
    print(f"[DeepCFR v2] Total traversals: {trainer.total_traversals}")

    # Final evaluation
    if args.final_eval_games > 0:
        try:
            from .evaluate import evaluate_vs_random
            win_rate = evaluate_vs_random(trainer.strategy_net, trainer.device,
                                         num_games=args.final_eval_games)
            print(f"[EVAL] Final vs Random: win_rate={win_rate:.1%} ({args.final_eval_games} games)")
        except Exception as e:
            print(f"[EVAL] Final eval error: {e}")


if __name__ == "__main__":
    main()
