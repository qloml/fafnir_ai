"""
Training script for Deep CFR Fafnir AI.

Usage:
    python -m cfr_ai.ai.train [options]

Options:
    --iterations N       Number of CFR iterations (default: 1000)
    --traversals N       Traversals per iteration (default: 200)
    --hidden N           Hidden layer size (default: 256)
    --lr FLOAT           Learning rate (default: 1e-3)
    --batch-size N       Training batch size (default: 2048)
    --train-steps N      Training steps per iteration (default: 1000)
    --max-depth N        Max traversal depth (default: 30)
    --augments N         Symmetry augmentations per sample (default: 3)
    --save-dir PATH      Checkpoint save directory
    --resume             Resume from checkpoint
    --device DEVICE      cpu/cuda/auto (default: auto)
    --workers N          Number of parallel workers (default: auto = CPU cores - 1)
"""
import argparse
import os
import time
import signal
import sys
from .trainer import DeepCFRTrainer


def main():
    parser = argparse.ArgumentParser(description="Deep CFR Training for Fafnir")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traversals", type=int, default=1000,
                        help="Traversals per iteration (1 traversal = 1 round)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=25,
                        help="Max depth per traversal (1 round is typically 10-20 turns)")
    parser.add_argument("--augments", type=int, default=3)
    parser.add_argument("--buffer-capacity", type=int, default=500_000,
                        help="Reservoir buffer capacity per network")
    parser.add_argument("--save-dir", type=str, default="cfr_ai/ai/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (0=auto, 1=single-process)")
    args = parser.parse_args()

    # Auto-detect workers: cap at 4 to limit memory usage
    if args.workers <= 0:
        cpu_count = os.cpu_count() or 4
        args.workers = min(4, max(1, cpu_count - 1))
        print(f"[DeepCFR] Auto-detected {cpu_count} CPU cores, using {args.workers} workers")

    trainer = DeepCFRTrainer(
        hidden_dim=args.hidden,
        lr=args.lr,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        train_steps_per_iter=args.train_steps,
        max_depth=args.max_depth,
        num_augments=args.augments,
        device=args.device,
        save_dir=args.save_dir,
        num_workers=args.workers,
    )

    if args.resume:
        trainer.load()

    # Graceful Ctrl+C handling: save before exit
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: force exit
            print("\n[DeepCFR] Force exit!")
            os._exit(1)
        interrupted = True
        print(f"\n[DeepCFR] Interrupt received! Saving checkpoint...")
        try:
            trainer.shutdown_pool()
            trainer.save()
        except Exception as e:
            print(f"[DeepCFR] Error during save: {e}")
        print(f"[DeepCFR] Saved. Exiting.")
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n{'='*60}")
    print(f"  Deep CFR Training for Fafnir")
    print(f"  Iterations: {args.iterations}")
    print(f"  Traversals/iter: {args.traversals}")
    print(f"  Hidden dim: {args.hidden}")
    print(f"  Workers: {args.workers}")
    print(f"  Device: {trainer.device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Ctrl+C to stop (progress will be saved)")
    print(f"{'='*60}\n")

    total_start = time.time()

    for i in range(args.iterations):
        if interrupted:
            break
        stats = trainer.run_iteration(num_traversals=args.traversals)

        # Save periodically
        if (i + 1) % args.save_every == 0:
            trainer.save()

    # Final save and cleanup
    if not interrupted:
        trainer.save()
    trainer.shutdown_pool()

    total_time = time.time() - total_start
    print(f"\n[DeepCFR] Training complete! Total time: {total_time:.1f}s")
    print(f"[DeepCFR] Total traversals: {trainer.total_traversals}")


if __name__ == "__main__":
    main()
