"""
Training script for Deep CFR Fafnir AI.

Usage:
    python -m cfr_ai.ai.train [options]

Options:
    --iterations N       Number of CFR iterations (default: 100)
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
"""
import argparse
import time
from .trainer import DeepCFRTrainer


def main():
    parser = argparse.ArgumentParser(description="Deep CFR Training for Fafnir")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--augments", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="cfr_ai/ai/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N iterations")
    args = parser.parse_args()

    trainer = DeepCFRTrainer(
        hidden_dim=args.hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        train_steps_per_iter=args.train_steps,
        max_depth=args.max_depth,
        num_augments=args.augments,
        device=args.device,
        save_dir=args.save_dir,
    )

    if args.resume:
        trainer.load()

    print(f"\n{'='*60}")
    print(f"  Deep CFR Training for Fafnir")
    print(f"  Iterations: {args.iterations}")
    print(f"  Traversals/iter: {args.traversals}")
    print(f"  Hidden dim: {args.hidden}")
    print(f"  Device: {trainer.device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for i in range(args.iterations):
        stats = trainer.run_iteration(num_traversals=args.traversals)

        # Save periodically
        if (i + 1) % args.save_every == 0:
            trainer.save()

    # Final save
    trainer.save()

    total_time = time.time() - total_start
    print(f"\n[DeepCFR] Training complete! Total time: {total_time:.1f}s")
    print(f"[DeepCFR] Total traversals: {trainer.total_traversals}")


if __name__ == "__main__":
    main()
