# rl/train.py
"""
FAFNIR RL Training Script — Self-play with MaskablePPO.

Usage:
    python rl/train.py                          # default settings
    python rl/train.py --total-steps 2000000    # longer training
    python rl/train.py --score-to-win 50        # harder games
"""
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.game_env import (
    OpponentManager, ModelOpponent, RandomOpponent,
    N_COLORS, MAX_BID_PER_COLOR,
)
from rl.game_env_fast import FafnirFastEnv


# ==========================================
# Self-play callback
# ==========================================

class SelfPlayCallback(BaseCallback):
    """
    Periodically updates the opponent to a saved checkpoint of the training model.
    Implements curriculum: starts vs random, then vs older self.
    """

    def __init__(self, save_path: str, update_freq: int = 20_000,
                 keep_random_ratio: float = 0.2, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.update_freq = update_freq
        self.keep_random_ratio = keep_random_ratio
        self._checkpoints = []
        self._win_count = 0
        self._loss_count = 0
        self._episode_count = 0

    def _on_step(self) -> bool:
        # Track wins/losses from info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                r = info["episode"]["r"]
                if r > 0:
                    self._win_count += 1
                elif r < 0:
                    self._loss_count += 1

        if self.n_calls % self.update_freq == 0 and self.n_calls > 0:
            # Save current model as potential opponent
            ckpt_path = os.path.join(self.save_path, f"selfplay_{self.n_calls}")
            self.model.save(ckpt_path)
            self._checkpoints.append(ckpt_path)

            # Keep only last 10 checkpoints for opponent sampling
            if len(self._checkpoints) > 10:
                self._checkpoints = self._checkpoints[-10:]

            # Update opponent: mix of random and past self
            new_opponent = None
            if np.random.random() < self.keep_random_ratio:
                new_opponent = RandomOpponent()
                opp_name = "Random"
            else:
                # Pick a random past checkpoint
                ckpt = np.random.choice(self._checkpoints)
                try:
                    loaded = MaskablePPO.load(ckpt)
                    new_opponent = ModelOpponent(loaded)
                    opp_name = os.path.basename(ckpt)
                except Exception as e:
                    print(f"[WARN] Failed to load checkpoint {ckpt}: {e}")
                    new_opponent = RandomOpponent()
                    opp_name = "Random (fallback)"

            # Broadcast new opponent to all parallel environments
            self.training_env.env_method("set_opponent", new_opponent)

            win_rate = (self._win_count / max(1, self._episode_count)) * 100
            if self.verbose:
                print(f"\n[SelfPlay] Step {self.n_calls}: "
                      f"opponent -> {opp_name} | "
                      f"Win rate: {win_rate:.1f}% "
                      f"({self._win_count}W/{self._loss_count}L/{self._episode_count}ep)")

            self._win_count = 0
            self._loss_count = 0
            self._episode_count = 0

        return True


class WinRateLogCallback(BaseCallback):
    """Logs win rate to tensorboard."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._wins = 0
        self._losses = 0
        self._draws = 0
        self._episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episodes += 1
                r = info["episode"]["r"]
                if r > 0.3:
                    self._wins += 1
                elif r < -0.3:
                    self._losses += 1
                else:
                    self._draws += 1

        if self.n_calls % self.log_freq == 0 and self._episodes > 0:
            wr = self._wins / max(1, self._episodes)
            lr_ = self._losses / max(1, self._episodes)
            self.logger.record("fafnir/win_rate", wr)
            self.logger.record("fafnir/loss_rate", lr_)
            self.logger.record("fafnir/episodes", self._episodes)
            self._wins = 0
            self._losses = 0
            self._draws = 0
            self._episodes = 0

        return True


class LiveViewCallback(BaseCallback):
    """Prints the live CUI game state from env 0 periodically and tracks win rate."""
    def __init__(self, display_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.display_freq = display_freq
        self.recent_wins = 0
        self.recent_losses = 0
        self.recent_episodes = 0

    def _on_step(self) -> bool:
        # Track win/loss from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.recent_episodes += 1
                r = info["episode"]["r"]
                if r > 0.3:
                    self.recent_wins += 1
                elif r < -0.3:
                    self.recent_losses += 1

        if self.n_calls % self.display_freq == 0:
            states = self.training_env.env_method("get_cui_state", indices=[0])
            wr = (self.recent_wins / max(1, self.recent_episodes)) * 100
            
            if states and states[0]:
                print(f"\n[LiveView - Step {self.n_calls:,}]")
                print(f"  >>> 最近の勝率 (Recent Win Rate): {wr:.1f}% ({self.recent_wins}勝 / {self.recent_losses}敗 / 計{self.recent_episodes}試合)")
                
            # 表示ごとにリセットして直近の勝率を測る
            self.recent_wins = 0
            self.recent_losses = 0
            self.recent_episodes = 0
            
        return True


# ==========================================
# Action masker wrapper
# ==========================================

from stable_baselines3.common.monitor import Monitor

def mask_fn(env: FafnirFastEnv) -> np.ndarray:
    return env.valid_action_mask()


def make_masked_env(score_to_win: int, max_turns: int):
    def _init():
        env = FafnirFastEnv(score_to_win=score_to_win, max_turns=max_turns)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)  # 勝率等の統計トラッキングに必須
        return env
    return _init


# ==========================================
# Main training
# ==========================================

def train(args):
    print("=" * 60)
    print("FAFNIR RL Training - Self-Play with MaskablePPO")
    print("=" * 60)
    print(f"  Total steps:    {args.total_steps:,}")
    print(f"  Score to win:   {args.score_to_win}")
    print(f"  Max turns/ep:   {args.max_turns}")
    print(f"  N envs:         {args.n_envs}")
    print(f"  Update freq:    {args.update_freq:,}")
    print(f"  Device:         {args.device}")
    print(f"  Save dir:       {args.save_dir}")
    print("=" * 60)

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Start with random opponent
    OpponentManager.set(RandomOpponent())

    # Create vectorized environment with multiple processes
    env = SubprocVecEnv([make_masked_env(args.score_to_win, args.max_turns)
                         for _ in range(args.n_envs)])

    # Resume or create new model
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device=args.device,
                                  tensorboard_log=log_dir)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device=args.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )

    # Callbacks
    selfplay_cb = SelfPlayCallback(
        save_path=ckpt_dir,
        update_freq=args.update_freq,
        keep_random_ratio=0.2,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=ckpt_dir,
        name_prefix="fafnir_rl",
        verbose=1,
    )

    winrate_cb = WinRateLogCallback(log_freq=5000)
    liveview_cb = LiveViewCallback(display_freq=500)

    # Train!
    print("\nStarting training...")
    t0 = time.time()

    model.learn(
        total_timesteps=args.total_steps,
        callback=[selfplay_cb, checkpoint_cb, winrate_cb, liveview_cb],
        progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete! Elapsed: {elapsed / 60:.1f} min")

    # Save final model
    final_path = os.path.join(args.save_dir, "fafnir_final")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    env.close()


def main():
    ap = argparse.ArgumentParser(description="Train FAFNIR RL agent")
    ap.add_argument("--total-steps", type=int, default=500_000,
                    help="Total training timesteps")
    ap.add_argument("--score-to-win", type=int, default=40,
                    help="Score to win (lower = shorter episodes)")
    ap.add_argument("--max-turns", type=int, default=500,
                    help="Max turns per episode before truncation")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="Number of parallel environments")
    ap.add_argument("--update-freq", type=int, default=20_000,
                    help="Steps between self-play opponent updates")
    ap.add_argument("--save-freq", type=int, default=50_000,
                    help="Steps between checkpoint saves")
    ap.add_argument("--save-dir", type=str, default="rl/output",
                    help="Directory for checkpoints and logs")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: auto, cpu, cuda")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to model to resume training from")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
