# rl/game_env_fast.py
"""
Fast FAFNIR Gymnasium environment using Numba JIT-compiled engine.
Drop-in replacement for game_env.FafnirEnv (for random opponent training).

Observation: 36-dim vector (see fast_engine.build_obs for layout).
"""
import numpy as np
import gymnasium
from gymnasium import spaces

from rl.fast_engine import (
    fast_reset, fast_step, build_obs, build_mask, warmup, _random_bid,
    N_COLORS, MAX_BID, OBS_DIM,
)
from rl.game_env import OpponentManager, ModelOpponent

# Warm up JIT on import
warmup()


class FafnirFastEnv(gymnasium.Env):
    """Fast FAFNIR env. Agent=P0, opponent managed by OpponentManager."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, score_to_win=40, max_turns=500, opponent=None, render_mode=None):
        super().__init__()
        self.score_to_win = np.int32(score_to_win)
        self.max_turns = np.int32(max_turns)
        self.render_mode = render_mode
        self._opponent = opponent

        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([MAX_BID + 1] * N_COLORS)

        self.hands = None
        self.bag = None
        self.trash = None
        self.offer = None
        self.scores = None
        self.state = None
        self.known = None

    def _get_opponent(self):
        if self._opponent is not None:
            return self._opponent
        return OpponentManager.get()

    def set_opponent(self, opponent):
        """Called by SubprocVecEnv to broadcast new opponent."""
        self._opponent = opponent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.hands, self.bag, self.trash, self.offer, self.scores, \
            self.state, self.known = fast_reset(self.score_to_win)
        obs = build_obs(self.hands, self.bag, self.trash, self.offer,
                        self.scores, self.state, self.known,
                        np.int32(0), self.score_to_win)
        return obs, self._info()

    def step(self, action):
        act = np.asarray(action, dtype=np.int64)

        # 1. Agent bid
        bid0 = np.zeros(N_COLORS, dtype=np.int32)
        for c in range(N_COLORS):
            if self.offer[c] > 0:
                bid0[c] = np.int32(0)
            else:
                bid0[c] = min(np.int32(act[c]), self.hands[0, c])

        # 2. Opponent bid
        opponent = self._get_opponent()
        if isinstance(opponent, ModelOpponent):
            opp_obs = build_obs(self.hands, self.bag, self.trash, self.offer,
                                self.scores, self.state, self.known,
                                np.int32(1), self.score_to_win)
            opp_mask = build_mask(self.hands, self.offer, np.int32(1))
            opp_act = opponent.choose_bid_from_obs(opp_obs, opp_mask)
            bid1 = np.zeros(N_COLORS, dtype=np.int32)
            for c in range(N_COLORS):
                if self.offer[c] > 0:
                    bid1[c] = np.int32(0)
                else:
                    bid1[c] = min(np.int32(opp_act[c]), self.hands[1, c])
        else:
            bid1 = _random_bid(self.hands[1], self.offer)

        # 3. Advance state
        reward, terminated, truncated = fast_step(
            self.hands, self.bag, self.trash, self.offer,
            self.scores, self.state, self.known, bid0, bid1,
            self.score_to_win, self.max_turns,
        )
        obs = build_obs(self.hands, self.bag, self.trash, self.offer,
                        self.scores, self.state, self.known,
                        np.int32(0), self.score_to_win)
        return obs, float(reward), bool(terminated), bool(truncated), self._info()

    def valid_action_mask(self):
        return build_mask(self.hands, self.offer, np.int32(0))

    def _info(self):
        bl = int(self.bag.sum()) if self.bag is not None else 0
        return {
            "scores": self.scores.copy() if self.scores is not None else np.zeros(2, dtype=np.int32),
            "round": int(self.state[1]) if self.state is not None else 0,
            "turn": int(self.state[2]) if self.state is not None else 0,
            "total_turns": int(self.state[3]) if self.state is not None else 0,
            "bag_left": bl,
            "caretaker": int(self.state[0]) if self.state is not None else 0,
        }

    def render(self):
        if self.render_mode == "human" and self.state is not None:
            print(self.get_cui_state())

    def get_cui_state(self) -> str:
        if self.state is None:
            return "Env not initialized"
        colors = ["GLD", "RED", "ORG", "YLW", "GRN", "BLU"]
        
        # 1. format offer
        offer_str = ""
        for i in range(N_COLORS):
            if self.offer[i] > 0:
                offer_str += f"{colors[i]}:{self.offer[i]} "
        if not offer_str:
            offer_str = "None"
            
        # 2. format scores and hands
        p0_hand = sum(self.hands[0])
        p1_hand = sum(self.hands[1])
        
        return (f"--- [LIVE GAME] R{self.state[1]} T{self.state[2]} ---\n"
                f"  Score: P0(AI)={self.scores[0]} vs P1(Opp)={self.scores[1]} | Bag: {self.bag.sum()}\n"
                f"  Offer: {offer_str}\n"
                f"  Hands: P0={p0_hand} stones, P1={p1_hand} stones\n"
                f"---------------------------------")
