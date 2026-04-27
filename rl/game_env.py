# rl/game_env.py
"""
FAFNIR ONLINE — Gymnasium environment for RL training.
High-speed game simulator (no Socket.IO), replicating server.py logic exactly.
"""
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

# ==========================================
# Constants (match server.py)
# ==========================================
GOLD = "gold"
COLORS_NAMES = ["gold", "red", "orange", "yellow", "green", "blue"]
N_COLORS = 6  # gold=0, red=1, orange=2, yellow=3, green=4, blue=5
COLOR_IDX = {c: i for i, c in enumerate(COLORS_NAMES)}

INITIAL_BAG = np.array([20, 12, 12, 12, 12, 12], dtype=np.int32)  # gold, r, o, y, g, b
TRASH_LIMIT = 6
SEED_TRASH_N = 3
POINT_CHIP = 1

MAX_BID_PER_COLOR = 10  # action values 0..MAX_BID_PER_COLOR

# ==========================================
# Fast game engine (numpy-based)
# ==========================================

def make_bag() -> np.ndarray:
    return INITIAL_BAG.copy()


def draw_one(bag: np.ndarray, rng: np.random.Generator) -> Optional[int]:
    total = int(bag.sum())
    if total == 0:
        return None
    r = rng.integers(0, total)
    cum = 0
    for i in range(N_COLORS):
        cum += bag[i]
        if r < cum:
            bag[i] -= 1
            return i
    return None


def draw_n(bag: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    drawn = np.zeros(N_COLORS, dtype=np.int32)
    for _ in range(n):
        c = draw_one(bag, rng)
        if c is None:
            break
        drawn[c] += 1
    return drawn


def setup_offer(bag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw 2 stones for the offer. If all same color, keep drawing 2 more until mixed or bag empty."""
    offer = np.zeros(N_COLORS, dtype=np.int32)
    total = int(bag.sum())
    if total == 0:
        return offer

    for _ in range(50):  # safety limit
        total = int(bag.sum())
        if total == 0:
            break
        draw_count = min(2, total)
        for _ in range(draw_count):
            c = draw_one(bag, rng)
            if c is not None:
                offer[c] += 1

        # Check if offer has more than 1 color
        n_colors = sum(1 for c in range(N_COLORS) if offer[c] > 0)
        if n_colors > 1 or int(bag.sum()) == 0:
            break

    return offer


def rank_colors_by_total(hands: np.ndarray) -> List[Tuple[int, int]]:
    """Rank non-gold colors (idx 1-5) by total count across both players.
    Returns list of (color_idx, total_count) sorted by (-count, priority).
    """
    totals = []
    for c in range(1, N_COLORS):  # skip gold
        t = int(hands[0, c] + hands[1, c])
        totals.append((c, t))
    totals.sort(key=lambda x: (-x[1], x[0]))
    return totals


def compute_hand_score(hands: np.ndarray, player: int, ranked: List[Tuple[int, int]]) -> int:
    """Compute round-end score addition for a player."""
    first_color = ranked[0][0] if ranked else None
    second_color = ranked[1][0] if len(ranked) > 1 else None

    score = int(hands[player, 0])  # gold = 1 per stone

    for c in range(1, N_COLORS):
        cnt = int(hands[player, c])
        if cnt == 0 or cnt >= 5:
            continue
        if c == first_color:
            score += cnt * 3
        elif c == second_color:
            score += cnt * 2
        else:
            score += cnt * (-1)

    return score


def clamp_score(x: int) -> int:
    return max(0, x)


# ==========================================
# Opponent policies
# ==========================================

class RandomOpponent:
    """Bids 0-3 random valid stones."""
    def choose_bid(self, hand: np.ndarray, offer: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        bid = np.zeros(N_COLORS, dtype=np.int32)
        # Pick random number of stones to bid (0-3)
        biddable = []
        for c in range(N_COLORS):
            if offer[c] == 0 and hand[c] > 0:
                for _ in range(hand[c]):
                    biddable.append(c)

        if not biddable:
            return bid

        n = rng.integers(1, min(4, len(biddable)) + 1)
        chosen = list(rng.choice(len(biddable), size=min(n, len(biddable)), replace=False))
        for idx in chosen:
            bid[biddable[idx]] += 1
        return bid


class ModelOpponent:
    """Uses a trained MaskablePPO model to choose bids."""
    def __init__(self, model):
        self.model = model

    def choose_bid(self, hand: np.ndarray, offer: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        # Build observation from opponent's perspective
        # This is called by the env with opponent's view
        obs = None  # will be set by env
        mask = None  # will be set by env
        raise NotImplementedError("Use choose_bid_from_obs instead")

    def choose_bid_from_obs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
        return action


# ==========================================
# Shared opponent manager (for self-play)
# ==========================================

class OpponentManager:
    _opponent = None

    @classmethod
    def set(cls, opponent):
        cls._opponent = opponent

    @classmethod
    def get(cls):
        return cls._opponent or RandomOpponent()


# ==========================================
# Gymnasium Environment
# ==========================================

class FafnirEnv(gymnasium.Env):
    """
    FAFNIR auction game environment for RL training.

    The agent is always player 0 from its own perspective.
    The opponent (player 1) is controlled by an external policy.

    Observation: 25-dim normalized float vector.
    Action: MultiDiscrete([MAX_BID+1] * 6) — bid count per color.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, score_to_win: int = 40, max_turns: int = 500,
                 opponent=None, render_mode=None):
        super().__init__()

        self.score_to_win = score_to_win
        self.max_turns = max_turns
        self._opponent = opponent
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(36,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [MAX_BID_PER_COLOR + 1] * N_COLORS
        )

        # Internal state
        self._rng = np.random.default_rng()
        self._reset_game()

    def _get_opponent(self):
        if self._opponent is not None:
            return self._opponent
        return OpponentManager.get()

    def _reset_game(self):
        self.bag = make_bag()
        self.hands = np.zeros((2, N_COLORS), dtype=np.int32)
        self.trash = np.zeros(N_COLORS, dtype=np.int32)
        self.offer = np.zeros(N_COLORS, dtype=np.int32)
        self.scores = np.zeros(2, dtype=np.int32)
        self.caretaker = 0
        self.round_num = 1
        self.turn_num = 1
        self.total_turns = 0
        self.last_winner = -1
        self.known = np.zeros((2, N_COLORS), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_game()
        self.caretaker = int(self._rng.integers(0, 2))

        # Deal initial hands
        for p in range(2):
            n = 11 if p == self.caretaker else 10
            self.hands[p] = draw_n(self.bag, n, self._rng)

        # Seed trash
        seeded = draw_n(self.bag, SEED_TRASH_N, self._rng)
        self.trash += seeded

        # Setup first offer
        self.offer = setup_offer(self.bag, self._rng)

        return self._get_obs(0), self._get_info()

    def _compute_potential(self, player: int) -> float:
        """Compute expected round-end score for a player given current hands."""
        ranked = rank_colors_by_total(self.hands)
        return float(compute_hand_score(self.hands, player, ranked))

    def _get_obs(self, player: int) -> np.ndarray:
        """Build 36-dim observation from player's perspective."""
        other = 1 - player
        obs = np.zeros(36, dtype=np.float32)

        # 0-5: my hand counts (normalized by max possible)
        for c in range(N_COLORS):
            obs[c] = self.hands[player, c] / max(1, INITIAL_BAG[c])

        # 6-11: offer counts (use 10 as max for multi-stone offers)
        for c in range(N_COLORS):
            obs[6 + c] = self.offer[c] / 10.0

        # 12-17: trash counts (normalized by limit)
        for c in range(N_COLORS):
            obs[12 + c] = self.trash[c] / float(TRASH_LIMIT)

        # 18-23: opponent's confirmed hand
        for c in range(N_COLORS):
            obs[18 + c] = self.known[other, c] / max(1, INITIAL_BAG[c])

        # 24: opponent's unknown stone count
        opp_total = int(self.hands[other].sum())
        opp_known_total = int(self.known[other].sum())
        opp_unknown = max(0, opp_total - opp_known_total)
        obs[24] = opp_unknown / 15.0

        # 25-30: my confirmed hand (what opponent knows about me)
        for c in range(N_COLORS):
            obs[25 + c] = self.known[player, c] / max(1, INITIAL_BAG[c])

        # 31: my score
        obs[31] = self.scores[player] / float(max(1, self.score_to_win))

        # 32: opponent score
        obs[32] = self.scores[other] / float(max(1, self.score_to_win))

        # 33: bag remaining
        obs[33] = int(self.bag.sum()) / float(max(1, INITIAL_BAG.sum()))

        # 34: am I caretaker
        obs[34] = 1.0 if self.caretaker == player else 0.0

        # 35: my hand's current potential score
        potential = self._compute_potential(player)
        obs[35] = (potential + 15.0) / 75.0

        return np.clip(obs, 0.0, 1.0)

    def _get_action_mask(self, player: int) -> np.ndarray:
        """Build action mask for MultiDiscrete.
        For each color: mask[color*(MAX_BID+1) + value] = valid?
        """
        mask_size = N_COLORS * (MAX_BID_PER_COLOR + 1)
        mask = np.zeros(mask_size, dtype=np.int8)

        for c in range(N_COLORS):
            base = c * (MAX_BID_PER_COLOR + 1)
            if self.offer[c] > 0:
                # Cannot bid colors in the offer — only 0 is valid
                mask[base] = 1
            else:
                hand_count = int(self.hands[player, c])
                max_valid = min(hand_count, MAX_BID_PER_COLOR)
                for v in range(max_valid + 1):
                    mask[base + v] = 1

        return mask

    def valid_action_mask(self) -> np.ndarray:
        """For MaskablePPO — returns mask for player 0 (the agent)."""
        return self._get_action_mask(0)

    def _action_to_bid(self, action: np.ndarray, player: int) -> np.ndarray:
        """Convert action array to valid bid, clamping to hand counts."""
        bid = np.zeros(N_COLORS, dtype=np.int32)
        for c in range(N_COLORS):
            if self.offer[c] > 0:
                bid[c] = 0  # can't bid offer colors
            else:
                bid[c] = min(int(action[c]), int(self.hands[player, c]))
        return bid

    def _resolve_auction(self, bid0: np.ndarray, bid1: np.ndarray) -> Optional[int]:
        """Resolve auction. Returns winner index or None if no bids."""
        count0 = int(bid0.sum())
        count1 = int(bid1.sum())
        max_bid = max(count0, count1)

        if max_bid == 0:
            # No bid — both lose 1 point, offer goes to trash
            self.scores[0] = clamp_score(self.scores[0] - 1)
            self.scores[1] = clamp_score(self.scores[1] - 1)
            self.trash += self.offer
            self.offer = np.zeros(N_COLORS, dtype=np.int32)
            return None

        # Determine winner (higher count wins, caretaker loses ties)
        counts = [count0, count1]
        candidates = [i for i, c in enumerate(counts) if c == max_bid]

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Tie — caretaker loses
            ct = self.caretaker
            if ct in candidates:
                non_ct = [i for i in candidates if i != ct]
                winner = non_ct[0] if non_ct else ct
            else:
                winner = min(candidates)

        loser = 1 - winner

        # Winner's bid goes to trash
        winner_bid = bid0 if winner == 0 else bid1
        loser_bid = bid1 if winner == 0 else bid0
        self.hands[winner] -= winner_bid
        self.trash += winner_bid

        # Winner gets offer stones
        offer_copy = self.offer.copy()
        self.hands[winner] += self.offer
        self.offer = np.zeros(N_COLORS, dtype=np.int32)

        # Update known hands tracking
        for c in range(N_COLORS):
            val = self.known[winner, c] + offer_copy[c] - winner_bid[c]
            self.known[winner, c] = max(0, val)
            self.known[loser, c] = max(self.known[loser, c], int(loser_bid[c]))

        # Winner gets point chip
        self.scores[winner] += POINT_CHIP

        # Caretaker becomes the winner
        self.caretaker = winner

        return winner

    def _check_trash_limit(self) -> bool:
        return bool(np.any(self.trash >= TRASH_LIMIT))

    def _check_bag_low(self) -> bool:
        return int(self.bag.sum()) < 2

    def _do_round_end(self):
        """Apply round-end scoring, then reset for new round."""
        ranked = rank_colors_by_total(self.hands)
        adds = [compute_hand_score(self.hands, p, ranked) for p in range(2)]

        for p in range(2):
            self.scores[p] = clamp_score(self.scores[p] + adds[p])

        # Reset for new round
        self.bag = make_bag()
        self.trash = np.zeros(N_COLORS, dtype=np.int32)
        self.hands = np.zeros((2, N_COLORS), dtype=np.int32)

        for p in range(2):
            n = 11 if p == self.caretaker else 10
            self.hands[p] = draw_n(self.bag, n, self._rng)

        seeded = draw_n(self.bag, SEED_TRASH_N, self._rng)
        self.trash += seeded

        self.offer = setup_offer(self.bag, self._rng)
        self.round_num += 1
        self.turn_num = 1

        # Reset known hands (new cards dealt)
        self.known = np.zeros((2, N_COLORS), dtype=np.int32)

        return adds

    def _check_game_end(self) -> Optional[int]:
        for p in range(2):
            if self.scores[p] >= self.score_to_win:
                return p
        return None

    def step(self, action):
        """One step = one auction turn.

        1. Agent (P0) submits bid via action
        2. Opponent (P1) submits bid
        3. Resolve auction
        4. Check round end / game end
        5. Setup next offer
        """
        self.total_turns += 1

        # Agent's bid
        bid0 = self._action_to_bid(np.array(action), 0)

        # Opponent's bid
        opponent = self._get_opponent()
        if isinstance(opponent, ModelOpponent):
            opp_obs = self._get_obs(1)
            opp_mask = self._get_action_mask(1)
            opp_action = opponent.choose_bid_from_obs(opp_obs, opp_mask)
            bid1 = self._action_to_bid(np.array(opp_action), 1)
        else:
            bid1 = opponent.choose_bid(self.hands[1], self.offer, self._rng)

        # Store scores and potential before resolution for reward calculation
        scores_before = self.scores.copy()
        pot_before = self._compute_potential(0)

        # Resolve auction
        winner = self._resolve_auction(bid0, bid1)
        self.last_winner = winner if winner is not None else -1

        # Action Tax: slight penalty for each stone spent to discourage early/wasteful discarding
        b0_sum = int(bid0.sum())
        action_tax = b0_sum * -0.005

        # Reward calculation
        reward = 0.0
        score_delta = (self.scores[0] - scores_before[0]) - (self.scores[1] - scores_before[1])
        reward += score_delta * 0.02
        
        # Apply action tax
        reward += action_tax

        # Potential-based reward shaping
        pot_after = self._compute_potential(0)
        reward += (pot_after - pot_before) * 0.005

        terminated = False
        truncated = False

        # Check game end (before round end processing)
        game_winner = self._check_game_end()
        if game_winner is not None:
            reward += 1.0 if game_winner == 0 else -1.0
            terminated = True
            return self._get_obs(0), reward, terminated, truncated, self._get_info()

        # Check round end conditions
        if self._check_trash_limit() or self._check_bag_low():
            adds = self._do_round_end()
            round_reward = (adds[0] - adds[1]) * 0.02
            reward += round_reward

            # Check game end after round scoring
            game_winner = self._check_game_end()
            if game_winner is not None:
                reward += 1.0 if game_winner == 0 else -1.0
                terminated = True
                return self._get_obs(0), reward, terminated, truncated, self._get_info()
        else:
            # Setup next offer
            self.offer = setup_offer(self.bag, self._rng)
            self.turn_num += 1

        # Truncation check
        if self.total_turns >= self.max_turns:
            truncated = True
            # Tie-break by score
            if self.scores[0] > self.scores[1]:
                reward += 0.5
            elif self.scores[0] < self.scores[1]:
                reward -= 0.5

        return self._get_obs(0), reward, terminated, truncated, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "scores": self.scores.copy(),
            "round": self.round_num,
            "turn": self.turn_num,
            "total_turns": self.total_turns,
            "bag_left": int(self.bag.sum()),
            "caretaker": self.caretaker,
        }

    def render(self):
        if self.render_mode == "human":
            print(f"R{self.round_num} T{self.turn_num} | "
                  f"Score: {self.scores[0]}-{self.scores[1]} | "
                  f"Hand: {self.hands[0]} vs {self.hands[1]} | "
                  f"Offer: {self.offer} | Bag: {self.bag.sum()} | "
                  f"Trash: {self.trash} | CT: {self.caretaker}")


# ==========================================
# Environment factory (for make_vec_env)
# ==========================================

def make_fafnir_env(score_to_win: int = 40, max_turns: int = 500, opponent=None):
    """Factory function for creating FafnirEnv instances."""
    def _init():
        return FafnirEnv(score_to_win=score_to_win, max_turns=max_turns,
                         opponent=opponent)
    return _init


# ==========================================
# Quick self-test
# ==========================================

if __name__ == "__main__":
    print("Running FafnirEnv self-test...")
    env = FafnirEnv(score_to_win=40, max_turns=500)

    wins = {0: 0, 1: 0, "draw": 0}
    total_episodes = 200

    for ep in range(total_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.valid_action_mask()
            # Random action respecting mask
            action = np.zeros(N_COLORS, dtype=np.int64)
            for c in range(N_COLORS):
                base = c * (MAX_BID_PER_COLOR + 1)
                valid = np.where(mask[base:base + MAX_BID_PER_COLOR + 1])[0]
                action[c] = env._rng.choice(valid)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        s = info["scores"]
        if s[0] > s[1]:
            wins[0] += 1
        elif s[1] > s[0]:
            wins[1] += 1
        else:
            wins["draw"] += 1

    print(f"Results over {total_episodes} episodes:")
    print(f"  P0 wins: {wins[0]} ({wins[0]/total_episodes*100:.1f}%)")
    print(f"  P1 wins: {wins[1]} ({wins[1]/total_episodes*100:.1f}%)")
    print(f"  Draws:   {wins['draw']} ({wins['draw']/total_episodes*100:.1f}%)")
    print("Self-test PASSED!")
