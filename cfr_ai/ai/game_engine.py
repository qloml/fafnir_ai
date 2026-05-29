"""
Fafnir game engine for Deep CFR self-play.
Standalone (no network) implementation for fast traversals.
"""
import random
from typing import List, Dict, Optional, Tuple, Any
from copy import deepcopy

# ============================================================
# Constants
# ============================================================
COLORS = ["red", "orange", "yellow", "green", "blue"]
GOLD = "gold"
ALL_COLORS = [GOLD] + COLORS  # index order: gold=0, red=1, orange=2, yellow=3, green=4, blue=5
COLOR_TO_IDX = {c: i for i, c in enumerate(ALL_COLORS)}
NUM_COLORS = len(ALL_COLORS)  # 6

STONE_COUNT = {GOLD: 20, "red": 12, "orange": 12, "yellow": 12, "green": 12, "blue": 12}
TOTAL_STONES = sum(STONE_COUNT.values())  # 80

TRASH_LIMIT = 6
SEED_TRASH_AT_ROUND_START = 3
SCORE_TO_WIN = 1000


# ============================================================
# Helper: stones list <-> count vector
# ============================================================
def stones_to_counts(stones: List[str]) -> List[int]:
    """Convert a list of stone color strings to a 6-element count vector."""
    counts = [0] * NUM_COLORS
    for s in stones:
        idx = COLOR_TO_IDX.get(s)
        if idx is not None:
            counts[idx] += 1
    return counts


def counts_to_stones(counts: List[int]) -> List[str]:
    """Convert a 6-element count vector back to a list of stone color strings."""
    stones = []
    for i, c in enumerate(ALL_COLORS):
        stones.extend([c] * counts[i])
    return stones


# ============================================================
# Game State (lightweight, copyable)
# ============================================================
class FafnirState:
    """
    Full game state for one match.
    Uses count vectors internally for efficiency.
    """
    __slots__ = [
        'hand',           # [2][6] - each player's hand counts
        'bag',            # List[str] - remaining bag (order matters for draws)
        'trash',          # [6] - trash counts per color
        'offer',          # [6] - current offer counts
        'caretaker',      # int (0 or 1) - who is caretaker
        'scores',         # [2] - player scores
        'phase',          # str: "BIDDING", "ROUND_END", "GAME_END"
        'round_num',      # int
        'turn_num',       # int
        'bid_history',    # list of past bids for information tracking
    ]

    def __init__(self):
        self.hand = [[0]*NUM_COLORS, [0]*NUM_COLORS]
        self.bag: List[str] = []
        self.trash = [0]*NUM_COLORS
        self.offer = [0]*NUM_COLORS
        self.caretaker = 0
        self.scores = [0, 0]
        self.phase = "BIDDING"
        self.round_num = 1
        self.turn_num = 1
        self.bid_history: List[Dict[str, Any]] = []

    def clone(self) -> 'FafnirState':
        s = FafnirState()
        s.hand = [self.hand[0][:], self.hand[1][:]]
        s.bag = self.bag[:]
        s.trash = self.trash[:]
        s.offer = self.offer[:]
        s.caretaker = self.caretaker
        s.scores = self.scores[:]
        s.phase = self.phase
        s.round_num = self.round_num
        s.turn_num = self.turn_num
        s.bid_history = self.bid_history[:]  # shallow copy of history
        return s

    def hand_total(self, player: int) -> int:
        return sum(self.hand[player])

    def bag_left(self) -> int:
        return len(self.bag)

    def trash_total(self) -> int:
        return sum(self.trash)


# ============================================================
# Game Logic
# ============================================================
def make_bag() -> List[str]:
    bag = []
    for c, n in STONE_COUNT.items():
        bag.extend([c] * n)
    random.shuffle(bag)
    return bag


def draw_one(state: FafnirState) -> Optional[str]:
    if not state.bag:
        return None
    return state.bag.pop()


def deal_initial_hands(state: FafnirState):
    """Deal initial hands: caretaker gets 11, other gets 10."""
    state.hand = [[0]*NUM_COLORS, [0]*NUM_COLORS]
    for i in range(2):
        n = 11 if i == state.caretaker else 10
        for _ in range(n):
            s = draw_one(state)
            if s is not None:
                state.hand[i][COLOR_TO_IDX[s]] += 1


def seed_trash(state: FafnirState, n: int = SEED_TRASH_AT_ROUND_START):
    """Draw n stones into trash at round start."""
    for _ in range(n):
        s = draw_one(state)
        if s is None:
            break
        state.trash[COLOR_TO_IDX[s]] += 1


def setup_offer(state: FafnirState) -> bool:
    """
    Draw stones for the offer. Keep drawing 2 at a time until
    at least 2 different colors appear, or bag is empty.
    """
    state.offer = [0] * NUM_COLORS
    if not state.bag:
        return False

    drawn_colors = set()
    total_drawn = 0

    while True:
        draw_n = min(2, len(state.bag))
        for _ in range(draw_n):
            s = draw_one(state)
            if s is not None:
                idx = COLOR_TO_IDX[s]
                state.offer[idx] += 1
                drawn_colors.add(idx)
                total_drawn += 1

        if len(drawn_colors) > 1 or len(state.bag) == 0:
            break

    return total_drawn > 0


def is_trash_limit_reached(state: FafnirState) -> bool:
    for i in range(NUM_COLORS):
        if state.trash[i] >= TRASH_LIMIT:
            return True
    return False


def should_force_round_end_by_bag(state: FafnirState) -> bool:
    return len(state.bag) < 2


def rank_colors_by_total(state: FafnirState) -> List[Tuple[int, int]]:
    """
    Rank non-gold colors by total count across both players' hands.
    Returns list of (color_idx, total_count), sorted by (-count, color_priority).
    Color priority: red=1 > orange=2 > yellow=3 > green=4 > blue=5
    """
    totals = []
    for i in range(1, NUM_COLORS):  # skip gold (index 0)
        total = state.hand[0][i] + state.hand[1][i]
        totals.append((i, total))
    # Sort by (-count, color_index) - lower index = higher priority
    totals.sort(key=lambda x: (-x[1], x[0]))
    return totals


def compute_hand_score(state: FafnirState, player: int) -> int:
    """Compute the hand score for a player at round end."""
    ranked = rank_colors_by_total(state)
    first_color = ranked[0][0] if ranked else -1
    second_color = ranked[1][0] if len(ranked) > 1 else -1

    score = state.hand[player][0]  # gold = 1 point each

    for i in range(1, NUM_COLORS):
        cnt = state.hand[player][i]
        if cnt == 0:
            continue
        if cnt >= 5:
            # 5+ of same color: no positive points from this color
            continue

        if i == first_color:
            mult = 3
        elif i == second_color:
            mult = 2
        else:
            mult = -1
        score += cnt * mult

    return score


def compute_visible_hand_potential(hand_counts: List[int]) -> float:
    """
    Estimate hand value using only the player's own hand.

    This is intentionally weaker than compute_hand_score(): the real color
    ranking depends on the opponent's private hand, which is unavailable to
    the bot on server_0424.py. Keeping dim 33 fair prevents train/inference
    observation drift.
    """
    ranked = [(i, hand_counts[i]) for i in range(1, NUM_COLORS)]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    first_color = ranked[0][0] if ranked else -1
    second_color = ranked[1][0] if len(ranked) > 1 else -1

    raw = hand_counts[0]
    for i in range(1, NUM_COLORS):
        cnt = hand_counts[i]
        if cnt == 0:
            continue
        if cnt >= 5:
            continue
        if i == first_color:
            mult = 3
        elif i == second_color:
            mult = 2
        else:
            mult = -1
        raw += cnt * mult

    return max(-1.0, min(1.0, (raw + 15.0) / 37.5 - 1.0))


def compute_expected_score(state: FafnirState, player: int) -> float:
    """
    Compute expected hand score for observation space (dim 33).
    This estimates "if the round ended now, how many points would I get?"
    Normalized to roughly [-1, 1] range (from [-15, +60]).
    """
    raw = compute_hand_score(state, player)
    # Normalize: map [-15, 60] -> [-1, 1]
    return max(-1.0, min(1.0, (raw + 15.0) / 37.5 - 1.0))


def clamp_score(x: int) -> int:
    return max(0, x)


# ============================================================
# Auction Resolution
# ============================================================
def determine_auction_winner(bid0: List[int], bid1: List[int], caretaker: int) -> Optional[int]:
    """Return the auction winner before mutating state, or None if both pass."""
    total0 = sum(bid0)
    total1 = sum(bid1)
    max_bid = max(total0, total1)
    if max_bid == 0:
        return None
    if total0 > total1:
        return 0
    if total1 > total0:
        return 1
    return 1 - caretaker


def resolve_auction(state: FafnirState, bid0: List[int], bid1: List[int]) -> Optional[int]:
    """
    Resolve an auction given bids as count vectors [6].
    Returns winner index (0 or 1) or None if both bid 0.
    Mutates state in place.
    """
    # Record bid history
    state.bid_history.append({
        'bids': [bid0[:], bid1[:]],
        'offer': state.offer[:],
        'caretaker': state.caretaker,
    })

    winner = determine_auction_winner(bid0, bid1, state.caretaker)

    # Both bid 0 -> offer goes to trash, both lose 1 point
    if winner is None:
        for i in range(NUM_COLORS):
            state.trash[i] += state.offer[i]
        state.offer = [0] * NUM_COLORS
        state.scores[0] = clamp_score(state.scores[0] - 1)
        state.scores[1] = clamp_score(state.scores[1] - 1)
        return None

    loser = 1 - winner

    # Winner's bid goes to trash, winner gets offer
    for i in range(NUM_COLORS):
        state.hand[winner][i] -= bid0[i] if winner == 0 else bid1[i]
        state.trash[i] += bid0[i] if winner == 0 else bid1[i]
        state.hand[winner][i] += state.offer[i]

    # Loser keeps their bid (returns to hand - no change needed)

    state.offer = [0] * NUM_COLORS
    state.scores[winner] += 1
    state.caretaker = winner

    return winner


# ============================================================
# Round End & Reset
# ============================================================
def do_round_end(state: FafnirState):
    """Apply round-end scoring and reset for new round."""
    # Calculate and apply hand scores
    for p in range(2):
        add = compute_hand_score(state, p)
        state.scores[p] = clamp_score(state.scores[p] + add)

    # Check game end
    if check_game_end(state):
        return

    # Reset for new round
    state.bag = make_bag()
    state.trash = [0] * NUM_COLORS
    state.offer = [0] * NUM_COLORS
    deal_initial_hands(state)
    seed_trash(state)
    setup_offer(state)
    state.round_num += 1
    state.turn_num = 1
    state.phase = "BIDDING"


def check_game_end(state: FafnirState) -> bool:
    """Check if game should end (score threshold)."""
    for i in range(2):
        if state.scores[i] >= SCORE_TO_WIN:
            state.phase = "GAME_END"
            return True
    return False


# ============================================================
# Full Turn Step
# ============================================================
def step_auction(state: FafnirState, bid0: List[int], bid1: List[int]):
    """
    Execute one full auction turn:
    1. Resolve auction
    2. Check trash limit / bag low -> round end
    3. Setup next offer if continuing
    """
    resolve_auction(state, bid0, bid1)

    state.turn_num += 1

    # Check game end by score
    if check_game_end(state):
        return

    # Check round end conditions
    if should_force_round_end_by_bag(state) or is_trash_limit_reached(state):
        do_round_end(state)
        return

    # Continue: setup next offer
    if sum(state.offer) == 0:
        ok = setup_offer(state)
        if not ok or should_force_round_end_by_bag(state):
            do_round_end(state)
            return


# ============================================================
# New Game
# ============================================================
def new_game(seed: Optional[int] = None) -> FafnirState:
    """Create a new game state, fully initialized."""
    if seed is not None:
        random.seed(seed)

    state = FafnirState()
    state.bag = make_bag()
    state.caretaker = random.randint(0, 1)
    deal_initial_hands(state)
    seed_trash(state)
    setup_offer(state)
    state.phase = "BIDDING"
    return state
