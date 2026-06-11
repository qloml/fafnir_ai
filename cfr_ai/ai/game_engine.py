"""
Fafnir game engine for Deep CFR self-play.
Standalone (no network) implementation for fast traversals.
"""
import random
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

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
INITIAL_BAG_COUNTS = [STONE_COUNT[c] for c in ALL_COLORS]
INITIAL_BAG_ARRAY = np.array(INITIAL_BAG_COUNTS, dtype=np.int32)

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
        'bag',            # [6] - remaining bag counts per color
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
        self.hand = np.zeros((2, NUM_COLORS), dtype=np.int32)
        self.bag = np.zeros(NUM_COLORS, dtype=np.int32)
        self.trash = np.zeros(NUM_COLORS, dtype=np.int32)
        self.offer = np.zeros(NUM_COLORS, dtype=np.int32)
        self.caretaker = 0
        self.scores = np.zeros(2, dtype=np.int32)
        self.phase = "BIDDING"
        self.round_num = 1
        self.turn_num = 1
        self.bid_history: List[Dict[str, Any]] = []

    def clone(self) -> 'FafnirState':
        s = FafnirState()
        s.hand = self.hand.copy()
        s.bag = self.bag.copy()
        s.trash = self.trash.copy()
        s.offer = self.offer.copy()
        s.caretaker = self.caretaker
        s.scores = self.scores.copy()
        s.phase = self.phase
        s.round_num = self.round_num
        s.turn_num = self.turn_num
        s.bid_history = self.bid_history[:]  # shallow copy of history
        return s

    def hand_total(self, player: int) -> int:
        return int(self.hand[player].sum())

    def bag_left(self) -> int:
        return int(self.bag.sum())

    def trash_total(self) -> int:
        return int(self.trash.sum())


# ============================================================
# Game Logic
# ============================================================
@njit(cache=True, fastmath=True)
def _draw_one_fast(bag):
    total = np.int32(0)
    for i in range(NUM_COLORS):
        total += bag[i]
    if total <= 0:
        return np.int32(-1)

    r = np.random.randint(0, total)
    cum = np.int32(0)
    for i in range(NUM_COLORS):
        cum += bag[i]
        if r < cum:
            bag[i] -= np.int32(1)
            return np.int32(i)
    return np.int32(-1)


@njit(cache=True, fastmath=True)
def _draw_n_fast(bag, n):
    drawn = np.zeros(NUM_COLORS, dtype=np.int32)
    for _ in range(n):
        c = _draw_one_fast(bag)
        if c < 0:
            break
        drawn[c] += np.int32(1)
    return drawn


@njit(cache=True, fastmath=True)
def _setup_offer_fast(bag):
    offer = np.zeros(NUM_COLORS, dtype=np.int32)
    total_drawn = np.int32(0)

    for _ in range(50):
        bag_left = np.int32(0)
        for c in range(NUM_COLORS):
            bag_left += bag[c]
        if bag_left <= 0:
            break

        draw_count = min(np.int32(2), bag_left)
        for _ in range(draw_count):
            c = _draw_one_fast(bag)
            if c >= 0:
                offer[c] += np.int32(1)
                total_drawn += np.int32(1)

        colors_in_offer = np.int32(0)
        for c in range(NUM_COLORS):
            if offer[c] > 0:
                colors_in_offer += np.int32(1)
        if colors_in_offer > 1:
            break

    return offer, total_drawn


@njit(cache=True, fastmath=True)
def _top_two_colors_fast(hand0, hand1):
    first_color = np.int32(1)
    second_color = np.int32(2)
    first_count = np.int32(-1)
    second_count = np.int32(-1)
    for c in range(1, NUM_COLORS):
        total = hand0[c] + hand1[c]
        if total > first_count or (total == first_count and c < first_color):
            second_color = first_color
            second_count = first_count
            first_color = np.int32(c)
            first_count = total
        elif total > second_count or (total == second_count and c < second_color):
            second_color = np.int32(c)
            second_count = total
    return first_color, second_color


@njit(cache=True, fastmath=True)
def _hand_score_fast(hands, player):
    first_color, second_color = _top_two_colors_fast(hands[0], hands[1])
    score = hands[player, 0]
    for c in range(1, NUM_COLORS):
        cnt = hands[player, c]
        if cnt == 0 or cnt >= 5:
            continue
        if c == first_color:
            score += cnt * np.int32(3)
        elif c == second_color:
            score += cnt * np.int32(2)
        else:
            score -= cnt
    return score


@njit(cache=True, fastmath=True)
def _visible_hand_potential_fast(hand_counts):
    first_color = np.int32(1)
    second_color = np.int32(2)
    first_count = np.int32(-1)
    second_count = np.int32(-1)
    for c in range(1, NUM_COLORS):
        cnt = hand_counts[c]
        if cnt > first_count or (cnt == first_count and c < first_color):
            second_color = first_color
            second_count = first_count
            first_color = np.int32(c)
            first_count = cnt
        elif cnt > second_count or (cnt == second_count and c < second_color):
            second_color = np.int32(c)
            second_count = cnt

    raw = hand_counts[0]
    for c in range(1, NUM_COLORS):
        cnt = hand_counts[c]
        if cnt == 0 or cnt >= 5:
            continue
        if c == first_color:
            raw += cnt * np.int32(3)
        elif c == second_color:
            raw += cnt * np.int32(2)
        else:
            raw -= cnt

    val = (np.float32(raw) + np.float32(15.0)) / np.float32(37.5) - np.float32(1.0)
    if val < np.float32(-1.0):
        return np.float32(-1.0)
    if val > np.float32(1.0):
        return np.float32(1.0)
    return val


@njit(cache=True, fastmath=True)
def _determine_winner_fast(bid0, bid1, caretaker):
    total0 = np.int32(0)
    total1 = np.int32(0)
    for c in range(NUM_COLORS):
        total0 += bid0[c]
        total1 += bid1[c]
    if total0 == 0 and total1 == 0:
        return np.int32(-1)
    if total0 > total1:
        return np.int32(0)
    if total1 > total0:
        return np.int32(1)
    return np.int32(1 - caretaker)


@njit(cache=True, fastmath=True)
def _resolve_auction_fast(hands, trash, offer, scores, caretaker, bid0, bid1):
    winner = _determine_winner_fast(bid0, bid1, caretaker)
    if winner < 0:
        for c in range(NUM_COLORS):
            trash[c] += offer[c]
            offer[c] = np.int32(0)
        scores[0] = max(np.int32(0), scores[0] - np.int32(1))
        scores[1] = max(np.int32(0), scores[1] - np.int32(1))
        return winner, caretaker

    w_bid = bid0 if winner == 0 else bid1
    for c in range(NUM_COLORS):
        hands[winner, c] -= w_bid[c]
        trash[c] += w_bid[c]
        hands[winner, c] += offer[c]
        offer[c] = np.int32(0)
    scores[winner] += np.int32(1)
    return winner, winner


@njit(cache=True, fastmath=True)
def _do_round_end_scoring_fast(hands, scores):
    add0 = _hand_score_fast(hands, np.int32(0))
    add1 = _hand_score_fast(hands, np.int32(1))
    scores[0] = max(np.int32(0), scores[0] + add0)
    scores[1] = max(np.int32(0), scores[1] + add1)
    return add0, add1


@njit(cache=True, fastmath=True)
def _reset_for_new_round_fast(hands, bag, trash, offer, caretaker):
    for c in range(NUM_COLORS):
        bag[c] = INITIAL_BAG_ARRAY[c]
        trash[c] = np.int32(0)
        offer[c] = np.int32(0)
        hands[0, c] = np.int32(0)
        hands[1, c] = np.int32(0)

    for p in range(2):
        n = np.int32(11) if p == caretaker else np.int32(10)
        drawn = _draw_n_fast(bag, n)
        for c in range(NUM_COLORS):
            hands[p, c] += drawn[c]

    seeded = _draw_n_fast(bag, np.int32(SEED_TRASH_AT_ROUND_START))
    for c in range(NUM_COLORS):
        trash[c] += seeded[c]

    new_offer, _ = _setup_offer_fast(bag)
    for c in range(NUM_COLORS):
        offer[c] = new_offer[c]


@njit(cache=True, fastmath=True)
def _step_auction_fast(hands, bag, trash, offer, scores, caretaker, bid0, bid1):
    winner, caretaker = _resolve_auction_fast(
        hands, trash, offer, scores, caretaker, bid0, bid1
    )

    if scores[0] >= SCORE_TO_WIN or scores[1] >= SCORE_TO_WIN:
        return caretaker, np.int8(0), np.int8(1)

    trash_hit = False
    for c in range(NUM_COLORS):
        if trash[c] >= TRASH_LIMIT:
            trash_hit = True
            break

    bag_left = np.int32(0)
    for c in range(NUM_COLORS):
        bag_left += bag[c]

    round_ended = trash_hit or bag_left < 2
    if not round_ended:
        offer_empty = True
        for c in range(NUM_COLORS):
            if offer[c] > 0:
                offer_empty = False
                break

        if offer_empty:
            new_offer, drawn = _setup_offer_fast(bag)
            for c in range(NUM_COLORS):
                offer[c] = new_offer[c]
            if drawn <= 0:
                round_ended = True
            else:
                bag_left = np.int32(0)
                for c in range(NUM_COLORS):
                    bag_left += bag[c]
                if bag_left < 2:
                    round_ended = True

    if round_ended:
        _do_round_end_scoring_fast(hands, scores)
        if scores[0] >= SCORE_TO_WIN or scores[1] >= SCORE_TO_WIN:
            return caretaker, np.int8(1), np.int8(1)
        _reset_for_new_round_fast(hands, bag, trash, offer, caretaker)
        return caretaker, np.int8(1), np.int8(0)

    return caretaker, np.int8(0), np.int8(0)


@njit(cache=True, fastmath=True)
def get_legal_mask_fast(hand, offer, action_table):
    n_actions = action_table.shape[0]
    mask = np.zeros(n_actions, dtype=np.bool_)
    for a in range(n_actions):
        legal = True
        for c in range(NUM_COLORS):
            if action_table[a, c] > hand[c]:
                legal = False
                break
            if offer[c] > 0 and action_table[a, c] > 0:
                legal = False
                break
        mask[a] = legal
    return mask


@njit(cache=True, fastmath=True)
def build_observation_fast_arrays(hands, bag, trash, offer, scores, caretaker,
                                  player, confirmed, round_num, turn_num):
    obs = np.zeros(42, dtype=np.float32)
    opp = np.int32(1 - player)
    for c in range(NUM_COLORS):
        obs[c] = np.float32(hands[player, c])
        obs[6 + c] = np.float32(offer[c])
        obs[12 + c] = np.float32(trash[c])
        obs[18 + c] = np.float32(confirmed[opp, c])
        obs[25 + c] = np.float32(confirmed[player, c])

    opp_total = np.int32(0)
    opp_confirmed_total = np.int32(0)
    for c in range(NUM_COLORS):
        opp_total += hands[opp, c]
        opp_confirmed_total += confirmed[opp, c]
    obs[24] = np.float32(max(np.int32(0), opp_total - opp_confirmed_total))

    bag_left = np.int32(0)
    for c in range(NUM_COLORS):
        bag_left += bag[c]
    obs[31] = np.float32(bag_left) / np.float32(TOTAL_STONES)
    obs[32] = np.float32(1.0) if caretaker == player else np.float32(0.0)
    obs[33] = _visible_hand_potential_fast(hands[player])
    obs[34] = np.float32(scores[player]) / np.float32(SCORE_TO_WIN)
    obs[35] = np.float32(scores[opp]) / np.float32(SCORE_TO_WIN)
    obs[36] = min(np.float32(round_num), np.float32(20.0)) / np.float32(20.0)
    obs[37] = min(np.float32(turn_num), np.float32(30.0)) / np.float32(30.0)

    offer_total = np.int32(0)
    my_total = np.int32(0)
    trash_total = np.int32(0)
    for c in range(NUM_COLORS):
        offer_total += offer[c]
        my_total += hands[player, c]
        trash_total += trash[c]
    obs[38] = np.float32(offer_total) / np.float32(10.0)
    obs[39] = np.float32(my_total) / np.float32(20.0)
    obs[40] = np.float32(opp_total) / np.float32(20.0)
    obs[41] = np.float32(trash_total) / np.float32(NUM_COLORS * TRASH_LIMIT)
    return obs


@njit(cache=True, fastmath=True)
def update_confirmed_fast(confirmed, winner, bid_winner, bid_loser, offer):
    if winner < 0:
        return
    loser = np.int32(1 - winner)
    for c in range(NUM_COLORS):
        val = confirmed[winner, c] - bid_winner[c]
        if val < 0:
            val = np.int32(0)
        confirmed[winner, c] = val + offer[c]
        if bid_loser[c] > confirmed[loser, c]:
            confirmed[loser, c] = bid_loser[c]


def make_bag() -> List[int]:
    return INITIAL_BAG_ARRAY.copy()


def draw_one(state: FafnirState) -> Optional[str]:
    total = int(state.bag.sum())
    if total <= 0:
        return None
    r = random.randrange(total)
    cum = 0
    for idx, n in enumerate(state.bag):
        cum += n
        if r < cum:
            state.bag[idx] -= 1
            return ALL_COLORS[idx]
    return None


def draw_one_idx(state: FafnirState) -> int:
    return int(_draw_one_fast(state.bag))


def deal_initial_hands(state: FafnirState):
    """Deal initial hands: caretaker gets 11, other gets 10."""
    state.hand.fill(0)
    for i in range(2):
        n = 11 if i == state.caretaker else 10
        state.hand[i] += _draw_n_fast(state.bag, n)


def seed_trash(state: FafnirState, n: int = SEED_TRASH_AT_ROUND_START):
    """Draw n stones into trash at round start."""
    state.trash += _draw_n_fast(state.bag, n)


def setup_offer(state: FafnirState) -> bool:
    """
    Draw stones for the offer. Keep drawing 2 at a time until
    at least 2 different colors appear, or bag is empty.
    """
    state.offer.fill(0)
    if state.bag_left() <= 0:
        return False

    offer, total_drawn = _setup_offer_fast(state.bag)
    state.offer[:] = offer
    return int(total_drawn) > 0


def is_trash_limit_reached(state: FafnirState) -> bool:
    return bool(np.any(state.trash >= TRASH_LIMIT))


def should_force_round_end_by_bag(state: FafnirState) -> bool:
    return state.bag_left() < 2


def rank_colors_by_total(state: FafnirState) -> List[Tuple[int, int]]:
    """
    Rank non-gold colors by total count across both players' hands.
    Returns list of (color_idx, total_count), sorted by (-count, color_priority).
    Color priority: red=1 > orange=2 > yellow=3 > green=4 > blue=5
    """
    totals = [(i, int(state.hand[0, i] + state.hand[1, i])) for i in range(1, NUM_COLORS)]
    totals.sort(key=lambda x: (-x[1], x[0]))
    return totals


def _top_two_colors_by_total(hand0: List[int], hand1: List[int]) -> Tuple[int, int]:
    first_color, second_color = _top_two_colors_fast(
        np.asarray(hand0, dtype=np.int32), np.asarray(hand1, dtype=np.int32)
    )
    return int(first_color), int(second_color)


def compute_hand_score(state: FafnirState, player: int) -> int:
    """Compute the hand score for a player at round end."""
    return int(_hand_score_fast(state.hand, player))


def compute_visible_hand_potential(hand_counts: List[int]) -> float:
    """
    Estimate hand value using only the player's own hand.

    This is intentionally weaker than compute_hand_score(): the real color
    ranking depends on the opponent's private hand, which is unavailable to
    the bot on server_0424.py. Keeping dim 33 fair prevents train/inference
    observation drift.
    """
    return float(_visible_hand_potential_fast(np.asarray(hand_counts, dtype=np.int32)))


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
    winner = int(_determine_winner_fast(
        np.asarray(bid0, dtype=np.int32), np.asarray(bid1, dtype=np.int32), caretaker
    ))
    if winner < 0:
        return None
    return winner


def resolve_auction(state: FafnirState, bid0: List[int], bid1: List[int]) -> Optional[int]:
    """
    Resolve an auction given bids as count vectors [6].
    Returns winner index (0 or 1) or None if both bid 0.
    Mutates state in place.
    """
    # Record bid history
    state.bid_history.append({
        'bids': [list(bid0), list(bid1)],
        'offer': state.offer.copy(),
        'caretaker': state.caretaker,
    })

    bid0_arr = np.asarray(bid0, dtype=np.int32)
    bid1_arr = np.asarray(bid1, dtype=np.int32)
    winner_raw, new_caretaker = _resolve_auction_fast(
        state.hand, state.trash, state.offer, state.scores, state.caretaker, bid0_arr, bid1_arr
    )
    state.caretaker = int(new_caretaker)
    if winner_raw < 0:
        return None
    return int(winner_raw)


# ============================================================
# Round End & Reset
# ============================================================
def do_round_end(state: FafnirState):
    """Apply round-end scoring and reset for new round."""
    _do_round_end_scoring_fast(state.hand, state.scores)

    # Check game end
    if check_game_end(state):
        return

    # Reset for new round
    _reset_for_new_round_fast(state.hand, state.bag, state.trash, state.offer, state.caretaker)
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
    bid0_arr = np.asarray(bid0, dtype=np.int32)
    bid1_arr = np.asarray(bid1, dtype=np.int32)
    new_caretaker, round_ended, game_ended = _step_auction_fast(
        state.hand, state.bag, state.trash, state.offer, state.scores,
        state.caretaker, bid0_arr, bid1_arr,
    )
    state.caretaker = int(new_caretaker)

    if game_ended:
        state.turn_num += 1
        state.phase = "GAME_END"
    elif round_ended:
        state.round_num += 1
        state.turn_num = 1
        state.phase = "BIDDING"
    else:
        state.turn_num += 1


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


def warmup() -> None:
    """Compile hot Numba paths before training/evaluation timing starts."""
    random_state = random.getstate()
    state = new_game()
    random.setstate(random_state)
    confirmed = np.zeros((2, NUM_COLORS), dtype=np.int32)
    _ = build_observation_fast_arrays(
        state.hand, state.bag, state.trash, state.offer, state.scores,
        state.caretaker, np.int32(0), confirmed, state.round_num, state.turn_num,
    )
    action_table = np.zeros((1, NUM_COLORS), dtype=np.int32)
    _ = get_legal_mask_fast(state.hand[0], state.offer, action_table)
    bid0 = np.zeros(NUM_COLORS, dtype=np.int32)
    bid1 = np.zeros(NUM_COLORS, dtype=np.int32)
    old_offer = state.offer.copy()
    update_confirmed_fast(confirmed, np.int32(0), bid0, bid1, old_offer)
    _step_auction_fast(
        state.hand, state.bag, state.trash, state.offer, state.scores,
        state.caretaker, bid0, bid1,
    )
