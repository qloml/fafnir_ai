"""
Action space management for Fafnir Deep CFR.

An "action" is a bid: which stones to discard from hand.
Represented as a 6-element count vector [gold, red, orange, yellow, green, blue].

We enumerate ALL possible bid combinations and assign each a unique integer ID.
At runtime, we mask out illegal actions based on:
  - Hand contents (can't bid what you don't have)
  - Offer colors (can't bid same color as offer)

Hand sizes:
  - Initial: 10-11 stones
  - Can grow to 20+ by winning large offers (same-color draws keep stacking)
  - Simulation shows: 98.8% of hands are ≤14, 99.99% are ≤19
  - Max observed: 24

We set max_total=15, max_per_color=12 (54,264 actions).
For rare cases where hand > 15, a fallback clips the bid to the NN range.
"""
from typing import List, Tuple, Dict, Optional
from itertools import product
import numpy as np

from .game_engine import NUM_COLORS, ALL_COLORS, COLOR_TO_IDX

# ============================================================
# Action Enumeration
# ============================================================
# Bid limits (NN output size):
#   max_total = 6 : covers practical bids (合計6個まで)
#   max_per_color = 6 : sufficient for any single-color bid
#   Rare larger bids are handled by clip_bid_to_range()
#   This gives 462 actions (vs 3,003 with max=8)
MAX_PER_COLOR_BID = 6
MAX_TOTAL_BID = 6


def _build_action_table(max_total: int, max_per_color: int) -> List[Tuple[int, ...]]:
    """
    Build a lookup table of all possible bid vectors.
    Each entry is a tuple of 6 ints (one per color).
    Includes the zero-bid (pass) action.
    """
    actions: List[Tuple[int, ...]] = []
    _enumerate_bids([], 0, max_total, max_per_color, actions)
    return actions


def _enumerate_bids(
    current: List[int],
    depth: int,
    remaining: int,
    max_per_color: int,
    out: List[Tuple[int, ...]],
):
    """Recursively enumerate bid vectors with total <= remaining."""
    if depth == NUM_COLORS:
        out.append(tuple(current))
        return
    max_this = min(max_per_color, remaining)
    for k in range(max_this + 1):
        current.append(k)
        _enumerate_bids(current, depth + 1, remaining - k, max_per_color, out)
        current.pop()


# Pre-compute the action table (once at import time)
print("[ActionSpace] Building action table...")
ACTION_TABLE: List[Tuple[int, ...]] = _build_action_table(MAX_TOTAL_BID, MAX_PER_COLOR_BID)
NUM_ACTIONS = len(ACTION_TABLE)
print(f"[ActionSpace] Total actions: {NUM_ACTIONS}")

# Pre-computed numpy array for vectorized legal mask computation
# Shape: [NUM_ACTIONS, NUM_COLORS]
ACTION_TABLE_NP = np.array(ACTION_TABLE, dtype=np.int32)

# Reverse mapping: tuple -> action_id
ACTION_TO_ID: Dict[Tuple[int, ...], int] = {a: i for i, a in enumerate(ACTION_TABLE)}

# Zero-bid (pass) action
PASS_ACTION_ID = ACTION_TO_ID.get((0, 0, 0, 0, 0, 0), 0)


# ============================================================
# Action Masking
# ============================================================
def get_legal_mask(hand: List[int], offer: List[int]) -> np.ndarray:
    """
    Returns a boolean mask of shape [NUM_ACTIONS].
    True = legal, False = illegal.

    An action is legal if:
    1. For each color, bid[color] <= hand[color]
    2. For colors present in offer, bid[color] == 0

    Uses vectorized NumPy operations for speed.
    """
    hand_arr = np.array(hand, dtype=np.int32)
    offer_arr = np.array(offer, dtype=np.int32)

    # Check 1: All bid counts <= hand counts (per color)
    # ACTION_TABLE_NP: [N, 6], hand_arr: [6] -> broadcast to [N, 6]
    hand_ok = np.all(ACTION_TABLE_NP <= hand_arr, axis=1)

    # Check 2: For colors in offer (offer > 0), bid must be 0
    forbidden_colors = offer_arr > 0  # shape [6], bool
    if forbidden_colors.any():
        # For forbidden colors, action must have 0
        forbidden_ok = np.all(ACTION_TABLE_NP[:, forbidden_colors] == 0, axis=1)
        mask = hand_ok & forbidden_ok
    else:
        mask = hand_ok

    return mask


def get_legal_action_ids(hand: List[int], offer: List[int]) -> List[int]:
    """Get list of legal action IDs."""
    mask = get_legal_mask(hand, offer)
    return list(np.where(mask)[0])


def action_id_to_counts(action_id: int) -> List[int]:
    """Convert action ID to count vector."""
    return list(ACTION_TABLE[action_id])


def counts_to_action_id(counts: List[int]) -> Optional[int]:
    """Convert count vector to action ID. None if out of NN range."""
    return ACTION_TO_ID.get(tuple(counts))


def action_id_to_stones(action_id: int) -> List[str]:
    """Convert action ID to list of stone color strings."""
    counts = ACTION_TABLE[action_id]
    stones: List[str] = []
    for i, c in enumerate(ALL_COLORS):
        stones.extend([c] * counts[i])
    return stones


def clip_bid_to_range(bid: List[int]) -> List[int]:
    """
    If a bid exceeds MAX_TOTAL_BID or MAX_PER_COLOR_BID,
    clip it down to fit the NN action space.
    Prioritizes keeping the bid total as high as possible
    by trimming the least valuable stones first.
    """
    # Clip per-color
    clipped = [min(b, MAX_PER_COLOR_BID) for b in bid]

    # Clip total
    total = sum(clipped)
    if total <= MAX_TOTAL_BID:
        return clipped

    # Need to reduce by (total - MAX_TOTAL_BID)
    # Remove from colors with highest count first (greedy)
    excess = total - MAX_TOTAL_BID
    # Sort colors by count (descending), remove from largest first
    indexed = sorted(range(NUM_COLORS), key=lambda c: -clipped[c])
    for c in indexed:
        if excess <= 0:
            break
        remove = min(clipped[c], excess)
        clipped[c] -= remove
        excess -= remove

    return clipped
