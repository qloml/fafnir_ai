"""
Observation space builder for Fafnir Deep CFR.

Observation vector (33 dimensions):
  [0-5]   : My hand counts (6 colors)
  [6-11]  : Current offer counts (6 colors)
  [12-17] : Trash counts (6 colors)
  [18-23] : Opponent's confirmed hand (6 colors) - deduced from bid history
  [24]    : Opponent's unknown card count
  [25-30] : My confirmed hand (6 colors) - what opponent knows I have
  [31]    : Am I caretaker? (0 or 1)
  [32]    : My visible hand potential (normalized, own hand only)
"""
import numpy as np
from typing import List, Dict, Any

from .game_engine import (
    FafnirState, NUM_COLORS, compute_visible_hand_potential,
    build_observation_fast_arrays, update_confirmed_fast,
)

# Observation dimensions
OBS_DIM = 33


# ============================================================
# Bid History Tracker
# ============================================================
class BidTracker:
    """
    Tracks bid history to infer confirmed hands.

    When a player bids stones and WINS, those stones are removed from
    their hand (go to trash) and they gain the offer stones.
    When a player bids stones and LOSES, the stones return to their hand
    (but we now KNOW they have those stones).

    This tracker accumulates "confirmed" information about what each
    player definitely has.
    """

    def __init__(self):
        # confirmed[player][color] = count of stones we KNOW they have
        self.confirmed = np.zeros((2, NUM_COLORS), dtype=np.int32)

    def update_from_auction(
        self,
        winner: int,  # None for no-bid
        bid_winner: List[int],  # winner's bid counts
        bid_loser: List[int],   # loser's bid counts
        offer: List[int],       # the offer that was won
        is_no_bid: bool = False
    ):
        """
        Update confirmed hands based on an auction result.

        For the winner:
          - They bid stones (now in trash) -> remove from confirmed
          - They gained offer stones -> add to confirmed

        For the loser:
          - They revealed their bid -> we now confirm those stones
          - Stones returned to hand -> confirmed += bid
        """
        if is_no_bid:
            return

        update_confirmed_fast(
            self.confirmed,
            np.int32(winner),
            np.asarray(bid_winner, dtype=np.int32),
            np.asarray(bid_loser, dtype=np.int32),
            np.asarray(offer, dtype=np.int32),
        )

    def reset(self):
        """Reset for new round."""
        self.confirmed.fill(0)


# ============================================================
# Observation Builder
# ============================================================
def build_observation(
    state: FafnirState,
    player: int,
    bid_tracker: BidTracker,
) -> np.ndarray:
    """
    Build the 33-dimensional observation vector for a player.
    All values are raw counts except for the normalized fields.
    """
    confirmed = np.asarray(bid_tracker.confirmed, dtype=np.int32)
    return build_observation_fast_arrays(
        state.hand, state.trash, state.offer, state.caretaker, player, confirmed,
    )


def build_observation_from_server_state(
    server_state: Dict[str, Any],
    my_index: int,
    bid_tracker: BidTracker,
) -> np.ndarray:
    """
    Build observation from server state_update payload.
    Used by the Socket.IO client.
    """
    from .game_engine import COLOR_TO_IDX, ALL_COLORS

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    opp = 1 - my_index

    players = server_state.get("players", [])
    if len(players) < 2 or my_index < 0:
        return obs

    me = players[my_index]
    them = players[opp]

    my_hand_counts = [0] * NUM_COLORS

    # [0-5] My hand
    hand = me.get("hand", [])
    if hand:
        for s in hand:
            idx = COLOR_TO_IDX.get(s)
            if idx is not None:
                obs[idx] += 1
                my_hand_counts[idx] += 1

    # [6-11] Offer
    offer = server_state.get("offer", [])
    for s in offer:
        idx = COLOR_TO_IDX.get(s)
        if idx is not None:
            obs[6 + idx] += 1

    # [12-17] Trash
    trash = server_state.get("trash", {})
    for c in ALL_COLORS:
        idx = COLOR_TO_IDX[c]
        obs[12 + idx] = trash.get(c, 0)

    # [18-23] Opponent's confirmed hand
    for c in range(NUM_COLORS):
        obs[18 + c] = bid_tracker.confirmed[opp][c]

    # [24] Opponent's unknown cards
    opp_total = them.get("hand_count", 0)
    opp_confirmed = sum(bid_tracker.confirmed[opp])
    obs[24] = max(0, opp_total - opp_confirmed)

    # [25-30] My confirmed hand
    for c in range(NUM_COLORS):
        obs[25 + c] = bid_tracker.confirmed[my_index][c]

    # [31] Am I caretaker?
    caretaker = server_state.get("caretaker", 0)
    obs[31] = 1.0 if caretaker == my_index else 0.0

    # [32] My visible hand potential (same fair feature as training)
    obs[32] = compute_visible_hand_potential(my_hand_counts)

    return obs
