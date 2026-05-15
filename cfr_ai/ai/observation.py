"""
Observation space builder for Fafnir Deep CFR.

Observation vector (34 dimensions):
  [0-5]   : My hand counts (6 colors)
  [6-11]  : Current offer counts (6 colors)
  [12-17] : Trash counts (6 colors)
  [18-23] : Opponent's confirmed hand (6 colors) - deduced from bid history
  [24]    : Opponent's unknown card count
  [25-30] : My confirmed hand (6 colors) - what opponent knows I have
  [31]    : Bag remaining count (normalized)
  [32]    : Am I caretaker? (0 or 1)
  [33]    : My expected score (normalized)
"""
import numpy as np
from typing import List, Dict, Any

from .game_engine import (
    FafnirState, NUM_COLORS, TOTAL_STONES, compute_expected_score
)


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
        self.confirmed = [[0]*NUM_COLORS, [0]*NUM_COLORS]

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

        loser = 1 - winner

        # Winner: we saw their bid, those went to trash, they got offer
        # After auction: winner's confirmed = old_confirmed - bid + offer
        for c in range(NUM_COLORS):
            # Remove bid from confirmed (they no longer have those)
            self.confirmed[winner][c] = max(0, self.confirmed[winner][c] - bid_winner[c])
            # Add offer to confirmed (they definitely have these now)
            self.confirmed[winner][c] += offer[c]

        # Loser: their bid was revealed, so we know they have those
        for c in range(NUM_COLORS):
            # We at least know they have bid_loser[c] of this color
            self.confirmed[loser][c] = max(self.confirmed[loser][c], bid_loser[c])

    def reset(self):
        """Reset for new round."""
        self.confirmed = [[0]*NUM_COLORS, [0]*NUM_COLORS]


# ============================================================
# Observation Builder
# ============================================================
def build_observation(
    state: FafnirState,
    player: int,
    bid_tracker: BidTracker,
) -> np.ndarray:
    """
    Build the 34-dimensional observation vector for a player.
    All values are raw counts except for the normalized fields.
    """
    obs = np.zeros(34, dtype=np.float32)
    opp = 1 - player

    # [0-5] My hand
    for c in range(NUM_COLORS):
        obs[c] = state.hand[player][c]

    # [6-11] Current offer
    for c in range(NUM_COLORS):
        obs[6 + c] = state.offer[c]

    # [12-17] Trash
    for c in range(NUM_COLORS):
        obs[12 + c] = state.trash[c]

    # [18-23] Opponent's confirmed hand
    for c in range(NUM_COLORS):
        obs[18 + c] = bid_tracker.confirmed[opp][c]

    # [24] Opponent's unknown card count
    opp_total = sum(state.hand[opp])
    opp_confirmed_total = sum(bid_tracker.confirmed[opp])
    obs[24] = max(0, opp_total - opp_confirmed_total)

    # [25-30] My confirmed hand (what opponent knows about me)
    for c in range(NUM_COLORS):
        obs[25 + c] = bid_tracker.confirmed[player][c]

    # [31] Bag remaining (normalized to [0, 1])
    obs[31] = state.bag_left() / TOTAL_STONES

    # [32] Am I caretaker?
    obs[32] = 1.0 if state.caretaker == player else 0.0

    # [33] My expected hand score (normalized)
    obs[33] = compute_expected_score(state, player)

    return obs


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

    obs = np.zeros(34, dtype=np.float32)
    opp = 1 - my_index

    players = server_state.get("players", [])
    if len(players) < 2 or my_index < 0:
        return obs

    me = players[my_index]
    them = players[opp]

    # [0-5] My hand
    hand = me.get("hand", [])
    if hand:
        for s in hand:
            idx = COLOR_TO_IDX.get(s)
            if idx is not None:
                obs[idx] += 1

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

    # [31] Bag remaining (normalized)
    bag_left = server_state.get("bag_left", 0)
    obs[31] = bag_left / TOTAL_STONES

    # [32] Am I caretaker?
    caretaker = server_state.get("caretaker", 0)
    obs[32] = 1.0 if caretaker == my_index else 0.0

    # [33] Expected score (approximate from available info)
    # Since we can't compute the exact expected score from server state
    # (we don't know opponent's hand), we estimate from our hand alone
    my_score = me.get("score", 0)
    opp_score = them.get("score", 0)
    score_diff = (my_score - opp_score)
    obs[33] = max(-1.0, min(1.0, score_diff / 30.0))

    return obs
