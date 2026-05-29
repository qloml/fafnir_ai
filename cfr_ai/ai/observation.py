"""
Observation space builder for Fafnir Deep CFR.

Observation vector (42 dimensions):
  [0-5]   : My hand counts (6 colors)
  [6-11]  : Current offer counts (6 colors)
  [12-17] : Trash counts (6 colors)
  [18-23] : Opponent's confirmed hand (6 colors) - deduced from bid history
  [24]    : Opponent's unknown card count
  [25-30] : My confirmed hand (6 colors) - what opponent knows I have
  [31]    : Bag remaining count (normalized)
  [32]    : Am I caretaker? (0 or 1)
  [33]    : My visible hand potential (normalized, own hand only)
  --- NEW (v2) ---
  [34]    : My game score (normalized: /1000)
  [35]    : Opponent's game score (normalized: /1000)
  [36]    : Round number (normalized: /20)
  [37]    : Turn number within round (normalized: /30)
  [38]    : Offer total stones (normalized: /10)
  [39]    : My hand total (normalized: /20)
  [40]    : Opponent's hand total (normalized: /20)
  [41]    : Trash total (normalized: /36, max = 6 colors × TRASH_LIMIT)
"""
import numpy as np
from typing import List, Dict, Any

from .game_engine import (
    FafnirState, NUM_COLORS, TOTAL_STONES, compute_visible_hand_potential,
    SCORE_TO_WIN, TRASH_LIMIT,
)

# Observation dimensions
OBS_DIM = 42


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
    Build the 42-dimensional observation vector for a player.
    All values are raw counts except for the normalized fields.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    opp = 1 - player

    # [0-5] My hand
    obs[0:6] = state.hand[player]

    # [6-11] Current offer
    obs[6:12] = state.offer

    # [12-17] Trash
    obs[12:18] = state.trash

    # [18-23] Opponent's confirmed hand
    obs[18:24] = bid_tracker.confirmed[opp]

    # [24] Opponent's unknown card count
    opp_total = sum(state.hand[opp])
    opp_confirmed_total = sum(bid_tracker.confirmed[opp])
    obs[24] = max(0, opp_total - opp_confirmed_total)

    # [25-30] My confirmed hand (what opponent knows about me)
    obs[25:31] = bid_tracker.confirmed[player]

    # [31] Bag remaining (normalized to [0, 1])
    obs[31] = state.bag_left() / TOTAL_STONES

    # [32] Am I caretaker?
    obs[32] = 1.0 if state.caretaker == player else 0.0

    # [33] My visible hand potential (normalized, own hand only)
    obs[33] = compute_visible_hand_potential(state.hand[player])

    # --- NEW (v2) ---

    # [34] My game score (normalized)
    obs[34] = state.scores[player] / SCORE_TO_WIN

    # [35] Opponent's game score (normalized)
    obs[35] = state.scores[opp] / SCORE_TO_WIN

    # [36] Round number (normalized, cap at 20)
    obs[36] = min(state.round_num, 20) / 20.0

    # [37] Turn number within round (normalized, cap at 30)
    obs[37] = min(state.turn_num, 30) / 30.0

    # [38] Offer total stones (normalized)
    obs[38] = sum(state.offer) / 10.0

    # [39] My hand total (normalized)
    obs[39] = sum(state.hand[player]) / 20.0

    # [40] Opponent's hand total (normalized)
    obs[40] = opp_total / 20.0

    # [41] Trash total (normalized by max trash capacity)
    obs[41] = sum(state.trash) / (NUM_COLORS * TRASH_LIMIT)

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

    # [31] Bag remaining (normalized)
    bag_left = server_state.get("bag_left", 0)
    obs[31] = bag_left / TOTAL_STONES

    # [32] Am I caretaker?
    caretaker = server_state.get("caretaker", 0)
    obs[32] = 1.0 if caretaker == my_index else 0.0

    my_score = me.get("score", 0)
    opp_score = them.get("score", 0)
    # [33] My visible hand potential (same fair feature as training)
    obs[33] = compute_visible_hand_potential(my_hand_counts)

    # --- NEW (v2) ---

    # [34] My game score (normalized)
    obs[34] = my_score / SCORE_TO_WIN

    # [35] Opponent's game score (normalized)
    obs[35] = opp_score / SCORE_TO_WIN

    # [36] Round number (normalized)
    round_num = server_state.get("round", 1)
    obs[36] = min(round_num, 20) / 20.0

    # [37] Turn number (normalized)
    turn_num = server_state.get("turn", 1)
    obs[37] = min(turn_num, 30) / 30.0

    # [38] Offer total stones (normalized)
    obs[38] = len(offer) / 10.0

    # [39] My hand total (normalized)
    obs[39] = sum(my_hand_counts) / 20.0

    # [40] Opponent's hand total (normalized)
    obs[40] = opp_total / 20.0

    # [41] Trash total (normalized)
    trash_total = sum(trash.get(c, 0) for c in ALL_COLORS)
    obs[41] = trash_total / (NUM_COLORS * TRASH_LIMIT)

    return obs
