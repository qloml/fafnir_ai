# clients/rl_bot.py
"""
FAFNIR RL Bot — Socket.IO client using a trained MaskablePPO model.
Same interface as ai_bot_sample.py but uses the RL model for decisions.

Usage:
    python clients/rl_bot.py --model rl/output/fafnir_final
    python clients/rl_bot.py --model rl/output/fafnir_final --url http://SERVER:8765 --room room1 --name RLBot
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import socketio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from rl.game_env import N_COLORS, MAX_BID_PER_COLOR, INITIAL_BAG, TRASH_LIMIT, COLORS_NAMES

sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "RLBot", "url": "http://127.0.0.1:8765", "model_path": ""}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
model: Optional[MaskablePPO] = None

_action_lock = asyncio.Lock()
_ok_sent_key: Optional[str] = None

AUTO_NEXT = True
THINK_DELAY = 0.05
DETERMINISTIC = True


# ==========================================
# State -> Observation conversion
# ==========================================

def _color_to_idx(color_str: str) -> int:
    """Convert color string to index."""
    s = str(color_str).lower().strip()
    for i, name in enumerate(COLORS_NAMES):
        if s == name:
            return i
    return -1


def _stones_to_counts(stones: Any) -> np.ndarray:
    """Convert list of color strings to count array."""
    counts = np.zeros(N_COLORS, dtype=np.int32)
    if not isinstance(stones, list):
        return counts
    for s in stones:
        idx = _color_to_idx(s)
        if 0 <= idx < N_COLORS:
            counts[idx] += 1
    return counts


def _trash_to_counts(trash: Any) -> np.ndarray:
    """Convert trash dict to count array."""
    counts = np.zeros(N_COLORS, dtype=np.int32)
    if not isinstance(trash, dict):
        return counts
    for name, n in trash.items():
        idx = _color_to_idx(name)
        if 0 <= idx < N_COLORS:
            counts[idx] = int(n)
    return counts


# Known-hand tracking (updated each auction from server state)
_known = np.zeros((2, N_COLORS), dtype=np.int32)  # confirmed stones per player
_prev_round = -1  # track round changes to reset known hands


def _rank_colors_from_hands(my_counts, opp_total_counts):
    """Approximate color ranking from known info. Uses my hand + best guess for opponent."""
    # For potential score, we need total counts across both players for non-gold colors
    totals = []
    for c in range(1, N_COLORS):
        t = int(my_counts[c]) + int(opp_total_counts[c])
        totals.append((c, t))
    totals.sort(key=lambda x: (-x[1], x[0]))
    return totals


def _compute_potential_from_counts(hand_counts, opp_counts):
    """Compute the expected round-end score for the player."""
    ranked = _rank_colors_from_hands(hand_counts, opp_counts)
    first_c = ranked[0][0] if ranked else 1
    second_c = ranked[1][0] if len(ranked) > 1 else 2

    score = int(hand_counts[0])  # gold
    for c in range(1, N_COLORS):
        cnt = int(hand_counts[c])
        if cnt == 0 or cnt >= 5:
            continue
        if c == first_c:
            score += cnt * 3
        elif c == second_c:
            score += cnt * 2
        else:
            score -= cnt
    return score


def update_known_from_state(st: Dict[str, Any], player_idx: int):
    """Update known-hand tracking from server state (call each state_update)."""
    global _known, _prev_round

    current_round = st.get("round", 1)
    if current_round != _prev_round:
        # Round changed — reset known hands
        _known[:] = 0
        _prev_round = current_round

    # Check last_result for new auction info
    lr = st.get("last_result") or {}
    winner = lr.get("winner")
    if winner is None:
        return

    try:
        wi = int(winner)
    except Exception:
        return
    li = 1 - wi

    # Get bids
    bids_by_player = lr.get("bids_by_player")
    if isinstance(bids_by_player, list) and len(bids_by_player) >= 2:
        w_bid = _stones_to_counts(bids_by_player[wi] or [])
        l_bid = _stones_to_counts(bids_by_player[li] or [])
    else:
        w_bid = _stones_to_counts(lr.get("winner_bid") or [])
        l_bid = _stones_to_counts(lr.get("loser_bid") or [])

    # Get offer that was competed for
    offer_counts = _stones_to_counts(st.get("last_offer") or lr.get("offer") or [])

    # Update known hands
    for c in range(N_COLORS):
        val = _known[wi, c] + offer_counts[c] - w_bid[c]
        _known[wi, c] = max(0, val)
        _known[li, c] = max(_known[li, c], int(l_bid[c]))


def state_to_obs(st: Dict[str, Any], player_idx: int, score_to_win: int = 1000) -> np.ndarray:
    """Convert server state_update dict to 36-dim observation vector."""
    obs = np.zeros(36, dtype=np.float32)
    ps = st.get("players") or []
    other_idx = 1 - player_idx

    # 0-5: my hand counts
    hand_counts = np.zeros(N_COLORS, dtype=np.int32)
    if player_idx < len(ps):
        hand = ps[player_idx].get("hand") or []
        hand_counts = _stones_to_counts(hand)
        for c in range(N_COLORS):
            obs[c] = hand_counts[c] / max(1, INITIAL_BAG[c])

    # 6-11: offer (use 10 as max for multi-stone offers)
    offer = st.get("offer") or []
    offer_counts = _stones_to_counts(offer)
    for c in range(N_COLORS):
        obs[6 + c] = offer_counts[c] / 10.0

    # 12-17: trash
    trash_counts = _trash_to_counts(st.get("trash") or {})
    for c in range(N_COLORS):
        obs[12 + c] = trash_counts[c] / float(TRASH_LIMIT)

    # 18-23: opponent's confirmed hand
    for c in range(N_COLORS):
        obs[18 + c] = _known[other_idx, c] / max(1, INITIAL_BAG[c])

    # 24: opponent's unknown stone count
    opp_total = 0
    if other_idx < len(ps):
        opp_total = ps[other_idx].get("hand_count", 0)
    opp_known_total = int(_known[other_idx].sum())
    opp_unknown = max(0, opp_total - opp_known_total)
    obs[24] = opp_unknown / 15.0

    # 25-30: my confirmed hand (what opponent knows about me)
    for c in range(N_COLORS):
        obs[25 + c] = _known[player_idx, c] / max(1, INITIAL_BAG[c])

    # 31: my score
    if player_idx < len(ps):
        obs[31] = ps[player_idx].get("score", 0) / float(max(1, score_to_win))

    # 32: opponent score
    if other_idx < len(ps):
        obs[32] = ps[other_idx].get("score", 0) / float(max(1, score_to_win))

    # 33: bag remaining
    obs[33] = st.get("bag_left", 0) / float(max(1, sum(INITIAL_BAG)))

    # 34: am I caretaker
    obs[34] = 1.0 if st.get("caretaker") == player_idx else 0.0

    # 35: my hand's current potential score
    # Approximate opponent hand for ranking: use known + distribute unknown evenly
    opp_approx = np.zeros(N_COLORS, dtype=np.int32)
    for c in range(N_COLORS):
        opp_approx[c] = _known[other_idx, c]
    potential = _compute_potential_from_counts(hand_counts, opp_approx)
    obs[35] = (potential + 15.0) / 75.0

    return np.clip(obs, 0.0, 1.0)


def state_to_mask(st: Dict[str, Any], player_idx: int) -> np.ndarray:
    """Build action mask from server state."""
    mask_size = N_COLORS * (MAX_BID_PER_COLOR + 1)
    mask = np.zeros(mask_size, dtype=np.int8)

    ps = st.get("players") or []
    if player_idx >= len(ps):
        # Fallback: allow only zeros
        for c in range(N_COLORS):
            mask[c * (MAX_BID_PER_COLOR + 1)] = 1
        return mask

    hand = ps[player_idx].get("hand") or []
    hand_counts = _stones_to_counts(hand)
    offer = st.get("offer") or []
    offer_counts = _stones_to_counts(offer)

    for c in range(N_COLORS):
        base = c * (MAX_BID_PER_COLOR + 1)
        if offer_counts[c] > 0:
            mask[base] = 1  # can only bid 0
        else:
            max_valid = min(int(hand_counts[c]), MAX_BID_PER_COLOR)
            for v in range(max_valid + 1):
                mask[base + v] = 1

    return mask


def action_to_stones(action: np.ndarray, hand: List[str]) -> List[str]:
    """Convert action (count per color) to list of stone strings for submit_bid."""
    hand_counts = _stones_to_counts(hand)
    stones: List[str] = []
    for c in range(N_COLORS):
        count = min(int(action[c]), int(hand_counts[c]))
        for _ in range(count):
            stones.append(COLORS_NAMES[c])
    return stones


# ==========================================
# Bot logic
# ==========================================

def phase_of(st: Dict[str, Any]) -> str:
    return str(st.get("phase") or "WAITING")


def current_bidder(st: Dict[str, Any]) -> Optional[int]:
    cb = st.get("current_bidder")
    try:
        return int(cb) if cb is not None else None
    except Exception:
        return None


async def _emit_safe(event: str, payload: Dict[str, Any]):
    if not sio.connected:
        return
    try:
        await sio.emit(event, payload)
    except Exception as e:
        print(f"[RL] emit error: {repr(e)}")


async def do_submit_bid(st: Dict[str, Any]):
    global model
    if model is None or my_index is None:
        return

    obs = state_to_obs(st, my_index)
    mask = state_to_mask(st, my_index)
    action, _ = model.predict(obs, action_masks=mask, deterministic=DETERMINISTIC)

    ps = st.get("players") or []
    hand = ps[my_index].get("hand") or [] if my_index < len(ps) else []
    stones = action_to_stones(action, hand)

    print(f"[RL] Bidding: {stones} (action={action})")
    await asyncio.sleep(THINK_DELAY)
    await _emit_safe("submit_bid", {"room_id": cfg["room"], "stones": stones})


async def do_ok_next(st: Dict[str, Any]):
    global _ok_sent_key
    ph = phase_of(st)
    key = f"{ph}:{st.get('round')}:{st.get('turn')}"
    if _ok_sent_key == key:
        return

    ps = st.get("players") or []
    if my_index is not None and my_index < len(ps):
        if ps[my_index].get("ok_ready"):
            _ok_sent_key = key
            return

    await asyncio.sleep(THINK_DELAY)
    await _emit_safe("proceed_phase", {"room_id": cfg["room"]})
    _ok_sent_key = key
    print(f"[RL] OK/Next ({ph})")


async def brain_loop():
    while True:
        st = last_state
        if st and my_index is not None and my_index >= 0:
            ph = phase_of(st)

            if ph == "BIDDING":
                cb = current_bidder(st)
                ps = st.get("players") or []
                submitted = False
                if my_index < len(ps):
                    submitted = ps[my_index].get("bid_submitted", False)
                if cb == my_index and not submitted:
                    async with _action_lock:
                        await do_submit_bid(st)

            elif ph in ("RESULT", "ROUND_END"):
                if AUTO_NEXT:
                    async with _action_lock:
                        await do_ok_next(st)

        await asyncio.sleep(0.10)


# ==========================================
# Socket events
# ==========================================

@sio.event
async def connect():
    print("[RL] Connected to server")
    await sio.emit("join_room", {
        "room_id": cfg["room"],
        "player_name": cfg["name"],
    })


@sio.event
async def disconnect():
    print("[RL] Disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print(f"[RL] Assigned index = {my_index}")


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key
    last_state = state
    # Update known-hand tracking
    if my_index is not None:
        update_known_from_state(state, my_index)
    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    msg = data.get("message") or data.get("reason") or "rejected"
    print(f"[RL] BID REJECTED: {msg}")


# ==========================================
# Main
# ==========================================

async def main():
    global model, DETERMINISTIC, AUTO_NEXT, THINK_DELAY

    ap = argparse.ArgumentParser(description="FAFNIR RL Bot (Socket.IO client)")
    ap.add_argument("--model", type=str, required=True, help="Path to trained model")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:8765")
    ap.add_argument("--room", type=str, default="room1")
    ap.add_argument("--name", type=str, default="RLBot")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--think-delay", type=float, default=0.05)
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name
    DETERMINISTIC = bool(args.deterministic)
    THINK_DELAY = args.think_delay

    # Load model
    print(f"[RL] Loading model: {args.model}")
    model = MaskablePPO.load(args.model)
    print(f"[RL] Model loaded successfully!")

    task_brain = None
    try:
        await sio.connect(cfg["url"], wait_timeout=15)
        task_brain = asyncio.create_task(brain_loop())
        await sio.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if task_brain:
            task_brain.cancel()
            try:
                await task_brain
            except Exception:
                pass
        try:
            if sio.connected:
                await sio.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
