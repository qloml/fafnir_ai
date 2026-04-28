# mppo_ai/clients/rl_bot_v1.py
"""
FAFNIR RL Bot (v1 — 25-dim observation) — Socket.IO client using a trained MaskablePPO model.
For use with models trained on the OLD 25-dimension observation space.

Usage:
    python mppo_ai/clients/rl_bot_v1.py --model mppo_ai/rl/output/completed/fafnir_final.zip
    python mppo_ai/clients/rl_bot_v1.py --model mppo_ai/rl/output/completed/fafnir_final.zip --url http://SERVER:8765 --room room1 --name RLBot_v1
"""
import asyncio
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import socketio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

# ==========================================
# Constants (matching old v1 engine)
# ==========================================
N_COLORS = 6
MAX_BID_PER_COLOR = 10
INITIAL_BAG = [20, 12, 12, 12, 12, 12]
TRASH_LIMIT = 6
COLORS_NAMES = ["gold", "red", "orange", "yellow", "green", "blue"]
BAG_TOTAL = sum(INITIAL_BAG)
OBS_DIM = 25

sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "RLBot_v1", "url": "http://127.0.0.1:8765", "model_path": ""}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
model: Optional[MaskablePPO] = None

_action_lock = asyncio.Lock()
_ok_sent_key: Optional[str] = None
_restart_sent = False

AUTO_NEXT = True
THINK_DELAY = 0.05
DETERMINISTIC = True


# ==========================================
# State -> Observation conversion (25-dim)
# ==========================================

def _color_to_idx(color_str: str) -> int:
    s = str(color_str).lower().strip()
    for i, name in enumerate(COLORS_NAMES):
        if s == name:
            return i
    return -1


def _stones_to_counts(stones: Any) -> np.ndarray:
    counts = np.zeros(N_COLORS, dtype=np.int32)
    if not isinstance(stones, list):
        return counts
    for s in stones:
        idx = _color_to_idx(s)
        if 0 <= idx < N_COLORS:
            counts[idx] += 1
    return counts


def _trash_to_counts(trash: Any) -> np.ndarray:
    counts = np.zeros(N_COLORS, dtype=np.int32)
    if not isinstance(trash, dict):
        return counts
    for name, n in trash.items():
        idx = _color_to_idx(name)
        if 0 <= idx < N_COLORS:
            counts[idx] = int(n)
    return counts


def state_to_obs(st: Dict[str, Any], player_idx: int, score_to_win: int = 30) -> np.ndarray:
    """Convert server state_update dict to 25-dim observation vector (v1 layout).

    Layout:
      0-5  : my hand (6 colors)
      6-11 : offer (6 colors)
      12-17: trash (6 colors)
      18   : my score
      19   : opponent score
      20   : opponent hand count
      21   : bag remaining
      22   : am I caretaker
      23   : round progress (turn / max_turns approx)
      24   : last auction winner == me
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    ps = st.get("players") or []
    other_idx = 1 - player_idx

    # 0-5: my hand
    if player_idx < len(ps):
        hand = ps[player_idx].get("hand") or []
        hand_counts = _stones_to_counts(hand)
        for c in range(N_COLORS):
            obs[c] = hand_counts[c] / max(1, INITIAL_BAG[c])

    # 6-11: offer
    offer = st.get("offer") or []
    offer_counts = _stones_to_counts(offer)
    for c in range(N_COLORS):
        obs[6 + c] = offer_counts[c] / 10.0

    # 12-17: trash
    trash_counts = _trash_to_counts(st.get("trash") or {})
    for c in range(N_COLORS):
        obs[12 + c] = trash_counts[c] / float(TRASH_LIMIT)

    # 18: my score
    if player_idx < len(ps):
        obs[18] = ps[player_idx].get("score", 0) / float(max(1, score_to_win))

    # 19: opponent score
    if other_idx < len(ps):
        obs[19] = ps[other_idx].get("score", 0) / float(max(1, score_to_win))

    # 20: opponent hand count (normalized)
    if other_idx < len(ps):
        obs[20] = ps[other_idx].get("hand_count", 0) / 15.0

    # 21: bag remaining
    obs[21] = st.get("bag_left", 0) / float(BAG_TOTAL)

    # 22: am I caretaker
    obs[22] = 1.0 if st.get("caretaker") == player_idx else 0.0

    # 23: round progress
    turn = st.get("turn", 1)
    obs[23] = min(1.0, turn / 50.0)

    # 24: last auction winner == me
    lr = st.get("last_result") or {}
    w = lr.get("winner")
    if w is not None:
        try:
            obs[24] = 1.0 if int(w) == player_idx else 0.0
        except Exception:
            pass

    return np.clip(obs, 0.0, 1.0)


def state_to_mask(st: Dict[str, Any], player_idx: int) -> np.ndarray:
    mask_size = N_COLORS * (MAX_BID_PER_COLOR + 1)
    mask = np.zeros(mask_size, dtype=np.int8)

    ps = st.get("players") or []
    if player_idx >= len(ps):
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
            mask[base] = 1
        else:
            max_valid = min(int(hand_counts[c]), MAX_BID_PER_COLOR)
            for v in range(max_valid + 1):
                mask[base + v] = 1

    return mask


def action_to_stones(action: np.ndarray, hand: List[str]) -> List[str]:
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
        print(f"[RL-v1] emit error: {repr(e)}")


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

    print(f"[RL-v1] Bidding: {stones} (action={action})")
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
    print(f"[RL-v1] OK/Next ({ph})")


async def do_restart_game():
    global _restart_sent, _ok_sent_key
    if _restart_sent:
        return
    _restart_sent = True
    _ok_sent_key = None
    print("[RL-v1] Game ended — sending restart_game")
    await asyncio.sleep(THINK_DELAY)
    await _emit_safe("restart_game", {"room_id": cfg["room"]})


async def brain_loop():
    global _restart_sent
    while True:
        try:
            st = last_state
            if st and my_index is not None and my_index >= 0:
                ph = phase_of(st)

                if ph == "BIDDING":
                    _restart_sent = False
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

                elif ph == "GAME_END":
                    async with _action_lock:
                        await do_restart_game()

        except Exception as e:
            print(f"[RL-v1] Error in brain_loop: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.10)


# ==========================================
# Socket events
# ==========================================

@sio.event
async def connect():
    print("[RL-v1] Connected to server")
    await sio.emit("join_room", {
        "room_id": cfg["room"],
        "player_name": cfg["name"],
    })


@sio.event
async def disconnect():
    print("[RL-v1] Disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print(f"[RL-v1] Assigned index = {my_index}")


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key
    last_state = state
    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    msg = data.get("message") or data.get("reason") or "rejected"
    print(f"[RL-v1] BID REJECTED: {msg}")


# ==========================================
# Main
# ==========================================

async def main():
    global model, DETERMINISTIC, AUTO_NEXT, THINK_DELAY

    ap = argparse.ArgumentParser(description="FAFNIR RL Bot v1 (25-dim, Socket.IO client)")
    ap.add_argument("--model", type=str, required=True, help="Path to trained model (25-dim)")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:8765")
    ap.add_argument("--room", type=str, default="room1")
    ap.add_argument("--name", type=str, default="RLBot_v1")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--think-delay", type=float, default=0.05)
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name
    DETERMINISTIC = bool(args.deterministic)
    THINK_DELAY = args.think_delay

    # Load model
    print(f"[RL-v1] Loading model: {args.model}")
    model = MaskablePPO.load(args.model)
    print(f"[RL-v1] Model loaded successfully! (25-dim obs)")

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
