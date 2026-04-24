# clients/ai_bot.py
import asyncio
import argparse
import random
from typing import Any, Dict, List, Optional

import socketio

sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "AI", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None

# anti-spam
_action_lock = asyncio.Lock()
_last_emit_ts = 0.0

# OK debounce per RESULT/ROUND_END instance
_ok_sent_key: Optional[str] = None

AUTO_NEXT = True
ALLOW_EMPTY_BID = False
THINK_DELAY = 0.01


def _loop_time() -> float:
    return asyncio.get_running_loop().time()


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def phase_of(st: Dict[str, Any]) -> str:
    return str(st.get("phase") or "WAITING")


def current_bidder(st: Dict[str, Any]) -> Optional[int]:
    cb = st.get("current_bidder", None)
    try:
        return int(cb)
    except Exception:
        return None


def players_of(st: Dict[str, Any]) -> List[Dict[str, Any]]:
    ps = st.get("players")
    return ps if isinstance(ps, list) else []


def me_view(st: Dict[str, Any]) -> Dict[str, Any]:
    ps = players_of(st)
    if my_index is None or my_index < 0 or my_index >= len(ps):
        return {}
    v = ps[my_index]
    return v if isinstance(v, dict) else {}


def my_hand(st: Dict[str, Any]) -> List[str]:
    hand = me_view(st).get("hand")
    return [x for x in hand] if isinstance(hand, list) else []


def my_bid_submitted(st: Dict[str, Any]) -> bool:
    return bool(me_view(st).get("bid_submitted", False))


def my_ok_ready(st: Dict[str, Any]) -> bool:
    return bool(me_view(st).get("ok_ready", False))


def offer_set(st: Dict[str, Any]) -> set:
    offer = safe_list(st.get("offer"))
    return set([x for x in offer if isinstance(x, str)])


def sanitize_bid(proposal: List[str], hand: List[str], forbidden: set) -> List[str]:
    """
    - Avoid colors listed in the offer.
    - Must have a multiset in hand.
    """
    tmp = hand[:]
    out: List[str] = []
    for s in proposal:
        if not isinstance(s, str):
            continue
        if s in forbidden:
            continue
        if s in tmp:
            out.append(s)
            tmp.remove(s)
    return out


def choose_bid(hand: List[str]) -> List[str]:
    """
    Prevent "no pending turn" by attempting to bid at least 1 coin (if empty bets are not allowed).
    """
    if not hand:
        return []

    if ALLOW_EMPTY_BID and random.random() < 0.15:
        return []

    # 1..3 coins
    n = random.randint(1, min(3, len(hand)))
    idxs = random.sample(range(len(hand)), n)
    return [hand[i] for i in idxs]


async def _emit_throttled(event: str, payload: Dict[str, Any], min_interval: float = 0.12):
    global _last_emit_ts
    async with _action_lock:
        dt = _loop_time() - _last_emit_ts
        if dt < min_interval:
            await asyncio.sleep(min_interval - dt)
        _last_emit_ts = _loop_time()
        await sio.emit(event, payload)


def _phase_key(st: Dict[str, Any]) -> str:
    """
    Use these as keys to prevent duplication:
    - RESULT: Use round/turn
    - ROUND_END: ​​Use round (when declaring the end of a round)
    """
    ph = phase_of(st)
    r = st.get("round", "?")
    t = st.get("turn", "?")
    if ph == "ROUND_END":
        return f"ROUND_END:r{r}"
    if ph == "RESULT":
        return f"RESULT:r{r}:t{t}"
    return f"{ph}:r{r}:t{t}"


async def do_submit_bid(st: Dict[str, Any], reason: str):
    hand = my_hand(st)
    forbidden = offer_set(st)

    proposal = choose_bid(hand)
    bid = sanitize_bid(proposal, hand, forbidden)

    # If you're not allowed to empty the area, sanitize it until it's empty -> find another one.
    if (not ALLOW_EMPTY_BID) and (len(bid) == 0) and hand:
        # Choose a coin that does not have at least one conflicting offer.
        candidates = [x for x in hand if x not in forbidden]
        if candidates:
            bid = [random.choice(candidates)]

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("submit_bid", {"room_id": cfg["room"], "stones": bid})
    print(f"[AI] submit ({reason}) stones={bid}")


async def do_ok_next(st: Dict[str, Any], reason: str):
    global _ok_sent_key
    key = _phase_key(st)
    if _ok_sent_key == key:
        return
    if my_ok_ready(st):
        # We already pressed it on our side.
        _ok_sent_key = key
        return

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("proceed_phase", {"room_id": cfg["room"]})
    _ok_sent_key = key
    print(f"[AI] OK/Next ({reason})")


async def brain_loop():
    """
    ✅ Key solution: Don't rely solely on events.
    This loop will always "move the game" on its own.
    """
    while True:
        st = last_state
        if st and my_index is not None and my_index >= 0:
            ph = phase_of(st)

            if ph == "BIDDING":
                cb = current_bidder(st)
                if cb == my_index and (not my_bid_submitted(st)):
                    await do_submit_bid(st, reason="brain_loop")

            elif ph in ("RESULT", "ROUND_END"):
                if AUTO_NEXT:
                    await do_ok_next(st, reason="brain_loop")

            # GAME_END: ​​Do nothing (allow the user to restart)
        await asyncio.sleep(0.10)


# ============ socket handlers ============

@sio.event
async def connect():
    print("[AI] connected")
    await _emit_throttled(
        "join_room",
        {"room_id": cfg["room"], "player_name": cfg["name"]},
        min_interval=0.0,
    )


@sio.event
async def disconnect():
    print("[AI] disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print("[AI] assigned index =", my_index)


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key
    last_state = state

    # reset OK debounce when leaving RESULT/ROUND_END
    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    """
    If not_your_turn -> Silent, wait for state_update/brain_loop
    Otherwise -> brain_loop will try again when it's your turn.
    """
    reason = data.get("reason") or ""
    msg = data.get("message") or reason
    print("[AI] BID REJECTED:", msg)
    # No need to do anything extra. Let brain_loop handle it.


# ============ main ============

async def main():
    global AUTO_NEXT, ALLOW_EMPTY_BID, THINK_MIN, THINK_MAX

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="AI")

    ap.add_argument("--auto-next", type=int, default=1, help="1=auto OK/Next in RESULT/ROUND_END, 0=disable")
    ap.add_argument("--allow-empty", type=int, default=0, help="1=allow empty bid sometimes, 0=force >=1 if possible")
    ap.add_argument("--think-min", type=float, default=0.05)
    ap.add_argument("--think-max", type=float, default=0.18)
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name

    AUTO_NEXT = bool(args.auto_next)
    ALLOW_EMPTY_BID = bool(args.allow_empty)
    THINK_MIN = float(args.think_min)
    THINK_MAX = float(args.think_max)

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
        # extra safety close (ลด unclosed session)
        try:
            eio = getattr(sio, "eio", None)
            if eio is not None and getattr(eio, "connected", False):
                await eio.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
