# clients/human_cli.py
import asyncio
import argparse
from typing import Dict, Optional, List, Any, Tuple

import socketio

sio = socketio.AsyncClient(reconnection=True)

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
cfg = {"room": "room1", "name": "Human"}

_last_render_key: Optional[Tuple] = None
_last_reject_msg: Optional[str] = None

# debounce OK/Next
_last_ok_phase_key: Optional[Tuple[Any, Any, Any]] = None

# ✅ Prevent duplicate join_room during reconnection.
_join_sent: bool = False


# =========================
# Safe emit (Use only during normal gameplay)
# =========================
async def safe_emit(event: str, data: Dict[str, Any]):
    if not sio.connected:
        print(f"[WARN] Not connected -> cannot emit '{event}'. (auto-reconnect running)")
        return
    try:
        await sio.emit(event, data)
    except Exception as e:
        print(f"[ERR] emit '{event}' failed:", repr(e))


# =========================
# UI helpers
# =========================
def clear_screen():
    print("\033[2J\033[H", end="")


def phase_of(st: Optional[Dict[str, Any]]) -> str:
    return str((st or {}).get("phase") or "WAITING")


def my_player(st: Dict[str, Any]) -> Dict[str, Any]:
    ps = st.get("players") or []
    if my_index is None or my_index >= len(ps):
        return {}
    return ps[my_index]


def my_hand(st: Dict[str, Any]) -> List[str]:
    return (my_player(st).get("hand") or [])


def my_bid_submitted(st: Dict[str, Any]) -> bool:
    return bool(my_player(st).get("bid_submitted"))


def current_bidder(st: Dict[str, Any]) -> Optional[int]:
    v = st.get("current_bidder")
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def is_my_turn_to_bid(st: Dict[str, Any]) -> bool:
    if my_index is None:
        return False
    cb = current_bidder(st)
    return cb is not None and cb == my_index


def render_key(st: Dict[str, Any]) -> Tuple:
    ps = st.get("players") or []
    players_sig = tuple(
        (p.get("score"), p.get("hand_count"), p.get("ok_ready"), p.get("bid_submitted"))
        for p in ps
    )
    # NOTE: last_result not included; but phase/turn changes when result happens anyway.
    return (
        st.get("room_id"),
        st.get("round"),
        st.get("turn"),
        st.get("phase"),
        tuple(st.get("offer") or []),
        tuple(st.get("last_offer") or []),
        str(st.get("status") or ""),
        st.get("caretaker"),
        st.get("current_bidder"),
        st.get("bag_left"),
        players_sig,
        _last_reject_msg,
        str(st.get("round_end_info") or ""),
        str(st.get("game_end_info") or ""),
        str(st.get("trash") or ""),
    )


def format_hand(hand: List[str]) -> str:
    return "  ".join([f"[{i}]{s}" for i, s in enumerate(hand)])


# =========================
# Trash board + Risk %
# =========================
def _pct(n: int, d: int) -> int:
    if d <= 0:
        return 0
    v = int(round((n / d) * 100))
    return max(0, min(100, v))


def compute_round_end_risk(trash: Dict[str, int], limit: int) -> Dict[str, Any]:
    order = ["gold", "red", "orange", "yellow", "green", "blue"]
    per = {c: _pct(int(trash.get(c, 0)), limit) for c in order}
    max_pct = max(per.values()) if per else 0
    leaders = [c for c in order if per.get(c, 0) == max_pct and max_pct > 0]
    return {"overall_pct": max_pct, "leaders": leaders, "per_color_pct": per}


def render_trash_board(trash: Dict[str, int], limit: int) -> str:
    order = ["gold", "red", "orange", "yellow", "green", "blue"]
    icon = {
        "gold": "💛",
        "red": "🔴",
        "orange": "🟠",
        "yellow": "🟡",
        "green": "🟢",
        "blue": "🔵",
    }

    risk = compute_round_end_risk(trash, limit)
    overall = risk["overall_pct"]
    leaders = risk["leaders"]
    leaders_txt = ", ".join([c.upper() for c in leaders]) if leaders else "-"

    lines = []
    lines.append("┌─────────────────────────────── TRASH BOARD ───────────────────────────────┐")
    lines.append(f"│ Round-End Risk: {overall:>3}%   Leader: {leaders_txt:<20} (progress to {limit}) │")
    lines.append("├────────┬─────────┬────────────────┬────────┬──────────────────────────────┤")
    lines.append("│  Icon  │  Color  │     Board      │ Count  │ Risk % (color progress)       │")
    lines.append("├────────┼─────────┼────────────────┼────────┼──────────────────────────────┤")

    for c in order:
        n = max(0, int(trash.get(c, 0)))
        filled = "■" * min(n, limit)
        empty = "□" * max(0, limit - n)
        bar = filled + empty

        pct = _pct(n, limit)
        mini_fill = int(round(pct / 10))
        mini = "■" * mini_fill + "□" * (10 - mini_fill)

        lines.append(
            f"│  {icon.get(c,'•')}   │ {c.upper():<7}│ {bar:<14} │ {n:>2}/{limit:<2} │ {pct:>3}% {mini}                 │"
        )

    lines.append("└────────┴─────────┴────────────────┴────────┴──────────────────────────────┘")
    return "\n".join(lines)


# =========================
# Result helpers
# =========================
_COLOR_ORDER = ["gold", "red", "orange", "yellow", "green", "blue"]
_COLOR_ICON = {
    "gold": "💛",
    "red": "🔴",
    "orange": "🟠",
    "yellow": "🟡",
    "green": "🟢",
    "blue": "🔵",
}


def stone_to_color(stone: Any) -> str:
    s = str(stone).strip().lower()
    for c in _COLOR_ORDER:
        if s.startswith(c):
            return c
    return "unknown"


def summarize_stones(stones: Any) -> str:
    if not stones:
        return "-"
    if not isinstance(stones, list):
        try:
            stones = list(stones)
        except Exception:
            return str(stones)

    counts: Dict[str, int] = {}
    for st in stones:
        c = stone_to_color(st)
        counts[c] = counts.get(c, 0) + 1

    parts = []
    for c in _COLOR_ORDER + ["unknown"]:
        if c in counts:
            icon = _COLOR_ICON.get(c, "•")
            parts.append(f"{icon}{c.upper()}x{counts[c]}")
    return " ".join(parts) if parts else "-"


def _safe_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    try:
        return list(v)
    except Exception:
        return []


def extract_bids_by_player(last_result: Dict[str, Any]) -> Dict[int, List[Any]]:
    """
    ✅ Supports both:
    - list: [p0_bid_list, p1_bid_list] (new format from the server)
    - dict: {0:[...],1:[...]} or {"0":[...],"1":[...]} (old format)
    """
    for key in ("bids_by_player", "bids_map", "bids_per_player"):
        v = last_result.get(key)

        # NEW: list
        if isinstance(v, list):
            out: Dict[int, List[Any]] = {}
            if len(v) > 0:
                out[0] = _safe_list(v[0])
            if len(v) > 1:
                out[1] = _safe_list(v[1])
            return out

        # old: dict
        if isinstance(v, dict):
            out2: Dict[int, List[Any]] = {}
            for k, vv in v.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                out2[idx] = _safe_list(vv)
            return out2

    return {}


def get_winner_loser_bids(last_result: Dict[str, Any], winner_idx: Optional[int]) -> Tuple[List[Any], List[Any]]:
    """
    ✅ Prefer server fields winner_bid/loser_bid if present.
    Fallback to bids_by_player.
    """
    wb = _safe_list(last_result.get("winner_bid"))
    lb = _safe_list(last_result.get("loser_bid"))
    if wb or lb:
        return wb, lb

    bids_map = extract_bids_by_player(last_result)
    if winner_idx is None:
        return [], []
    loser_idx = 1 - winner_idx
    return _safe_list(bids_map.get(winner_idx)), _safe_list(bids_map.get(loser_idx))


# =========================
# Input / parsing
# =========================
async def async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def parse_indices(text: str) -> Optional[List[int]]:
    """
    ✅ You must type "no bid" to avoid bidding.
    """
    t = text.strip().lower()
    if t == "":
        return None

    if t in ("no bid", "nobid"):
        return []

    if t.startswith("bid "):
        t = t[4:].strip()

    parts = t.split()
    for p in parts:
        if not p.isdigit():
            return None
    return [int(x) for x in parts]


# =========================
# UI renderer
# =========================
def draw_ui(st: Optional[Dict[str, Any]]):
    global _last_render_key

    if not st or my_index is None:
        clear_screen()
        print("FAFNIR ONLINE (CLI)")
        print("Connecting.../waiting state...")
        print("print: quit To exit")
        return

    key = render_key(st)
    if _last_render_key == key:
        return
    _last_render_key = key

    clear_screen()
    phase = phase_of(st)
    ps = st.get("players") or []
    limit = int(st.get("trash_limit") or 6)

    cb = current_bidder(st)
    cb_name = "-"
    if isinstance(cb, int) and 0 <= cb < len(ps):
        cb_name = str(ps[cb].get("name") or f"P{cb}")

    print("FAFNIR ONLINE (CLI)")
    print(
        f"Room: {st.get('room_id')}  Round: {st.get('round')}  Turn: {st.get('turn')}  "
        f"Phase: {phase}  Caretaker: {st.get('caretaker')}  CurrentBidder: {cb_name}  Bag: {st.get('bag_left')}"
    )
    print(render_trash_board(st.get("trash") or {}, limit))
    print("-" * 90)

    offer = st.get("offer") or []
    last_offer = st.get("last_offer") or []
    last_result = st.get("last_result") or {}
    status = st.get("status") or ""

    if _last_reject_msg:
        print("❌ BID REJECTED:", _last_reject_msg)
        print("-" * 90)

    def pname(i: int) -> str:
        if isinstance(i, int) and 0 <= i < len(ps):
            return str(ps[i].get("name") or f"P{i}")
        return f"P{i}"

    if phase == "BIDDING":
        print(f"OFFER (Competing for this round): {offer}")
        print("rules: cannot bid 'same color on OFFER'")
        if not is_my_turn_to_bid(st):
            print(f"⏳ Waiting turn... Now bidding: {cb_name}")
            print("Enter = refresh  |  quit = exit")
        else:
            print("✅ It's YOUR turn to bid now.")
            print("How to bid: 0 2 5")
            print('No bid : type ->  no bid')

    elif phase == "RESULT":
        w = last_result.get("winner")
        bids_count = last_result.get("bids_count")

        try:
            wi = int(w) if w is not None else None
        except Exception:
            wi = None

        print(f"RESULT: This round is competitive: {last_offer}")
        print(f"summarize: winner={w}  bids_count={bids_count}")

        if wi in (0, 1):
            loser = 1 - wi

            # ✅ HERE: get bids correctly (winner_bid/loser_bid OR bids_by_player list/dict)
            wbid, lbid = get_winner_loser_bids(last_result, wi)

            print(f"🏅 Winner: {pname(wi)}")
            print(f"   🪙 bid (winner): {summarize_stones(wbid)}")
            print(f"🥈 Loser : {pname(loser)}")
            print(f"   🪙 bid (loser) : {summarize_stones(lbid)}")
        else:
            print("🏅 Winner: -")
            print("🥈 Loser : -")

        print()
        print("OK/Next: press Enter (both players must press once)")

    elif phase == "ROUND_END":
        info = st.get("round_end_info") or {}
        trigger = info.get("trigger_color")
        ranked = info.get("ranked")
        adds = info.get("adds") or []
        seeded = info.get("seeded_trash")
        rw = info.get("round_winner")

        print("=== ROUND END SUMMARY ===")
        print(f"Trigger color: {trigger}")
        print(f"Color ranking (total in hands): {ranked}")
        print(f"Score change this round (adds): {adds}")
        print("-" * 90)

        if isinstance(rw, int):
            add_txt = f"+{adds[rw]}" if (rw < len(adds)) else "+"
            print(f"🏅 Round Winner: {pname(rw)} ({add_txt})")
        else:
            print("🏅 Round Winner: -")

        print(f"Seeded trash (next round seed): {seeded}")
        print()
        print("OK/Next: press Enter (both players must press once)")

    elif phase == "GAME_END":
        info = st.get("game_end_info") or {}
        reason = info.get("reason")
        winner = info.get("winner")

        print("=== GAME END ===")
        print("reason:", reason)
        if isinstance(winner, int):
            print("🏆 WINNER:", pname(winner), f"(index={winner})")
        else:
            print("🏆 WINNER: -")
        print("print: restart = new game | quit = exit")

    print("STATUS:", status)
    print("-" * 90)

    print("PLAYERS:")
    for i, pl in enumerate(ps):
        tag = "YOU" if i == my_index else f"P{i}"
        submitted = "✔" if pl.get("bid_submitted") else " "
        ok = "✔" if pl.get("ok_ready") else " "
        turn_mark = "👉" if (cb == i and phase == "BIDDING") else "  "
        print(
            f"{turn_mark} {tag:>3} | {pl.get('name'):<10} score={pl.get('score'):<3} "
            f"hand={pl.get('hand_count'):<3} ok={ok}  bid={submitted}"
        )
    print("-" * 90)

    if phase != "GAME_END":
        hand = my_hand(st)
        print("YOUR HAND:")
        print(format_hand(hand))
        print("-" * 90)


# =========================
# Socket events
# =========================
async def _join_room_with_retry():
    global _join_sent
    if _join_sent:
        return
    payload = {"room_id": cfg["room"], "player_name": cfg["name"]}

    for _ in range(3):
        if not sio.connected:
            await asyncio.sleep(0.2)
            continue
        try:
            await sio.emit("join_room", payload)
            _join_sent = True
            return
        except Exception:
            await asyncio.sleep(0.25)

    print("[ERR] join_room failed after retries.")


@sio.event
async def connect():
    global _join_sent
    _join_sent = False
    await _join_room_with_retry()


@sio.event
async def disconnect():
    global _join_sent
    _join_sent = False
    print("NOTICE: Disconnected from server.")


@sio.event
async def connect_error(data):
    print("[NET] connect_error:", data)


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index, _last_render_key
    my_index = int(data["index"])
    _last_render_key = None
    print("[HUMAN] assigned index =", my_index)


@sio.on("bid_rejected")
async def bid_rejected(data):
    global _last_reject_msg, _last_render_key
    _last_reject_msg = str(data.get("message") or data.get("reason") or "rejected")
    _last_render_key = None
    if last_state:
        draw_ui(last_state)


@sio.on("state_update")
async def state_update(state):
    global last_state, _last_reject_msg, _last_ok_phase_key
    last_state = state

    if phase_of(state) != "BIDDING":
        _last_reject_msg = None

    if phase_of(state) == "BIDDING":
        _last_ok_phase_key = None

    draw_ui(last_state)


# =========================
# Main loop
# =========================
async def main():
    global _last_render_key, _last_reject_msg, _last_ok_phase_key

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="Human")
    args = ap.parse_args()

    cfg["room"] = args.room
    cfg["name"] = args.name

    try:
        await sio.connect(args.url, wait_timeout=15)
    except Exception as e:
        print("[NET] initial connect failed:", repr(e))

    await _join_room_with_retry()

    try:
        while True:
            if my_index is None or last_state is None:
                cmd = (await async_input("> ")).strip().lower()
                if cmd == "quit":
                    break
                continue

            p = phase_of(last_state)

            if p == "GAME_END":
                cmd = (await async_input("GAME_END > ")).strip().lower()
                if cmd == "quit":
                    break
                if cmd == "restart":
                    await safe_emit("restart_game", {"room_id": cfg["room"]})
                    _last_render_key = None
                continue

            if p in ("RESULT", "ROUND_END"):
                phase_key = (last_state.get("round"), last_state.get("turn"), p)
                cmd = await async_input(f"{p} > ")
                low = cmd.strip().lower()

                if low == "quit":
                    break

                if low in ("", "ok", "next"):
                    if _last_ok_phase_key == phase_key:
                        continue
                    _last_ok_phase_key = phase_key

                    await safe_emit("proceed_phase", {"room_id": cfg["room"]})
                    _last_render_key = None
                continue

            if p == "BIDDING":
                if not is_my_turn_to_bid(last_state):
                    cmd = (await async_input("(waiting turn) Enter=refresh | quit > ")).strip().lower()
                    if cmd == "quit":
                        break
                    _last_render_key = None
                    draw_ui(last_state)
                    continue

                cmd = await async_input("BID > ")
                low = cmd.strip().lower()
                if low == "quit":
                    break
                if low == "show":
                    _last_render_key = None
                    draw_ui(last_state)
                    continue

                idxs = parse_indices(cmd)
                if idxs is None:
                    continue

                seen = set()
                uniq = []
                for i in idxs:
                    if i not in seen:
                        seen.add(i)
                        uniq.append(i)

                hand = my_hand(last_state)
                stones = [hand[i] for i in uniq if 0 <= i < len(hand)]

                _last_reject_msg = None
                _last_render_key = None

                await safe_emit("submit_bid", {"room_id": cfg["room"], "stones": stones})
                continue

            cmd = (await async_input("> ")).strip().lower()
            if cmd == "quit":
                break

    finally:
        try:
            if sio.connected:
                await sio.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
