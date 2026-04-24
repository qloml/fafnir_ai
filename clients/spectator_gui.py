# clients/spectator_gui.py
import asyncio
import argparse
from typing import Any, Dict, Optional, List, Tuple

import socketio
import tkinter as tk
from tkinter import ttk

import os

sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "Spectator", "url": "http://127.0.0.1:8765"}

last_state: Optional[Dict[str, Any]] = None
action_queue: List[Dict[str, Any]] = []
_join_sent: bool = False

# ======== Coin rendering config ========
COIN_COLORS = {
    "gold": "#f6d365",
    "red": "#ff5c5c",
    "orange": "#ff9f43",
    "yellow": "#ffd93d",
    "green": "#2ecc71",
    "blue": "#4dabf7",
}
COIN_TEXT = {
    "gold": "G",
    "red": "R",
    "orange": "O",
    "yellow": "Y",
    "green": "G",
    "blue": "B",
}
COIN_ORDER = ["gold", "red", "orange", "yellow", "green", "blue"]


def safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    return []


def short_payload(payload: Any, limit: int = 170) -> str:
    s = str(payload)
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def count_map(stones: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for s in stones:
        s = str(s).lower()
        d[s] = d.get(s, 0) + 1
    return d


def fmt_bid_summary_plain(stones: List[str]) -> str:
    """No emoji (avoid gray emoji issue in Tk). Example: GOLDx1 REDx2"""
    stones = [str(s).lower() for s in stones]
    cm = count_map(stones)
    parts: List[str] = []
    for c in COIN_ORDER:
        n = cm.get(c, 0)
        if n:
            parts.append(f"{c.upper()}x{n}")
    return " ".join(parts) if parts else "-"


def extract_bids(bids_by_player: Any) -> Tuple[List[str], List[str]]:
    """
    Supported:
      - list: [bidP0, bidP1]
      - dict: {0:[...],1:[...]}
      - dict: {"0":[...],"1":[...]}
    """
    if isinstance(bids_by_player, list):
        b0 = safe_list(bids_by_player[0]) if len(bids_by_player) > 0 else []
        b1 = safe_list(bids_by_player[1]) if len(bids_by_player) > 1 else []
        return b0, b1

    if isinstance(bids_by_player, dict):
        b0 = bids_by_player.get(0, None)
        if b0 is None:
            b0 = bids_by_player.get("0", [])
        b1 = bids_by_player.get(1, None)
        if b1 is None:
            b1 = bids_by_player.get("1", [])
        return safe_list(b0), safe_list(b1)

    return [], []


def norm_color_list(xs: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(xs, list):
        return out
    for s in xs:
        if not isinstance(s, str):
            continue
        out.append(s.lower().strip())
    return out


def history_lines_from_actions(actions: List[Dict[str, Any]], limit: int = 200) -> List[str]:
    """
    Create a history of "each round" from the action kind=resolve_after
    Shortened format but using full words in colors: GOLD/RED/...
    R1 T2 | WIN Bee (P1) vs LOSE Apple (P0) | WBid: ORANGEx1 | LBid: GOLDx1
    """
    lines: List[str] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        if a.get("kind") != "resolve_after":
            continue
        payload = a.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        rnd = payload.get("round", "?")
        trn = payload.get("turn", "?")
        w = payload.get("winner", None)

        lr = payload.get("last_result") or {}
        if not isinstance(lr, dict):
            lr = {}

        # no-bid case
        if w is None:
            lines.append(f"R{rnd} T{trn} | NO BID (offer->trash, both -1)")
            continue

        try:
            wi = int(w)
        except Exception:
            wi = 0
        li = 1 - wi

        winner_name = str(payload.get("winner_name") or lr.get("winner_name") or f"P{wi}")
        loser_name = str(lr.get("loser_name") or f"P{li}")

        wbid = norm_color_list(lr.get("winner_bid", []))
        lbid = norm_color_list(lr.get("loser_bid", []))

        # fallback if server doesn't provide winner_bid/loser_bid
        if (not wbid) or (not lbid):
            p0b, p1b = extract_bids(lr.get("bids_by_player"))
            if not wbid:
                wbid = p0b if wi == 0 else p1b
            if not lbid:
                lbid = p0b if li == 0 else p1b

        lines.append(
            f"R{rnd} T{trn} | WIN {winner_name} (P{wi}) vs LOSE {loser_name} (P{li}) | "
            f"WBid: {fmt_bid_summary_plain(wbid)} | LBid: {fmt_bid_summary_plain(lbid)}"
        )

    if len(lines) > limit:
        lines = lines[-limit:]
    return lines


class CoinStrip(ttk.Frame):
    def __init__(self, master, title: str, coin_size: int = 18, height_rows: int = 1):
        super().__init__(master)
        self.coin_size = coin_size
        self.height_rows = height_rows

        self.label = ttk.Label(self, text=title, font=("Segoe UI", 9, "bold"))
        self.label.pack(anchor="w")

        self.canvas = tk.Canvas(self, height=self._canvas_height(), highlightthickness=1)
        self.canvas.pack(fill="x", expand=True)

        self._stones: List[str] = []

    def _canvas_height(self) -> int:
        pad = 6
        return self.height_rows * (self.coin_size + pad) + pad

    def set_title(self, title: str):
        self.label.config(text=title)

    def draw(self, stones: List[str]):
        self._stones = stones[:]
        self.canvas.delete("all")

        pad = 6
        size = self.coin_size
        w = self.canvas.winfo_width()
        if w <= 1:
            w = 900

        per_row = max(10, (w - pad) // (size + pad))
        per_row = min(per_row, 44)

        for idx, raw in enumerate(stones):
            s = str(raw).lower()
            row = idx // per_row
            col = idx % per_row
            x0 = pad + col * (size + pad)
            y0 = pad + row * (size + pad)
            x1 = x0 + size
            y1 = y0 + size

            fill = COIN_COLORS.get(s, "#cccccc")
            self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline="#333333", width=1)
            t = COIN_TEXT.get(s, "?")
            self.canvas.create_text(
                (x0 + x1) // 2,
                (y0 + y1) // 2,
                text=t,
                fill="#111111",
                font=("Segoe UI", 9, "bold"),
            )

        total_rows = (len(stones) + per_row - 1) // per_row
        height = max(self.height_rows, total_rows) * (size + pad) + pad
        self.canvas.config(scrollregion=(0, 0, w, height))

    def redraw(self):
        self.draw(self._stones)


class SpectatorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FAFNIR Spectator (Coins View)")
        self.root.geometry("1200x780")

        self.var_top = tk.StringVar(value="Connecting...")
        self.var_status = tk.StringVar(value="-")

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.var_top, font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.var_status, font=("Segoe UI", 10)).pack(anchor="w", pady=(6, 0))

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        # ===== Players =====
        self.players_box = ttk.LabelFrame(left, text="Players", padding=10)
        self.players_box.pack(fill="x", pady=(0, 10))

        self.p0_line = tk.StringVar(value="P0: -")
        self.p1_line = tk.StringVar(value="P1: -")
        ttk.Label(self.players_box, textvariable=self.p0_line, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.p0_hand_strip = CoinStrip(self.players_box, "P0 hand", coin_size=18, height_rows=2)
        self.p0_hand_strip.pack(fill="x", pady=(4, 6))
        self.p0_bid_strip = CoinStrip(self.players_box, "P0 bid (current)", coin_size=18, height_rows=1)
        self.p0_bid_strip.pack(fill="x", pady=(0, 8))

        ttk.Label(self.players_box, textvariable=self.p1_line, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.p1_hand_strip = CoinStrip(self.players_box, "P1 hand", coin_size=18, height_rows=2)
        self.p1_hand_strip.pack(fill="x", pady=(4, 6))
        self.p1_bid_strip = CoinStrip(self.players_box, "P1 bid (current)", coin_size=18, height_rows=1)
        self.p1_bid_strip.pack(fill="x", pady=(0, 8))

        self.turn = tk.StringVar(value="Current bidder: -")
        ttk.Label(self.players_box, textvariable=self.turn, font=("Segoe UI", 11)).pack(anchor="w", pady=(6, 0))

        # ===== State (Tabbed) =====
        self.state_box = ttk.LabelFrame(left, text="State", padding=10)
        self.state_box.pack(fill="both", expand=True)

        self.nb = ttk.Notebook(self.state_box)
        self.nb.pack(fill="both", expand=True)

        self.tab_summary = ttk.Frame(self.nb, padding=8)
        self.tab_lastbid = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.tab_summary, text="Summary")
        self.nb.add(self.tab_lastbid, text="Last bid")

        # --- Summary tab ---
        self.offer_strip = CoinStrip(self.tab_summary, "Offer", coin_size=22, height_rows=1)
        self.offer_strip.pack(fill="x", pady=(0, 8))

        self.last_offer_strip = CoinStrip(self.tab_summary, "Last offer", coin_size=22, height_rows=1)
        self.last_offer_strip.pack(fill="x", pady=(0, 8))

        self.last_result_text = tk.StringVar(value="Last result: -")
        ttk.Label(self.tab_summary, textvariable=self.last_result_text, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # ✅ Remove the words "Result Summary" and move them up (use a regular frame so it doesn't cover the heading).
        self.summary_box = ttk.Frame(self.tab_summary, padding=6)
        self.summary_box.pack(fill="x", pady=(2, 0))

        self.sum_winner = tk.StringVar(value="🏅 Winner : -")
        self.sum_loser = tk.StringVar(value="🥈 Loser  : -")
        self.winner_bid_plain = tk.StringVar(value="  ↳ bid (winner) : -")
        self.loser_bid_plain = tk.StringVar(value="  ↳ bid (loser)  : -")

        ttk.Label(self.summary_box, textvariable=self.sum_winner, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(self.summary_box, textvariable=self.winner_bid_plain, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 4))
        self.winner_bid_strip = CoinStrip(self.summary_box, "bid (winner) coins", coin_size=18, height_rows=1)
        self.winner_bid_strip.pack(fill="x", pady=(0, 10))

        ttk.Label(self.summary_box, textvariable=self.sum_loser, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(self.summary_box, textvariable=self.loser_bid_plain, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 4))
        self.loser_bid_strip = CoinStrip(self.summary_box, "bid (loser) coins", coin_size=18, height_rows=1)
        self.loser_bid_strip.pack(fill="x")

        # --- Last bid tab ---
        self.last_p0_bid_strip = CoinStrip(self.tab_lastbid, "Last bid (P0)", coin_size=20, height_rows=1)
        self.last_p0_bid_strip.pack(fill="x", pady=(0, 10))

        self.last_p1_bid_strip = CoinStrip(self.tab_lastbid, "Last bid (P1)", coin_size=20, height_rows=1)
        self.last_p1_bid_strip.pack(fill="x", pady=(0, 10))

        self.used_strip = CoinStrip(self.tab_lastbid, "Used by winner", coin_size=20, height_rows=1)
        self.used_strip.pack(fill="x", pady=(0, 0))

        # ===== Right: Trash + Connections + History =====
        self.trash_box = ttk.LabelFrame(right, text="Trash (limit 6)", padding=10)
        self.trash_box.pack(fill="x", pady=(0, 10))

        self.trash_labels: Dict[str, tk.StringVar] = {}
        for c in COIN_ORDER:
            row = ttk.Frame(self.trash_box)
            row.pack(fill="x", pady=2)

            icon = tk.Canvas(row, width=18, height=18, highlightthickness=0)
            icon.pack(side="left", padx=(0, 6))
            icon.create_oval(1, 1, 17, 17, fill=COIN_COLORS.get(c, "#ccc"), outline="#333", width=1)

            v = tk.StringVar(value=f"{c.upper():<6}: 0/6")
            self.trash_labels[c] = v
            ttk.Label(row, textvariable=v, font=("Consolas", 11)).pack(side="left", anchor="w")

        self.conn_box = ttk.LabelFrame(right, text="Connections", padding=10)
        self.conn_box.pack(fill="x", pady=(0, 10))
        self.conn_list = tk.Listbox(self.conn_box, height=6)
        self.conn_list.pack(fill="x")

        self.action_box = ttk.LabelFrame(right, text="History (Rounds)", padding=10)
        self.action_box.pack(fill="both", expand=True)
        self.action_list = tk.Listbox(self.action_box)
        self.action_list.pack(fill="both", expand=True)

        self.root.bind("<Configure>", self._on_resize)

    def _on_resize(self, _evt):
        self.p0_hand_strip.redraw()
        self.p1_hand_strip.redraw()
        self.p0_bid_strip.redraw()
        self.p1_bid_strip.redraw()
        self.offer_strip.redraw()
        self.last_offer_strip.redraw()
        self.winner_bid_strip.redraw()
        self.loser_bid_strip.redraw()
        self.last_p0_bid_strip.redraw()
        self.last_p1_bid_strip.redraw()
        self.used_strip.redraw()

    def update(self, st: Dict[str, Any]):
        room_id = st.get("room_id", "?")
        phase = st.get("phase", "?")
        rnd = st.get("round", 0)
        turn = st.get("turn", 0)
        caretaker = st.get("caretaker", 0)
        bag_left = st.get("bag_left", 0)
        cb = st.get("current_bidder", 0)

        self.var_top.set(
            f"Room {room_id} | Phase: {phase} | Round {rnd} Turn {turn} | Caretaker: P{caretaker} | Bag left: {bag_left}"
        )
        self.var_status.set("STATUS: " + str(st.get("status", "-")))
        self.turn.set(f"Current bidder: P{cb}")

        ps = st.get("players") or []
        p0_name = ps[0].get("name", "P0") if len(ps) >= 1 else "P0"
        p1_name = ps[1].get("name", "P1") if len(ps) >= 2 else "P1"

        if len(ps) >= 1:
            p0 = ps[0]
            self.p0_line.set(f"P0: {p0_name} | score={p0.get('score',0)} | hand={p0.get('hand_count',0)}")
            self.p0_hand_strip.draw(safe_list(p0.get("hand")))
            self.p0_bid_strip.draw(safe_list(p0.get("last_bid")))
        else:
            self.p0_line.set("P0: -")
            self.p0_hand_strip.draw([])
            self.p0_bid_strip.draw([])

        if len(ps) >= 2:
            p1 = ps[1]
            self.p1_line.set(f"P1: {p1_name} | score={p1.get('score',0)} | hand={p1.get('hand_count',0)}")
            self.p1_hand_strip.draw(safe_list(p1.get("hand")))
            self.p1_bid_strip.draw(safe_list(p1.get("last_bid")))
        else:
            self.p1_line.set("P1: -")
            self.p1_hand_strip.draw([])
            self.p1_bid_strip.draw([])

        self.offer_strip.draw(safe_list(st.get("offer")))
        self.last_offer_strip.draw(safe_list(st.get("last_offer")))

        lr = st.get("last_result") or {}
        winner = lr.get("winner", None)
        loser = lr.get("loser", None)
        bids_count = lr.get("bids_count", None)

        # raw bids_by_player (fallback/debug)
        p0_last_bid, p1_last_bid = extract_bids(lr.get("bids_by_player"))

        # explicit winner/loser bids (new server)
        server_winner_bid = norm_color_list(lr.get("winner_bid", []))
        server_loser_bid = norm_color_list(lr.get("loser_bid", []))

        wi: Optional[int] = None
        li: Optional[int] = None
        if winner is not None:
            try:
                wi = int(winner)
            except Exception:
                wi = 0
            li = 1 - wi

        # fallback if server didn't provide
        if (not server_winner_bid) and (wi is not None):
            server_winner_bid = p0_last_bid if wi == 0 else p1_last_bid
        if (not server_loser_bid) and (li is not None):
            server_loser_bid = p0_last_bid if li == 0 else p1_last_bid

        if winner is None:
            winner_name = "-"
            loser_name = "-"
            loser_idx = None
            winner_bid: List[str] = []
            loser_bid: List[str] = []
        else:
            if wi is None:
                wi = 0
                li = 1

            if loser is None:
                loser_idx = li
            else:
                try:
                    loser_idx = int(loser)
                except Exception:
                    loser_idx = li

            winner_name = str(lr.get("winner_name") or (p0_name if wi == 0 else p1_name))
            loser_name = str(lr.get("loser_name") or (p0_name if loser_idx == 0 else p1_name))
            winner_bid = server_winner_bid[:]
            loser_bid = server_loser_bid[:]

        self.last_result_text.set(
            f"Last result: WINNER={winner_name} (P{winner}) | LOSER={loser_name} (P{loser_idx}) | bids_count={bids_count}"
        )

        self.sum_winner.set(f"🏅 Winner : {winner_name}")
        self.sum_loser.set(f"🥈 Loser  : {loser_name}")

        self.winner_bid_plain.set(f"  ↳ bid (winner) : {fmt_bid_summary_plain(winner_bid)}")
        self.loser_bid_plain.set(f"  ↳ bid (loser)  : {fmt_bid_summary_plain(loser_bid)}")

        self.winner_bid_strip.draw(winner_bid)
        self.loser_bid_strip.draw(loser_bid)

        # Last bid tab strips
        self.last_p0_bid_strip.set_title(f"Last bid (P0: {p0_name})")
        self.last_p1_bid_strip.set_title(f"Last bid (P1: {p1_name})")
        self.last_p0_bid_strip.draw(p0_last_bid)
        self.last_p1_bid_strip.draw(p1_last_bid)

        used = safe_list(lr.get("used"))
        self.used_strip.draw(used)

        # Trash
        limit = int(st.get("trash_limit", 6) or 6)
        trash = st.get("trash") or {}
        for c in COIN_ORDER:
            n = int(trash.get(c, 0) or 0)
            self.trash_labels[c].set(f"{c.upper():<6}: {n}/{limit}")

        # Connections
        self.conn_list.delete(0, tk.END)
        for it in (st.get("connections") or []):
            role = it.get("role", "?")
            nm = it.get("name", "Unknown")
            sid = it.get("sid", "------")
            sec = it.get("connected_for_sec", 0)
            self.conn_list.insert(tk.END, f"{role:<9} {nm:<12} sid={sid} {sec}s")

        # History (Rounds)
        self.action_list.delete(0, tk.END)
        lines = history_lines_from_actions(action_queue, limit=200)
        for line in lines:
            self.action_list.insert(tk.END, line)
        if lines:
            self.action_list.yview_moveto(1.0)


async def _join_room_with_retry():
    global _join_sent
    if _join_sent:
        return
    payload = {"room_id": cfg["room"], "player_name": cfg["name"], "spectator": True}
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


@sio.event
async def connect():
    global _join_sent
    _join_sent = False
    await _join_room_with_retry()


@sio.event
async def disconnect():
    global _join_sent
    _join_sent = False


@sio.on("state_update")
async def state_update(state):
    global last_state
    last_state = state
    if state and not action_queue:
        for a in (state.get("action_log") or [])[-200:]:
            action_queue.append(a)


@sio.on("action")
async def on_action(entry):
    action_queue.append(entry)
    if len(action_queue) > 600:
        del action_queue[:-600]


async def net_loop():
    await sio.connect(cfg["url"], wait_timeout=15)
    await _join_room_with_retry()
    await sio.wait()


async def ui_loop(ui: SpectatorUI):
    while True:
        if last_state:
            ui.update(last_state)
        ui.root.update()
        await asyncio.sleep(0.05)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="Spectator")
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name

    ui = SpectatorUI()
    task_net = asyncio.create_task(net_loop())
    task_ui = asyncio.create_task(ui_loop(ui))

    try:
        await asyncio.gather(task_net, task_ui)
    finally:
        if sio.connected:
            await sio.disconnect()


if __name__ == "__main__":
    
    os.environ['TCL_LIBRARY'] = r'C:\Users\icebo\AppData\Local\Programs\Python\Python311\tcl\tcl8.6'
    asyncio.run(main())
