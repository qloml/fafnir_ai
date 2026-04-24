# server.py
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import socketio
from fastapi import FastAPI

# =========================
# Constants
# =========================
COLORS = ["red", "orange", "yellow", "green", "blue"]  # priority order for tie-break
GOLD = "gold"
STONE_COUNT = {GOLD: 20, "red": 12, "orange": 12, "yellow": 12, "green": 12, "blue": 12}

TRASH_LIMIT = 6
SEED_TRASH_AT_ROUND_START = 3

POINT_CHIP = 1
SCORE_TO_WIN = 1000


# =========================
# Data models
# =========================
@dataclass
class Player:
    name: str
    stones: List[str] = field(default_factory=list)
    revealed_bid: List[str] = field(default_factory=list)
    bid_submitted: bool = False
    score: int = 0
    ok_ready: bool = False


@dataclass
class Game:
    players: List[Player]
    caretaker: int = 0
    bag: List[str] = field(default_factory=list)

    trash: Dict[str, int] = field(default_factory=dict)
    trash_pile: List[str] = field(default_factory=list)

    offer: List[str] = field(default_factory=list)

    round: int = 1
    turn: int = 1
    phase: str = "BIDDING"  # BIDDING / RESULT / ROUND_END / GAME_END
    status: str = "Waiting..."
    created_ts: float = field(default_factory=lambda: time.time())

    last_offer: List[str] = field(default_factory=list)
    last_result: Dict[str, Any] = field(default_factory=dict)

    round_end_pending: bool = False
    round_end_info: Dict[str, Any] = field(default_factory=dict)

    game_end_info: Dict[str, Any] = field(default_factory=dict)

    # ✅ sequential bidding: who is allowed to submit now
    current_bidder: int = 0

    def bag_left(self) -> int:
        return len(self.bag)


# =========================
# Helpers
# =========================
def ts() -> str:
    return time.strftime("%H:%M:%S")


def clamp_score(x: int) -> int:
    return max(0, int(x))


def make_bag() -> List[str]:
    bag: List[str] = []
    for c, n in STONE_COUNT.items():
        bag.extend([c] * n)
    random.shuffle(bag)
    return bag


def init_trash() -> Dict[str, int]:
    return {c: 0 for c in STONE_COUNT.keys()}


def draw_one(game: Game) -> Optional[str]:
    if not game.bag:
        return None
    return game.bag.pop()


def line_up_trash(game: Game, stones: List[str]):
    for s in stones:
        game.trash[s] = game.trash.get(s, 0) + 1
    game.trash_pile.extend(stones)


def seed_trash_at_round_start(game: Game, n: int = SEED_TRASH_AT_ROUND_START) -> List[str]:
    seeded: List[str] = []
    for _ in range(n):
        s = draw_one(game)
        if s is None:
            break
        seeded.append(s)
        game.trash[s] = game.trash.get(s, 0) + 1
        game.trash_pile.append(s)
    return seeded


def deal_initial_hands(game: Game):
    for p in game.players:
        p.stones.clear()
        p.revealed_bid.clear()
        p.bid_submitted = False
        p.ok_ready = False

    for i, p in enumerate(game.players):
        n = 11 if i == game.caretaker else 10
        for _ in range(n):
            s = draw_one(game)
            if s is not None:
                p.stones.append(s)


def setup_offer(game: Game) -> bool:
    """
    Offer: draw 2 stones. If both same color, return and redraw until >=2 colors (or bag low).
    Returns True if offer drawn (>=1), False if bag empty.
    """
    game.offer = []
    if len(game.bag) == 0:
        return False

    while True:
        if len(game.bag) == 0:
            break

        draw_n = min(2, len(game.bag))
        stones: List[str] = []
        for _ in range(draw_n):
            s = draw_one(game)
            if s is not None:
                stones.append(s)

        # accept if <2 (bag low) OR 2 stones but not same color
        if len(stones) < 2 or len(set(stones)) > 1:
            game.offer = stones
            break

        game.bag.extend(stones)
        random.shuffle(game.bag)

    return bool(game.offer)


def is_trash_limit_reached(game: Game, limit: int = TRASH_LIMIT) -> bool:
    for c in COLORS + [GOLD]:
        if int(game.trash.get(c, 0)) >= limit:
            return True
    return False


def trash_trigger_color(game: Game, limit: int = TRASH_LIMIT) -> Optional[str]:
    for c in COLORS + [GOLD]:
        if int(game.trash.get(c, 0)) >= limit:
            return c
    return None


def trash_risk_percent(game: Game, limit: int = TRASH_LIMIT) -> int:
    mx = 0
    for c in COLORS + [GOLD]:
        mx = max(mx, int(game.trash.get(c, 0)))
    return int(round((mx / max(1, limit)) * 100))


# =========================
# ✅ SERVER LOG HELPERS
# =========================
_COLOR_ORDER = ["gold", "red", "orange", "yellow", "green", "blue"]
_COLOR_ICON = {"gold": "💛", "red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢", "blue": "🔵"}


def _summ_counts(stones: List[str]) -> str:
    counts: Dict[str, int] = {}
    for s in stones:
        c = str(s).lower()
        counts[c] = counts.get(c, 0) + 1
    parts = []
    for c in _COLOR_ORDER:
        if counts.get(c, 0) > 0:
            parts.append(f"{_COLOR_ICON.get(c,'•')}{c.upper()}x{counts[c]}")
    return " ".join(parts) if parts else "-"


def log_hands(game: Game, title: str = ""):
    if title:
        print(title)
    for i, p in enumerate(game.players):
        print(f"  HAND P{i} {p.name}: {_summ_counts(p.stones)}")
        print(f"    list: {p.stones}")


def _get_bids_from_last_result(lr: Dict[str, Any], idx: int) -> List[str]:
    """
    Supports both list bids_by_player [p0, p1] or dict {0:[],1:[]} or {"0":[],"1":[]}
    """
    bbp = lr.get("bids_by_player")
    if isinstance(bbp, list):
        if 0 <= idx < len(bbp):
            return list(bbp[idx] or [])
        return []
    if isinstance(bbp, dict):
        if idx in bbp:
            return list(bbp.get(idx) or [])
        sidx = str(idx)
        if sidx in bbp:
            return list(bbp.get(sidx) or [])
        return []
    return []


def log_auction_result(game: Game):
    lr = game.last_result or {}
    w = lr.get("winner")
    l = lr.get("loser")
    offer = lr.get("offer")
    used = lr.get("used") or []
    bids_count = lr.get("bids_count")

    print("┌────────────────────────────── AUCTION RESULT ──────────────────────────────┐")
    print(f"│ Round={game.round} Turn={game.turn}  Caretaker={game.caretaker}  Offer={offer}")
    print(f"│ bids_count={bids_count}")
    for i, p in enumerate(game.players):
        b = _get_bids_from_last_result(lr, i)
        print(f"│  P{i} {p.name:<10} bid: {_summ_counts(b)}  | list={b}")
    print(f"│ winner={w} loser={l}  used_by_winner={_summ_counts(used)}  | list={used}")
    print(f"│ score: P0={game.players[0].score}  P1={game.players[1].score}")
    print(f"│ trash: {game.trash}   (limit={TRASH_LIMIT})")
    print("└────────────────────────────────────────────────────────────────────────────┘")


# =========================
# Scoring
# =========================
def rank_colors_by_total_in_hands(game: Game) -> List[Tuple[str, int]]:
    totals = {c: 0 for c in COLORS}
    for p in game.players:
        for s in p.stones:
            if s in COLORS:
                totals[s] += 1
    priority = list(COLORS)
    return sorted(totals.items(), key=lambda x: (-x[1], priority.index(x[0])))


def compute_hand_score_for_player(game: Game, player: Player) -> int:
    ranked = rank_colors_by_total_in_hands(game)
    first = ranked[0][0] if ranked else None
    second = ranked[1][0] if len(ranked) > 1 else None

    VALUE_FIRST = 3
    VALUE_SECOND = 2
    VALUE_OTHER = -1

    score = 0
    score += player.stones.count(GOLD)

    for c in COLORS:
        cnt = player.stones.count(c)
        if cnt == 0:
            continue
        if cnt >= 5:
            continue

        if c == first:
            mult = VALUE_FIRST
        elif c == second:
            mult = VALUE_SECOND
        else:
            mult = VALUE_OTHER
        score += cnt * mult

    return score


def compute_round_scores(game: Game) -> Tuple[List[Tuple[str, int]], List[int]]:
    ranked = rank_colors_by_total_in_hands(game)
    adds = [compute_hand_score_for_player(game, p) for p in game.players]
    return ranked, adds


def apply_round_scoring(game: Game) -> Tuple[List[Tuple[str, int]], List[int]]:
    ranked, adds = compute_round_scores(game)
    for p, add in zip(game.players, adds):
        p.score = clamp_score(p.score + add)
    return ranked, adds


def round_winner_from_adds(game: Game, adds: List[int]) -> int:
    best = max(adds) if adds else 0
    cands = [i for i, a in enumerate(adds) if a == best]
    if len(cands) == 1:
        return cands[0]
    return game.caretaker if game.caretaker in cands else min(cands)


# =========================
# Endgame rules
# =========================
def check_game_end(game: Game) -> Optional[Dict[str, Any]]:
    for i, p in enumerate(game.players):
        if p.score >= SCORE_TO_WIN:
            return {"reason": "SCORE_1000", "winner": i, "score_to_win": SCORE_TO_WIN}
    return None


# =========================
# Round reset
# =========================
def reset_all_stones_into_bag_and_redeal(game: Game):
    game.bag = make_bag()
    game.trash = init_trash()
    game.trash_pile.clear()
    game.offer = []

    deal_initial_hands(game)

    seeded = seed_trash_at_round_start(game, SEED_TRASH_AT_ROUND_START)
    game.round_end_info["seeded_trash"] = seeded

    setup_offer(game)
    game.current_bidder = 0


def should_force_round_end_by_bag(game: Game) -> bool:
    return len(game.bag) < 2


# =========================
# Auction
# =========================
def resolve_auction(game: Game) -> Optional[int]:
    """
    ✅ เก็บ bid ของรอบล่าสุดไว้ใน last_result แบบ JSON-safe
        เพื่อให้ "ผู้เล่น" และ "spectator" เห็น winner/loser bid ได้แน่นอน
    """
    # snapshot bids BEFORE clearing
    bids_p0 = game.players[0].revealed_bid[:]
    bids_p1 = game.players[1].revealed_bid[:]
    bids_by_player = [bids_p0, bids_p1]  # ✅ list, JSON-safe

    bids_count = [len(bids_p0), len(bids_p1)]
    max_bid = max(bids_count) if bids_count else 0

    game.last_offer = game.offer[:]
    game.last_result = {
        "bids_count": bids_count[:],
        "winner": None,
        "loser": None,
        "used": [],
        "offer": game.offer[:],
        "bids_by_player": bids_by_player,   # ✅ list [p0, p1]
        "winner_bid": [],
        "loser_bid": [],
        "winner_name": None,
        "loser_name": None,
    }

    # no-bid
    if max_bid == 0:
        for p in game.players:
            p.score = clamp_score(p.score - 1)
        line_up_trash(game, game.offer)
        game.offer = []
        for p in game.players:
            p.revealed_bid.clear()

        # (The bids_by_player and bids_count files are still available for the client to display.)
        return None

    # tie-break (caretaker loses ties)
    candidates = [i for i, b in enumerate(bids_count) if b == max_bid]
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        ct = game.caretaker
        if ct in candidates:
            non_ct = [i for i in candidates if i != ct]
            winner = min(non_ct) if non_ct else ct
        else:
            winner = min(candidates)

    loser = 1 - winner

    used = game.players[winner].revealed_bid[:]

    for s in used:
        try:
            game.players[winner].stones.remove(s)
        except ValueError:
            pass
    if used:
        line_up_trash(game, used)

    if game.offer:
        game.players[winner].stones.extend(game.offer)
    game.offer = []

    game.players[winner].score += POINT_CHIP

    # clear revealed_bid (turn ends)
    for p in game.players:
        p.revealed_bid.clear()

    # caretaker -> last auction winner
    game.caretaker = winner

    # ✅ fill last_result fully (so clients can show winner/loser bids)
    game.last_result["winner"] = winner
    game.last_result["loser"] = loser
    game.last_result["used"] = used[:]
    game.last_result["winner_bid"] = bids_by_player[winner][:]
    game.last_result["loser_bid"] = bids_by_player[loser][:]
    game.last_result["winner_name"] = game.players[winner].name
    game.last_result["loser_name"] = game.players[loser].name

    return winner


# =========================
# Rooms
# =========================
class Room:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.sids: List[str] = []         # players (max 2)
        self.spectators: List[str] = []   # spectators
        self.game: Optional[Game] = None
        self.names_by_sid: Dict[str, str] = {}
        self.join_ts_by_sid: Dict[str, float] = {}

        self.action_seq: int = 0
        self.action_log: List[Dict[str, Any]] = []

    def is_full(self) -> bool:
        return len(self.sids) >= 2

    def all_sids(self) -> List[str]:
        return list(self.sids) + list(self.spectators)


rooms: Dict[str, Room] = {}


def get_room(room_id: str) -> Room:
    if room_id not in rooms:
        rooms[room_id] = Room(room_id)
    return rooms[room_id]


# =========================
# Socket / FastAPI
# =========================
app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)


@app.get("/")
def root():
    return {"ok": True, "rooms": list(rooms.keys())}


async def broadcast_room_notice(room: Room, kind: str, message: str):
    for sid in room.all_sids():
        await sio.emit(
            "room_notice",
            {"kind": kind, "message": message, "room_id": room.room_id, "ts": time.time()},
            to=sid,
        )


async def emit_action(room: Room, kind: str, payload: Dict[str, Any]):
    """
    Emit action feed to everyone (players + spectators).
    Also keep a short in-memory log (last 200).
    """
    room.action_seq += 1
    entry = {"seq": room.action_seq, "ts": time.time(), "kind": kind, "payload": payload}
    room.action_log.append(entry)
    if len(room.action_log) > 200:
        room.action_log = room.action_log[-200:]

    for sid in room.all_sids():
        await sio.emit("action", entry, to=sid)


def start_game_if_ready(room: Room):
    if room.game is not None:
        return
    if len(room.sids) < 2:
        return

    caretaker = random.randint(0, 1)
    p0 = room.names_by_sid.get(room.sids[0], "Player0")
    p1 = room.names_by_sid.get(room.sids[1], "Player1")
    players = [Player(name=p0), Player(name=p1)]

    game = Game(players=players, caretaker=caretaker, bag=make_bag(), trash=init_trash())
    deal_initial_hands(game)
    seed_trash_at_round_start(game, SEED_TRASH_AT_ROUND_START)
    setup_offer(game)

    game.phase = "BIDDING"
    game.status = "Both connected. Submit your bid."
    game.current_bidder = 0
    room.game = game

    print(f"[{ts()}][SERVER] GAME START room={room.room_id} caretaker={game.caretaker} offer={game.offer}")
    log_hands(game, title=f"[{ts()}][SERVER] INITIAL HANDS:")


def start_new_game(room: Room):
    if len(room.sids) < 2:
        room.game = None
        return

    caretaker = random.randint(0, 1)
    p0_name = room.names_by_sid.get(room.sids[0], "Player0")
    p1_name = room.names_by_sid.get(room.sids[1], "Player1")
    players = [Player(name=p0_name), Player(name=p1_name)]

    game = Game(players=players, caretaker=caretaker, bag=make_bag(), trash=init_trash())
    deal_initial_hands(game)
    seed_trash_at_round_start(game, SEED_TRASH_AT_ROUND_START)
    setup_offer(game)

    game.round = 1
    game.turn = 1
    game.phase = "BIDDING"
    game.status = "NEW GAME! Submit your bid."
    game.last_offer = []
    game.last_result = {}
    game.round_end_pending = False
    game.round_end_info = {}
    game.game_end_info = {}
    game.current_bidder = 0

    room.game = game

    print(f"[{ts()}][SERVER] NEW GAME room={room.room_id} caretaker={game.caretaker} offer={game.offer}")
    log_hands(game, title=f"[{ts()}][SERVER] INITIAL HANDS (NEW GAME):")


def build_state(room: Room, viewer_index: int, viewer_mode: str) -> Dict[str, Any]:
    """
    viewer_mode:
      - "player": see own hand & own revealed_bid only
      - "spectator_full": see ALL hands & ALL bids
    """
    game = room.game

    connections = []
    for sid in room.sids:
        connections.append(
            {
                "sid": sid[-6:],
                "name": room.names_by_sid.get(sid, "Unknown"),
                "role": "player",
                "connected_for_sec": int(time.time() - room.join_ts_by_sid.get(sid, time.time())),
            }
        )
    for sid in room.spectators:
        connections.append(
            {
                "sid": sid[-6:],
                "name": room.names_by_sid.get(sid, "Unknown"),
                "role": "spectator",
                "connected_for_sec": int(time.time() - room.join_ts_by_sid.get(sid, time.time())),
            }
        )

    if game is None:
        return {
            "room_id": room.room_id,
            "phase": "WAITING",
            "round": 0,
            "turn": 0,
            "caretaker": 0,
            "current_bidder": 0,
            "offer": [],
            "trash": init_trash(),
            "trash_pile": [],
            "trash_limit": TRASH_LIMIT,
            "trash_risk_percent": 0,
            "bag_left": 0,
            "players": [],
            "status": "Waiting for players...",
            "last_offer": [],
            "last_result": {},
            "round_end_pending": False,
            "round_end_info": {},
            "game_end_info": {},
            "connections": connections,
            "action_log": room.action_log[-80:],
            "ts": time.time(),
        }

    players_view = []
    for i, p in enumerate(game.players):
        view = {
            "name": p.name,
            "score": p.score,
            "hand_count": len(p.stones),
            "bid_submitted": p.bid_submitted,
            "ok_ready": p.ok_ready,
        }

        if viewer_mode == "spectator_full":
            view["hand"] = p.stones[:]
            view["last_bid"] = p.revealed_bid[:]
        else:
            view["hand"] = p.stones[:] if i == viewer_index else None
            view["last_bid"] = p.revealed_bid[:] if i == viewer_index else None

        players_view.append(view)

    return {
        "room_id": room.room_id,
        "phase": game.phase,
        "round": game.round,
        "turn": game.turn,
        "caretaker": game.caretaker,
        "current_bidder": game.current_bidder,
        "offer": game.offer[:],
        "trash": dict(game.trash),
        "trash_pile": game.trash_pile[:],
        "trash_limit": TRASH_LIMIT,
        "trash_risk_percent": trash_risk_percent(game, TRASH_LIMIT),
        "bag_left": game.bag_left(),
        "players": players_view,
        "status": game.status,
        "last_offer": game.last_offer[:],
        # ✅ players WILL receive winner/loser + bids here too
        "last_result": dict(game.last_result),
        "round_end_pending": bool(game.round_end_pending),
        "round_end_info": dict(game.round_end_info),
        "game_end_info": dict(game.game_end_info),
        "connections": connections,
        "action_log": room.action_log[-80:],
        "ts": time.time(),
    }


async def emit_state(room: Room):
    for i, sid in enumerate(room.sids):
        await sio.emit("state_update", build_state(room, i, "player"), to=sid)
    for sid in room.spectators:
        await sio.emit("state_update", build_state(room, -1, "spectator_full"), to=sid)


# =========================
# Round end
# =========================
def start_round_end(game: Game, trigger_reason: str, trigger_color: Optional[str], last_bid_winner: Optional[int]):
    ranked, adds = apply_round_scoring(game)
    rw = round_winner_from_adds(game, adds)

    game.round_end_pending = True
    game.round_end_info = {
        "trigger_reason": trigger_reason,
        "trigger_color": trigger_color,
        "ranked": ranked,
        "adds": adds,
        "round_winner": rw,
        "seeded_trash": [],
        "trash_limit": TRASH_LIMIT,
        "last_bid_winner": last_bid_winner,
    }

    reset_all_stones_into_bag_and_redeal(game)

    game.round += 1
    game.phase = "ROUND_END"
    game.status = f"ROUND END! reason={trigger_reason} winner=P{rw} caretaker={game.caretaker}. Press OK/Next."

    print(f"[{ts()}][SERVER] ROUND_END reason={trigger_reason} trigger_color={trigger_color} rw={rw} caretaker={game.caretaker}")
    log_hands(game, title=f"[{ts()}][SERVER] HANDS AFTER ROUND RESET:")


# =========================
# Events
# =========================
@sio.event
async def connect(sid, environ, auth):
    print(f"[{ts()}][SERVER] connect sid={sid}")
    await sio.emit("server_info", {"msg": "connected"}, to=sid)


@sio.event
async def disconnect(sid):
    print(f"[{ts()}][SERVER] disconnect sid={sid}")
    for room_id, room in list(rooms.items()):
        if sid in room.sids or sid in room.spectators:
            name = room.names_by_sid.get(sid, "Unknown")
            dur = int(time.time() - room.join_ts_by_sid.get(sid, time.time()))
            role = "spectator" if sid in room.spectators else "player"

            if sid in room.sids:
                room.sids.remove(sid)
                room.game = None
            if sid in room.spectators:
                room.spectators.remove(sid)

            room.names_by_sid.pop(sid, None)
            room.join_ts_by_sid.pop(sid, None)

            print(f"[{ts()}][SERVER] {name} left room={room_id} role={role} after {dur}s")

            await emit_action(room, "disconnect", {"name": name, "role": role, "room_id": room_id, "dur_s": dur})
            await broadcast_room_notice(room, "leave", f"{name} Leave the room. {room_id} ({dur}s)")
            await emit_state(room)

            if not room.sids and not room.spectators:
                rooms.pop(room_id, None)
            return


@sio.event
async def join_room(sid, data: Dict[str, Any]):
    room_id = str(data.get("room_id", "room1"))
    name = str(data.get("player_name", "Player"))
    is_spectator = bool(data.get("spectator", False))

    room = get_room(room_id)

    room.names_by_sid[sid] = name
    room.join_ts_by_sid[sid] = time.time()

    if is_spectator:
        if sid not in room.spectators:
            room.spectators.append(sid)

        await sio.emit("player_assigned", {"room_id": room_id, "index": -1, "role": "spectator"}, to=sid)
        await emit_action(room, "join_spectator", {"name": name, "room_id": room_id})
        await broadcast_room_notice(room, "join", f"{name} joined as SPECTATOR in room {room_id}")
        await emit_state(room)
        return

    if room.is_full() and sid not in room.sids:
        await sio.emit("join_error", {"error": "Room full (2 players)"}, to=sid)
        print(f"[{ts()}][SERVER] join denied sid={sid} room={room_id} (full)")
        return

    if sid not in room.sids:
        room.sids.append(sid)

    start_game_if_ready(room)

    if room.game is not None:
        idx = room.sids.index(sid)
        room.game.players[idx].name = name
        room.game.status = "Both connected. Submit your bid."

    idx_now = room.sids.index(sid)
    print(f"[{ts()}][SERVER] join_room sid={sid} room={room_id} idx={idx_now} name={name} sids={len(room.sids)}")

    await sio.emit("player_assigned", {"room_id": room_id, "index": idx_now, "role": "player"}, to=sid)
    await emit_action(room, "join_player", {"name": name, "room_id": room_id, "slot": idx_now})
    await broadcast_room_notice(room, "join", f"{name} join room {room_id} (slot {idx_now})")
    await emit_state(room)


@sio.event
async def submit_bid(sid, data: Dict[str, Any]):
    room_id = str(data.get("room_id", "room1"))
    room = rooms.get(room_id)
    if not room or not room.game or sid not in room.sids:
        return

    game = room.game
    if game.phase != "BIDDING":
        return

    player_index = room.sids.index(sid)
    p = game.players[player_index]

    if player_index != game.current_bidder:
        await sio.emit("bid_rejected", {"reason": "not_your_turn", "message": "Not your turn to bid."}, to=sid)
        return

    if p.bid_submitted:
        return

    stones = data.get("stones", [])
    if not isinstance(stones, list):
        await sio.emit("bid_rejected", {"reason": "invalid_payload", "message": "stones It must be list"}, to=sid)
        return

    forbidden = set(game.offer)
    clash = [s for s in stones if isinstance(s, str) and s in forbidden]
    if clash:
        await sio.emit(
            "bid_rejected",
            {
                "reason": "offer_color_conflict",
                "message": f"Cannot bid same color on OFFER: {sorted(list(forbidden))} | you send: {clash}",
                "offer": game.offer[:],
                "clash": clash,
            },
            to=sid,
        )
        return

    hand_tmp = p.stones[:]
    invalid: List[str] = []
    filtered: List[str] = []
    for s in stones:
        if not isinstance(s, str):
            continue
        if s in hand_tmp:
            filtered.append(s)
            hand_tmp.remove(s)
        else:
            invalid.append(s)

    if invalid:
        await sio.emit(
            "bid_rejected",
            {"reason": "not_in_hand", "message": f"There is a stone that is not in my hand: {invalid}"},
            to=sid,
        )
        return

    p.revealed_bid = filtered
    p.bid_submitted = True
    p.ok_ready = False

    print(f"[{ts()}][SERVER] submit_bid from P{player_index} {p.name} -> bid={filtered}")
    await emit_action(
        room,
        "submit_bid",
        {
            "room_id": room_id,
            "player_index": player_index,
            "player_name": p.name,
            "bid": filtered[:],
            "round": game.round,
            "turn": game.turn,
            "phase": game.phase,
            "current_bidder_before": player_index,
        },
    )

    other = 1 - player_index
    if len(game.players) == 2 and not game.players[other].bid_submitted:
        game.current_bidder = other

    if len(game.players) == 2 and all(pp.bid_submitted for pp in game.players):
        await emit_action(
            room,
            "resolve_before",
            {
                "room_id": room_id,
                "round": game.round,
                "turn": game.turn,
                "offer": game.offer[:],
                "hands": [pp.stones[:] for pp in game.players],
                "bids": [pp.revealed_bid[:] for pp in game.players],
                "trash": dict(game.trash),
                "caretaker": game.caretaker,
            },
        )

        w = resolve_auction(game)

        log_auction_result(game)
        log_hands(game, title=f"[{ts()}][SERVER] HANDS AFTER AUCTION:")

        # ✅ IMPORTANT: expose winner/loser + bids BOTH inside last_result and top-level
        await emit_action(
            room,
            "resolve_after",
            {
                "room_id": room_id,
                "round": game.round,
                "turn": game.turn,
                "winner": w,
                "winner_name": (game.players[w].name if w is not None else None),

                # new way (spectator_gui uses this)
                "last_result": dict(game.last_result),

                # backward compat (player cli often reads these directly)
                "bids_count": list(game.last_result.get("bids_count") or []),
                "loser": game.last_result.get("loser"),
                "loser_name": game.last_result.get("loser_name"),
                "winner_bid": list(game.last_result.get("winner_bid") or []),
                "loser_bid": list(game.last_result.get("loser_bid") or []),

                "hands": [pp.stones[:] for pp in game.players],
                "trash": dict(game.trash),
                "caretaker": game.caretaker,
                "score": [pp.score for pp in game.players],
            },
        )

        game.turn += 1

        for pp in game.players:
            pp.bid_submitted = False
            pp.ok_ready = False
            pp.revealed_bid = []

        game.current_bidder = 0

        end = check_game_end(game)
        if end:
            game.phase = "GAME_END"
            game.game_end_info = end
            game.status = f"GAME END: {end['reason']} winner={end['winner']} (type restart to play again)"
            await emit_action(room, "game_end", {"room_id": room_id, **end, "score": [pp.score for pp in game.players]})
            await emit_state(room)
            return

        if should_force_round_end_by_bag(game):
            start_round_end(game, trigger_reason="BAG_LOW", trigger_color=None, last_bid_winner=w)
            await emit_action(
                room,
                "round_end_start",
                {"room_id": room_id, "reason": "BAG_LOW", "trigger_color": None, "last_bid_winner": w, "round_end_info": dict(game.round_end_info)},
            )
            await emit_state(room)
            return

        if is_trash_limit_reached(game, TRASH_LIMIT):
            trigger = trash_trigger_color(game, TRASH_LIMIT)
            start_round_end(game, trigger_reason="TRASH_6", trigger_color=trigger, last_bid_winner=w)
            await emit_action(
                room,
                "round_end_start",
                {"room_id": room_id, "reason": "TRASH_6", "trigger_color": trigger, "last_bid_winner": w, "round_end_info": dict(game.round_end_info)},
            )
        else:
            game.phase = "RESULT"
            if w is None:
                game.status = "RESULT: No one bid. Offer->trash, both -1. Press OK/Next."
            else:
                game.status = f"RESULT: {game.players[w].name} won. Press OK/Next."

            if not game.offer:
                setup_offer(game)

    await emit_state(room)


@sio.event
async def proceed_phase(sid, data: Dict[str, Any]):
    room_id = str(data.get("room_id", "room1"))
    room = rooms.get(room_id)
    if not room or not room.game or sid not in room.sids:
        return

    game = room.game
    if game.phase == "GAME_END":
        return

    if game.phase not in ("RESULT", "ROUND_END"):
        return

    player_index = room.sids.index(sid)
    game.players[player_index].ok_ready = True

    await emit_action(
        room,
        "ok_next",
        {
            "room_id": room_id,
            "player_index": player_index,
            "player_name": game.players[player_index].name,
            "phase": game.phase,
            "round": game.round,
            "turn": game.turn,
            "ok_ready": [pp.ok_ready for pp in game.players],
        },
    )

    if not all(p.ok_ready for p in game.players):
        await emit_state(room)
        return

    for p in game.players:
        p.ok_ready = False
        p.revealed_bid = []
        p.bid_submitted = False

    if game.round_end_pending:
        game.round_end_pending = False

    if game.phase == "ROUND_END":
        game.phase = "BIDDING"
        game.status = "New round started. Submit your bid."
        game.current_bidder = 0
        await emit_action(room, "phase_change", {"room_id": room_id, "to": "BIDDING", "from": "ROUND_END"})
        await emit_state(room)
        return

    if not game.offer:
        ok = setup_offer(game)
        if not ok or should_force_round_end_by_bag(game):
            start_round_end(game, trigger_reason="BAG_LOW", trigger_color=None, last_bid_winner=None)
            await emit_action(
                room,
                "round_end_start",
                {"room_id": room_id, "reason": "BAG_LOW", "trigger_color": None, "last_bid_winner": None, "round_end_info": dict(game.round_end_info)},
            )
            await emit_state(room)
            return

    end = check_game_end(game)
    if end:
        game.phase = "GAME_END"
        game.game_end_info = end
        game.status = f"GAME END: {end['reason']} winner={end['winner']} (type restart to play again)"
        await emit_action(room, "game_end", {"room_id": room_id, **end, "score": [pp.score for pp in game.players]})
        await emit_state(room)
        return

    game.phase = "BIDDING"
    game.status = "New bidding. Submit your bid."
    game.current_bidder = 0

    await emit_action(room, "phase_change", {"room_id": room_id, "to": "BIDDING", "from": "RESULT"})
    await emit_state(room)


@sio.event
async def restart_game(sid, data: Dict[str, Any]):
    room_id = str(data.get("room_id", "room1"))
    room = rooms.get(room_id)
    if not room or sid not in room.sids:
        return
    if not room.game or room.game.phase != "GAME_END":
        await sio.emit("restart_rejected", {"message": "The game isn't over yet"}, to=sid)
        return

    by = room.names_by_sid.get(sid, "Unknown")
    await emit_action(room, "restart_game", {"room_id": room_id, "by": by})

    start_new_game(room)
    await emit_state(room)
