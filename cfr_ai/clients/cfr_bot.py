"""
Deep CFR Bot Client for Fafnir.

Connects to the game server via Socket.IO and plays using
the trained strategy network.

Usage:
    python cfr_ai/ai/cfr_bot.py --url http://127.0.0.1:8765 --room room1 --name DeepCFR
"""
import asyncio
import argparse
import random
import numpy as np
from typing import Any, Dict, List, Optional

import socketio

import sys
import os
# Add project root to sys.path so we can import from cfr_ai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cfr_ai.ai.action_space import (
    NUM_ACTIONS, get_legal_mask, action_id_to_stones, action_id_to_counts,
    PASS_ACTION_ID, ACTION_TABLE,
)
from cfr_ai.ai.observation import BidTracker, build_observation_from_server_state, NUM_COLORS, OBS_DIM
from cfr_ai.ai.game_engine import COLOR_TO_IDX, ALL_COLORS
from cfr_ai.ai.networks import StrategyNetwork, RegretNetwork, masked_softmax, regret_matching

import torch

# ============================================================
# Bot Configuration
# ============================================================
sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "DeepCFR", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
bid_tracker = BidTracker()
prev_round = 0
_last_tracker_key: Optional[tuple] = None

# Networks
strategy_net: Optional[StrategyNetwork] = None
regret_net: Optional[RegretNetwork] = None
device = torch.device("cpu")

# Anti-spam
_action_lock = asyncio.Lock()
_last_emit_ts = 0.0
_ok_sent_key: Optional[str] = None

AUTO_NEXT = True
THINK_DELAY = 0.01
TEMPERATURE = 0.3  # Lower = more deterministic
POLICY_MODE = "strategy"  # strategy or regret
DETERMINISTIC = False


def _loop_time() -> float:
    return asyncio.get_running_loop().time()


# ============================================================
# Server State Helpers
# ============================================================
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


def offer_colors(st: Dict[str, Any]) -> List[str]:
    offer = st.get("offer", [])
    return [x for x in offer if isinstance(x, str)]


def hand_to_counts(hand: List[str]) -> List[int]:
    counts = [0] * NUM_COLORS
    for s in hand:
        idx = COLOR_TO_IDX.get(s)
        if idx is not None:
            counts[idx] += 1
    return counts


def offer_to_counts(offer: List[str]) -> List[int]:
    counts = [0] * NUM_COLORS
    for s in offer:
        idx = COLOR_TO_IDX.get(s)
        if idx is not None:
            counts[idx] += 1
    return counts


def fair_state_view(st: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the subset of server state allowed for fair BIDDING decisions.

    server_0424.py may include sequential-bid leaks in action_log and opponent
    views. Action selection must behave as simultaneous bidding, so this view
    removes unresolved bids and any opponent hand/last_bid data.
    """
    fair = dict(st)
    fair["action_log"] = []
    fair["last_result"] = {}

    players = []
    for i, p in enumerate(players_of(st)):
        src = p if isinstance(p, dict) else {}
        view = {
            "name": src.get("name"),
            "score": src.get("score", 0),
            "hand_count": src.get("hand_count", 0),
            "bid_submitted": False,
            "ok_ready": False,
            "last_bid": None,
        }
        if my_index is not None and i == my_index:
            hand = src.get("hand", [])
            view["hand"] = hand[:] if isinstance(hand, list) else []
            view["bid_submitted"] = src.get("bid_submitted", False)
            view["ok_ready"] = src.get("ok_ready", False)
        else:
            view["hand"] = None
        players.append(view)

    fair["players"] = players
    return fair


# ============================================================
# Action Selection
# ============================================================
def choose_action(st: Dict[str, Any]) -> List[str]:
    """Use strategy network to choose bid."""
    fair_st = fair_state_view(st)
    hand = my_hand(fair_st)
    offer = offer_colors(fair_st)

    hand_counts = hand_to_counts(hand)
    offer_counts = offer_to_counts(offer)

    # Get legal action mask
    mask = get_legal_mask(hand_counts, offer_counts)

    # Build observation
    obs = build_observation_from_server_state(fair_st, my_index, bid_tracker)

    # Get policy from network
    with torch.inference_mode():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if POLICY_MODE == "regret" and regret_net is not None:
            regrets = regret_net(obs_tensor).cpu().numpy()[0]
            probs = regret_matching(regrets, mask)
        else:
            logits = strategy_net(obs_tensor).cpu().numpy()[0]
            probs = masked_softmax(logits, mask, TEMPERATURE)

    # Sample action
    legal_actions = np.where(mask)[0]
    if len(legal_actions) == 0:
        return []

    legal_probs = probs[legal_actions]
    legal_probs = legal_probs / (legal_probs.sum() + 1e-10)

    if DETERMINISTIC:
        action_id = int(legal_actions[np.argmax(legal_probs)])
    else:
        action_id = np.random.choice(legal_actions, p=legal_probs)

    # Convert to stone list
    stones = action_id_to_stones(int(action_id))

    # Log top actions
    top_k = min(5, len(legal_actions))
    sorted_ids = legal_actions[np.argsort(-legal_probs)][:top_k]
    top_str = ", ".join([
        f"{action_id_to_stones(int(a))}({probs[a]:.2f})" for a in sorted_ids
    ])
    print(f"[CFR] Top actions: {top_str}")
    print(f"[CFR] Chosen: {stones} (p={probs[action_id]:.3f})")

    return stones


def choose_action_random(st: Dict[str, Any]) -> List[str]:
    """Fallback random action if no model loaded."""
    fair_st = fair_state_view(st)
    hand = my_hand(fair_st)
    offer = offer_colors(fair_st)
    offer_set = set(offer)

    candidates = [s for s in hand if s not in offer_set]
    if not candidates:
        return []

    n = random.randint(0, min(3, len(candidates)))
    if n == 0:
        return []
    return random.sample(candidates, n)


# ============================================================
# Bid Tracker Updates
# ============================================================
def update_tracker_from_state(st: Dict[str, Any]):
    """Update bid tracker from auction results."""
    global prev_round, _last_tracker_key

    current_round = st.get("round", 0)

    # Reset tracker on new round
    if current_round != prev_round:
        bid_tracker.reset()
        _last_tracker_key = None
        prev_round = current_round

    # Update from last result
    lr = st.get("last_result", {})
    if not lr:
        return

    winner = lr.get("winner")
    if winner is None:
        return
    try:
        winner = int(winner)
    except Exception:
        return

    bids_by_player = lr.get("bids_by_player", [])
    if not isinstance(bids_by_player, list) or len(bids_by_player) < 2:
        return

    offer_stones = lr.get("offer", [])
    key = (
        current_round,
        st.get("turn", 0),
        winner,
        tuple(tuple(b or []) for b in bids_by_player[:2]),
        tuple(offer_stones or []),
    )
    if key == _last_tracker_key:
        return

    bid0 = hand_to_counts(bids_by_player[0] if bids_by_player[0] else [])
    bid1 = hand_to_counts(bids_by_player[1] if bids_by_player[1] else [])
    offer_c = offer_to_counts(offer_stones if offer_stones else [])

    loser = 1 - winner
    bid_winner = bid0 if winner == 0 else bid1
    bid_loser = bid0 if loser == 0 else bid1

    bid_tracker.update_from_auction(winner, bid_winner, bid_loser, offer_c)
    _last_tracker_key = key


# ============================================================
# Socket Handlers
# ============================================================
async def _emit_throttled(event: str, payload: Dict[str, Any], min_interval: float = 0.12):
    global _last_emit_ts
    async with _action_lock:
        dt = _loop_time() - _last_emit_ts
        if dt < min_interval:
            await asyncio.sleep(min_interval - dt)
        _last_emit_ts = _loop_time()
        await sio.emit(event, payload)


def _phase_key(st: Dict[str, Any]) -> str:
    ph = phase_of(st)
    r = st.get("round", "?")
    t = st.get("turn", "?")
    if ph == "ROUND_END":
        return f"ROUND_END:r{r}"
    if ph == "RESULT":
        return f"RESULT:r{r}:t{t}"
    return f"{ph}:r{r}:t{t}"


async def do_submit_bid(st: Dict[str, Any], reason: str):
    if strategy_net is not None:
        stones = choose_action(st)
    else:
        stones = choose_action_random(st)

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("submit_bid", {"room_id": cfg["room"], "stones": stones})
    print(f"[CFR] submit ({reason}) stones={stones}")


async def do_ok_next(st: Dict[str, Any], reason: str):
    global _ok_sent_key
    key = _phase_key(st)
    if _ok_sent_key == key:
        return
    if my_ok_ready(st):
        _ok_sent_key = key
        return

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("proceed_phase", {"room_id": cfg["room"]})
    _ok_sent_key = key
    print(f"[CFR] OK/Next ({reason})")


async def brain_loop():
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

            elif ph == "GAME_END":
                # Auto restart
                await asyncio.sleep(0.5)
                await _emit_throttled("restart_game", {"room_id": cfg["room"]})

        await asyncio.sleep(0.10)


@sio.event
async def connect():
    print("[CFR] connected")
    await _emit_throttled(
        "join_room",
        {"room_id": cfg["room"], "player_name": cfg["name"]},
        min_interval=0.0,
    )


@sio.event
async def disconnect():
    print("[CFR] disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print("[CFR] assigned index =", my_index)


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key
    last_state = state

    # Update bid tracker
    update_tracker_from_state(state)

    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    reason = data.get("reason") or ""
    msg = data.get("message") or reason
    print("[CFR] BID REJECTED:", msg)


# ============================================================
# Main
# ============================================================
async def main():
    global strategy_net, regret_net, device, TEMPERATURE, POLICY_MODE, DETERMINISTIC

    parser = argparse.ArgumentParser(description="Deep CFR Bot for Fafnir")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--room", default="room1")
    parser.add_argument("--name", default="DeepCFR")
    parser.add_argument("--checkpoint", default="cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (lower=more deterministic)")
    parser.add_argument("--policy", choices=["strategy", "regret"], default="strategy",
                        help="Use strategy_net softmax or regret_net regret matching")
    parser.add_argument("--deterministic", action="store_true",
                        help="Choose argmax action instead of sampling")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name
    TEMPERATURE = args.temperature
    POLICY_MODE = args.policy
    DETERMINISTIC = args.deterministic
    device = torch.device(args.device)

    # Load model
    import os
    if os.path.exists(args.checkpoint):
        print(f"[CFR] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        hidden_dim = ckpt.get('hidden_dim', 192)
        ckpt_obs_dim = ckpt.get('obs_dim')
        if ckpt_obs_dim != OBS_DIM:
            raise RuntimeError(
                f"Checkpoint obs_dim={ckpt_obs_dim} is incompatible with current OBS_DIM={OBS_DIM}. "
                "Retrain CFR after the observation layout change."
            )
        strategy_net = StrategyNetwork(
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            hidden=hidden_dim,
        ).to(device)
        strategy_net.load_state_dict(ckpt['strategy_net'])
        strategy_net.eval()
        if POLICY_MODE == "regret":
            if 'regret_net' not in ckpt:
                raise RuntimeError("Checkpoint does not contain regret_net for --policy regret")
            regret_net = RegretNetwork(
                obs_dim=OBS_DIM,
                num_actions=NUM_ACTIONS,
                hidden=hidden_dim,
            ).to(device)
            regret_net.load_state_dict(ckpt['regret_net'])
            regret_net.eval()
        print(f"[CFR] Model loaded (iter={ckpt.get('iteration', '?')}, policy={POLICY_MODE})")
    else:
        print(f"[CFR] WARNING: No checkpoint at {args.checkpoint}, using random play")
        strategy_net = None

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
