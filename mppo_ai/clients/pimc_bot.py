# mppo_ai/clients/pimc_bot.py
"""
FAFNIR PIMC Bot — PPO model with 1-step lookahead search.

Action space: MultiDiscrete([MAX_BID_PER_COLOR+1] * N_COLORS)
  => model.predict() returns shape-(N_COLORS,) array, e.g. [0,2,0,1,0,0]
  => This is the bid count per color, directly used as bid vector.

PIMC strategy:
  For each determinization of opponent's hidden hand:
    - Sample K candidate bids from the PPO policy (stochastic)
    - Also test the greedy (deterministic) PPO bid as one candidate
    - Simulate each candidate bid 1 step ahead using fast_engine
    - Evaluate resulting state with PPO value network
  Select the candidate with highest average value across all determinizations.

Usage:
    python mppo_ai/clients/pimc_bot.py --model mppo_ai/rl/output/fafnir_final
    python mppo_ai/clients/pimc_bot.py --model mppo_ai/rl/output/fafnir_final --url http://SERVER:8765 --room room1 --name PIMC_Bot
"""
import asyncio
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import socketio
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from mppo_ai.rl.game_env import (
    N_COLORS, MAX_BID_PER_COLOR, INITIAL_BAG, TRASH_LIMIT, COLORS_NAMES
)
from mppo_ai.rl.fast_engine import fast_step, build_obs, _random_bid

sio = socketio.AsyncClient(reconnection=True)

cfg = {"room": "room1", "name": "PIMC_Bot", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
model: Optional[MaskablePPO] = None

_action_lock = asyncio.Lock()
_ok_sent_key: Optional[str] = None
_restart_sent = False

AUTO_NEXT = True
THINK_DELAY = 0.05
SEARCH_TIME = 0.8   # seconds per bid decision
N_CANDIDATES = 8    # stochastic samples from policy per determinization

# ── Known-hand tracking ────────────────────────────────────────────────────
_known = np.zeros((2, N_COLORS), dtype=np.int32)
_prev_round = -1
_last_processed_auction = None


# ══════════════════════════════════════════════════════════════════════════
# Helpers — identical to rl_bot_v2.py
# ══════════════════════════════════════════════════════════════════════════

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


def update_known_from_state(st: Dict[str, Any], player_idx: int):
    global _known, _prev_round, _last_processed_auction

    current_round = st.get("round", 1)
    if current_round != _prev_round:
        _known[:] = 0
        _prev_round = current_round
        _last_processed_auction = None

    lr = st.get("last_result") or {}
    winner = lr.get("winner")
    if winner is None:
        return

    auction_id = f"{current_round}-{st.get('turn', 1)}"
    if _last_processed_auction == auction_id:
        return
    _last_processed_auction = auction_id

    try:
        wi = int(winner)
    except Exception:
        return
    li = 1 - wi

    bids_by_player = lr.get("bids_by_player")
    if isinstance(bids_by_player, list) and len(bids_by_player) >= 2:
        w_bid = _stones_to_counts(bids_by_player[wi] or [])
        l_bid = _stones_to_counts(bids_by_player[li] or [])
    else:
        w_bid = _stones_to_counts(lr.get("winner_bid") or [])
        l_bid = _stones_to_counts(lr.get("loser_bid") or [])

    offer_counts = _stones_to_counts(st.get("last_offer") or lr.get("offer") or [])

    for c in range(N_COLORS):
        val = _known[wi, c] + offer_counts[c] - w_bid[c]
        _known[wi, c] = max(0, val)
        _known[li, c] = max(_known[li, c], int(l_bid[c]))


def state_to_mask(st: Dict[str, Any], player_idx: int) -> np.ndarray:
    mask_size = N_COLORS * (MAX_BID_PER_COLOR + 1)
    mask = np.zeros(mask_size, dtype=np.int8)
    ps = st.get("players") or []
    if player_idx >= len(ps):
        return mask

    hand_counts = _stones_to_counts(ps[player_idx].get("hand") or [])
    offer_counts = _stones_to_counts(st.get("offer") or [])

    for c in range(N_COLORS):
        base = c * (MAX_BID_PER_COLOR + 1)
        if offer_counts[c] > 0:
            mask[base] = 1
        else:
            max_valid = min(int(hand_counts[c]), MAX_BID_PER_COLOR)
            for v in range(max_valid + 1):
                mask[base + v] = 1
    return mask


def bid_array_to_stones(bid: np.ndarray, hand: List[str]) -> List[str]:
    """Convert bid count-per-color array to list of stone strings."""
    hand_counts = _stones_to_counts(hand)
    stones: List[str] = []
    for c in range(N_COLORS):
        count = min(int(bid[c]), int(hand_counts[c]))
        for _ in range(count):
            stones.append(COLORS_NAMES[c])
    return stones


def clamp_bid(bid: np.ndarray, hand_counts: np.ndarray, offer_counts: np.ndarray) -> np.ndarray:
    """Clamp bid so it does not exceed hand, and zeros out offer colors."""
    result = np.zeros(N_COLORS, dtype=np.int32)
    for c in range(N_COLORS):
        if offer_counts[c] > 0:
            result[c] = 0
        else:
            result[c] = min(int(bid[c]), int(hand_counts[c]))
    return result


# ══════════════════════════════════════════════════════════════════════════
# PIMC Search
# ══════════════════════════════════════════════════════════════════════════

def _build_state_obs(
    hands, bag, trash, offer, scores, known,
    caretaker, round_n, turn_n, player_idx
) -> np.ndarray:
    """Build a 36-dim obs for fast_engine.build_obs."""
    state = np.array([caretaker, round_n, turn_n, 0, -1], dtype=np.int32)
    return build_obs(
        hands, bag, trash, offer, scores, state, known,
        np.int32(player_idx), np.int32(40)
    )


def _get_value_batch(obs_batch: np.ndarray) -> np.ndarray:
    """
    Evaluate a batch of observations with PPO value network.
    obs_batch: shape (B, 36)
    Returns: shape (B,) float32
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
    with torch.no_grad():
        values = model.policy.predict_values(obs_tensor)
    return values.cpu().numpy().flatten()


def _sample_candidate_bids(
    obs: np.ndarray,
    mask: np.ndarray,
    hand_counts: np.ndarray,
    offer_counts: np.ndarray,
    n_samples: int
) -> List[np.ndarray]:
    """
    Sample candidate bids from the PPO policy.
    One deterministic + (n_samples-1) stochastic.
    All returned bids are clamped to be legal.
    Uses model.predict() which returns MultiDiscrete shape-(N_COLORS,).
    """
    candidates: List[np.ndarray] = []

    # Deterministic (greedy) bid
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    candidates.append(clamp_bid(action.astype(np.int32), hand_counts, offer_counts))

    # Stochastic samples
    for _ in range(n_samples - 1):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        candidates.append(clamp_bid(action.astype(np.int32), hand_counts, offer_counts))

    # Deduplicate
    unique: List[np.ndarray] = []
    seen = set()
    for b in candidates:
        key = tuple(b.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def run_pimc_search(st: Dict[str, Any], player_idx: int, time_limit: float) -> np.ndarray:
    """
    PIMC 1-step lookahead search.
    Returns the best bid as a MultiDiscrete action array (shape N_COLORS).
    """
    ps = st.get("players") or []
    other_idx = 1 - player_idx

    # ── Parse current game state ──────────────────────────────────────────
    my_hand_str = (ps[player_idx].get("hand") or []) if player_idx < len(ps) else []
    my_hand = _stones_to_counts(my_hand_str)
    offer = _stones_to_counts(st.get("offer") or [])
    trash = _trash_to_counts(st.get("trash") or {})

    my_score  = ps[player_idx].get("score", 0) if player_idx < len(ps) else 0
    opp_score = ps[other_idx].get("score", 0)  if other_idx < len(ps) else 0

    # fast_engine always indexes as [player0, player1]
    scores = np.zeros(2, dtype=np.int32)
    scores[player_idx] = my_score
    scores[other_idx]  = opp_score

    caretaker = st.get("caretaker", 0)
    round_n   = st.get("round", 1)
    turn_n    = st.get("turn", 1)

    opp_total_stones  = (ps[other_idx].get("hand_count", 0) if other_idx < len(ps) else 0)
    opp_known         = _known[other_idx].copy()
    opp_unknown_count = max(0, opp_total_stones - int(np.sum(opp_known)))

    # ── Build unseen stone pool (stones not in my hand / trash / offer / known_opp) ──
    unseen_list: List[int] = []
    for c in range(N_COLORS):
        used = int(my_hand[c]) + int(trash[c]) + int(offer[c]) + int(opp_known[c])
        available = max(0, int(INITIAL_BAG[c]) - used)
        unseen_list.extend([c] * available)
    unseen_arr = np.array(unseen_list, dtype=np.int32)

    # ── Get my current obs + mask ─────────────────────────────────────────
    mask = state_to_mask(st, player_idx)

    # Build obs from current (real) state to sample policy candidates
    cur_hands = np.zeros((2, N_COLORS), dtype=np.int32)
    cur_hands[player_idx] = my_hand.copy()
    # Opponent's real hand is hidden — use known as proxy for obs building only
    cur_hands[other_idx]  = opp_known.copy()
    cur_state = np.array([caretaker, round_n, turn_n, 0, -1], dtype=np.int32)
    cur_known = _known.copy()
    cur_obs = build_obs(
        cur_hands, np.zeros(N_COLORS, dtype=np.int32),
        trash, offer, scores, cur_state, cur_known,
        np.int32(player_idx), np.int32(40)
    )

    # ── Sample candidate bids from policy ────────────────────────────────
    candidates = _sample_candidate_bids(cur_obs, mask, my_hand, offer, N_CANDIDATES)

    if len(candidates) == 1:
        print(f"[PIMC] Only 1 legal candidate — returning immediately.")
        return candidates[0]

    # ── Accumulate value estimates across determinizations ────────────────
    # action_sum[i] = total value accumulated for candidate i
    action_sum    = np.zeros(len(candidates), dtype=np.float64)
    action_visits = np.zeros(len(candidates), dtype=np.int32)

    start_time = time.perf_counter()
    det_count  = 0

    while time.perf_counter() - start_time < time_limit:
        det_count += 1

        # 1. Determinize opponent's hidden hand
        if opp_unknown_count > 0 and len(unseen_arr) >= opp_unknown_count:
            perm = np.random.permutation(len(unseen_arr))
            chosen_idx = perm[:opp_unknown_count]
            opp_hand = opp_known.copy()
            for ci in chosen_idx:
                opp_hand[unseen_arr[ci]] += 1
            bag_indices = perm[opp_unknown_count:]
        else:
            opp_hand = opp_known.copy()
            bag_indices = np.arange(len(unseen_arr))

        bag = np.zeros(N_COLORS, dtype=np.int32)
        for ci in bag_indices:
            bag[unseen_arr[ci]] += 1

        # Build simulated hands
        sim_hands_base = np.zeros((2, N_COLORS), dtype=np.int32)
        sim_hands_base[player_idx] = my_hand.copy()
        sim_hands_base[other_idx]  = opp_hand.copy()

        # 2. Collect post-step obs for all candidates in a batch
        obs_list: List[np.ndarray] = []
        term_values: List[Optional[float]] = []

        for bid in candidates:
            sim_hands  = sim_hands_base.copy()
            sim_bag    = bag.copy()
            sim_trash  = trash.copy()
            sim_offer  = offer.copy()
            sim_scores = scores.copy()
            sim_state  = np.array([caretaker, round_n, turn_n, 0, -1], dtype=np.int32)
            sim_known  = _known.copy()

            # Opponent uses random policy during lookahead
            opp_bid = _random_bid(sim_hands[other_idx], sim_offer)

            bid0 = bid     if player_idx == 0 else opp_bid
            bid1 = opp_bid if player_idx == 0 else bid

            reward, term, _ = fast_step(
                sim_hands, sim_bag, sim_trash, sim_offer,
                sim_scores, sim_state, sim_known,
                bid0, bid1, np.int32(40), np.int32(500)
            )

            if term:
                # Terminal: +1 if I won, -1 if I lost
                v = 1.0 if sim_scores[player_idx] >= 40 else -1.0
                # fast_step reward is from player 0's perspective; flip for player 1
                r = float(reward) if player_idx == 0 else -float(reward)
                term_values.append(r + v)
                obs_list.append(None)
            else:
                post_obs = build_obs(
                    sim_hands, sim_bag, sim_trash, sim_offer,
                    sim_scores, sim_state, sim_known,
                    np.int32(player_idx), np.int32(40)
                )
                obs_list.append(post_obs)
                term_values.append(None)  # will fill from batch

        # 3. Batch-evaluate non-terminal states
        non_term_indices = [i for i, tv in enumerate(term_values) if tv is None]
        if non_term_indices:
            batch = np.stack([obs_list[i] for i in non_term_indices])
            values = _get_value_batch(batch)
            for rank, i in enumerate(non_term_indices):
                term_values[i] = float(values[rank])

        # 4. Accumulate
        for i, v in enumerate(term_values):
            action_sum[i]    += v
            action_visits[i] += 1

    # ── Pick best candidate ───────────────────────────────────────────────
    best_idx = 0
    best_avg = -1e9
    for i in range(len(candidates)):
        if action_visits[i] > 0:
            avg = action_sum[i] / action_visits[i]
            if avg > best_avg:
                best_avg = avg
                best_idx = i

    best_bid = candidates[best_idx]
    print(f"[PIMC] {det_count} determinizations | "
          f"best bid={best_bid.tolist()} avg_value={best_avg:.3f}")
    return best_bid


# ══════════════════════════════════════════════════════════════════════════
# Socket / bot infrastructure (same pattern as rl_bot_v2.py)
# ══════════════════════════════════════════════════════════════════════════

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
        print(f"[PIMC] emit error: {repr(e)}")


async def do_submit_bid(st: Dict[str, Any]):
    if model is None or my_index is None:
        return

    print("[PIMC] Searching for best bid...")
    loop = asyncio.get_running_loop()
    best_bid = await loop.run_in_executor(None, run_pimc_search, st, my_index, SEARCH_TIME)

    ps = st.get("players") or []
    hand = (ps[my_index].get("hand") or []) if my_index < len(ps) else []
    stones = bid_array_to_stones(best_bid, hand)

    print(f"[PIMC] Bidding: {stones}")
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
    print(f"[PIMC] OK/Next ({ph})")


async def do_restart_game():
    global _restart_sent, _known, _prev_round, _last_processed_auction, _ok_sent_key
    if _restart_sent:
        return
    _restart_sent = True
    _known[:] = 0
    _prev_round = -1
    _last_processed_auction = None
    _ok_sent_key = None
    print("[PIMC] Game ended — sending restart_game")
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
            print(f"[PIMC] Error in brain_loop: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(0.10)


# ══════════════════════════════════════════════════════════════════════════
# Socket.IO events
# ══════════════════════════════════════════════════════════════════════════

@sio.event
async def connect():
    print("[PIMC] Connected to server")
    await sio.emit("join_room", {"room_id": cfg["room"], "player_name": cfg["name"]})


@sio.event
async def disconnect():
    print("[PIMC] Disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print(f"[PIMC] Assigned index = {my_index}")


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key, my_index
    last_state = state

    # Auto-correct index by name (same as rl_bot_v2.py)
    ps = state.get("players") or []
    my_name = cfg["name"]
    for i, p in enumerate(ps):
        if p.get("name") == my_name:
            if my_index != i:
                print(f"[PIMC] Index corrected: {my_index} -> {i}")
                my_index = i
            break

    if my_index is not None:
        update_known_from_state(state, my_index)

    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    msg = data.get("message") or data.get("reason") or "rejected"
    print(f"[PIMC] BID REJECTED: {msg}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

async def main():
    global model, SEARCH_TIME, THINK_DELAY, N_CANDIDATES

    ap = argparse.ArgumentParser(description="FAFNIR PIMC Bot")
    ap.add_argument("--model",       type=str,   required=True)
    ap.add_argument("--url",         type=str,   default="http://127.0.0.1:8765")
    ap.add_argument("--room",        type=str,   default="room1")
    ap.add_argument("--name",        type=str,   default="PIMC_Bot")
    ap.add_argument("--search-time", type=float, default=0.8,
                    help="Seconds for PIMC search per bid (default 0.8)")
    ap.add_argument("--candidates",  type=int,   default=8,
                    help="Policy samples per determinization (default 8)")
    args = ap.parse_args()

    cfg["url"]  = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name
    SEARCH_TIME  = args.search_time
    N_CANDIDATES = args.candidates

    print(f"[PIMC] Loading model: {args.model}")
    model = MaskablePPO.load(args.model)
    print(f"[PIMC] Model loaded! search_time={SEARCH_TIME}s candidates={N_CANDIDATES}")

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
