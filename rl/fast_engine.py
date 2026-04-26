# rl/fast_engine.py
"""
Numba JIT-compiled FAFNIR game engine for fast RL training.
All game logic as pure functions operating on numpy arrays.

Observation space: 36 dimensions (see build_obs for details).
"""
import numpy as np
from numba import njit

N_COLORS = 6
TRASH_LIMIT = 6
SEED_TRASH_N = 3
MAX_BID = 10

# State indices: state_ints = [caretaker, round, turn, total_turns, last_winner]
S_CT = 0; S_RND = 1; S_TURN = 2; S_TOTAL = 3; S_LAST_W = 4
STATE_SIZE = 5
INITIAL_BAG = np.array([20, 12, 12, 12, 12, 12], dtype=np.int32)
BAG_TOTAL = 80

OBS_DIM = 36

# ---------- primitives ----------

@njit(cache=True)
def _draw_one(bag):
    total = np.int32(0)
    for i in range(N_COLORS):
        total += bag[i]
    if total == 0:
        return np.int32(-1)
    r = np.random.randint(0, total)
    cum = np.int32(0)
    for i in range(N_COLORS):
        cum += bag[i]
        if r < cum:
            bag[i] -= np.int32(1)
            return np.int32(i)
    return np.int32(-1)


@njit(cache=True)
def _draw_n(bag, n):
    drawn = np.zeros(N_COLORS, dtype=np.int32)
    for _ in range(n):
        c = _draw_one(bag)
        if c < 0:
            break
        drawn[c] += np.int32(1)
    return drawn


@njit(cache=True)
def _setup_offer(bag):
    offer = np.zeros(N_COLORS, dtype=np.int32)
    for _ in range(50):
        total = np.int32(0)
        for i in range(N_COLORS):
            total += bag[i]
        if total == 0:
            break
        s1 = _draw_one(bag)
        s2 = np.int32(-1)
        if total >= 2 and s1 >= 0:
            s2 = _draw_one(bag)
        if s2 < 0 or s1 != s2:
            if s1 >= 0:
                offer[s1] += np.int32(1)
            if s2 >= 0:
                offer[s2] += np.int32(1)
            break
        else:
            bag[s1] += np.int32(1)
            bag[s2] += np.int32(1)
    return offer


@njit(cache=True)
def _clamp(x):
    return max(np.int32(0), x)


@njit(cache=True)
def _random_bid(hand, offer):
    bid = np.zeros(N_COLORS, dtype=np.int32)
    n_biddable = np.int32(0)
    for c in range(N_COLORS):
        if offer[c] == 0:
            n_biddable += hand[c]
    if n_biddable == 0:
        return bid
    biddable = np.zeros(n_biddable, dtype=np.int32)
    idx = 0
    for c in range(N_COLORS):
        if offer[c] == 0:
            for _ in range(hand[c]):
                biddable[idx] = np.int32(c)
                idx += 1
    n = np.random.randint(1, min(4, n_biddable) + 1)
    for i in range(min(n, n_biddable)):
        j = np.random.randint(i, n_biddable)
        tmp = biddable[i]; biddable[i] = biddable[j]; biddable[j] = tmp
    for i in range(min(n, n_biddable)):
        bid[biddable[i]] += np.int32(1)
    return bid


# ---------- scoring ----------

@njit(cache=True)
def _rank_colors(hands):
    ranked_idx = np.zeros(5, dtype=np.int32)
    ranked_cnt = np.zeros(5, dtype=np.int32)
    for i in range(5):
        c = np.int32(i + 1)
        ranked_idx[i] = c
        ranked_cnt[i] = hands[0, c] + hands[1, c]
    for i in range(5):
        for j in range(i + 1, 5):
            swap = False
            if ranked_cnt[j] > ranked_cnt[i]:
                swap = True
            elif ranked_cnt[j] == ranked_cnt[i] and ranked_idx[j] < ranked_idx[i]:
                swap = True
            if swap:
                ti = ranked_idx[i]; tc = ranked_cnt[i]
                ranked_idx[i] = ranked_idx[j]; ranked_cnt[i] = ranked_cnt[j]
                ranked_idx[j] = ti; ranked_cnt[j] = tc
    return ranked_idx, ranked_cnt


@njit(cache=True)
def _hand_score(hands, player, first_c, second_c):
    score = hands[player, 0]  # gold
    for c in range(1, N_COLORS):
        cnt = hands[player, c]
        if cnt == 0 or cnt >= 5:
            continue
        if c == first_c:
            score += cnt * np.int32(3)
        elif c == second_c:
            score += cnt * np.int32(2)
        else:
            score -= cnt
    return score


@njit(cache=True)
def _compute_potential(hands, player):
    """Compute the expected round-end score for a player given current hands.
    This is used for potential-based reward shaping.
    """
    ranked_idx, _ = _rank_colors(hands)
    first_c = ranked_idx[0]
    second_c = ranked_idx[1]
    return _hand_score(hands, player, first_c, second_c)


# ---------- known hand tracking ----------

@njit(cache=True)
def _update_known_hands(known, winner, loser, w_bid, offer, l_bid):
    """Update known-hand tracking arrays after an auction.

    known: shape (2, N_COLORS) — minimum stones each player is known to hold.

    Rules:
    - Winner gained offer (public) and lost w_bid to trash (public).
      known[winner] += offer - w_bid, clamped >= 0 per color.
    - Loser's bid was revealed but stones stay. We know they have at least l_bid.
      known[loser] = max(known[loser], l_bid) per color.
    """
    for c in range(N_COLORS):
        # Winner: net change from offer gained minus bid lost
        val = known[winner, c] + offer[c] - w_bid[c]
        known[winner, c] = max(np.int32(0), val)
        # Loser: bid was revealed, they still have those stones
        if l_bid[c] > known[loser, c]:
            known[loser, c] = l_bid[c]


@njit(cache=True)
def _reset_known_hands(known):
    """Reset known hands at round start (all cards re-dealt from bag)."""
    for p in range(2):
        for c in range(N_COLORS):
            known[p, c] = np.int32(0)


# ---------- round end ----------

@njit(cache=True)
def _do_round_end(hands, bag, trash, offer, scores, state, known):
    ranked_idx, _ = _rank_colors(hands)
    first_c = ranked_idx[0]
    second_c = ranked_idx[1]
    add0 = _hand_score(hands, 0, first_c, second_c)
    add1 = _hand_score(hands, 1, first_c, second_c)
    scores[0] = _clamp(scores[0] + add0)
    scores[1] = _clamp(scores[1] + add1)
    round_reward = np.float32(add0 - add1) * np.float32(0.02)

    # Reset for new round
    for c in range(N_COLORS):
        bag[c] = INITIAL_BAG[c]
        trash[c] = np.int32(0)
        hands[0, c] = np.int32(0)
        hands[1, c] = np.int32(0)
        offer[c] = np.int32(0)
    ct = state[S_CT]
    for p in range(2):
        n = 11 if p == ct else 10
        drawn = _draw_n(bag, n)
        for c2 in range(N_COLORS):
            hands[p, c2] += drawn[c2]
    seeded = _draw_n(bag, SEED_TRASH_N)
    for c2 in range(N_COLORS):
        trash[c2] += seeded[c2]
    new_offer = _setup_offer(bag)
    for c2 in range(N_COLORS):
        offer[c2] = new_offer[c2]
    state[S_RND] += np.int32(1)
    state[S_TURN] = np.int32(1)

    # Reset known hands (new cards dealt)
    _reset_known_hands(known)

    return round_reward


# ---------- obs / mask ----------

@njit(cache=True)
def build_obs(hands, bag, trash, offer, scores, state, known, player, score_to_win):
    """Build 36-dim observation vector for the given player.

    Layout:
      0- 5: My hand (6) — normalized by max possible per color
      6-11: Current offer (6) — normalized by 10
     12-17: Trash (6) — normalized by TRASH_LIMIT
     18-23: Opponent's confirmed hand (6) — normalized by max possible
     24   : Opponent's unknown stone count (1) — normalized by 15
     25-30: My confirmed hand (what opponent knows about me) (6) — normalized
     31   : My score — normalized by score_to_win
     32   : Opponent's score — normalized by score_to_win
     33   : Bag remaining — normalized by BAG_TOTAL
     34   : Am I the caretaker? (0 or 1)
     35   : My hand's current potential score — normalized
    """
    other = np.int32(1 - player)
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # 0-5: My hand
    for c in range(N_COLORS):
        obs[c] = np.float32(hands[player, c]) / max(np.float32(1), np.float32(INITIAL_BAG[c]))

    # 6-11: Offer (use 10 as max to handle potential multi-stone offers)
    for c in range(N_COLORS):
        obs[6 + c] = np.float32(offer[c]) / np.float32(10)

    # 12-17: Trash
    for c in range(N_COLORS):
        obs[12 + c] = np.float32(trash[c]) / np.float32(TRASH_LIMIT)

    # 18-23: Opponent's confirmed hand
    for c in range(N_COLORS):
        obs[18 + c] = np.float32(known[other, c]) / max(np.float32(1), np.float32(INITIAL_BAG[c]))

    # 24: Opponent's unknown stone count
    opp_total = np.int32(0)
    opp_known_total = np.int32(0)
    for c in range(N_COLORS):
        opp_total += hands[other, c]
        opp_known_total += known[other, c]
    opp_unknown = max(np.int32(0), opp_total - opp_known_total)
    obs[24] = np.float32(opp_unknown) / np.float32(15)

    # 25-30: My confirmed hand (what opponent can see/deduce about me)
    for c in range(N_COLORS):
        obs[25 + c] = np.float32(known[player, c]) / max(np.float32(1), np.float32(INITIAL_BAG[c]))

    # 31: My score
    obs[31] = np.float32(scores[player]) / max(np.float32(1), np.float32(score_to_win))

    # 32: Opponent's score
    obs[32] = np.float32(scores[other]) / max(np.float32(1), np.float32(score_to_win))

    # 33: Bag remaining
    bl = np.int32(0)
    for c in range(N_COLORS):
        bl += bag[c]
    obs[33] = np.float32(bl) / np.float32(BAG_TOTAL)

    # 34: Am I the caretaker?
    obs[34] = np.float32(1) if state[S_CT] == player else np.float32(0)

    # 35: My hand's current potential score (normalized)
    potential = _compute_potential(hands, player)
    # Normalize: theoretical range is roughly -15 to +60, use 30 as midpoint scale
    obs[35] = np.float32(potential + 15) / np.float32(75)

    # Clamp all to [0, 1]
    for i in range(OBS_DIM):
        obs[i] = max(np.float32(0), min(np.float32(1), obs[i]))
    return obs


@njit(cache=True)
def build_mask(hands, offer, player):
    ms = N_COLORS * (MAX_BID + 1)
    mask = np.zeros(ms, dtype=np.int8)
    for c in range(N_COLORS):
        base = c * (MAX_BID + 1)
        if offer[c] > 0:
            mask[base] = np.int8(1)
        else:
            mx = min(np.int32(hands[player, c]), np.int32(MAX_BID))
            for v in range(mx + 1):
                mask[base + v] = np.int8(1)
    return mask


# ---------- main step ----------

@njit(cache=True)
def fast_reset(score_to_win):
    bag = INITIAL_BAG.copy()
    hands = np.zeros((2, N_COLORS), dtype=np.int32)
    trash = np.zeros(N_COLORS, dtype=np.int32)
    scores = np.zeros(2, dtype=np.int32)
    ct = np.int32(np.random.randint(0, 2))
    for p in range(2):
        n = 11 if p == ct else 10
        drawn = _draw_n(bag, n)
        for c in range(N_COLORS):
            hands[p, c] += drawn[c]
    seeded = _draw_n(bag, SEED_TRASH_N)
    for c in range(N_COLORS):
        trash[c] += seeded[c]
    offer = _setup_offer(bag)
    state = np.array([ct, 1, 1, 0, -1], dtype=np.int32)
    # Initialize known hands (nothing known at game start)
    known = np.zeros((2, N_COLORS), dtype=np.int32)
    return hands, bag, trash, offer, scores, state, known


@njit(cache=True)
def fast_step(hands, bag, trash, offer, scores, state, known,
              bid0, bid1, score_to_win, max_turns):
    """Returns (reward, terminated, truncated). Modifies arrays in-place."""
    state[S_TOTAL] += np.int32(1)
    ct = state[S_CT]

    # Compute potential BEFORE the step (for reward shaping)
    pot_before = _compute_potential(hands, np.int32(0))

    s0_before = scores[0]; s1_before = scores[1]
    c0 = np.int32(0); c1 = np.int32(0)
    for c in range(N_COLORS):
        c0 += bid0[c]; c1 += bid1[c]

    mx = max(c0, c1)
    winner = np.int32(-1)

    if mx == 0:
        scores[0] = _clamp(scores[0] - np.int32(1))
        scores[1] = _clamp(scores[1] - np.int32(1))
        for c in range(N_COLORS):
            trash[c] += offer[c]; offer[c] = np.int32(0)
    else:
        if c0 > c1:
            winner = np.int32(0)
        elif c1 > c0:
            winner = np.int32(1)
        else:
            winner = np.int32(1) if ct == 0 else np.int32(0)
        loser = np.int32(1 - winner)
        w_bid = bid0 if winner == 0 else bid1
        l_bid = bid1 if winner == 0 else bid0

        for c in range(N_COLORS):
            hands[winner, c] -= w_bid[c]
            trash[c] += w_bid[c]
            hands[winner, c] += offer[c]
            offer[c] = np.int32(0)
        scores[winner] += np.int32(1)
        state[S_CT] = winner

        # Update known hands
        _update_known_hands(known, winner, loser, w_bid, offer, l_bid)

    state[S_LAST_W] = winner

    # Score-based reward
    reward = np.float32(scores[0] - s0_before - (scores[1] - s1_before)) * np.float32(0.02)

    # Potential-based reward shaping: reward change in expected hand value
    pot_after = _compute_potential(hands, np.int32(0))
    reward += np.float32(pot_after - pot_before) * np.float32(0.005)

    # Game end check
    if scores[0] >= score_to_win:
        return reward + np.float32(1.0), True, False
    if scores[1] >= score_to_win:
        return reward - np.float32(1.0), True, False

    # Trash / bag check
    trash_hit = False
    for c in range(N_COLORS):
        if trash[c] >= TRASH_LIMIT:
            trash_hit = True; break
    bl = np.int32(0)
    for c in range(N_COLORS):
        bl += bag[c]
    bag_low = bl < 2

    if trash_hit or bag_low:
        rr = _do_round_end(hands, bag, trash, offer, scores, state, known)
        reward += rr
        if scores[0] >= score_to_win:
            return reward + np.float32(1.0), True, False
        if scores[1] >= score_to_win:
            return reward - np.float32(1.0), True, False
    else:
        new_offer = _setup_offer(bag)
        for c2 in range(N_COLORS):
            offer[c2] = new_offer[c2]
        state[S_TURN] += np.int32(1)

    if state[S_TOTAL] >= max_turns:
        trunc_r = np.float32(0)
        if scores[0] > scores[1]:
            trunc_r = np.float32(0.5)
        elif scores[0] < scores[1]:
            trunc_r = np.float32(-0.5)
        return reward + trunc_r, False, True

    return reward, False, False


# ---------- warmup (call once to trigger JIT) ----------

def warmup():
    """Pre-compile all Numba functions."""
    hands, bag, trash, offer, scores, state, known = fast_reset(np.int32(30))
    _ = build_obs(hands, bag, trash, offer, scores, state, known,
                  np.int32(0), np.int32(30))
    _ = build_mask(hands, offer, np.int32(0))
    bid0 = np.zeros(N_COLORS, dtype=np.int32)
    bid1 = np.zeros(N_COLORS, dtype=np.int32)
    _ = fast_step(hands, bag, trash, offer, scores, state, known,
                  bid0, bid1, np.int32(30), np.int32(500))


if __name__ == "__main__":
    import time
    print("Warming up JIT...")
    warmup()
    print("Benchmarking...")
    t0 = time.perf_counter()
    episodes = 1000
    total_steps = 0
    for _ in range(episodes):
        h, b, tr, o, sc, st, kn = fast_reset(np.int32(30))
        done = False
        while not done:
            mask = build_mask(h, o, np.int32(0))
            act = np.zeros(N_COLORS, dtype=np.int64)
            for c in range(N_COLORS):
                base = c * (MAX_BID + 1)
                valid = np.where(mask[base:base + MAX_BID + 1])[0]
                act[c] = np.random.choice(valid)
            
            bid0 = np.zeros(N_COLORS, dtype=np.int32)
            for c in range(N_COLORS):
                if o[c] > 0:
                    bid0[c] = 0
                else:
                    bid0[c] = min(np.int32(act[c]), h[0, c])
                    
            bid1 = _random_bid(h[1], o)
            r, term, trunc = fast_step(h, b, tr, o, sc, st, kn,
                                       bid0, bid1, np.int32(30), np.int32(500))
            done = term or trunc
            total_steps += 1
    elapsed = time.perf_counter() - t0
    print(f"{episodes} episodes, {total_steps} steps in {elapsed:.2f}s")
    print(f"Speed: {total_steps / elapsed:.0f} steps/sec")
