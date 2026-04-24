# rl/fast_engine.py
"""
Numba JIT-compiled FAFNIR game engine for fast RL training.
All game logic as pure functions operating on numpy arrays.
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


# ---------- round end ----------

@njit(cache=True)
def _do_round_end(hands, bag, trash, offer, scores, state):
    ranked_idx, _ = _rank_colors(hands)
    first_c = ranked_idx[0]
    second_c = ranked_idx[1]
    add0 = _hand_score(hands, 0, first_c, second_c)
    add1 = _hand_score(hands, 1, first_c, second_c)
    scores[0] = _clamp(scores[0] + add0)
    scores[1] = _clamp(scores[1] + add1)
    round_reward = np.float32(add0 - add1) * np.float32(0.02)

    # Reset
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
    return round_reward


# ---------- obs / mask ----------

@njit(cache=True)
def build_obs(hands, bag, trash, offer, scores, state, player, score_to_win):
    other = np.int32(1 - player)
    obs = np.zeros(25, dtype=np.float32)
    for c in range(N_COLORS):
        obs[c] = np.float32(hands[player, c]) / max(np.float32(1), np.float32(INITIAL_BAG[c]))
    for c in range(N_COLORS):
        obs[6 + c] = np.float32(offer[c]) / np.float32(2)
    for c in range(N_COLORS):
        obs[12 + c] = np.float32(trash[c]) / np.float32(TRASH_LIMIT)
    obs[18] = np.float32(scores[player]) / max(np.float32(1), np.float32(score_to_win))
    obs[19] = np.float32(scores[other]) / max(np.float32(1), np.float32(score_to_win))
    oht = np.int32(0)
    for c in range(N_COLORS):
        oht += hands[other, c]
    obs[20] = np.float32(oht) / np.float32(15)
    bl = np.int32(0)
    for c in range(N_COLORS):
        bl += bag[c]
    obs[21] = np.float32(bl) / np.float32(BAG_TOTAL)
    obs[22] = np.float32(1) if state[S_CT] == player else np.float32(0)
    obs[23] = min(np.float32(state[S_RND]) / np.float32(20), np.float32(1))
    obs[24] = np.float32(1) if state[S_LAST_W] == player else np.float32(0)
    for i in range(25):
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
    return hands, bag, trash, offer, scores, state


@njit(cache=True)
def fast_step(hands, bag, trash, offer, scores, state,
              bid0, bid1, score_to_win, max_turns):
    """Returns (reward, terminated, truncated). Modifies arrays in-place."""
    state[S_TOTAL] += np.int32(1)
    ct = state[S_CT]

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
        for c in range(N_COLORS):
            hands[winner, c] -= w_bid[c]
            trash[c] += w_bid[c]
            hands[winner, c] += offer[c]
            offer[c] = np.int32(0)
        scores[winner] += np.int32(1)
        state[S_CT] = winner

    state[S_LAST_W] = winner
    reward = np.float32(scores[0] - s0_before - (scores[1] - s1_before)) * np.float32(0.02)

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
        rr = _do_round_end(hands, bag, trash, offer, scores, state)
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
    hands, bag, trash, offer, scores, state = fast_reset(np.int32(30))
    _ = build_obs(hands, bag, trash, offer, scores, state, np.int32(0), np.int32(30))
    _ = build_mask(hands, offer, np.int32(0))
    bid0 = np.zeros(N_COLORS, dtype=np.int32)
    bid1 = np.zeros(N_COLORS, dtype=np.int32)
    _ = fast_step(hands, bag, trash, offer, scores, state, bid0, bid1, np.int32(30), np.int32(500))


if __name__ == "__main__":
    import time
    print("Warming up JIT...")
    warmup()
    print("Benchmarking...")
    t0 = time.perf_counter()
    episodes = 1000
    total_steps = 0
    for _ in range(episodes):
        h, b, tr, o, sc, st = fast_reset(np.int32(30))
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
            r, term, trunc = fast_step(h, b, tr, o, sc, st, bid0, bid1, np.int32(30), np.int32(500))
            done = term or trunc
            total_steps += 1
    elapsed = time.perf_counter() - t0
    print(f"{episodes} episodes, {total_steps} steps in {elapsed:.2f}s")
    print(f"Speed: {total_steps / elapsed:.0f} steps/sec")
