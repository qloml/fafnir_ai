# cfr_ai/ai/fast_engine.py
"""
Numba JIT-compiled game engine for Deep CFR fast traversals.

Based on mppo_ai/rl/fast_engine.py but adapted for CFR's needs:
- Returns full game state arrays (not just rewards)
- Supports state cloning for traversal branching
- Computes hand scores for terminal value calculation

All game logic as pure functions operating on numpy arrays.
~100x faster than the pure Python game_engine.py.
"""
import numpy as np
from numba import njit

N_COLORS = 6
TRASH_LIMIT = 6
SEED_TRASH_N = 3
SCORE_TO_WIN = 1000

INITIAL_BAG = np.array([20, 12, 12, 12, 12, 12], dtype=np.int32)
BAG_TOTAL = 80


# ============================================================
# Primitives
# ============================================================

@njit(cache=True, fastmath=True)
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


@njit(cache=True, fastmath=True)
def _draw_n(bag, n):
    drawn = np.zeros(N_COLORS, dtype=np.int32)
    for _ in range(n):
        c = _draw_one(bag)
        if c < 0:
            break
        drawn[c] += np.int32(1)
    return drawn


@njit(cache=True, fastmath=True)
def _setup_offer(bag):
    offer = np.zeros(N_COLORS, dtype=np.int32)
    for _ in range(50):
        total = np.int32(0)
        for i in range(N_COLORS):
            total += bag[i]
        if total == 0:
            break
        draw_count = min(np.int32(2), total)
        for _ in range(draw_count):
            c = _draw_one(bag)
            if c >= 0:
                offer[c] += np.int32(1)
        n_colors_in_offer = np.int32(0)
        for i in range(N_COLORS):
            if offer[i] > 0:
                n_colors_in_offer += np.int32(1)
        if n_colors_in_offer > 1:
            break
        bag_left = np.int32(0)
        for i in range(N_COLORS):
            bag_left += bag[i]
        if bag_left == 0:
            break
    return offer


@njit(cache=True, fastmath=True)
def _clamp(x):
    return max(np.int32(0), x)


# ============================================================
# Scoring
# ============================================================

@njit(cache=True, fastmath=True)
def _rank_colors(hands):
    """Rank non-gold colors by total count across both players."""
    ranked_idx = np.zeros(5, dtype=np.int32)
    ranked_cnt = np.zeros(5, dtype=np.int32)
    for i in range(5):
        c = np.int32(i + 1)
        ranked_idx[i] = c
        ranked_cnt[i] = hands[0, c] + hands[1, c]
    # Bubble sort by (-count, color_priority)
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


@njit(cache=True, fastmath=True)
def hand_score(hands, player, first_c, second_c):
    """Compute hand score for a player given 1st/2nd color rankings."""
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


@njit(cache=True, fastmath=True)
def compute_hand_score_full(hands, player):
    """Compute hand score with auto-ranking."""
    ranked_idx, _ = _rank_colors(hands)
    return hand_score(hands, player, ranked_idx[0], ranked_idx[1])


# ============================================================
# Game State Management
# ============================================================

@njit(cache=True, fastmath=True)
def new_game():
    """Create a new game. Returns (hands, bag, trash, offer, scores, caretaker)."""
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
    return hands, bag, trash, offer, scores, ct


@njit(cache=True, fastmath=True)
def clone_state(hands, bag, trash, offer, scores, ct):
    """Deep copy all game state arrays."""
    return (hands.copy(), bag.copy(), trash.copy(),
            offer.copy(), scores.copy(), ct)


# ============================================================
# Auction Resolution
# ============================================================

@njit(cache=True, fastmath=True)
def resolve_auction(hands, bag, trash, offer, scores, ct, bid0, bid1):
    """
    Resolve one auction. Modifies arrays in-place.

    Returns:
        winner: int (-1 if no bid)
        new_ct: int (updated caretaker)
        round_ended: bool
        bag_low: bool
    """
    c0 = np.int32(0)
    c1 = np.int32(0)
    for c in range(N_COLORS):
        c0 += bid0[c]
        c1 += bid1[c]

    mx = max(c0, c1)
    winner = np.int32(-1)

    if mx == 0:
        # Both pass: offer → trash, both lose 1 point
        scores[0] = _clamp(scores[0] - np.int32(1))
        scores[1] = _clamp(scores[1] - np.int32(1))
        for c in range(N_COLORS):
            trash[c] += offer[c]
            offer[c] = np.int32(0)
    else:
        if c0 > c1:
            winner = np.int32(0)
        elif c1 > c0:
            winner = np.int32(1)
        else:
            # Tie: non-caretaker wins
            winner = np.int32(1) if ct == 0 else np.int32(0)

        loser = np.int32(1 - winner)
        w_bid = bid0 if winner == 0 else bid1

        for c in range(N_COLORS):
            hands[winner, c] -= w_bid[c]
            trash[c] += w_bid[c]
            hands[winner, c] += offer[c]
            offer[c] = np.int32(0)

        scores[winner] += np.int32(1)
        ct = winner

    # Check round end conditions
    trash_hit = False
    for c in range(N_COLORS):
        if trash[c] >= TRASH_LIMIT:
            trash_hit = True
            break

    bl = np.int32(0)
    for c in range(N_COLORS):
        bl += bag[c]
    bag_low = bl < 2

    round_ended = trash_hit or bag_low

    if not round_ended:
        # Setup next offer
        new_offer = _setup_offer(bag)
        for c in range(N_COLORS):
            offer[c] = new_offer[c]

    return winner, ct, round_ended


@njit(cache=True, fastmath=True)
def do_round_end_scoring(hands, scores):
    """Apply round-end hand scoring to scores. Returns (add0, add1)."""
    ranked_idx, _ = _rank_colors(hands)
    first_c = ranked_idx[0]
    second_c = ranked_idx[1]
    add0 = hand_score(hands, 0, first_c, second_c)
    add1 = hand_score(hands, 1, first_c, second_c)
    scores[0] = _clamp(scores[0] + add0)
    scores[1] = _clamp(scores[1] + add1)
    return add0, add1


@njit(cache=True, fastmath=True)
def reset_for_new_round(hands, bag, trash, offer, ct):
    """Reset game state for a new round (does NOT update scores)."""
    for c in range(N_COLORS):
        bag[c] = INITIAL_BAG[c]
        trash[c] = np.int32(0)
        hands[0, c] = np.int32(0)
        hands[1, c] = np.int32(0)
        offer[c] = np.int32(0)

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


# ============================================================
# Legal Action Mask (for CFR action space)
# ============================================================

@njit(cache=True, fastmath=True)
def get_legal_mask_fast(hand_p, offer, action_table):
    """
    Compute legal action mask using pre-built action table.

    Args:
        hand_p: shape (6,) - player's hand counts
        offer: shape (6,) - current offer counts
        action_table: shape (N, 6) - all possible bid vectors

    Returns:
        mask: shape (N,) - boolean mask
    """
    n_actions = action_table.shape[0]
    mask = np.zeros(n_actions, dtype=np.int8)

    for a in range(n_actions):
        legal = True
        for c in range(N_COLORS):
            if action_table[a, c] > hand_p[c]:
                legal = False
                break
            if offer[c] > 0 and action_table[a, c] > 0:
                legal = False
                break
        if legal:
            mask[a] = np.int8(1)

    return mask


# ============================================================
# Observation Builder (42-dim, matching cfr_ai/ai/observation.py)
# ============================================================

@njit(cache=True, fastmath=True)
def build_obs(hands, bag, trash, offer, scores, ct, player,
              confirmed, round_num, turn_num):
    """
    Build 42-dim observation vector for CFR.

    Layout matches cfr_ai/ai/observation.py:
      [0-5]   My hand
      [6-11]  Offer
      [12-17] Trash
      [18-23] Opponent confirmed hand
      [24]    Opponent unknown count
      [25-30] My confirmed hand
      [31]    Bag remaining (normalized)
      [32]    Am I caretaker?
      [33]    My expected score (normalized)
      [34]    My game score (normalized /1000)
      [35]    Opponent game score (normalized /1000)
      [36]    Round number (normalized /20)
      [37]    Turn number (normalized /30)
      [38]    Offer total (normalized /10)
      [39]    My hand total (normalized /20)
      [40]    Opponent hand total (normalized /20)
      [41]    Trash total (normalized /36)
    """
    OBS_DIM = 42
    other = np.int32(1 - player)
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # [0-5] My hand
    for c in range(N_COLORS):
        obs[c] = np.float32(hands[player, c])

    # [6-11] Offer
    for c in range(N_COLORS):
        obs[6 + c] = np.float32(offer[c])

    # [12-17] Trash
    for c in range(N_COLORS):
        obs[12 + c] = np.float32(trash[c])

    # [18-23] Opponent confirmed hand
    for c in range(N_COLORS):
        obs[18 + c] = np.float32(confirmed[other, c])

    # [24] Opponent unknown count
    opp_total = np.int32(0)
    opp_confirmed_total = np.int32(0)
    for c in range(N_COLORS):
        opp_total += hands[other, c]
        opp_confirmed_total += confirmed[other, c]
    obs[24] = np.float32(max(np.int32(0), opp_total - opp_confirmed_total))

    # [25-30] My confirmed hand
    for c in range(N_COLORS):
        obs[25 + c] = np.float32(confirmed[player, c])

    # [31] Bag remaining
    bl = np.int32(0)
    for c in range(N_COLORS):
        bl += bag[c]
    obs[31] = np.float32(bl) / np.float32(BAG_TOTAL)

    # [32] Am I caretaker?
    obs[32] = np.float32(1) if ct == player else np.float32(0)

    # [33] Expected score
    ranked_idx, _ = _rank_colors(hands)
    raw = hand_score(hands, player, ranked_idx[0], ranked_idx[1])
    obs[33] = max(np.float32(-1.0), min(np.float32(1.0),
                  (np.float32(raw) + np.float32(15.0)) / np.float32(37.5) - np.float32(1.0)))

    # [34] My game score
    obs[34] = np.float32(scores[player]) / np.float32(SCORE_TO_WIN)

    # [35] Opponent game score
    obs[35] = np.float32(scores[other]) / np.float32(SCORE_TO_WIN)

    # [36] Round number
    obs[36] = min(np.float32(round_num), np.float32(20)) / np.float32(20)

    # [37] Turn number
    obs[37] = min(np.float32(turn_num), np.float32(30)) / np.float32(30)

    # [38] Offer total
    offer_total = np.int32(0)
    for c in range(N_COLORS):
        offer_total += offer[c]
    obs[38] = np.float32(offer_total) / np.float32(10)

    # [39] My hand total
    my_total = np.int32(0)
    for c in range(N_COLORS):
        my_total += hands[player, c]
    obs[39] = np.float32(my_total) / np.float32(20)

    # [40] Opponent hand total
    obs[40] = np.float32(opp_total) / np.float32(20)

    # [41] Trash total
    trash_total = np.int32(0)
    for c in range(N_COLORS):
        trash_total += trash[c]
    obs[41] = np.float32(trash_total) / np.float32(N_COLORS * TRASH_LIMIT)

    return obs


# ============================================================
# Confirmed hand tracking (Numba-compatible)
# ============================================================

@njit(cache=True, fastmath=True)
def update_confirmed(confirmed, winner, loser, w_bid, l_bid, offer_snap):
    """Update confirmed hand tracking after an auction."""
    for c in range(N_COLORS):
        val = confirmed[winner, c] + offer_snap[c] - w_bid[c]
        confirmed[winner, c] = max(np.int32(0), val)
        if l_bid[c] > confirmed[loser, c]:
            confirmed[loser, c] = l_bid[c]


@njit(cache=True, fastmath=True)
def reset_confirmed(confirmed):
    """Reset confirmed hands for new round."""
    for p in range(2):
        for c in range(N_COLORS):
            confirmed[p, c] = np.int32(0)


# ============================================================
# Warmup (trigger JIT compilation)
# ============================================================

def warmup():
    """Pre-compile all Numba functions."""
    hands, bag, trash, offer, scores, ct = new_game()
    confirmed = np.zeros((2, N_COLORS), dtype=np.int32)

    obs = build_obs(hands, bag, trash, offer, scores, ct,
                    np.int32(0), confirmed, np.int32(1), np.int32(1))

    bid0 = np.zeros(N_COLORS, dtype=np.int32)
    bid1 = np.zeros(N_COLORS, dtype=np.int32)

    winner, new_ct, ended = resolve_auction(
        hands, bag, trash, offer, scores, ct, bid0, bid1)

    _ = compute_hand_score_full(hands, np.int32(0))
    _ = do_round_end_scoring(hands, scores)

    h2, b2, t2, o2, s2, c2 = clone_state(hands, bag, trash, offer, scores, ct)

    update_confirmed(confirmed, np.int32(0), np.int32(1), bid0, bid1, offer)
    reset_confirmed(confirmed)

    print("[CFR fast_engine] JIT warmup complete")


if __name__ == "__main__":
    import time
    print("Warming up JIT...")
    warmup()

    print("Benchmarking...")
    t0 = time.perf_counter()
    episodes = 1000
    total_steps = 0

    for _ in range(episodes):
        hands, bag, trash, offer, scores, ct = new_game()
        confirmed = np.zeros((2, N_COLORS), dtype=np.int32)
        done = False
        while not done:
            bid0 = np.zeros(N_COLORS, dtype=np.int32)
            bid1 = np.zeros(N_COLORS, dtype=np.int32)
            # Random bids (1 stone each)
            for p_bid, hand_p in [(bid0, hands[0]), (bid1, hands[1])]:
                biddable = []
                for c in range(N_COLORS):
                    if offer[c] == 0 and hand_p[c] > 0:
                        biddable.append(c)
                if biddable:
                    c_chosen = biddable[np.random.randint(0, len(biddable))]
                    p_bid[c_chosen] = np.int32(1)

            winner, ct, ended = resolve_auction(
                hands, bag, trash, offer, scores, ct, bid0, bid1)

            if ended:
                do_round_end_scoring(hands, scores)
                done = True
            total_steps += 1

    elapsed = time.perf_counter() - t0
    print(f"{episodes} episodes, {total_steps} steps in {elapsed:.2f}s")
    print(f"Speed: {total_steps / elapsed:.0f} steps/sec")
