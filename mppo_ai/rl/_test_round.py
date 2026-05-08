"""Quick test: verify episodes terminate at round-end, not score_to_win."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mppo_ai.rl.fast_engine import *
from mppo_ai.rl.fast_engine import _random_bid
import numpy as np

episodes = 0
total_steps = 0
for _ in range(100):
    h, b, tr, o, sc, st, kn = fast_reset(np.int32(30))
    done = False
    steps = 0
    while not done:
        bid0 = np.zeros(N_COLORS, dtype=np.int32)
        bid1 = _random_bid(h[1], o)
        r, term, trunc = fast_step(h, b, tr, o, sc, st, kn,
                                    bid0, bid1, np.int32(30), np.int32(500))
        done = term or trunc
        steps += 1
    episodes += 1
    total_steps += steps

print(f"{episodes} round-episodes, avg {total_steps/episodes:.1f} steps/episode")
print(f"(Should be ~8-15 steps, NOT 50-100)")
