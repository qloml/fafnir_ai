"""
Color symmetry data augmentation for Fafnir.

Since the 5 non-gold colors (red, orange, yellow, green, blue) are
functionally identical (except for tie-breaking), we can swap their
identities to create augmented training data.

This effectively multiplies training data by up to 5! = 120x.
In practice, we use a subset of permutations for efficiency.
"""
import random
import numpy as np
from typing import List, Tuple
from itertools import permutations

from .game_engine import NUM_COLORS  # 6 (gold + 5 colors)


# The 5 non-gold colors occupy indices 1-5 in our count vectors.
# Gold (index 0) is NEVER swapped.
NON_GOLD_INDICES = [1, 2, 3, 4, 5]

# All permutations of non-gold indices (5! = 120)
ALL_PERMS = list(permutations(NON_GOLD_INDICES))

# Cached action-to-id mapping (built lazily on first use)
_ACTION_TO_ID: dict | None = None
_ACTION_TO_ID_SIZE: int = 0


def _get_action_to_id(action_table):
    """Return cached {action_tuple: id} dict, building it once on first call."""
    global _ACTION_TO_ID, _ACTION_TO_ID_SIZE
    if _ACTION_TO_ID is None or _ACTION_TO_ID_SIZE != len(action_table):
        _ACTION_TO_ID = {a: i for i, a in enumerate(action_table)}
        _ACTION_TO_ID_SIZE = len(action_table)
    return _ACTION_TO_ID


def apply_color_permutation(
    obs: np.ndarray,
    perm: Tuple[int, ...],
) -> np.ndarray:
    """
    Apply a color permutation to an observation vector (34 or 42 dim).

    perm is a permutation of [1,2,3,4,5] (non-gold indices).
    Gold (index 0) is never moved.

    Observation layout:
      [0-5]   my hand      -> swap indices 1-5
      [6-11]  offer        -> swap indices 1-5
      [12-17] trash        -> swap indices 1-5
      [18-23] opp confirmed -> swap indices 1-5
      [24]    opp unknown count -> unchanged (scalar)
      [25-30] my confirmed  -> swap indices 1-5
      [31]    bag remaining -> unchanged
      [32]    is caretaker  -> unchanged
      [33]    expected score -> unchanged
      [34-41] v2 features (scores, round, turn, totals) -> unchanged (scalars)
    """
    new_obs = obs.copy()

    # Build full index mapping: 0->0, perm[0]->1, perm[1]->2, ...
    # perm = (p1, p2, p3, p4, p5) means: old index p1 -> new index 1, etc.
    idx_map = [0] + list(perm)

    # Apply to each 6-element color block
    color_blocks = [
        (0, 6),    # my hand
        (6, 12),   # offer
        (12, 18),  # trash
        (18, 24),  # opp confirmed
        (25, 31),  # my confirmed
    ]

    for start, end in color_blocks:
        original = obs[start:end].copy()
        for new_idx in range(6):
            old_idx = idx_map[new_idx]
            new_obs[start + new_idx] = original[old_idx]

    return new_obs


def apply_color_permutation_to_action_regrets(
    regrets: np.ndarray,
    perm: Tuple[int, ...],
    action_table: List[Tuple[int, ...]],
) -> np.ndarray:
    """
    Apply color permutation to regret/strategy vectors.

    When we permute colors, action (gold=1, red=2, ...) becomes a
    different action ID. We need to remap the regret values accordingly.
    """
    new_regrets = np.zeros_like(regrets)

    # Build inverse permutation for action remapping
    inv_perm = [0] * 6
    inv_perm[0] = 0
    for new_idx, old_idx in enumerate(perm):
        inv_perm[old_idx] = new_idx + 1
    inv_perm_list = inv_perm

    # For each action, find its permuted counterpart
    action_to_id = _get_action_to_id(action_table)

    for aid, action in enumerate(action_table):
        # Permute the action
        new_action = [0] * 6
        for c in range(6):
            new_action[inv_perm_list[c]] = action[c]
        new_action_tuple = tuple(new_action)

        new_aid = action_to_id.get(new_action_tuple)
        if new_aid is not None:
            new_regrets[new_aid] = regrets[aid]

    return new_regrets


def random_color_permutation() -> Tuple[int, ...]:
    """Get a random permutation of non-gold color indices."""
    perm = list(NON_GOLD_INDICES)
    random.shuffle(perm)
    return tuple(perm)


def augment_sample(
    obs: np.ndarray,
    regrets: np.ndarray,
    action_table: List[Tuple[int, ...]],
    num_augments: int = 4,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate augmented (obs, regret) pairs by applying random color permutations.

    Returns a list of (augmented_obs, augmented_regrets) tuples.
    The original sample is NOT included in the output.
    """
    seen = set()
    results = []

    for _ in range(num_augments * 3):  # oversample to handle duplicates
        if len(results) >= num_augments:
            break

        perm = random_color_permutation()
        if perm in seen or perm == tuple(NON_GOLD_INDICES):
            continue
        seen.add(perm)

        aug_obs = apply_color_permutation(obs, perm)
        aug_regrets = apply_color_permutation_to_action_regrets(
            regrets, perm, action_table
        )
        results.append((aug_obs, aug_regrets))

    return results


def augment_sample_sparse(
    obs: np.ndarray,
    action_id: int,
    value: float,
    action_table: List[Tuple[int, ...]],
    num_augments: int = 4,
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Sparse-optimized augmentation: O(1) per permutation instead of O(NUM_ACTIONS).

    Instead of permuting a full 3003-element regret array, directly permute
    the single (action_id, value) pair.

    Returns list of (augmented_obs, new_action_id, value) tuples.
    """
    action_to_id = _get_action_to_id(action_table)
    original_action = action_table[action_id]

    seen = set()
    results = []

    for _ in range(num_augments * 3):
        if len(results) >= num_augments:
            break

        perm = random_color_permutation()
        if perm in seen or perm == tuple(NON_GOLD_INDICES):
            continue
        seen.add(perm)

        # Permute observation
        aug_obs = apply_color_permutation(obs, perm)

        # Permute the single action: O(1) instead of O(3003)
        inv_perm = [0] * 6
        inv_perm[0] = 0
        for new_idx, old_idx in enumerate(perm):
            inv_perm[old_idx] = new_idx + 1

        new_action = tuple(original_action[inv_perm[c]] for c in range(6))
        new_aid = action_to_id.get(new_action)

        if new_aid is not None:
            results.append((aug_obs, new_aid, value))

    return results

