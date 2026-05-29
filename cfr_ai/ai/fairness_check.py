"""
Fair-play invariance checks for cfr_bot.

These checks verify that BIDDING decisions do not depend on server_0424.py
leaks such as unresolved opponent bids, opponent hands, or action_log payloads.
They are lightweight and do not require a trained checkpoint.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch

from cfr_ai.ai.action_space import NUM_ACTIONS
from cfr_ai.ai.observation import BidTracker, build_observation_from_server_state
from cfr_ai.clients import cfr_bot


ALL_TRASH = {
    "gold": 1,
    "red": 0,
    "orange": 2,
    "yellow": 1,
    "green": 0,
    "blue": 0,
}


class DummyStrategyNet:
    """Deterministic torch-compatible policy used only for invariance checks."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        base = torch.linspace(-1.0, 1.0, NUM_ACTIONS, device=obs.device)
        obs_factor = 1.0 + obs[:, 33:34] * 0.05
        return base.unsqueeze(0).repeat(obs.shape[0], 1) * obs_factor


def _base_state() -> Dict[str, Any]:
    return {
        "room_id": "fairness",
        "phase": "BIDDING",
        "round": 2,
        "turn": 4,
        "caretaker": 1,
        "current_bidder": 0,
        "offer": ["orange", "yellow"],
        "trash": dict(ALL_TRASH),
        "bag_left": 52,
        "players": [
            {
                "name": "FairCFR",
                "score": 11,
                "hand_count": 6,
                "hand": ["gold", "red", "red", "green", "blue", "blue"],
                "bid_submitted": False,
                "ok_ready": False,
                "last_bid": [],
            },
            {
                "name": "Opponent",
                "score": 9,
                "hand_count": 5,
                "hand": None,
                "bid_submitted": False,
                "ok_ready": False,
                "last_bid": None,
            },
        ],
        "last_result": {},
        "action_log": [],
    }


def _leaked_state() -> Dict[str, Any]:
    st = copy.deepcopy(_base_state())
    st["players"][1]["hand"] = ["blue", "blue", "green", "gold", "red"]
    st["players"][1]["bid_submitted"] = True
    st["players"][1]["last_bid"] = ["blue", "green"]
    st["last_result"] = {
        "winner": 1,
        "offer": ["gold", "red"],
        "bids_by_player": [[], ["blue", "green"]],
    }
    st["action_log"] = [
        {
            "kind": "submit_bid",
            "payload": {
                "player_index": 1,
                "bid": ["blue", "green"],
                "hands": [
                    ["gold", "red", "red", "green", "blue", "blue"],
                    ["blue", "blue", "green", "gold", "red"],
                ],
            },
        },
        {
            "kind": "resolve_before",
            "payload": {
                "hands": [
                    ["gold", "red", "red", "green", "blue", "blue"],
                    ["blue", "blue", "green", "gold", "red"],
                ],
                "bids": [[], ["blue", "green"]],
            },
        },
    ]
    return st


def _assert_same_list(a: List[str], b: List[str], label: str) -> None:
    if a != b:
        raise AssertionError(f"{label} differs: {a!r} != {b!r}")


def check_fair_state_view() -> None:
    cfr_bot.my_index = 0
    fair_base = cfr_bot.fair_state_view(_base_state())
    fair_leak = cfr_bot.fair_state_view(_leaked_state())
    if fair_base != fair_leak:
        raise AssertionError("fair_state_view changed when only leaked fields changed")


def check_observation_invariance() -> None:
    cfr_bot.my_index = 0
    tracker = BidTracker()
    obs_base = build_observation_from_server_state(
        cfr_bot.fair_state_view(_base_state()), 0, tracker
    )
    obs_leak = build_observation_from_server_state(
        cfr_bot.fair_state_view(_leaked_state()), 0, tracker
    )
    np.testing.assert_array_equal(obs_base, obs_leak)


def check_action_invariance() -> None:
    cfr_bot.my_index = 0
    cfr_bot.bid_tracker = BidTracker()
    cfr_bot.strategy_net = DummyStrategyNet()
    cfr_bot.device = torch.device("cpu")
    cfr_bot.TEMPERATURE = 0.3

    np.random.seed(20260529)
    action_base = cfr_bot.choose_action(_base_state())
    np.random.seed(20260529)
    action_leak = cfr_bot.choose_action(_leaked_state())

    _assert_same_list(action_base, action_leak, "chosen action")


def check_tracker_idempotence() -> None:
    cfr_bot.my_index = 0
    cfr_bot.bid_tracker = BidTracker()
    cfr_bot.prev_round = 2
    cfr_bot._last_tracker_key = None

    st = copy.deepcopy(_base_state())
    st["phase"] = "RESULT"
    st["last_result"] = {
        "winner": 0,
        "offer": ["orange", "yellow"],
        "bids_by_player": [["red", "red"], ["blue"]],
    }

    cfr_bot.update_tracker_from_state(st)
    once = copy.deepcopy(cfr_bot.bid_tracker.confirmed)
    cfr_bot.update_tracker_from_state(st)
    twice = copy.deepcopy(cfr_bot.bid_tracker.confirmed)

    if once != twice:
        raise AssertionError(f"tracker update is not idempotent: {once!r} != {twice!r}")


def run_all() -> None:
    checks = [
        check_fair_state_view,
        check_observation_invariance,
        check_action_invariance,
        check_tracker_idempotence,
    ]
    for check in checks:
        check()
        print(f"[FAIRNESS] {check.__name__}: OK")
    print("[FAIRNESS] all checks passed")


if __name__ == "__main__":
    run_all()
