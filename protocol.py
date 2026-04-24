# protocol.py
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import time

@dataclass
class PlayerView:
    name: str = "Player"
    score: int = 0
    hand_count: int = 0
    hand: Optional[List[str]] = None
    last_bid_count: int = 0
    last_bid: Optional[List[str]] = None

@dataclass
class GameStateView:
    room_id: str
    phase: str = "BIDDING"   # BIDDING / RESULT / ROUND_END / GAME_END
    round: int = 1
    caretaker: int = 0
    offer: List[str] = field(default_factory=list)
    trash: Dict[str, int] = field(default_factory=dict)
    bag_left: int = 0
    turn_index: int = 0
    players: List[PlayerView] = field(default_factory=list)
    status: str = "Waiting..."
    ts: float = field(default_factory=lambda: time.time())

def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
