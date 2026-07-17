from dataclasses import dataclass
from typing import Tuple

from .color import Color


@dataclass
class RingTarget:
    coordinate: Tuple[int, int]
    color: Color
    confidence: float = 100.0
    is_predicted: bool = False
