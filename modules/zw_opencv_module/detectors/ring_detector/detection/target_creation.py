from typing import Tuple

from modules.zw_opencv_module.models.ring import RingTarget
from modules.zw_opencv_module.models.color import Color


def create_ring_target(
    center: Tuple[float, float], color: Color, confidence: float = 100.0,
    is_predicted: bool = False,
) -> RingTarget:
    cx, cy = center
    return RingTarget(
        coordinate=(int(cx), int(cy)),
        color=color,
        confidence=confidence,
        is_predicted=is_predicted,
    )
