from typing import Optional, Tuple

from ....models.color import Color
from ....models.cargo import CargoItem


def create_cargo_item(
    center: Tuple[float, float],
    color: Color,
    index: int = 0,
    radius: Optional[float] = None,
    is_predicted: bool = False,
    confidence: float = 100.0,
) -> CargoItem:
    return CargoItem(
        index=index,
        color=color,
        batch=0,
        coordinate=(int(center[0]), int(center[1])),
        radius=radius,
        is_predicted=is_predicted,
        confidence=confidence,
    )
