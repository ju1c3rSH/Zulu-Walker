from typing import Optional, Tuple

from ....models.color import Color
from ....models.cargo import CargoItem


def create_cargo_item(
    center: Tuple[float, float],
    color: Color,
    index: int = 0,
    radius: Optional[float] = None,
) -> CargoItem:
    return CargoItem(
        index=index,
        color=color,
        batch=0,
        coordinate=(int(center[0]), int(center[1])),
        radius=radius,
    )
