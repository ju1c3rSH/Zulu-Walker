from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .color import Color


SLOT_COUNT = 3
SLOT_COLOR_MAP = {0: Color.RED, 1: Color.GREEN, 2: Color.BLUE}
COLOR_SLOT_MAP = {Color.RED: 0, Color.GREEN: 1, Color.BLUE: 2}


class CargoZone:
    RAW = 2
    ROUGH = 3
    TEMP = 4
    ON_ROBOT = 5


@dataclass
class CargoItem:
    index: int
    color: Color
    batch: int
    coordinate: Optional[Tuple[int, int]] = None
    available: bool = True
    zone: int = CargoZone.RAW

    def update_position(self, coord: Tuple[int, int]) -> None:
        self.coordinate = coord

    def move_to_zone(self, zone: int) -> None:
        self.zone = zone

    def pick(self) -> None:
        self.available = False
        self.zone = CargoZone.ON_ROBOT

    def place(self, zone: int) -> None:
        self.zone = zone

    def reset(self) -> None:
        self.available = True
        self.coordinate = None
        self.zone = CargoZone.RAW

    @property
    def is_detected(self) -> bool:
        return self.coordinate is not None

    @property
    def is_on_robot(self) -> bool:
        return self.zone == CargoZone.ON_ROBOT

    @property
    def slot_index(self) -> int:
        return COLOR_SLOT_MAP[self.color]


@dataclass
class CargoSet:
    items: List[CargoItem] = field(default_factory=list)

    @classmethod
    def create_standard(cls) -> "CargoSet":
        batch1 = [
            CargoItem(i, c, 1)
            for i, c in enumerate([Color.RED, Color.GREEN, Color.BLUE])
        ]
        batch2 = [
            CargoItem(i + 3, c, 2)
            for i, c in enumerate([Color.RED, Color.GREEN, Color.BLUE])
        ]
        return cls(items=batch1 + batch2)

    def get_by_index(self, idx: int) -> Optional[CargoItem]:
        for item in self.items:
            if item.index == idx:
                return item
        return None

    def get_by_color(self, color: Color, batch: Optional[int] = None) -> List[CargoItem]:
        result = [item for item in self.items if item.color == color]
        if batch is not None:
            result = [item for item in result if item.batch == batch]
        return result

    def get_batch(self, batch_num: int) -> List[CargoItem]:
        return [item for item in self.items if item.batch == batch_num]

    def get_available(self) -> List[CargoItem]:
        return [item for item in self.items if item.available]

    def get_detected(self) -> List[CargoItem]:
        return [item for item in self.items if item.is_detected]

    def get_by_zone(self, zone: int) -> List[CargoItem]:
        return [item for item in self.items if item.zone == zone]

    def reset_all(self) -> None:
        for item in self.items:
            item.reset()
