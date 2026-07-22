from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameReady:
    camera_id: str
    frame_index: int
    timestamp: float


@dataclass
class ServoData:
    error_x: int
    error_y: int
    flags: int
    visual_state: int


@dataclass
class TargetFound:
    confidence: float
    center_x: int
    center_y: int
    color_id: Optional[int] = None


@dataclass
class TargetLost:
    pass


@dataclass
class ReadyToPick:
    pass


@dataclass
class ReadyToPlace:
    pass


@dataclass
class QRResult:
    qr_str: str


@dataclass
class FrameResult:
    all_results: dict
