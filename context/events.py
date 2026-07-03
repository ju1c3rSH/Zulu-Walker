from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class McuCmdReceived:
    cmd_id: int
    args: bytes = field(default=b"")


@dataclass
class ArrivedEvent:
    zone_id: int


@dataclass
class ActionDoneEvent:
    action_id: int
    result: int


@dataclass
class HeartbeatEvent:
    seq: int
    mission_state: int
    visual_state: int


@dataclass
class EmergencyStopEvent:
    reason: int


@dataclass
class RequestSyncEvent:
    requested_state: int


@dataclass
class FrameReady:
    camera_id: str
    frame_index: int
    timestamp: float


@dataclass
class ServoData:
    error_x: int
    error_y: int
    distance_mm: int
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
class ColorResult:
    color_id: int
    confidence: int


@dataclass
class FrameResult:
    all_results: dict     # {camera_id: {task_name: VisionResult}}
