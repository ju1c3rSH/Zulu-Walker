from dataclasses import dataclass


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
