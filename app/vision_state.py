from enum import IntEnum


class VisionState(IntEnum):
    IDLE = 0
    CALIB = 1
    STREAMING = 2
    ERROR = 3
