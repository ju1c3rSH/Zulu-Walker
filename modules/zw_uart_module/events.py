from dataclasses import dataclass


@dataclass
class HeartbeatEvent:
    seq: int
    mission_state: int
    visual_state: int


@dataclass
class EmergencyStopEvent:
    reason: int


@dataclass
class CmdRequestEvent:
    data_type: int
    min_interval_ms: int
    reserved: int = 0   # EOD2026: expected task-code id (digit1/2/3) for device streams


@dataclass
class CmdStopEvent:
    pass
