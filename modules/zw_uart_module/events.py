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


@dataclass
class CmdStopEvent:
    pass
