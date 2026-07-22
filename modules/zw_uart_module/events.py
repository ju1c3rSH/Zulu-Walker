from dataclasses import dataclass, field

from .protocol import ActionId


@dataclass
class McuCmdReceived:
    cmd_id: int
    args: bytes = field(default=b"")


@dataclass
class ArrivedEvent:
    zone_id: int


@dataclass
class ActionDoneEvent:
    action_id: ActionId
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
