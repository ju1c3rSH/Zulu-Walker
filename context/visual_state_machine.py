# -*- coding: utf-8 -*-
"""
视觉跟踪状态机

- 5 个状态：IDLE → SEARCH → TRACKING → RECOVERY → FAIL
- SEARCH → TRACKING 和 TRACKING → SEARCH 由 on_execute 根据帧计数自驱动
- 其余转换保持事件驱动（START, STOP, RESET）
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass, field
from time import time

from utils.state_machine.base import BaseStateMachine, State

_SEARCH_DETECT_THRESHOLD = 10
_TRACKING_LOST_THRESHOLD = 5


@dataclass
class VisualContext:
    """视觉状态机上下文"""

    confidence: float = 0.0
    target_center: Optional[tuple] = None
    target_found: bool = False
    error_code: int = 0

    consecutive_detected_frames: int = 0
    consecutive_lost_frames: int = 0

    state_entry_time: float = 0.0
    percent_error_x: int = 0
    percent_error_y: int = 0

    custom: Dict[str, Any] = field(default_factory=dict)

    def reset_stats(self):
        self.consecutive_detected_frames = 0
        self.consecutive_lost_frames = 0
        self.percent_error_x = 0
        self.percent_error_y = 0


class IdleState(State):
    def on_enter(self, context: VisualContext, from_state: str) -> None:
        from utils.debug_console import DebugConsole
        DebugConsole().set("visual_state", "IDLE")
        print("[VisualStateMachine] Enter IDLE")
        context.reset_stats()
        context.error_code = 0
        context.target_found = False

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class SearchState(State):
    def on_enter(self, context: VisualContext, from_state: str) -> None:
        from utils.debug_console import DebugConsole
        DebugConsole().set("visual_state", "SEARCH")
        print("[VisualStateMachine] Enter SEARCH")
        context.reset_stats()
        context.state_entry_time = time()

    def on_execute(self, context: VisualContext) -> Optional[str]:
        if context.consecutive_detected_frames >= _SEARCH_DETECT_THRESHOLD:
            return "TRACKING"
        return None

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class TrackingState(State):
    def on_enter(self, context: VisualContext, from_state: str) -> None:
        from utils.debug_console import DebugConsole
        DebugConsole().set("visual_state", "TRACKING")
        print("[VisualStateMachine] Enter TRACKING")
        context.consecutive_detected_frames = 0

    def on_execute(self, context: VisualContext) -> Optional[str]:
        if context.consecutive_lost_frames >= _TRACKING_LOST_THRESHOLD:
            return "SEARCH"
        return None

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class RecoveryState(State):
    def on_enter(self, context: VisualContext, from_state: str) -> None:
        from utils.debug_console import DebugConsole
        DebugConsole().set("visual_state", "RECOVERY")
        print("[VisualStateMachine] Enter RECOVERY")
        context.reset_stats()

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class FailState(State):
    def on_enter(self, context: VisualContext, from_state: str) -> None:
        from utils.debug_console import DebugConsole
        DebugConsole().set("visual_state", f"FAIL({context.error_code})")
        print(f"[VisualStateMachine] Enter FAIL (error_code={context.error_code})")

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        context.error_code = 0


class VisualStateMachine(BaseStateMachine):
    """视觉跟踪状态机。

    状态转换（自驱动部分）:
      SEARCH → TRACKING: consecutive_detected_frames >= 10
      TRACKING → SEARCH: consecutive_lost_frames >= 5

    状态转换（事件驱动部分）:
      IDLE → SEARCH: START 事件
      RECOVERY → TRACKING: TARGET_RECOVERED 事件
      RECOVERY → SEARCH: RECOVERY_FAILED 事件
      任意活动状态 → IDLE: STOP 事件
      FAIL → IDLE: RESET 事件
    """

    class States:
        IDLE = "IDLE"
        SEARCH = "SEARCH"
        TRACKING = "TRACKING"
        RECOVERY = "RECOVERY"
        FAIL = "FAIL"

    class Events:
        START = "START"
        STOP = "STOP"
        RESET = "RESET"
        TARGET_FOUND = "TARGET_FOUND"
        TARGET_LOST = "TARGET_LOST"
        TARGET_RECOVERED = "TARGET_RECOVERED"
        RECOVERY_FAILED = "RECOVERY_FAILED"

    def __init__(self):
        super().__init__()
        self.context = VisualContext()

        self._setup_states()
        self._setup_transitions()
        self.set_initial_state(self.States.IDLE)

    def _setup_states(self) -> None:
        self.register_state(self.States.IDLE, IdleState())
        self.register_state(self.States.SEARCH, SearchState())
        self.register_state(self.States.TRACKING, TrackingState())
        self.register_state(self.States.RECOVERY, RecoveryState())
        self.register_state(self.States.FAIL, FailState())

    def _setup_transitions(self) -> None:
        self.register_transition(
            self.States.IDLE, self.States.SEARCH,
            event=self.Events.START
        )

        self.register_transition(
            self.States.SEARCH, self.States.TRACKING,
            event=self.Events.TARGET_FOUND
        )

        self.register_transition(
            self.States.TRACKING, self.States.SEARCH,
            event=self.Events.TARGET_LOST
        )

        self.register_transition(
            self.States.RECOVERY, self.States.TRACKING,
            event=self.Events.TARGET_RECOVERED
        )

        self.register_transition(
            self.States.RECOVERY, self.States.SEARCH,
            event=self.Events.RECOVERY_FAILED
        )

        for state in [self.States.SEARCH, self.States.TRACKING, self.States.RECOVERY]:
            self.register_transition(
                state, self.States.IDLE,
                event=self.Events.STOP
            )

        self.register_transition(
            self.States.FAIL, self.States.IDLE,
            event=self.Events.RESET
        )

    def start(self) -> bool:
        return self.trigger(self.Events.START)

    def stop(self) -> bool:
        return self.trigger(self.Events.STOP)

    def reset_machine(self) -> bool:
        return self.trigger(self.Events.RESET)

    def is_idle(self) -> bool:
        return self.is_in_state(self.States.IDLE)

    def is_searching(self) -> bool:
        return self.is_in_state(self.States.SEARCH)

    def is_tracking(self) -> bool:
        return self.is_in_state(self.States.TRACKING)

    def is_recovering(self) -> bool:
        return self.is_in_state(self.States.RECOVERY)

    def is_failed(self) -> bool:
        return self.is_in_state(self.States.FAIL)

    def is_active(self) -> bool:
        return self.current_state in (
            self.States.SEARCH, self.States.TRACKING, self.States.RECOVERY
        )

    def get_info(self) -> dict:
        return {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "state_duration": self.state_duration,
            "state_frame_count": self.state_frame_count,
            "context": {
                "target_found": self.context.target_found,
                "confidence": self.context.confidence,
                "target_center": self.context.target_center,
                "consecutive_detected": self.context.consecutive_detected_frames,
                "consecutive_lost": self.context.consecutive_lost_frames,
                "error_code": self.context.error_code,
            }
        }
