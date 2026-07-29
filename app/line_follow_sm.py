# -*- coding: utf-8 -*-
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
from time import time

from framework.state_machine.base import BaseStateMachine, State
from framework.log import fw_log as log_print


LineFollowStateNames = {
    0: "IDLE",
    1: "WAIT_START",
    2: "LINE_FOLLOW",
    3: "FINISHED",
    4: "ERROR",
}


@dataclass
class LineFollowContext:
    error_code: int = 0
    error_msg: str = ""
    state_entry_time: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        self.error_code = 0
        self.error_msg = ""


class IdleState(State):
    def on_enter(self, context: LineFollowContext, from_state: str) -> None:
        try:
            from utils.debug_console import DebugConsole
            DebugConsole().set("line_follow_state", "IDLE")
        except ImportError:
            pass
        log_print("[LineFollowSM] Enter IDLE")

    def on_execute(self, context: LineFollowContext) -> Optional[str]:
        return None

    def on_exit(self, context: LineFollowContext, to_state: str) -> None:
        pass


class WaitStartState(State):
    def on_enter(self, context: LineFollowContext, from_state: str) -> None:
        log_print("[LineFollowSM] Enter WAIT_START")

    def on_execute(self, context: LineFollowContext) -> Optional[str]:
        return None

    def on_exit(self, context: LineFollowContext, to_state: str) -> None:
        pass


class LineFollowState(State):
    def on_enter(self, context: LineFollowContext, from_state: str) -> None:
        log_print("[LineFollowSM] Enter LINE_FOLLOW")

    def on_execute(self, context: LineFollowContext) -> Optional[str]:
        return None

    def on_exit(self, context: LineFollowContext, to_state: str) -> None:
        pass


class FinishedState(State):
    def on_enter(self, context: LineFollowContext, from_state: str) -> None:
        log_print("[LineFollowSM] Enter FINISHED")

    def on_execute(self, context: LineFollowContext) -> Optional[str]:
        return None

    def on_exit(self, context: LineFollowContext, to_state: str) -> None:
        pass


class ErrorState(State):
    def on_enter(self, context: LineFollowContext, from_state: str) -> None:
        log_print(f"[LineFollowSM] Enter ERROR (error_code={context.error_code})")

    def on_execute(self, context: LineFollowContext) -> Optional[str]:
        return None

    def on_exit(self, context: LineFollowContext, to_state: str) -> None:
        context.error_code = 0
        context.error_msg = ""


class Ti2026StateMachine(BaseStateMachine):

    class States:
        IDLE = "IDLE"
        WAIT_START = "WAIT_START"
        LINE_FOLLOW = "LINE_FOLLOW"
        FINISHED = "FINISHED"
        ERROR = "ERROR"

    class Events:
        START = "START"
        FINISHED = "FINISHED"
        ERROR = "ERROR"
        RESET = "RESET"

    def __init__(self):
        super().__init__()
        self.context = LineFollowContext()

        self._state_id_to_name = LineFollowStateNames
        self._name_to_state_id = {v: k for k, v in LineFollowStateNames.items()}

        self._setup_states()
        self._setup_transitions()
        self.set_initial_state(self.States.WAIT_START)

    def _setup_states(self) -> None:
        self.register_state(self.States.IDLE, IdleState())
        self.register_state(self.States.WAIT_START, WaitStartState())
        self.register_state(self.States.LINE_FOLLOW, LineFollowState())
        self.register_state(self.States.FINISHED, FinishedState())
        self.register_state(self.States.ERROR, ErrorState())

    def _setup_transitions(self) -> None:
        self.register_transition(
            self.States.IDLE, self.States.WAIT_START,
            event=self.Events.START
        )
        self.register_transition(
            self.States.WAIT_START, self.States.LINE_FOLLOW,
            event=self.Events.START
        )
        self.register_transition(
            self.States.LINE_FOLLOW, self.States.FINISHED,
            event=self.Events.FINISHED
        )
        for state in [self.States.LINE_FOLLOW, self.States.WAIT_START, self.States.FINISHED]:
            self.register_transition(
                state, self.States.ERROR,
                event=self.Events.ERROR
            )
        self.register_transition(
            self.States.ERROR, self.States.IDLE,
            event=self.Events.RESET
        )

    def start(self) -> bool:
        return self.trigger(self.Events.START)

    def on_finished(self) -> bool:
        return self.trigger(self.Events.FINISHED)

    def set_error(self, code: int, msg: str = "") -> bool:
        self.context.error_code = code
        self.context.error_msg = msg
        return self.trigger(self.Events.ERROR)

    def reset_machine(self) -> bool:
        self.context.reset()
        return self.trigger(self.Events.RESET)

    @property
    def current_state_id(self) -> int:
        name = self.current_state
        for sid, sname in self._state_id_to_name.items():
            if sname == name:
                return sid
        return -1
