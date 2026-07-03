# -*- coding: utf-8 -*-
"""
Mission-level state machine for the logistics robot competition.

This state machine runs on both Orange Pi and STM32. It describes the
high-level task flow: read QR code, pick/place materials in order,
transfer between raw/rough/temp zones, stack, and return home.

Synchronization rules:
- STM32 is the master of mission flow (odometry + actuators).
- Orange Pi mirrors this state machine and updates it via UART frames.
- Whoever triggers a transition must notify the other side.
- On state conflict or lost heartbeat, both sides enter ERROR.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from time import time

from utils.state_machine.base import BaseStateMachine, State
from modules.zw_opencv_module.models.color import Color
from modules.zw_uart_module.protocol import VisualFlags


# ===== Shared constants (must match STM32 firmware) =====

class MissionState:
    """Mission state IDs — must stay in sync with STM32 and protocol.py"""
    IDLE = 0
    WAIT_START = 1
    READ_QR = 2
    NAV_TO_RAW = 3
    ALIGN_RAW = 4
    PICK_RAW = 5
    CHECK_LOAD = 6
    NAV_TO_ROUGH = 7
    ALIGN_ROUGH = 8
    PLACE_ROUGH = 9
    # 这里可能要加中间State
    NAV_TO_TEMP = 10
    ALIGN_TEMP = 11
    PLACE_TEMP = 12
    NAV_TO_RAW_SECOND = 13
    ALIGN_RAW_SECOND = 14
    PICK_RAW_SECOND = 15
    CHECK_LOAD_SECOND = 16
    NAV_TO_ROUGH_SECOND = 17
    ALIGN_ROUGH_SECOND = 18
    PLACE_ROUGH_SECOND = 19
    NAV_TO_TEMP_SECOND = 20
    ALIGN_TEMP_SECOND = 21
    PLACE_TEMP_STACK = 22
    RETURN_HOME = 23
    FINISHED = 24
    ERROR = 25


class VisualState:
    """Visual sub-state IDs — must stay in sync with protocol.py"""
    IDLE = 0
    SEARCH = 1
    TRACKING = 2
    RECOVERY = 3
    FAIL = 4


class Zone:
    """Zone IDs used in ARRIVED / PICK / SET frames"""
    START = 0
    QR_BOARD = 1
    RAW = 2
    ROUGH = 3
    TEMP = 4


# ===== Context =====

@dataclass
class MissionContext:
    """Shared context across mission states."""

    # Task from QR code, e.g. "123+231"
    qr_result: str = ""

    # Parsed task queues
    first_batch: List[int] = field(default_factory=list)
    second_batch: List[int] = field(default_factory=list)

    # Progress
    current_batch: int = 0          # 0=none, 1=first, 2=second
    current_index: int = 0          # index within batch (0..2)
    cargo_count: int = 0            # materials currently on robot (0..3)

    # Visual feedback
    visual_state: int = VisualState.IDLE
    visual_flags: int = 0
    target_color: int = 0
    target_found: bool = False
    ready_to_pick: bool = False
    ready_to_place: bool = False
    visual_fail: bool = False
    cargo_confirmed: bool = False
    color_mismatch: bool = False

    # Zone / navigation
    current_zone: int = Zone.START
    last_arrived_zone: int = Zone.START

    # Errors / timeouts
    error_code: int = 0
    error_msg: str = ""
    state_entry_time: float = 0.0

    # Custom data
    custom: Dict[str, Any] = field(default_factory=dict)

    def reset(self):
        """Reset context to initial values (keep QR result optionally)."""
        self.first_batch.clear()
        self.second_batch.clear()
        self.current_batch = 0
        self.current_index = 0
        self.cargo_count = 0
        self.visual_state = VisualState.IDLE
        self.visual_flags = 0
        self.target_color = 0
        self.target_found = False
        self.ready_to_pick = False
        self.ready_to_place = False
        self.visual_fail = False
        self.cargo_confirmed = False
        self.color_mismatch = False
        self.current_zone = Zone.START
        self.last_arrived_zone = Zone.START
        self.error_code = 0
        self.error_msg = ""

    def parse_qr(self, qr_str: str) -> bool:
        """Parse QR string like '123+231' into batches."""
        parts = qr_str.strip().split('+')
        if len(parts) != 2:
            return False
        try:
            self.first_batch = [int(c) for c in parts[0] if c in '123']
            self.second_batch = [int(c) for c in parts[1] if c in '123']
            if len(self.first_batch) != 3 or len(self.second_batch) != 3:
                return False
        except ValueError:
            return False
        self.qr_result = qr_str
        return True

    def current_target_color(self) -> Optional[int]:
        """Return the color that should be picked/placed now."""
        batch = self._current_batch_list()
        if not batch or self.current_index >= len(batch):
            return None
        return batch[self.current_index]

    def advance_target(self):
        """Move to next target in current batch."""
        batch = self._current_batch_list()
        if batch and self.current_index < len(batch) - 1:
            self.current_index += 1
        else:
            self.current_index = 0

    def is_batch_complete(self) -> bool:
        """True when all 3 materials in current batch are handled."""
        batch = self._current_batch_list()
        return self.current_index >= len(batch) - 1 and self.cargo_count == 0

    def _current_batch_list(self) -> List[int]:
        if self.current_batch == 1:
            return self.first_batch
        if self.current_batch == 2:
            return self.second_batch
        return []

    def update_visual_flags(self, flags: int):
        self.visual_flags = flags
        self.target_found = bool(flags & VisualFlags.TARGET_FOUND)
        self.ready_to_pick = bool(flags & VisualFlags.READY_TO_PICK)
        self.ready_to_place = bool(flags & VisualFlags.READY_TO_PLACE)
        self.visual_fail = bool(flags & VisualFlags.VISUAL_FAIL)
        self.cargo_confirmed = bool(flags & VisualFlags.CARGO_CONFIRMED)
        self.color_mismatch = bool(flags & VisualFlags.COLOR_MISMATCH)


# ===== State implementations =====

class _IdleState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter IDLE")
        ctx.reset()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _WaitStartState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter WAIT_START")

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _ReadQrState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter READ_QR")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        # Transition triggered externally by on_qr_result() -> QR_OK event
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _NavToRawState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter NAV_TO_RAW")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        # STM32 arrives at RAW zone -> triggers ARRIVED_RAW
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _AlignRawState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter ALIGN_RAW")
        ctx.state_entry_time = time()
        ctx.ready_to_pick = False
        ctx.target_found = False

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        # Transition to PICK_RAW when visual reports READY_TO_PICK
        if ctx.ready_to_pick and not ctx.color_mismatch:
            return MissionStateNames.PICK_RAW
        if ctx.visual_fail:
            return MissionStateNames.ERROR
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _PickRawState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter PICK_RAW")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        # STM32 finishes pick action -> ACTION_DONE
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _CheckLoadState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter CHECK_LOAD")
        ctx.state_entry_time = time()
        ctx.cargo_confirmed = False

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        # STM32 checks load sensors; visual can also confirm.
        if ctx.cargo_confirmed:
            ctx.cargo_count += 1
            return MissionStateNames.NAV_TO_ROUGH
        if ctx.state_entry_time + 3.0 < time() and not ctx.cargo_confirmed:
            ctx.error_code = 1
            return MissionStateNames.ERROR
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _NavToRoughState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter NAV_TO_ROUGH")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _AlignRoughState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter ALIGN_ROUGH")
        ctx.state_entry_time = time()
        ctx.ready_to_place = False

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        if ctx.ready_to_place:
            return MissionStateNames.PLACE_ROUGH
        if ctx.visual_fail:
            return MissionStateNames.ERROR
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _PlaceRoughState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter PLACE_ROUGH")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        if ctx.cargo_count > 0:
            ctx.cargo_count -= 1


class _NavToTempState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter NAV_TO_TEMP")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _AlignTempState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter ALIGN_TEMP")
        ctx.state_entry_time = time()
        ctx.ready_to_place = False

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        if ctx.ready_to_place:
            return MissionStateNames.PLACE_TEMP
        if ctx.visual_fail:
            return MissionStateNames.ERROR
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _PlaceTempState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter PLACE_TEMP")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        if ctx.cargo_count > 0:
            ctx.cargo_count -= 1


class _ReturnHomeState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter RETURN_HOME")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        if ctx.current_zone == Zone.START:
            return MissionStateNames.FINISHED
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _FinishedState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter FINISHED")

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


class _ErrorState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print(f"[MissionSM] Enter ERROR code={ctx.error_code} msg={ctx.error_msg}")

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        ctx.error_code = 0
        ctx.error_msg = ""


# String name mapping for BaseStateMachine
MissionStateNames = {
    MissionState.IDLE: "IDLE",
    MissionState.WAIT_START: "WAIT_START",
    MissionState.READ_QR: "READ_QR",
    MissionState.NAV_TO_RAW: "NAV_TO_RAW",
    MissionState.ALIGN_RAW: "ALIGN_RAW",
    MissionState.PICK_RAW: "PICK_RAW",
    MissionState.CHECK_LOAD: "CHECK_LOAD",
    MissionState.NAV_TO_ROUGH: "NAV_TO_ROUGH",
    MissionState.ALIGN_ROUGH: "ALIGN_ROUGH",
    MissionState.PLACE_ROUGH: "PLACE_ROUGH",
    MissionState.NAV_TO_TEMP: "NAV_TO_TEMP",
    MissionState.ALIGN_TEMP: "ALIGN_TEMP",
    MissionState.PLACE_TEMP: "PLACE_TEMP",
    MissionState.NAV_TO_RAW_SECOND: "NAV_TO_RAW_SECOND",
    MissionState.ALIGN_RAW_SECOND: "ALIGN_RAW_SECOND",
    MissionState.PICK_RAW_SECOND: "PICK_RAW_SECOND",
    MissionState.CHECK_LOAD_SECOND: "CHECK_LOAD_SECOND",
    MissionState.NAV_TO_ROUGH_SECOND: "NAV_TO_ROUGH_SECOND",
    MissionState.ALIGN_ROUGH_SECOND: "ALIGN_ROUGH_SECOND",
    MissionState.PLACE_ROUGH_SECOND: "PLACE_ROUGH_SECOND",
    MissionState.NAV_TO_TEMP_SECOND: "NAV_TO_TEMP_SECOND",
    MissionState.ALIGN_TEMP_SECOND: "ALIGN_TEMP_SECOND",
    MissionState.PLACE_TEMP_STACK: "PLACE_TEMP_STACK",
    MissionState.RETURN_HOME: "RETURN_HOME",
    MissionState.FINISHED: "FINISHED",
    MissionState.ERROR: "ERROR",
}


class MissionStateMachine(BaseStateMachine):
    """
    Mission-level state machine for logistics robot.

    Mirrors on Orange Pi and STM32. Events can be triggered locally
    or by incoming UART frames.
    """

    class Events:
        START = "START"                         # WAIT_START -> READ_QR
        QR_OK = "QR_OK"                         # READ_QR -> NAV_TO_RAW
        ARRIVED_RAW = "ARRIVED_RAW"             # NAV_TO_RAW -> ALIGN_RAW
        READY_TO_PICK = "READY_TO_PICK"         # ALIGN_RAW -> PICK_RAW
        PICK_DONE = "PICK_DONE"                 # PICK_RAW -> CHECK_LOAD
        LOAD_CONFIRMED = "LOAD_CONFIRMED"       # CHECK_LOAD -> NAV_TO_ROUGH
        ARRIVED_ROUGH = "ARRIVED_ROUGH"         # NAV_TO_ROUGH -> ALIGN_ROUGH
        READY_TO_PLACE = "READY_TO_PLACE"       # ALIGN_ROUGH -> PLACE_ROUGH
        PLACE_DONE = "PLACE_DONE"               # PLACE_ROUGH -> next
        ARRIVED_TEMP = "ARRIVED_TEMP"           # NAV_TO_TEMP -> ALIGN_TEMP
        ALL_PLACED = "ALL_PLACED"               # batch done -> next phase
        RETURNED_HOME = "RETURNED_HOME"         # RETURN_HOME -> FINISHED
        RESET = "RESET"                         # ERROR -> IDLE
        ERROR = "ERROR"                         # any -> ERROR

    def __init__(self):
        super().__init__()
        self.context = MissionContext()
        self._state_id_to_name: Dict[int, str] = MissionStateNames
        self._name_to_state_id: Dict[str, int] = {v: k for k, v in MissionStateNames.items()}

        self._setup_states()
        self._setup_transitions()
        self.set_initial_state(MissionStateNames[MissionState.WAIT_START])

    def _setup_states(self) -> None:
        self.register_state(MissionStateNames[MissionState.IDLE], _IdleState())
        self.register_state(MissionStateNames[MissionState.WAIT_START], _WaitStartState())
        self.register_state(MissionStateNames[MissionState.READ_QR], _ReadQrState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_RAW], _NavToRawState())
        self.register_state(MissionStateNames[MissionState.ALIGN_RAW], _AlignRawState())
        self.register_state(MissionStateNames[MissionState.PICK_RAW], _PickRawState())
        self.register_state(MissionStateNames[MissionState.CHECK_LOAD], _CheckLoadState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_ROUGH], _NavToRoughState())
        self.register_state(MissionStateNames[MissionState.ALIGN_ROUGH], _AlignRoughState())
        self.register_state(MissionStateNames[MissionState.PLACE_ROUGH], _PlaceRoughState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_TEMP], _NavToTempState())
        self.register_state(MissionStateNames[MissionState.ALIGN_TEMP], _AlignTempState())
        self.register_state(MissionStateNames[MissionState.PLACE_TEMP], _PlaceTempState())
        self.register_state(MissionStateNames[MissionState.RETURN_HOME], _ReturnHomeState())
        self.register_state(MissionStateNames[MissionState.FINISHED], _FinishedState())
        self.register_state(MissionStateNames[MissionState.ERROR], _ErrorState())

    def _setup_transitions(self) -> None:
        # WAIT_START -> READ_QR
        self.register_transition(
            MissionStateNames[MissionState.WAIT_START],
            MissionStateNames[MissionState.READ_QR],
            event=self.Events.START
        )

        # READ_QR -> NAV_TO_RAW
        self.register_transition(
            MissionStateNames[MissionState.READ_QR],
            MissionStateNames[MissionState.NAV_TO_RAW],
            event=self.Events.QR_OK
        )

        # NAV_TO_RAW -> ALIGN_RAW
        self.register_transition(
            MissionStateNames[MissionState.NAV_TO_RAW],
            MissionStateNames[MissionState.ALIGN_RAW],
            event=self.Events.ARRIVED_RAW
        )

        # ALIGN_RAW -> PICK_RAW
        self.register_transition(
            MissionStateNames[MissionState.ALIGN_RAW],
            MissionStateNames[MissionState.PICK_RAW],
            event=self.Events.READY_TO_PICK
        )

        # PICK_RAW -> CHECK_LOAD
        self.register_transition(
            MissionStateNames[MissionState.PICK_RAW],
            MissionStateNames[MissionState.CHECK_LOAD],
            event=self.Events.PICK_DONE
        )

        # CHECK_LOAD -> NAV_TO_ROUGH
        self.register_transition(
            MissionStateNames[MissionState.CHECK_LOAD],
            MissionStateNames[MissionState.NAV_TO_ROUGH],
            event=self.Events.LOAD_CONFIRMED
        )

        # NAV_TO_ROUGH -> ALIGN_ROUGH
        self.register_transition(
            MissionStateNames[MissionState.NAV_TO_ROUGH],
            MissionStateNames[MissionState.ALIGN_ROUGH],
            event=self.Events.ARRIVED_ROUGH
        )

        # ALIGN_ROUGH -> PLACE_ROUGH
        self.register_transition(
            MissionStateNames[MissionState.ALIGN_ROUGH],
            MissionStateNames[MissionState.PLACE_ROUGH],
            event=self.Events.READY_TO_PLACE
        )

        # PLACE_ROUGH -> NAV_TO_TEMP (simplified: after one place)
        self.register_transition(
            MissionStateNames[MissionState.PLACE_ROUGH],
            MissionStateNames[MissionState.NAV_TO_TEMP],
            event=self.Events.PLACE_DONE
        )

        # NAV_TO_TEMP -> ALIGN_TEMP
        self.register_transition(
            MissionStateNames[MissionState.NAV_TO_TEMP],
            MissionStateNames[MissionState.ALIGN_TEMP],
            event=self.Events.ARRIVED_TEMP
        )

        # ALIGN_TEMP -> PLACE_TEMP
        self.register_transition(
            MissionStateNames[MissionState.ALIGN_TEMP],
            MissionStateNames[MissionState.PLACE_TEMP],
            event=self.Events.READY_TO_PLACE
        )

        # PLACE_TEMP -> RETURN_HOME (simplified full run)
        self.register_transition(
            MissionStateNames[MissionState.PLACE_TEMP],
            MissionStateNames[MissionState.RETURN_HOME],
            event=self.Events.ALL_PLACED
        )

        # RETURN_HOME -> FINISHED
        self.register_transition(
            MissionStateNames[MissionState.RETURN_HOME],
            MissionStateNames[MissionState.FINISHED],
            event=self.Events.RETURNED_HOME
        )

        # ERROR -> WAIT_START (reset for next run)
        self.register_transition(
            MissionStateNames[MissionState.ERROR],
            MissionStateNames[MissionState.WAIT_START],
            event=self.Events.RESET
        )

        # IDLE -> WAIT_START (after explicit reset or pre-init)
        self.register_transition(
            MissionStateNames[MissionState.IDLE],
            MissionStateNames[MissionState.WAIT_START],
            event=self.Events.START
        )

        # Any active state -> ERROR
        for sid in [
            MissionState.WAIT_START, MissionState.READ_QR,
            MissionState.NAV_TO_RAW, MissionState.ALIGN_RAW, MissionState.PICK_RAW,
            MissionState.CHECK_LOAD, MissionState.NAV_TO_ROUGH,
            MissionState.ALIGN_ROUGH, MissionState.PLACE_ROUGH,
            MissionState.NAV_TO_TEMP, MissionState.ALIGN_TEMP,
            MissionState.PLACE_TEMP, MissionState.RETURN_HOME,
        ]:
            self.register_transition(
                MissionStateNames[sid],
                MissionStateNames[MissionState.ERROR],
                event=self.Events.ERROR
            )

    # ===== Public event handlers =====

    def start(self) -> bool:
        """Start button pressed."""
        return self.trigger(self.Events.START)

    def on_qr_result(self, qr_str: str) -> bool:
        """Handle QR code result from vision."""
        if not self.context.parse_qr(qr_str):
            self.context.error_code = 10
            self.context.error_msg = f"Invalid QR: {qr_str}"
            return self.trigger(self.Events.ERROR)
        self.context.current_batch = 1
        self.context.current_index = 0
        return self.trigger(self.Events.QR_OK)

    def on_arrived(self, zone_id: int) -> bool:
        """Handle ARRIVED_AT_ZONE from STM32."""
        self.context.last_arrived_zone = zone_id
        self.context.current_zone = zone_id

        state = self.current_state
        if zone_id == Zone.RAW:
            if state in (MissionStateNames[MissionState.NAV_TO_RAW], MissionStateNames[MissionState.NAV_TO_RAW_SECOND]):
                return self.trigger(self.Events.ARRIVED_RAW)
        elif zone_id == Zone.ROUGH:
            if state in (MissionStateNames[MissionState.NAV_TO_ROUGH], MissionStateNames[MissionState.NAV_TO_ROUGH_SECOND]):
                return self.trigger(self.Events.ARRIVED_ROUGH)
        elif zone_id == Zone.TEMP:
            if state in (MissionStateNames[MissionState.NAV_TO_TEMP], MissionStateNames[MissionState.NAV_TO_TEMP_SECOND]):
                return self.trigger(self.Events.ARRIVED_TEMP)
        elif zone_id == Zone.START:
            if state == MissionStateNames[MissionState.RETURN_HOME]:
                return self.trigger(self.Events.RETURNED_HOME)
        return False

    def on_action_done(self, action_id: int, result: int) -> bool:
        """Handle ACTION_DONE from STM32."""
        if result != 0:  # not OK
            self.context.error_code = action_id
            self.context.error_msg = f"Action {action_id} failed with {result}"
            return self.trigger(self.Events.ERROR)

        state = self.current_state
        if state == MissionStateNames[MissionState.PICK_RAW] and action_id == 1:
            return self.trigger(self.Events.PICK_DONE)
        if state == MissionStateNames[MissionState.PLACE_ROUGH] and action_id == 2:
            return self.trigger(self.Events.PLACE_DONE)
        if state == MissionStateNames[MissionState.PLACE_TEMP] and action_id == 3:
            return self.trigger(self.Events.PLACE_DONE)
        return False

    def on_visual_status(self, visual_state: int, flags: int) -> bool:
        """Handle STATUS_FROM_VISION frame."""
        self.context.visual_state = visual_state
        self.context.update_visual_flags(flags)

        state = self.current_state
        if self.context.visual_fail:
            self.context.error_code = 30
            self.context.error_msg = "Visual failure"
            return self.trigger(self.Events.ERROR)

        if state == MissionStateNames[MissionState.ALIGN_RAW] and self.context.ready_to_pick:
            return self.trigger(self.Events.READY_TO_PICK)
        if state in (MissionStateNames[MissionState.ALIGN_ROUGH], MissionStateNames[MissionState.ALIGN_TEMP]) and self.context.ready_to_place:
            return self.trigger(self.Events.READY_TO_PLACE)
        if state == MissionStateNames[MissionState.CHECK_LOAD] and self.context.cargo_confirmed:
            return self.trigger(self.Events.LOAD_CONFIRMED)

        return False

    def set_error(self, code: int, msg: str) -> bool:
        """Force error state."""
        self.context.error_code = code
        self.context.error_msg = msg
        return self.trigger(self.Events.ERROR)

    def reset_machine(self) -> bool:
        """Reset from ERROR to IDLE."""
        return self.trigger(self.Events.RESET)

    # ===== Helpers =====

    @property
    def current_state_id(self) -> int:
        return self._name_to_state_id.get(self.current_state, MissionState.ERROR)

    def is_in_state_id(self, state_id: int) -> bool:
        return self.current_state == MissionStateNames.get(state_id)

    def get_info(self) -> dict:
        return {
            "state": self.current_state,
            "state_id": self.current_state_id,
            "previous_state": self.previous_state,
            "qr": self.context.qr_result,
            "batch": self.context.current_batch,
            "index": self.context.current_index,
            "cargo_count": self.context.cargo_count,
            "visual_state": self.context.visual_state,
            "visual_flags": self.context.visual_flags,
            "zone": self.context.current_zone,
            "duration": self.state_duration,
        }


if __name__ == "__main__":
    sm = MissionStateMachine()
    print("Initial:", sm.get_info())

    sm.start()
    print("After START:", sm.get_info())

    sm.on_qr_result("123+231")
    print("After QR:", sm.get_info())
    print("First batch:", sm.context.first_batch)
    print("Second batch:", sm.context.second_batch)
