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
from modules.zw_opencv_module.models.cargo import CargoSet
from modules.zw_uart_module.protocol import ActionId, VisualFlags


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
    NAV_TO_TEMP_SECOND = 20
    PLACE_TEMP_STACK = 22
    RETURN_HOME = 23
    FINISHED = 24
    ERROR = 25
    PICK_ROUGH = 26


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

    # Cargo tracking (global single instance)
    cargo_set: Optional[CargoSet] = None

    # Batch order parsed from QR, e.g. [Color.RED, Color.GREEN, Color.BLUE]
    current_batch_order: List[Color] = field(default_factory=list)
    first_batch_order: List[Color] = field(default_factory=list)
    second_batch_order: List[Color] = field(default_factory=list)

    # Progress
    current_batch: int = 0          # 1=first, 2=second
    current_step: int = 0           # 0→1→2 within current batch
    cargo_count: int = 0            # materials currently on robot (0..3)
    picking_from_rough: bool = False  # ROUGH 区处于取料阶段(True)还是放料阶段(False)
    place_action_done: bool = False   # PLACE 状态下 MCU 动作完成

    # Visual feedback
    visual_state: int = VisualState.IDLE
    visual_flags: int = 0
    target_color: Color = Color.RED
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
        self.current_batch_order.clear()
        self.current_batch = 0
        self.current_step = 0
        self.cargo_count = 0
        self.picking_from_rough = False
        self.place_action_done = False
        if self.cargo_set:
            self.cargo_set.reset_all()
        self.visual_state = VisualState.IDLE
        self.visual_flags = 0
        self.target_color = Color.RED
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
        """Parse QR string like '123+231' into batches. (Stub — full impl TBD)"""
        parts = qr_str.strip().split('+')
        if len(parts) != 2:
            return False
        try:
            first = [int(c) for c in parts[0] if c in '123']
            second = [int(c) for c in parts[1] if c in '123']
            if len(first) != 3 or len(second) != 3:
                return False
            self.current_batch_order = [Color(v) for v in first]
            # 这里记录下两个批次的顺序，方便后续使用
            self.first_batch_order = [Color(v) for v in first]
            self.second_batch_order = [Color(v) for v in second]

            # second batch stored in cargo_set via CargoItem.batch
        except (ValueError, TypeError):
            return False
        self.qr_result = qr_str
        return True

    def current_target_color(self) -> Optional[Color]:
        """Return the color that should be picked/placed now."""
        if not self.current_batch_order or self.current_step >= len(self.current_batch_order):
            return None
        return self.current_batch_order[self.current_step]

    def advance_target(self):
        """Move to next target in current batch."""
        if self.current_step < len(self.current_batch_order) - 1:
            self.current_step += 1
        else:
            self.current_step = 0

    def is_batch_complete(self) -> bool:
        """True when all 3 materials in current batch are handled."""
        return self.current_step >= len(self.current_batch_order) - 1 and self.cargo_count == 0

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
        ctx.cargo_confirmed = True


class _PickRoughState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter PICK_ROUGH")
        ctx.state_entry_time = time()

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        return None

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        ctx.cargo_confirmed = True


class _CheckLoadState(State):
    def on_enter(self, ctx: MissionContext, from_state: str) -> None:
        print("[MissionSM] Enter CHECK_LOAD")
        ctx.state_entry_time = time()
        ctx.cargo_confirmed = False

    def on_execute(self, ctx: MissionContext) -> Optional[str]:
        if ctx.cargo_confirmed:
            ctx.cargo_count += 1
            ctx.current_step += 1
            ctx.target_color = ctx.current_batch_order[ctx.current_step] if ctx.current_step < 3 else Color.RED
            if ctx.current_step < 3:
                if ctx.current_zone == Zone.RAW:
                    return MissionStateNames.ALIGN_RAW
                else:
                    return MissionStateNames.ALIGN_ROUGH
            else:
                ctx.current_step = 0
                if ctx.current_zone == Zone.RAW:
                    return MissionStateNames.NAV_TO_ROUGH
                else:
                    ctx.picking_from_rough = False
                    return MissionStateNames.NAV_TO_TEMP
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
        if ctx.picking_from_rough and ctx.ready_to_pick and not ctx.color_mismatch:
            return MissionStateNames.PICK_ROUGH
        if not ctx.picking_from_rough and ctx.ready_to_place:
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
        if not ctx.place_action_done:
            return None
        ctx.place_action_done = False
        if ctx.cargo_count > 0:
            return MissionStateNames.ALIGN_ROUGH
        ctx.picking_from_rough = True
        ctx.current_step = 0
        ctx.target_color = ctx.current_batch_order[0] if ctx.current_batch_order else Color.RED
        return MissionStateNames.ALIGN_ROUGH

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


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
        if not ctx.place_action_done:
            return None
        ctx.place_action_done = False
        if ctx.cargo_count > 0:
            return MissionStateNames.ALIGN_TEMP
        return MissionStateNames.RETURN_HOME

    def on_exit(self, ctx: MissionContext, to_state: str) -> None:
        pass


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
    MissionState.PICK_ROUGH: "PICK_ROUGH",
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
        PICK_DONE = "PICK_DONE"                 # PICK_{RAW,ROUGH} -> CHECK_LOAD
        ARRIVED_ROUGH = "ARRIVED_ROUGH"         # NAV_TO_ROUGH -> ALIGN_ROUGH
        ARRIVED_TEMP = "ARRIVED_TEMP"           # NAV_TO_TEMP -> ALIGN_TEMP
        RETURNED_HOME = "RETURNED_HOME"         # RETURN_HOME -> FINISHED
        RESET = "RESET"                         # ERROR -> WAIT_START
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
        self.register_state(MissionStateNames[MissionState.PICK_ROUGH], _PickRoughState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_TEMP], _NavToTempState())
        self.register_state(MissionStateNames[MissionState.ALIGN_TEMP], _AlignTempState())
        self.register_state(MissionStateNames[MissionState.PLACE_TEMP], _PlaceTempState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_RAW_SECOND], _NavToRawState())
        self.register_state(MissionStateNames[MissionState.NAV_TO_TEMP_SECOND], _NavToTempState())
        self.register_state(MissionStateNames[MissionState.PLACE_TEMP_STACK], _PlaceTempState())
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

        # PICK_RAW -> CHECK_LOAD
        self.register_transition(
            MissionStateNames[MissionState.PICK_RAW],
            MissionStateNames[MissionState.CHECK_LOAD],
            event=self.Events.PICK_DONE
        )

        # NAV_TO_ROUGH -> ALIGN_ROUGH
        self.register_transition(
            MissionStateNames[MissionState.NAV_TO_ROUGH],
            MissionStateNames[MissionState.ALIGN_ROUGH],
            event=self.Events.ARRIVED_ROUGH
        )

        # NAV_TO_TEMP -> ALIGN_TEMP
        self.register_transition(
            MissionStateNames[MissionState.NAV_TO_TEMP],
            MissionStateNames[MissionState.ALIGN_TEMP],
            event=self.Events.ARRIVED_TEMP
        )

        # PICK_ROUGH -> CHECK_LOAD
        self.register_transition(
            MissionStateNames[MissionState.PICK_ROUGH],
            MissionStateNames[MissionState.CHECK_LOAD],
            event=self.Events.PICK_DONE
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
            MissionState.PLACE_TEMP, MissionState.PICK_ROUGH, MissionState.RETURN_HOME,
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
        self.context.current_step = 0
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

    def on_action_done(self, action_id: ActionId, result: int) -> bool:
        """Handle ACTION_DONE from STM32."""
        if result != 0:  # not OK
            self.context.error_code = action_id
            self.context.error_msg = f"Action {action_id} failed with {result}"
            return self.trigger(self.Events.ERROR)

        state = self.current_state
        if state == MissionStateNames[MissionState.PICK_RAW] and action_id == ActionId.PICK_RAW:
            return self.trigger(self.Events.PICK_DONE)
        if state == MissionStateNames[MissionState.PLACE_ROUGH] and action_id == ActionId.PLACE_ROUGH:
            self.context.cargo_count -= 1
            self.context.place_action_done = True
            return False
        if state == MissionStateNames[MissionState.PICK_ROUGH] and action_id == ActionId.PICK_ROUGH:
            return self.trigger(self.Events.PICK_DONE)
        if state == MissionStateNames[MissionState.PLACE_TEMP] and action_id == ActionId.PLACE_TEMP:
            self.context.cargo_count -= 1
            self.context.place_action_done = True
            return False
        return False

    def on_visual_status(self, visual_state: int, flags: int) -> bool:
        """Handle STATUS_FROM_VISION frame."""
        self.context.visual_state = visual_state
        self.context.update_visual_flags(flags)

        if self.context.visual_fail:
            self.context.error_code = 30
            self.context.error_msg = "Visual failure"
            return self.trigger(self.Events.ERROR)

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
            "step": self.context.current_step,
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
    print("Batch order:", sm.context.current_batch_order)
