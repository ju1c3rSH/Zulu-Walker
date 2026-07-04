import threading
import time
from typing import Optional

from .event_bus import EventBus
from .events import (
    McuCmdReceived, ArrivedEvent, ActionDoneEvent,
    HeartbeatEvent, EmergencyStopEvent, RequestSyncEvent,
    FrameResult, ServoData, QRResult, ColorResult,
)
from .mission_state_machine import (
    MissionStateMachine, MissionContext,
    MissionState, VisualState, Zone,
)
from utils.state_machine.visual_state_machine import (
    VisualStateMachine, VisualStateMachine as VSM,
)
from modules.zw_uart_module.protocol import (
    build_status_from_vision_frame, build_visual_servo_data_frame,
    build_qr_result_frame, build_color_result_frame,
    build_heartbeat_frame,
    CMD_START_QR, CMD_STOP_VISUAL,
)
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.cargo import CargoSet
from modules.zw_opencv_module.processors.base import ColorTrackable


_VISUAL_STATE_TO_INT = {
    VSM.States.IDLE: 0,
    VSM.States.SEARCH: 1,
    VSM.States.TRACKING: 2,
    VSM.States.RECOVERY: 3,
    VSM.States.FAIL: 4,
}
_INT_TO_VISUAL_STATE = {v: k for k, v in _VISUAL_STATE_TO_INT.items()}

_READY_THRESHOLD = 10
_HEARTBEAT_INTERVAL = 0.1
_HEARTBEAT_TIMEOUT = 0.3


class MissionCoordinator:

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.mission_sm = MissionStateMachine()
        self.visual_sm = VisualStateMachine()
        self.mission_sm.context.cargo_set = CargoSet.create_standard()

        self._uart_sender: Optional[callable] = None
        self._camera_manager = None
        self._qr_decoded = False
        self._active_task: Optional[str] = None  # "qr_detect" / "track_cargo" / "ring_track"

        self._ready_frames = 0
        self._heartbeat_seq = 0
        self._last_mcu_heartbeat = 0.0
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

    # ===== bridge =====

    def connect_camera(self, camera_manager) -> None:
        self._camera_manager = camera_manager

    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            try:
                return self._uart_sender(frame)
            except Exception:
                return False
        return False

    # ===== lifecycle =====

    def start(self) -> None:
        self._running = True
        self._wire_events()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self._send_initial_status()

    def stop(self) -> None:
        self._running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)

    # ===== event wiring =====

    def _wire_events(self) -> None:
        self.event_bus.subscribe(McuCmdReceived, self._on_mcu_cmd)
        self.event_bus.subscribe(ArrivedEvent, self._on_arrived)
        self.event_bus.subscribe(ActionDoneEvent, self._on_action_done)
        self.event_bus.subscribe(HeartbeatEvent, self._on_heartbeat)
        self.event_bus.subscribe(EmergencyStopEvent, self._on_emergency)
        self.event_bus.subscribe(RequestSyncEvent, self._on_request_sync)
        self.event_bus.subscribe(FrameResult, self._on_vision_results)
        self.event_bus.subscribe(QRResult, self._on_qr_result_event)
        self.event_bus.subscribe(ColorResult, self._on_color_result_event)

    # ===== startup =====

    def _send_initial_status(self) -> None:
        self._send(build_status_from_vision_frame(
            self.mission_sm.current_state_id,
            0,  # visual_state=IDLE
            0,  # flags=0
            self.mission_sm.context.cargo_count,
        ))

    # ===== MCU commands =====

    def _on_mcu_cmd(self, event: McuCmdReceived) -> None:
        cmd, args = event.cmd_id, event.args

        if cmd == CMD_START_QR:
            self._qr_decoded = False
            self.mission_sm.start()
            self._activate_task("qr_detect")

        elif not self._qr_decoded:
            return  # QR 门控

        elif cmd == CMD_STOP_VISUAL:
            self._deactivate_all_visual()

    def _activate_task(self, task_name: str, color: Optional[Color] = None) -> None:
        if not self._camera_manager:
            return

        cm = self._camera_manager
        all_tasks = ["qr_detect", "track_cargo", "ring_track"]

        # disable all
        for cam in cm.cameras.values():
            for name in all_tasks:
                cam.disable_task(name)

        # enable the target one
        for cam_id, cam in cm.cameras.items():
            is_qr = cam_id.endswith("_qr")
            is_cargo = cam_id.endswith("_cargo")

            if task_name == "qr_detect" and is_qr:
                cam.enable_task(task_name)
                self._active_task = task_name
                break
            elif task_name in ("track_cargo", "ring_track") and is_cargo:
                cam.enable_task(task_name)
                if color is not None:
                    t = cam.get_task(task_name)
                    if t and isinstance(t.processor, ColorTrackable):
                        t.processor.set_target_color(color)
                self.visual_sm.start()  # IDLE → SEARCH
                self._active_task = task_name
                self._ready_frames = 0
                break

    def _deactivate_all_visual(self) -> None:
        if not self._camera_manager:
            return
        all_tasks = ["track_cargo", "ring_track"]
        for cam in self._camera_manager.cameras.values():
            for name in all_tasks:
                cam.disable_task(name)
        self.visual_sm.stop()
        self._active_task = None
        self._ready_frames = 0

        # notify MCU
        self._send(build_status_from_vision_frame(
            self.mission_sm.current_state_id,
            0,  # visual=IDLE
            0,
            self.mission_sm.context.cargo_count,
        ))

    # ===== MCU events =====

    def _on_arrived(self, event: ArrivedEvent) -> None:
        self.mission_sm.on_arrived(event.zone_id)

    def _on_action_done(self, event: ActionDoneEvent) -> None:
        self.mission_sm.on_action_done(event.action_id, event.result)

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        self._last_mcu_heartbeat = time.monotonic()

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        self.mission_sm.set_error(99, f"Emergency stop: reason={event.reason}")

    def _on_request_sync(self, event: RequestSyncEvent) -> None:
        pass  # Phase 4 暂不实现完整同步

    # ===== vision results =====

    def _on_qr_result_event(self, event: QRResult) -> None:
        success = self.mission_sm.on_qr_result(event.qr_str)
        if success:
            self._qr_decoded = True
            self._send(build_qr_result_frame(event.qr_str))
            self._send(build_status_from_vision_frame(
                self.mission_sm.current_state_id,
                0, 0,
                self.mission_sm.context.cargo_count,
            ))

    def _on_color_result_event(self, event: ColorResult) -> None:
        self._send(build_color_result_frame(event.color_id, event.confidence))

    def _on_vision_results(self, event: FrameResult) -> None:
        for camera_id, results in event.all_results.items():
            for task_name, vision_result in results.items():
                if vision_result is None or not vision_result.success:
                    continue
                data = vision_result.result_data
                if not data:
                    continue

                if task_name == "qr_detect":
                    self._handle_track_frame(data)
                elif task_name in ("track_cargo", "ring_track"):
                    self._handle_track_frame(data)

    def _handle_track_frame(self, data: dict) -> None:
        ctx = self.visual_sm.context
        target_found = data.get("target_found", False)

        if target_found:
            ctx.target_found = True
            ctx.consecutive_detected_frames += 1
            ctx.consecutive_lost_frames = 0
            ctx.percent_error_x = data.get("percent_error_x", 0)
            ctx.percent_error_y = data.get("percent_error_y", 0)
            ctx.target_distance_mm = data.get("target_distance_mm", 0.0)
            ctx.confidence = 1.0
        else:
            ctx.target_found = False
            ctx.consecutive_lost_frames += 1
            ctx.consecutive_detected_frames = 0
            ctx.confidence = 0.0

        # servo data
        if self.visual_sm.is_tracking():
            self._send(build_visual_servo_data_frame(
                ctx.percent_error_x,
                ctx.percent_error_y,
                int(ctx.target_distance_mm),
                self._visual_state_int(),
            ))

        # visual state transitions
        if self.visual_sm.is_searching() and ctx.consecutive_detected_frames >= _READY_THRESHOLD:
            self.visual_sm.trigger(VSM.Events.TARGET_FOUND)
            self._ready_frames += 1

        elif self.visual_sm.is_tracking() and ctx.consecutive_lost_frames >= 5:
            self.visual_sm.trigger(VSM.Events.TARGET_LOST)
            self._ready_frames = 0

        # ready-to-pick / ready-to-place
        if self.visual_sm.is_tracking() and ctx.consecutive_detected_frames > 0:
            self._ready_frames += 1
        else:
            self._ready_frames = max(0, self._ready_frames - 1)

        if self._ready_frames >= _READY_THRESHOLD:
            flags = 0
            from modules.zw_uart_module.protocol import VisualFlags
            flags |= VisualFlags.TARGET_FOUND
            state = self.mission_sm.current_state
            if state in ("ALIGN_RAW",):
                flags |= VisualFlags.READY_TO_PICK
            elif state in ("ALIGN_ROUGH", "ALIGN_TEMP"):
                flags |= VisualFlags.READY_TO_PLACE
            self.mission_sm.on_visual_status(self._visual_state_int(), flags)
            self._send(build_status_from_vision_frame(
                self.mission_sm.current_state_id,
                self._visual_state_int(),
                flags,
                self.mission_sm.context.cargo_count,
            ))
            self._ready_frames = 0

    def _visual_state_int(self) -> int:
        return _VISUAL_STATE_TO_INT.get(self.visual_sm.current_state, 0)

    # ===== heartbeat =====

    def _heartbeat_loop(self) -> None:
        self._last_mcu_heartbeat = time.monotonic()
        while self._running:
            time.sleep(_HEARTBEAT_INTERVAL)
            self._heartbeat_seq = (self._heartbeat_seq + 1) % 256

            self._send(build_heartbeat_frame(
                self._heartbeat_seq,
                self.mission_sm.current_state_id,
                self._visual_state_int(),
            ))

            if time.monotonic() - self._last_mcu_heartbeat > _HEARTBEAT_TIMEOUT:
                self.mission_sm.set_error(40, "MCU heartbeat lost")

    # ===== debug =====

    def get_info(self) -> dict:
        cs = self.mission_sm.context.cargo_set
        return {
            "mission": self.mission_sm.get_info(),
            "visual_state": self.visual_sm.current_state,
            "qr_decoded": self._qr_decoded,
            "active_task": self._active_task,
            "ready_frames": self._ready_frames,
            "cargo_batch1": [
                {"index": i.index, "color": i.color.name, "available": i.available}
                for i in cs.get_batch(1)
            ] if cs else [],
            "cargo_batch2": [
                {"index": i.index, "color": i.color.name, "available": i.available}
                for i in cs.get_batch(2)
            ] if cs else [],
        }
