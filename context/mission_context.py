import threading
import time
from collections import deque
from typing import Optional, Callable

from modules.zw_opencv_module.vision_manager import VisionManager
from utils.log_util import LoggerFactory, log_print

from .event_bus import EventBus
from .events import (
    McuCmdReceived, ArrivedEvent, ActionDoneEvent,
    HeartbeatEvent, EmergencyStopEvent, RequestSyncEvent,
    QRResult,
)
from .mission_state_machine import (
    MissionStateMachine, MissionContext,
    MissionState, MissionStateNames, VisualState, Zone,
)
from .visual_state_machine import VisualStateMachine
from utils.state_machine.bridge import StateActionBridge
from modules.zw_uart_module.protocol import (
    build_status_from_vision_frame, build_visual_servo_data_frame,
    build_qr_result_frame, build_heartbeat_frame,
    build_color_result_frame,
    CMD_START_QR, CMD_STOP_VISUAL,
    CMD_START_RING_DISCOVERY, CMD_DISCOVERY_DONE,
    VisualFlags,
)
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.cargo import CargoSet
from modules.zw_opencv_module.processors.base import ColorTrackable, VisionResult
from modules.zw_opencv_module.processors.cargo_processor import TrackCargoProcessor


_VISUAL_STATE_TO_INT = {
    VisualStateMachine.States.IDLE: 0,
    VisualStateMachine.States.SEARCH: 1,
    VisualStateMachine.States.TRACKING: 2,
    VisualStateMachine.States.RECOVERY: 3,
    VisualStateMachine.States.FAIL: 4,
}

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
        self._vision_manager: Optional[VisionManager] = None
        self._qr_decoded = False
        self._active_task: Optional[str] = None

        self._ready_frames = 0
        self._ready_latched = False
        self._ready_flag = 0
        self._discovery_ready_frames = 0
        self._discovery_ready_latched = False
        self._discovery_color_sent = False
        self._sm_queue: Deque[Callable] = deque()
        self._sm_lock = threading.Lock()
        self._heartbeat_seq = 0
        self._last_mcu_heartbeat = 0.0
        self._is_linked = False
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_servo_log_ts: float = 0.0

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def is_link_active(self) -> bool:
        return self._is_linked
    
    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            try:
                ok = self._uart_sender(frame)
                if ok and len(frame) >= 4 and frame[0] == 0xAA:
                    self._log_uart_tx(frame)
                return ok
            except Exception:
                return False
        return False

    def _log_uart_tx(self, frame: bytes):
        frame_type = frame[2]
        payload = frame[3:-1]

        if frame_type == 0x11:  # STATUS_FROM_VISION
            state_id, visual, flags, cargo = payload[0], payload[1], payload[2], payload[3]
            state_name = MissionStateNames.get(state_id, f"UNKNOWN({state_id})")
            log_print(f"[UART TX] STATUS state={state_id}({state_name}) visual={visual} flags=0x{flags:02x} cargo={cargo}")

        elif frame_type == 0x12:  # QR_RESULT
            qr_str = payload[1:1 + payload[0]].decode("ascii", errors="replace")
            log_print(f"[UART TX] QR_RESULT qr=\"{qr_str}\"")

        elif frame_type == 0x13:  # COLOR_RESULT
            log_print(f"[UART TX] COLOR_RESULT color={payload[0]} conf={payload[1]}")

        elif frame_type == 0x16:  # REQUEST_SYNC
            state = payload[0]
            state_name = MissionStateNames.get(state, f"UNKNOWN({state})")
            log_print(f"[UART TX] REQUEST_SYNC state={state}({state_name})")

        elif frame_type == 0x17:  # VISUAL_SERVO_DATA — 限速 1s
            now = time.time()
            if now - self._last_servo_log_ts >= 1.0:
                self._last_servo_log_ts = now
                err_x = int.from_bytes(payload[0:2], "little", signed=True)
                err_y = int.from_bytes(payload[2:4], "little", signed=True)
                log_print(f"[UART TX] VISUAL_SERVO err_x={err_x} err_y={err_y} flags=0x{payload[4]:02x} state={payload[5]}")

        elif frame_type == 0x18:  # EMERGENCY_STOP
            log_print(f"[UART TX] EMERGENCY_STOP reason={payload[0]}")

        elif frame_type == 0x01:  # ERROR
            err_value = int.from_bytes(payload[1:3], "little", signed=True)
            log_print(f"[UART TX] ERROR type={payload[0]} value={err_value}")

    def _enqueue_sm(self, fn: Callable) -> None:
        with self._sm_lock:
            self._sm_queue.append(fn)

    def loop(self) -> None:
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

        with self._sm_lock:
            while self._sm_queue:
                self._sm_queue.popleft()()
        self.mission_sm.run_to_completion()

    # ===== lifecycle =====

    def start(self) -> None:
        self._running = True
        self._wire_events()
        self._wire_state_actions()
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
        self.event_bus.subscribe(QRResult, self._on_qr_result_event)

    def _wire_state_actions(self) -> None:
        bridge = StateActionBridge(self.mission_sm)

        bridge.when_enter("READ_QR",
            lambda: self._activate_task("qr_detect"))

        bridge.when_enter({"ALIGN_RAW"},
            lambda: self._activate_task("track_cargo", self._current_target_color()))

        bridge.when_enter("RING_DISCOVERY",
            self._deactivate_all_visual)

        bridge.when_enter({
            "NAV_TO_RAW", "NAV_TO_ROUGH", "NAV_TO_TEMP",
            "NAV_TO_RAW_SECOND",
            "RETURN_HOME", "WAIT_START", "FINISHED", "ERROR", "IDLE",
            "PICK_RAW", "PICK_ROUGH", "PLACE_ROUGH", "PLACE_TEMP", "CHECK_LOAD",
            "ALIGN_ROUGH", "ALIGN_TEMP",
        }, self._deactivate_all_visual)

    def _current_target_color(self) -> Optional[Color]:
        return self.mission_sm.context.current_target_color()

    # ===== startup =====

    def _send_initial_status(self) -> None:
        self._send(build_status_from_vision_frame(
            self.mission_sm.current_state_id,
            0,
            0,
            self.mission_sm.context.cargo_count,
        ))

    # ===== MCU commands =====

    def _on_mcu_cmd(self, event: McuCmdReceived) -> None:
        cmd, args = event.cmd_id, event.args

        if cmd == CMD_START_QR:
            self._qr_decoded = False
            self._enqueue_sm(lambda: self.mission_sm.start())

        elif not self._qr_decoded:
            return

        elif cmd == CMD_STOP_VISUAL:
            self._deactivate_all_visual()

        elif cmd == CMD_START_RING_DISCOVERY:
            try:
                color_id = args[0] if len(args) >= 1 else 0
                color = Color(color_id)
            except (ValueError, KeyError):
                return
            captured = color
            self._enqueue_sm(lambda c=captured: self._handle_ring_discovery_cmd(c))

        elif cmd == CMD_DISCOVERY_DONE:
            self._enqueue_sm(lambda: self.mission_sm.on_discovery_done())

    def _handle_ring_discovery_cmd(self, color: Color) -> None:
        if self.mission_sm.current_state != "RING_DISCOVERY":
            return
        self.visual_sm.stop()
        self.visual_sm.start()
        self._discovery_ready_frames = 0
        self._discovery_ready_latched = False
        self._discovery_color_sent = False
        self._activate_task("ring_discovery", color)
        self.mission_sm.context.discovery_color = color
        self.mission_sm.context.discovery_active = True
        self.mission_sm.run_to_completion()

    def _activate_task(self, task_name: str, color: Optional[Color] = None) -> None:
        if not self._vision_manager:
            return

        vm = self._vision_manager
        all_pipelines = ["cam_qr", "cam_cargo"]

        for pid in all_pipelines:
            vm.disable_task(pid, "qr_detect")
            vm.disable_task(pid, "track_cargo")
            vm.disable_task(pid, "ring_discovery")

        if task_name == "qr_detect":
            vm.enable_task("cam_qr", task_name)
            self._active_task = task_name
        elif task_name in ("track_cargo", "ring_discovery"):
            vm.enable_task("cam_cargo", task_name)
            vm.release_pipeline("cam_qr")
            if color is not None:
                self.mission_sm.context.target_color = color
                pipe = vm.get_pipeline("cam_cargo")
                if pipe:
                    t = pipe.get_task(task_name)
                    if t and isinstance(t.processor, ColorTrackable):
                        t.processor.set_target_color(color)
                    if t and isinstance(t.processor, TrackCargoProcessor):
                        t.processor.register_getter("mission_ctx", lambda: (
                            self.mission_sm.context.cargo_set,
                            self.mission_sm.context.current_batch,
                            self.mission_sm.context.current_target_color(),
                        ))
            self.visual_sm.start()
            self._active_task = task_name
            self._ready_frames = 0
            self._ready_latched = False
            self._ready_flag = 0
            self._discovery_ready_frames = 0
            self._discovery_ready_latched = False
            from utils.debug_console import DebugConsole
            dc = DebugConsole()
            color_name = color.name if color else "N/A"
            dc.set("target_color", color_name)
            dc.set("active_task", task_name)
            dc.log(
                f"[MissionSM] Activate task={task_name} color={color_name} "
                f"batch={self.mission_sm.context.current_batch} "
                f"step={self.mission_sm.context.current_step}"
            )

    def _deactivate_all_visual(self) -> None:
        if self._vision_manager:
            vm = self._vision_manager
            all_pipelines = ["cam_qr", "cam_cargo"]
            all_tasks = ["qr_detect", "track_cargo", "ring_discovery"]
            for pid in all_pipelines:
                for name in all_tasks:
                    vm.disable_task(pid, name)
                for name in all_tasks:
                    pipe = vm.get_pipeline(pid)
                    if pipe:
                        t = pipe.get_task(name)
                        if t and hasattr(t.processor, "clear_getters"):
                            t.processor.clear_getters()
        self.visual_sm.stop()
        self._active_task = None
        self._ready_frames = 0
        self._ready_latched = False
        self._ready_flag = 0
        self._discovery_ready_frames = 0
        self._discovery_ready_latched = False
        self._discovery_color_sent = False
        self.mission_sm.context.discovery_active = False

        self._send(build_status_from_vision_frame(
            self.mission_sm.current_state_id,
            0,
            0,
            self.mission_sm.context.cargo_count,
        ))

    # ===== MCU events =====

    def _on_arrived(self, event: ArrivedEvent) -> None:
        captured_zone_id = event.zone_id
        self._enqueue_sm(lambda: self.mission_sm.on_arrived(captured_zone_id))

    def _on_action_done(self, event: ActionDoneEvent) -> None:
        captured_id = event.action_id
        captured_result = event.result
        if captured_result == 0:
            self._ready_latched = False
            self._ready_flag = 0
        self._enqueue_sm(lambda: self.mission_sm.on_action_done(captured_id, captured_result))

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        self._last_mcu_heartbeat = time.monotonic()
        self._is_linked = True

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        captured_reason = event.reason
        self._enqueue_sm(lambda: self.mission_sm.set_error(
            99, f"Emergency stop: reason={captured_reason}"))

    def _on_request_sync(self, event: RequestSyncEvent) -> None:
        pass

    # ===== vision results =====

    def _on_qr_result_event(self, event: QRResult) -> None:
        captured_qr = event.qr_str
        self._enqueue_sm(lambda: self._apply_qr_result(captured_qr))

    def _apply_qr_result(self, qr_str: str) -> None:
        success = self.mission_sm.on_qr_result(qr_str)
        if success:
            self._qr_decoded = True
            self._send(build_qr_result_frame(qr_str))
            self._send(build_status_from_vision_frame(
                self.mission_sm.current_state_id,
                0, 0,
                self.mission_sm.context.cargo_count,
            ))

    def _process_vision_results(self, all_results: dict) -> None:
        for camera_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if not isinstance(vision_result, VisionResult):
                    continue
                data = vision_result.result_data if vision_result.success else {}
                if task_name == "qr_detect":
                    if not vision_result.success or not data:
                        continue
                    self._handle_qr_result(data)
                elif task_name in ("track_cargo", "ring_discovery"):
                    self._handle_track_result(data)

    def _handle_qr_result(self, data: dict) -> None:
        result = data.get("result", {})
        qr_str = result.get("qr_data", "")
        if qr_str:
            self._on_qr_result_event(QRResult(qr_str))

    def _handle_track_result(self, data: dict) -> None:
        ctx = self.visual_sm.context
        target_found = data.get("target_found", False)

        if target_found:
            ctx.target_found = True
            ctx.percent_error_x = data.get("percent_error_x", 0)
            ctx.percent_error_y = data.get("percent_error_y", 0)
            ctx.consecutive_detected_frames += 1
            ctx.consecutive_lost_frames = 0
            coord = data.get("coordinate")
            if coord:
                ctx.target_center = coord
        else:
            ctx.target_found = False
            ctx.percent_error_x = 0
            ctx.percent_error_y = 0
            ctx.consecutive_lost_frames += 1
            ctx.consecutive_detected_frames = 0
            if ctx.consecutive_lost_frames % 60 == 1:
                from utils.debug_console import DebugConsole
                DebugConsole().log(
                    f"[VisualSM] {self.visual_sm.current_state} "
                    f"lost={ctx.consecutive_lost_frames} detected={ctx.consecutive_detected_frames}"
                )

        self.visual_sm.update()

        pe_x = ctx.percent_error_x
        pe_y = ctx.percent_error_y

        if self.mission_sm.context.discovery_active:
            self._handle_discovery_result(target_found, pe_x, pe_y, data)
            return

        flags = 0
        if target_found:
            flags |= VisualFlags.TARGET_FOUND

        if not self._ready_latched:
            if self.visual_sm.is_tracking() and target_found \
               and abs(ctx.percent_error_x) <= 250 and abs(ctx.percent_error_y) <= 250:
                self._ready_frames += 1
            else:
                self._ready_frames = max(0, self._ready_frames - 1)

            if self._ready_frames >= _READY_THRESHOLD:
                self._ready_latched = True
                self._ready_flag = 0
                state = self.mission_sm.current_state
                picking = self.mission_sm.context.picking_from_rough
                if state in ("ALIGN_RAW",) or (state == "ALIGN_ROUGH" and picking):
                    self._ready_flag = VisualFlags.READY_TO_PICK
                elif state in ("ALIGN_ROUGH", "ALIGN_TEMP") and not (state == "ALIGN_ROUGH" and picking):
                    self._ready_flag = VisualFlags.READY_TO_PLACE
                flags |= self._ready_flag
                # 捕获当前值，入队推状态机
                captured_flags = flags
                captured_vis = self._visual_state_int()
                self._enqueue_sm(lambda: self._apply_visual_status(captured_vis, captured_flags))
                self._ready_frames = 0
        else:
            flags |= self._ready_flag

        self._send(build_visual_servo_data_frame(pe_x, pe_y, flags, self._visual_state_int()))

    def _apply_visual_status(self, visual_state: int, flags: int) -> None:
        self.mission_sm.on_visual_status(visual_state, flags)
        self.mission_sm.run_to_completion()

    def _handle_discovery_result(self, target_found: bool, pe_x: int, pe_y: int, data: dict) -> None:
        flags = 0
        if target_found:
            flags |= VisualFlags.TARGET_FOUND

        if not self._discovery_ready_latched:
            if self.visual_sm.is_tracking() and target_found:
                conf = data.get("confidence", 0)
                if conf >= 80:
                    self._discovery_ready_frames += 1
                elif self._discovery_ready_frames % 60 == 1:
                    from utils.debug_console import DebugConsole
                    DebugConsole().log(
                        f"[VisualSM] discovery: target_found conf={conf}<80, "
                        f"waiting for complete ring"
                    )
            else:
                self._discovery_ready_frames = max(0, self._discovery_ready_frames - 1)

            if self._discovery_ready_frames >= _READY_THRESHOLD:
                self._discovery_ready_latched = True
        else:
            flags |= VisualFlags.RING_CENTERED

        if self._discovery_ready_latched:
            flags |= VisualFlags.RING_CENTERED
            color = self.mission_sm.context.discovery_color
            if color and not self._discovery_color_sent:
                conf = data.get("confidence", 100)
                self._send(build_color_result_frame(color.value, int(conf)))
                self._discovery_color_sent = True

        self._send(build_visual_servo_data_frame(pe_x, pe_y, flags, self._visual_state_int()))

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
                self._is_linked = False

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
            "link_active": self._is_linked,
        }
