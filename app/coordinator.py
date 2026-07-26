"""
Line-following mission coordinator.

Bridges vision results -> UART servo data for STM32.
Simple: receive line_follow error_x -> send to MCU via VISUAL_SERVO_DATA.
AI inference results: logged for now (TODO: MCU integration).
"""
import threading
import time
from collections import deque
from typing import Optional, Callable, Deque

from modules.zw_opencv_module.vision_manager import VisionManager
from utils.log_util import log_print

from framework.event_bus import EventBus
from modules.zw_uart_module.events import (
    HeartbeatEvent,
    EmergencyStopEvent,
)
from app.line_follow_sm import LineFollowStateMachine, LineFollowStateNames
from modules.zw_uart_module.protocol import (
    build_visual_servo_data_frame,
    build_heartbeat_frame,
    build_emergency_stop_frame,
)
from modules.zw_opencv_module.processors.base import VisionResult


_HEARTBEAT_INTERVAL = 0.1
_HEARTBEAT_TIMEOUT = 0.3


class LineFollowCoordinator:

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.state_machine = LineFollowStateMachine()
        self.state_machine.context

        self._uart_sender: Optional[callable] = None
        self._vision_manager: Optional[VisionManager] = None

        self._sm_queue: Deque[Callable] = deque()
        self._sm_lock = threading.Lock()

        self._heartbeat_seq = 0
        self._last_mcu_heartbeat = 0.0
        self._is_linked = False
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_servo_log_ts: float = 0.0
        self._last_det_count: int = 0
        self._last_fps: float = 0.0
        self._last_fps_time: float = 0.0

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def is_link_active(self) -> bool:
        return self._is_linked

    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            try:
                return self._uart_sender(frame)
            except Exception:
                return False
        return False

    def _enqueue_sm(self, fn: Callable) -> None:
        with self._sm_lock:
            self._sm_queue.append(fn)

    def loop(self) -> None:
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

            now = time.monotonic()
            if now - self._last_fps_time >= 1.0:
                self._last_fps = self._vision_manager.get_pipeline_fps("cam_main")
                self._last_fps_time = now

        with self._sm_lock:
            while self._sm_queue:
                self._sm_queue.popleft()()
        self.state_machine.run_to_completion()

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

    def _wire_events(self) -> None:
        self.event_bus.subscribe(HeartbeatEvent, self._on_heartbeat)
        self.event_bus.subscribe(EmergencyStopEvent, self._on_emergency)

    def _send_initial_status(self) -> None:
        self._send(build_heartbeat_frame(0, self.state_machine.current_state_id, 0))

    # ===== vision results =====

    def _process_vision_results(self, all_results: dict) -> None:
        for pipeline_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if not isinstance(vision_result, VisionResult):
                    continue
                data = vision_result.result_data if vision_result.success else {}

                if task_name == "line_follow":
                    self._handle_line_follow_result(data)
                elif task_name == "ai_inference":
                    self._handle_ai_result(data)

    def _handle_line_follow_result(self, data: dict) -> None:
        target_found = data.get("target_found", False)
        pe_x = data.get("percent_error_x", 0)

        if self.state_machine.current_state == "LINE_FOLLOW":
            flags = 1 if target_found else 0
            state = self.state_machine.current_state_id
            self._send(build_visual_servo_data_frame(pe_x, 0, flags, state))

    def _handle_ai_result(self, data: dict) -> None:
        detections = data.get("detections", [])
        self._last_det_count = len(detections)

    # ===== MCU events =====

    def _on_heartbeat(self, event: HeartbeatEvent) -> None:
        self._last_mcu_heartbeat = time.monotonic()
        self._is_linked = True

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        captured_reason = event.reason
        self._enqueue_sm(
            lambda: self.state_machine.set_error(99, f"Emergency stop: reason={captured_reason}")
        )

    # ===== heartbeat =====

    def _heartbeat_loop(self) -> None:
        self._last_mcu_heartbeat = time.monotonic()
        while self._running:
            time.sleep(_HEARTBEAT_INTERVAL)
            self._heartbeat_seq = (self._heartbeat_seq + 1) % 256
            self._send(
                build_heartbeat_frame(
                    self._heartbeat_seq,
                    self.state_machine.current_state_id,
                    0,
                )
            )
            if time.monotonic() - self._last_mcu_heartbeat > _HEARTBEAT_TIMEOUT:
                self._is_linked = False

    # ===== debug =====

    def get_info(self) -> dict:
        return {
            "state": self.state_machine.current_state,
            "state_id": self.state_machine.current_state_id,
            "link_active": self._is_linked,
            "det_count": self._last_det_count,
            "fps": self._last_fps,
        }
