"""
Line-following mission coordinator (v3.0 master-slave protocol).

Bridges vision results -> UART data streaming for MSPM0 master.
Slave role: waits for CMD_REQUEST/CMD_STOP from master, streams data when active.
"""
import threading
import time
from collections import deque
from typing import Optional, Callable, Deque

from modules.zw_opencv_module.vision_manager import VisionManager
from utils.log_util import log_print

try:
    from maix import sys as maix_sys
    _HAVE_MAIX_SYS = True
except ImportError:
    maix_sys = None
    _HAVE_MAIX_SYS = False

from framework.event_bus import EventBus
from modules.zw_uart_module.events import (
    EmergencyStopEvent,
    CmdRequestEvent,
    CmdStopEvent,
)
from app.line_follow_sm import Ti2026StateMachine
from modules.zw_uart_module.protocol import (
    build_cmd_ack_frame,
    build_cmd_nack_frame,
    build_data_stream_frame,
    DATA_PAYLOAD_SIZES,
    SUPPORTED_DATA_TYPES,
    DATA_LINE_POSITION,
    DATA_TARGET_POSITION,
    DATA_TARGET_COUNT,
    DATA_DETECTION_STATUS,
    DATA_ALL_TARGETS,
    DATA_SEGMENTATION_MASK,
    NACK_UNSUPPORTED_TYPE,
    NACK_NOT_READY,
)
from modules.zw_opencv_module.processors.base import VisionResult


_CMD_TIMEOUT = 5.0

_DATA_TYPE_MODEL = {
    DATA_TARGET_POSITION:   "yolo11n",
    DATA_TARGET_COUNT:      "yolo11n",
    DATA_ALL_TARGETS:       "yolo11n",
    DATA_SEGMENTATION_MASK: "plate_seg",
}


class Ti2026Coordinator:

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.state_machine = Ti2026StateMachine()

        self._uart_sender: Optional[callable] = None
        self._vision_manager: Optional[VisionManager] = None
        self._ai = None

        self._sm_queue: Deque[Callable] = deque()
        self._sm_lock = threading.Lock()

        # Master-slave streaming state
        self._streaming_type = 0
        self._stream_seq = 0
        self._last_cmd_time = 0.0
        self._master_linked = False
        self._cmd_lock = threading.Lock()

        self._running = False
        self._wdt_feed = lambda: None
        self._wdt_count = 0
        self._last_fps: float = 0.0
        self._last_fps_time: float = 0.0
        self._mem_log_counter: int = 0

        # Vision result cache
        self._latest_line: dict = {}
        self._latest_ai: dict = {}

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def set_ai(self, ai) -> None:
        self._ai = ai

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

        now = time.monotonic()
        with self._cmd_lock:
            streaming = self._streaming_type
            if streaming != 0:
                if now - self._last_cmd_time > _CMD_TIMEOUT:
                    self._streaming_type = 0
                    self._master_linked = False
                    streaming = 0
                else:
                    payload = self._build_stream_payload(streaming)
                    if payload:
                        frame = build_data_stream_frame(
                            self._stream_seq, streaming, payload)
                        self._send(frame)
                        self._stream_seq = (self._stream_seq + 1) & 0xFF

        self._wdt_feed()
        self._wdt_count += 1
        if self._wdt_count % 200 == 0:
            log_print(f"[WDT] coord feed #{self._wdt_count}")

        self._mem_log_counter += 1
        if self._mem_log_counter >= 300:
            self._mem_log_counter = 0
            self._log_memory()

    def start(self) -> None:
        self._running = True
        self._wire_events()
        self._last_cmd_time = time.monotonic()

    def stop(self) -> None:
        self._running = False

    def _wire_events(self) -> None:
        self.event_bus.subscribe(CmdRequestEvent, self._on_cmd_request)
        self.event_bus.subscribe(CmdStopEvent, self._on_cmd_stop)
        self.event_bus.subscribe(EmergencyStopEvent, self._on_emergency)

    # ===== vision results =====

    def _process_vision_results(self, all_results: dict) -> None:
        for pipeline_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if not isinstance(vision_result, VisionResult):
                    continue
                data = vision_result.result_data if vision_result.success else {}
                if task_name == "line_follow":
                    self._latest_line = data
                elif task_name == "ai_inference":
                    self._latest_ai = data

    # ===== Master events =====

    def _on_cmd_request(self, event: CmdRequestEvent) -> None:
        data_type = event.data_type
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._master_linked = True

        if data_type not in SUPPORTED_DATA_TYPES:
            frame = build_cmd_nack_frame(data_type, NACK_UNSUPPORTED_TYPE)
            self._send(frame)
            return

        if self._ai:
            nick = _DATA_TYPE_MODEL.get(data_type)
            if nick and self._ai.active_model != nick:
                if not self._ai.switch(nick):
                    frame = build_cmd_nack_frame(data_type, NACK_NOT_READY)
                    self._send(frame)
                    return

        with self._cmd_lock:
            self._streaming_type = data_type
            self._stream_seq = 0
        payload_size = DATA_PAYLOAD_SIZES.get(data_type, 0)
        if payload_size is None:
            payload_size = 0
        frame = build_cmd_ack_frame(data_type, 60, payload_size)
        self._send(frame)

    def _on_cmd_stop(self, event: CmdStopEvent) -> None:
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._streaming_type = 0
            self._stream_seq = 0
        frame = build_cmd_ack_frame(0x00, 0, 0)
        self._send(frame)

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        captured_reason = event.reason
        with self._cmd_lock:
            self._streaming_type = 0
        self._enqueue_sm(
            lambda: self.state_machine.set_error(99, f"Emergency stop: reason={captured_reason}")
        )

    # ===== streaming payload builders =====

    def _build_stream_payload(self, data_type: int) -> Optional[bytes]:
        if data_type == DATA_LINE_POSITION:
            return self._build_line_position_payload()
        elif data_type == DATA_TARGET_POSITION:
            return self._build_target_position_payload()
        elif data_type == DATA_TARGET_COUNT:
            return self._build_target_count_payload()
        elif data_type == DATA_DETECTION_STATUS:
            return self._build_detection_status_payload()
        elif data_type == DATA_ALL_TARGETS:
            return self._build_all_targets_payload()
        elif data_type == DATA_SEGMENTATION_MASK:
            return self._build_seg_mask_payload()
        return None

    def _build_line_position_payload(self) -> bytes:
        data = self._latest_line
        pe_x = data.get("percent_error_x", 0)
        pe_y = data.get("percent_error_y", 0)
        target_found = data.get("target_found", False)
        flags = 1 if target_found else 0
        state = self.state_machine.current_state_id
        return (pe_x.to_bytes(2, 'little', signed=True) +
                pe_y.to_bytes(2, 'little', signed=True) +
                bytes([flags, state]))

    def _build_target_position_payload(self) -> bytes:
        data = self._latest_ai
        detections = data.get("detections", [])
        if detections:
            best = max(detections, key=lambda d: d.score)
            x = int(best.x)
            y = int(best.y)
            conf = max(0, min(255, int(best.score * 255)))
            flags = 0x01
        else:
            x, y, conf = 0, 0, 0
            flags = 0x00
        return (x.to_bytes(2, 'little', signed=True) +
                y.to_bytes(2, 'little', signed=True) +
                bytes([conf, flags]))

    def _build_target_count_payload(self) -> bytes:
        data = self._latest_ai
        detections = data.get("detections", [])
        return bytes([len(detections)])

    def _build_detection_status_payload(self) -> bytes:
        data = self._latest_ai
        detections = data.get("detections", [])
        count = len(detections)
        visual_state = 0
        visual_flags = 0
        return (bytes([visual_state, visual_flags]) +
                count.to_bytes(2, 'little'))

    def _build_all_targets_payload(self) -> bytes:
        data = self._latest_ai
        detections = data.get("detections", [])
        count = min(len(detections), 16)
        payload = bytes([count])
        for d in detections[:count]:
            payload += (int(d.x).to_bytes(2, 'little', signed=True) +
                        int(d.y).to_bytes(2, 'little', signed=True) +
                        bytes([d.class_id]))
        return payload

    def _build_seg_mask_payload(self) -> bytes:
        data = self._latest_ai
        segments = data.get("segments", [])
        count = min(len(segments), 4)
        payload = bytes([count])
        for s in segments[:count]:
            payload += (
                bytes([s["class_id"]]) +
                int(s["center_x"]).to_bytes(2, 'little') +
                int(s["center_y"]).to_bytes(2, 'little') +
                min(s["area_px"], 65535).to_bytes(2, 'little')
            )
        return payload

    # ===== memory logging =====

    def _log_memory(self) -> None:
        if not _HAVE_MAIX_SYS:
            return
        try:
            info = maix_sys.memory_info()
            user = info.get("used", 0) / 1048576
            user_total = info.get("total", 0) / 1048576
            cmm = info.get("cmm_used", 0) / 1048576
            cmm_total = info.get("cmm_total", 0) / 1048576
            log_print(
                f"[MEM] user={user:.0f}/{user_total:.0f}MB "
                f"CMM={cmm:.0f}/{cmm_total:.0f}MB"
            )
        except Exception as e:
            log_print(f"[MEM] error: {e}")

    # ===== debug =====

    def get_info(self) -> dict:
        with self._cmd_lock:
            linked = self._master_linked
        detections = self._latest_ai.get("detections", [])
        return {
            "state": self.state_machine.current_state,
            "state_id": self.state_machine.current_state_id,
            "link_active": linked,
            "det_count": len(detections),
            "fps": self._last_fps,
        }
