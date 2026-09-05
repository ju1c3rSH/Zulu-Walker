"""
EOD2026 vision coordinator (v3.0 master-slave subscription protocol).

Slave role. The MCU owns the mission state machine and drives vision through
single-active-subscription semantics (panzer docs/design.md §2.7):
  CMD_REQUEST(type, min_interval, reserved=expected_id) -> ACK -> stream
  CMD_STOP -> ACK(0x00) -> idle (processors disabled, no model burns CPU)
  master silence > 5s -> auto-idle.

Data types are the EOD2026 table (protocol.py); payloads:
  0x01 QR_TASK     one-shot event stream: push once per new decode, else silent
  0x02-0x04 device streams: error_x/error_y (±5000) + id + latched flags
  0x05/0x06 zone streams:   error_x/error_y (±5000) + flags
All latching/debounce lives in EodAlignProcessor; this class never interprets
pixels (rule: coordinator consumes VisionResult dicts only).

Threading: UART RX callbacks publish events on the RX thread; subscription
gating (task enable/disable, processor.set_mode) is enqueued to the main loop
via _sm_queue — the vision thread only ever sees atomic bool flips and a
single-writer set_mode call between frames.
"""
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

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
from modules.zw_uart_module.protocol import (
    build_cmd_ack_frame,
    build_cmd_nack_frame,
    build_data_stream_frame,
    build_qr_task_sub_payload,
    build_device_sub_payload,
    build_zone_sub_payload,
    DATA_PAYLOAD_SIZES,
    SUPPORTED_DATA_TYPES,
    DATA_TYPE_NAMES,
    DATA_TYPE_MODES,
    DATA_QR_TASK,
    DATA_BOMB_DEV,
    DATA_TARGET_DEV,
    DATA_HOSTAGE_DEV,
    DATA_BUCKET_DEV,
    DATA_HOME_DEV,
    NACK_UNSUPPORTED_TYPE,
    NACK_NOT_READY,
)
from modules.zw_opencv_module.processors.base import VisionResult

_CMD_TIMEOUT = 5.0
_PIPELINE_ID = "cam_main"
_TASK_QR = "qr_task"
_TASK_ALIGN = "eod_align"

# data_type -> (optional) AI model nickname to hot-switch before ACKing
_DATA_TYPE_MODEL: dict[int, str] = {
    # DATA_TARGET_DEV: "eod_target",   # enable when a custom YOLO model exists
    # DATA_HOSTAGE_DEV: "eod_hostage",
}


class EodCoordinator:

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self._uart_sender: Optional[Callable[[bytes], bool]] = None
        self._vision_manager: Optional[VisionManager] = None
        self._ai = None

        # gating mutations from the RX thread are applied in the main loop
        self._sm_queue: Deque[Callable] = deque()
        self._sm_lock = threading.Lock()

        # master-slave streaming state
        self._streaming_type = 0
        self._stream_min_interval_ms = 0
        self._stream_seq = 0
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._master_linked = False
        self._cmd_lock = threading.Lock()

        # QR one-shot event state (§3 rule 1: push on new decode, dedup content)
        self._qr_pending = False
        self._qr_last_sent = ""

        self._stream_log_count = 0
        self._last_streamed_type = 0

        # task gating cache (avoid redundant TaskManager calls)
        self._task_enabled: dict[str, bool] = {_TASK_QR: False, _TASK_ALIGN: False}

        self._running = False
        self._wdt_feed: Callable[[], None] = lambda: None
        self._wdt_count = 0
        self._last_fps: float = 0.0
        self._last_fps_time: float = 0.0
        self._mem_log_counter: int = 0

        # vision result caches (written in loop(), read in loop())
        self._latest_qr: dict = {}
        self._latest_eod: dict = {}
        self._frame_width: int = 640
        self._frame_height: int = 640

    # ===== wiring =====

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_width = width
        self._frame_height = height

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def set_uart_sender(self, sender: Callable[[bytes], bool]) -> None:
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

    def _enqueue(self, fn: Callable) -> None:
        with self._sm_lock:
            self._sm_queue.append(fn)

    # ===== main loop =====

    def loop(self) -> None:
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

            now = time.monotonic()
            if now - self._last_fps_time >= 1.0:
                self._last_fps = self._vision_manager.get_pipeline_fps(_PIPELINE_ID)
                self._last_fps_time = now

        with self._sm_lock:
            while self._sm_queue:
                self._sm_queue.popleft()()

        self._wdt_feed()
        self._wdt_count += 1
        if self._wdt_count % 200 == 0:
            log_print(f"[WDT] coord feed #{self._wdt_count}")

        now = time.monotonic()
        with self._cmd_lock:
            streaming = self._streaming_type
            if streaming != 0:
                if now - self._last_cmd_time > _CMD_TIMEOUT:
                    log_print("[UART] master silent > 5s -> auto idle")
                    self._streaming_type = 0
                    self._master_linked = False
                    streaming = 0
                    self._enqueue(self._gate_idle)

        if streaming != 0:
            payload = self._build_stream_payload(streaming)
            if payload is not None:
                if self._send_allowed(streaming, now):
                    current_seq = self._stream_seq
                    frame = build_data_stream_frame(current_seq, streaming, payload)
                    if self._send(frame):
                        self._stream_seq = (current_seq + 1) & 0xFF
                        self._last_send_time = now
                        self._stream_log_count += 1
                        if streaming != self._last_streamed_type or self._stream_log_count >= 30:
                            self._stream_log_count = 0
                            self._last_streamed_type = streaming
                            type_name = DATA_TYPE_NAMES.get(streaming, f"0x{streaming:02X}")
                            log_print(f"[UART TX] DATA_STREAM seq={current_seq} "
                                      f"type={type_name} {self._stream_payload_desc(streaming)}")

        self._mem_log_counter += 1
        if self._mem_log_counter >= 300:
            self._mem_log_counter = 0
            self._log_memory()

    def _send_allowed(self, data_type: int, now: float) -> bool:
        """min_interval throttling; interval 0 = event-driven (no throttle)."""
        if self._stream_min_interval_ms <= 0:
            return True
        return (now - self._last_send_time) * 1000.0 >= self._stream_min_interval_ms

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
        for _pipeline_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if not isinstance(vision_result, VisionResult):
                    continue
                data = vision_result.result_data if vision_result.success else {}
                if task_name == _TASK_QR:
                    self._latest_qr = data or {}
                elif task_name == _TASK_ALIGN:
                    self._latest_eod = data or {}

    # ===== master events =====

    def _on_cmd_request(self, event: CmdRequestEvent) -> None:
        data_type = event.data_type
        dt_name = DATA_TYPE_NAMES.get(data_type, f"0x{data_type:02X}")
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._master_linked = True

        if data_type not in SUPPORTED_DATA_TYPES:
            log_print(f"[UART TX] CMD_NACK type={dt_name} reason=UNSUPPORTED_TYPE")
            self._send(build_cmd_nack_frame(data_type, NACK_UNSUPPORTED_TYPE))
            return

        if self._ai:
            nick = _DATA_TYPE_MODEL.get(data_type)
            if nick and getattr(self._ai, "active_model", None) != nick:
                if not self._ai.switch(nick):
                    log_print(f"[UART TX] CMD_NACK type={dt_name} reason=NOT_READY")
                    self._send(build_cmd_nack_frame(data_type, NACK_NOT_READY))
                    return

        mode = DATA_TYPE_MODES.get(data_type)
        if mode is not None:
            self._enqueue(lambda m=mode, e=event: self._gate_align(m, e))
        else:   # QR
            self._enqueue(lambda: self._gate_qr())

        with self._cmd_lock:
            self._streaming_type = data_type
            self._stream_seq = 0
            self._stream_min_interval_ms = event.min_interval_ms
            self._last_send_time = 0.0
            if data_type == DATA_QR_TASK:
                self._qr_pending = True
        payload_size = DATA_PAYLOAD_SIZES.get(data_type, 0)
        ack_freq = self._ack_freq_hz(event.min_interval_ms)
        log_print(f"[UART TX] CMD_ACK type={dt_name} freq={ack_freq}Hz size={payload_size}B "
                  f"interval={event.min_interval_ms}ms expected_id={event.reserved}")
        self._send(build_cmd_ack_frame(data_type, ack_freq, payload_size))

    def _on_cmd_stop(self, event: CmdStopEvent) -> None:
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._streaming_type = 0
            self._stream_seq = 0
            self._qr_pending = False
            self._qr_last_sent = ""
        log_print("[UART TX] CMD_ACK type=STOP")
        self._send(build_cmd_ack_frame(0x00, 0, 0))
        self._enqueue(self._gate_idle)

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        with self._cmd_lock:
            self._streaming_type = 0
            self._qr_pending = False
        log_print(f"[UART] EMERGENCY_STOP reason={event.reason} -> idle")
        self._enqueue(self._gate_idle)

    # ===== subscription gating (runs in main loop) =====

    def _get_align_processor(self):
        if not self._vision_manager:
            return None
        pipe = self._vision_manager.get_pipeline(_PIPELINE_ID)
        if pipe is None:
            return None
        task = pipe.task_manager.get_task(_TASK_ALIGN)
        return task.processor if task else None

    def _set_task_enabled(self, task_name: str, enabled: bool) -> None:
        if not self._vision_manager or self._task_enabled.get(task_name) == enabled:
            return
        if enabled:
            self._vision_manager.enable_task(_PIPELINE_ID, task_name)
        else:
            self._vision_manager.disable_task(_PIPELINE_ID, task_name)
        self._task_enabled[task_name] = enabled

    def _gate_align(self, mode: str, event: CmdRequestEvent) -> None:
        self._set_task_enabled(_TASK_QR, False)
        self._set_task_enabled(_TASK_ALIGN, True)
        processor = self._get_align_processor()
        if processor is not None:
            processor.set_mode(mode, event.reserved)

    def _gate_qr(self) -> None:
        self._set_task_enabled(_TASK_ALIGN, False)
        self._set_task_enabled(_TASK_QR, True)

    def _gate_idle(self) -> None:
        self._set_task_enabled(_TASK_QR, False)
        self._set_task_enabled(_TASK_ALIGN, False)
        processor = self._get_align_processor()
        if processor is not None:
            processor.reset()

    # ===== streaming payload builders =====

    def _build_stream_payload(self, data_type: int) -> Optional[bytes]:
        if data_type == DATA_QR_TASK:
            return self._build_qr_payload()
        eod = self._latest_eod
        if data_type in (DATA_BOMB_DEV, DATA_TARGET_DEV, DATA_HOSTAGE_DEV):
            return build_device_sub_payload(
                int(eod.get("error_x", 0)), int(eod.get("error_y", 0)),
                int(eod.get("id", 0)), int(eod.get("flags", 0)))
        if data_type in (DATA_BUCKET_DEV, DATA_HOME_DEV):
            return build_zone_sub_payload(
                int(eod.get("error_x", 0)), int(eod.get("error_y", 0)),
                int(eod.get("flags", 0)))
        return None

    def _build_qr_payload(self) -> Optional[bytes]:
        """One-shot semantics: None (silence) until a fresh valid decode."""
        data = self._latest_qr
        code = str(data.get("code", ""))
        if (self._qr_pending and data.get("latched")
                and data.get("code_domain_valid") and code != self._qr_last_sent):
            self._qr_pending = False
            self._qr_last_sent = code
            log_print(f"[QR] pushing task code {code!r}")
            return build_qr_task_sub_payload(True, code)
        if data.get("changed") and data.get("latched") and data.get("code_domain_valid"):
            # a different code appeared mid-subscription: re-arm (still content-deduped)
            self._qr_pending = True
        return None

    def _stream_payload_desc(self, data_type: int) -> str:
        if data_type == DATA_QR_TASK:
            return f"code={self._latest_qr.get('code', '')!r}"
        eod = self._latest_eod
        return (f"mode={eod.get('mode', '-')} ex={eod.get('error_x', 0):+d} "
                f"ey={eod.get('error_y', 0):+d} id={eod.get('id', 0)} "
                f"flags={eod.get('flags', 0):02b} raw_found={eod.get('raw_found', False)}")

    def _ack_freq_hz(self, min_interval_ms: int) -> int:
        if min_interval_ms <= 0:
            return 60
        return max(1, min(60, int(1000 / min_interval_ms)))

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
            mode = DATA_TYPE_NAMES.get(self._streaming_type, "IDLE") \
                if self._streaming_type else "IDLE"
        return {
            "state": mode,
            "state_id": self._streaming_type,
            "link_active": linked,
            "det_count": 1 if self._latest_eod.get("raw_found") else 0,
            "fps": self._last_fps,
        }
