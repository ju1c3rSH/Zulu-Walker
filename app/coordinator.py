"""
Ti2026 mission coordinator (v3.0 master-slave protocol).

Bridges vision results -> UART data streaming for MSPM0 master.
Slave role: waits for CMD_REQUEST/CMD_STOP from master, streams data when active.
Manages VisionState for operational mode tracking and PC recording signaling.
"""
import threading
import time
from typing import Optional, Callable

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
    DATA_PAYLOAD_SIZES,
    SUPPORTED_DATA_TYPES,
    DATA_TYPE_NAMES, NACK_REASON_NAMES,
    DATA_LINE_POSITION,
    DATA_TARGET_POSITION,
    DATA_TARGET_COUNT,
    DATA_DETECTION_STATUS,
    DATA_ALL_TARGETS,
    DATA_SEGMENTATION_MASK,
    DATA_PENDULUM_POSITION,
    NACK_UNSUPPORTED_TYPE,
    NACK_NOT_READY,
)
from modules.zw_opencv_module.processors.base import VisionResult
from modules.zw_opencv_module.detectors.pendulum_calibrator import RailCalibration
from app.vision_state import VisionState



_DATA_TYPE_MODEL = {
    DATA_TARGET_POSITION:   "yolo11n",
    DATA_TARGET_COUNT:      "yolo11n",
    DATA_ALL_TARGETS:       "yolo11n",
    DATA_SEGMENTATION_MASK: "plate_seg",
}

# Ball spatial gating constants
# MAX_DISPLACEMENT_PX = (MAX_BALL_SPEED / MIN_FPS) * pixels_per_cm * SAFETY_FACTOR
#                    = (30 / 40) * 25.6 * 2.0 ≈ 38.4 → 40 px @ default calibration
_MAX_BALL_SPEED_CM_S = 30.0       # max physical ball speed (cm/s)
_BALL_MIN_FPS = 40.0              # worst-case pipeline framerate
_VEL_SAFETY_FACTOR = 2.0          # pixel displacement safety factor
_BALL_ARM_FRAMES = 2              # consecutive valid detections to start sending
_BALL_DROP_FRAMES = 3             # consecutive invalid detections to stop sending

# α-β filter parameters (steady-state Kalman for 1D constant-velocity model)
_AB_ALPHA = 0.7                  # position smoothing gain (0=full smooth, 1=raw measurement)
_AB_BETA = 0.3                   # velocity tracking gain
_AB_V_MIN_LOCK = 25.0            # lock threshold (~1 cm/s @ 25.6 px/cm)
_AB_LOCK_FRAMES = 3              # consecutive slow frames to enter LOCKED
_AB_UNLOCK_JUMP_PX = 25.0        # single-frame position jump to exit LOCKED (~1 cm)
                                 #   Handles sudden acceleration (pendulum tilt -> immediate large displacement)
_AB_UNLOCK_BIAS_PX = 10.0        # cumulative residual to exit LOCKED (~0.4 cm)
                                 #   Handles slow creep / direction reversal: individual frames stay
                                 #   below JUMP threshold, but accumulated displacement triggers unlock.
                                 #   Set >3σ measurement noise (~9 px) to prevent false unlock from jitter.
                                 #   At 2 cm/s ball speed: unlocks in ~12 frames (~0.3s)


class Ti2026Coordinator:

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self._vision_state = VisionState.IDLE
        self._recording_test_id: int = 0
        self._test_id: int = 0
        self._record_cmd_sender: Optional[Callable] = None

        self._uart_sender: Optional[callable] = None
        self._vision_manager: Optional[VisionManager] = None
        self._ai = None

        # Master-slave streaming state
        self._streaming_type = 0
        self._stream_seq = 0
        self._last_cmd_time = 0.0
        self._master_linked = False
        self._cmd_lock = threading.Lock()

        self._stream_log_count = 0
        self._last_streamed_type = 0

        self._running = False
        self._wdt_feed = lambda: None
        self._wdt_count = 0
        self._last_fps: float = 0.0
        self._last_fps_time: float = 0.0
        self._mem_log_counter: int = 0

        # PC heartbeat detector
        self._pc_heartbeat: Optional['PcHeartbeatDetector'] = None

        # Vision result cache
        self._latest_line: dict = {}
        self._latest_ai: dict = {}
        self._ai_seq: int = 0
        self._ai_new: bool = False
        self._pixels_per_cm: float = 50.0
        self._frame_width: int = 640
        self._frame_height: int = 640
        self._rail_calib: Optional[RailCalibration] = None
        self._origin_from_ball: bool = False

        # α-β filter state (X axis only for pendulum 1D groove)
        self._ab_x: float = 0.0
        self._ab_vx: float = 0.0
        self._ab_ts_ns: int = 0
        self._ab_ready: bool = False
        self._ab_locked: bool = False
        self._ab_lock_count: int = 0
        self._ab_bias_sum: float = 0.0

        # ball detection spatial gating + hysteresis
        self._ball_armed: bool = False
        self._ball_hit_count: int = 0
        self._ball_miss_count: int = 0
        self._ball_last_cx: Optional[float] = None
        self._ball_last_cy: Optional[float] = None

    def set_pendulum_calibration(self, pixels_per_cm: float, frame_width: int = 640, frame_height: int = 640) -> None:
        self._pixels_per_cm = pixels_per_cm
        self._frame_width = frame_width
        self._frame_height = frame_height

    def set_rail_calibration(self, calib: RailCalibration) -> None:
        self._rail_calib = calib

    def get_rail_calibration(self) -> Optional[RailCalibration]:
        return self._rail_calib

    def is_origin_exact(self) -> bool:
        return self._origin_from_ball

    def calibrate_origin_from_ball(self) -> bool:
        calib = self._rail_calib
        if calib is None or not calib.calibrated:
            return False
        data = self._latest_ai
        detections = data.get("detections", [])
        ball = max((d for d in detections if d.class_id == 0),
                   key=lambda d: d.score, default=None)
        if ball is None:
            return False
        cx = ball.x + ball.w / 2
        cy = ball.y + ball.h / 2
        self._rail_calib = calib.replace_origin(cx, cy)
        self._origin_from_ball = True
        return True

    def get_last_ball_bbox(self) -> Optional[tuple]:
        data = self._latest_ai
        detections = data.get("detections", [])
        ball = max((d for d in detections if d.class_id == 0),
                   key=lambda d: d.score, default=None)
        if ball is None:
            return None
        return (ball.x, ball.y, ball.w, ball.h)

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def set_record_cmd_sender(self, sender: callable) -> None:
        self._record_cmd_sender = sender

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def set_ai(self, ai) -> None:
        self._ai = ai

    def set_pc_heartbeat(self, detector: 'PcHeartbeatDetector') -> None:
        self._pc_heartbeat = detector
        if detector is not None:
            detector.set_on_connected(self._on_pc_connected)

    def _on_pc_connected(self) -> None:
        if not self._running:
            return
        if self._vision_state != VisionState.STREAMING or self._recording_test_id == 0:
            return
        log_print("[Recording] PC reconnected, retry notify")
        threading.Thread(target=self._try_notify_pc, daemon=True).start()

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            try:
                return self._uart_sender(frame)
            except Exception:
                return False
        return False

    @property
    def vision_state(self) -> VisionState:
        return self._vision_state

    def change_state(self, new_state: VisionState) -> None:
        """线程安全的状态转换，自动处理录制生命周期。"""
        if self._vision_state == new_state:
            return
        with self._cmd_lock:
            old_state = self._vision_state
            log_print(f"[State] {old_state.name} \u2192 {new_state.name}")

            if old_state == VisionState.STREAMING and new_state != VisionState.STREAMING:
                self._stop_recording()
            elif new_state == VisionState.STREAMING and old_state != VisionState.STREAMING:
                self._start_recording()

            self._vision_state = new_state

    def _start_recording(self) -> None:
        self._test_id += 1
        self._recording_test_id = self._test_id
        if self._vision_manager:
            self._vision_manager.set_test_id(self._test_id)
        self._try_notify_pc()

    def _try_notify_pc(self) -> None:
        if self._record_cmd_sender:
            target_ip = self._pc_heartbeat.pc_ip if self._pc_heartbeat else None
            self._record_cmd_sender("start", self._test_id, target_ip)

    def _stop_recording(self) -> None:
        if self._recording_test_id != 0:
            if self._record_cmd_sender:
                target_ip = self._pc_heartbeat.pc_ip if self._pc_heartbeat else None
                self._record_cmd_sender("stop", self._recording_test_id, target_ip)
            self._recording_test_id = 0

    def loop(self) -> None:
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

            now = time.monotonic()
            if now - self._last_fps_time >= 1.0:
                self._last_fps = self._vision_manager.get_pipeline_fps("cam_main")
                self._last_fps_time = now
                infer_fps = self._latest_ai.get("infer_fps", 0.0)
                infer_ms = self._latest_ai.get("infer_avg_ms", 0.0)
                if infer_fps > 0:
                    log_print(f"[AI] YOLO infer: {infer_fps:.1f} fps, {infer_ms:.2f} ms")

        self._wdt_feed()
        self._wdt_count += 1
        if self._wdt_count % 200 == 0:
            log_print(f"[WDT] coord feed #{self._wdt_count}")

        with self._cmd_lock:
            streaming = self._streaming_type
            if streaming != 0 and self._vision_state != VisionState.CALIB:
                payload = self._build_stream_payload(streaming)
                if payload:
                    current_seq = self._stream_seq
                    frame = build_data_stream_frame(
                        current_seq, streaming, payload)
                    self._send(frame)
                    self._stream_seq = (current_seq + 1) & 0xFF

                    self._stream_log_count += 1
                    type_changed = (streaming != self._last_streamed_type)
                    if type_changed or self._stream_log_count >= 30:
                        self._stream_log_count = 0
                        self._last_streamed_type = streaming
                        type_name = DATA_TYPE_NAMES.get(streaming, f"0x{streaming:02X}")
                        desc = self._stream_payload_desc(streaming)
                        log_print(f"[UART TX] DATA_STREAM seq={current_seq} type={type_name} {desc}")

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
                    self._ai_seq += 1
                    self._ai_new = True

    # ===== Master events =====

    def _on_cmd_request(self, event: CmdRequestEvent) -> None:
        data_type = event.data_type
        dt_name = DATA_TYPE_NAMES.get(data_type, f"0x{data_type:02X}")
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._master_linked = True

        if data_type not in SUPPORTED_DATA_TYPES:
            log_print(f"[UART TX] CMD_NACK type={dt_name} reason=UNSUPPORTED_TYPE")
            frame = build_cmd_nack_frame(data_type, NACK_UNSUPPORTED_TYPE)
            self._send(frame)
            return

        if self._ai:
            nick = _DATA_TYPE_MODEL.get(data_type)
            if nick and self._ai.active_model != nick:
                if not self._ai.switch(nick):
                    log_print(f"[UART TX] CMD_NACK type={dt_name} reason=NOT_READY")
                    frame = build_cmd_nack_frame(data_type, NACK_NOT_READY)
                    self._send(frame)
                    return

        with self._cmd_lock:
            self._streaming_type = data_type
            self._stream_seq = 0
        payload_size = DATA_PAYLOAD_SIZES.get(data_type, 0)
        if payload_size is None:
            payload_size = 0
        log_print(f"[UART TX] CMD_ACK type={dt_name} freq=60fps size={payload_size}B")

        if self._vision_state == VisionState.CALIB:
            self._send(build_cmd_ack_frame(data_type, 60, payload_size))
            return

        frame = build_cmd_ack_frame(data_type, 60, payload_size)
        self._send(frame)

        self.change_state(VisionState.STREAMING)

    def _on_cmd_stop(self, event: CmdStopEvent) -> None:
        if self._vision_state == VisionState.CALIB:
            return
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._streaming_type = 0
            self._stream_seq = 0
        log_print("[UART TX] CMD_ACK type=STOP")
        frame = build_cmd_ack_frame(0x00, 0, 0)
        self._send(frame)

        self.change_state(VisionState.IDLE)

    def _on_emergency(self, event: EmergencyStopEvent) -> None:
        with self._cmd_lock:
            self._streaming_type = 0
        self.change_state(VisionState.ERROR)
        log_print(f"[Emergency] Emergency stop: reason={event.reason}")

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
        elif data_type == DATA_PENDULUM_POSITION:
            return self._build_pendulum_position_payload()
        return None

    def _stream_payload_desc(self, data_type: int) -> str:
        ai = self._latest_ai
        line = self._latest_line
        detections = ai.get("detections", [])

        if data_type == DATA_LINE_POSITION:
            pe_x = line.get("percent_error_x", 0)
            pe_y = line.get("percent_error_y", 0)
            found = 1 if line.get("target_found", False) else 0
            state = int(self._vision_state)
            return f"pe_x={pe_x} pe_y={pe_y} found={found} state={state}"

        if data_type == DATA_TARGET_POSITION:
            if detections:
                best = max(detections, key=lambda d: d.score)
                return f"x={int(best.x)} y={int(best.y)} conf={int(best.score * 255)} found=1"
            return "x=0 y=0 conf=0 found=0"

        if data_type == DATA_TARGET_COUNT:
            return f"count={len(detections)}"

        if data_type == DATA_DETECTION_STATUS:
            return f"vstate=0 vflags=0 count={len(detections)}"

        if data_type == DATA_ALL_TARGETS:
            count = min(len(detections), 16)
            items = " ".join(
                f"[{i}:cls={d.class_id} x={int(d.x)} y={int(d.y)}]"
                for i, d in enumerate(detections[:count])
            )
            return f"count={count} {items}"

        if data_type == DATA_SEGMENTATION_MASK:
            segments = ai.get("segments", [])
            count = min(len(segments), 4)
            items = " ".join(
                f"[{i}:cls={s['class_id']} cx={s['center_x']} cy={s['center_y']} area={s['area_px']}]"
                for i, s in enumerate(segments[:count])
            )
            return f"count={count} {items}"

        if data_type == DATA_PENDULUM_POSITION:
            if self._ball_armed and self._ball_last_cx is not None:
                calib = self._rail_calib
                if calib is not None and calib.calibrated:
                    dist_px = calib.project(self._ball_last_cx, self._ball_last_cy)
                else:
                    dist_px = self._ball_last_cx - self._frame_width / 2.0
                half = max(self._frame_width, self._frame_height) / 2.0
                pe_x = int((dist_px / half) * 5000.0)
                ball_cm = dist_px / self._pixels_per_cm
                v_px = self._ab_vx if self._ab_ready else 0.0
                ball_v = int(round(v_px / self._pixels_per_cm * 100))
                return f"pe_x={pe_x} ball_cm={ball_cm:.1f} found=1 v={ball_v}"
            return "pe_x=0 ball_cm=0 found=0 v=0"

        return ""

    def _build_line_position_payload(self) -> bytes:
        data = self._latest_line
        pe_x = data.get("percent_error_x", 0)
        pe_y = data.get("percent_error_y", 0)
        target_found = data.get("target_found", False)
        flags = 1 if target_found else 0
        state = int(self._vision_state)
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
        return bytes([min(len(detections), 255)])

    def _build_detection_status_payload(self) -> bytes:
        data = self._latest_ai
        detections = data.get("detections", [])
        count = len(detections)
        visual_state = int(self._vision_state)
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

    @property
    def _max_displacement_px(self) -> float:
        """Euclidean spatial gating threshold (pixels).

        MAX_DISPLACEMENT_PX = (MAX_BALL_SPEED / MIN_FPS) * pixels_per_cm * SAFETY_FACTOR
        """
        return (_MAX_BALL_SPEED_CM_S / _BALL_MIN_FPS) * self._pixels_per_cm * _VEL_SAFETY_FACTOR

    # ===== ball detection helpers =====

    def _on_ball_invalid(self) -> None:
        self._ball_hit_count = 0
        self._ball_miss_count += 1
        if self._ball_miss_count >= _BALL_DROP_FRAMES:
            self._ball_armed = False
            self._ball_last_cx = None
            self._ball_last_cy = None
            self._ab_ball_filter_reset()

    # ===== α-β filter (X-axis only, pendulum 1D groove) =====

    def _ab_ball_filter_reset(self) -> None:
        self._ab_ready = False
        self._ab_locked = False
        self._ab_lock_count = 0
        self._ab_bias_sum = 0.0
        self._ab_x = 0.0
        self._ab_vx = 0.0
        self._ab_ts_ns = 0

    def _ab_ball_filter_apply(
        self, zx: float, is_new_frame: bool, t_ns: int
    ) -> tuple[float, float, bool]:
        """Apply α-β filter to 1D position measurement.

        Returns (x_filtered, vx_filtered, is_locked).
        is_new_frame=True triggers UPDATE step, else PREDICT (extrapolate only,
        does not mutate internal state).
        """
        if not self._ab_ready:
            if is_new_frame:
                self._ab_x = zx
                self._ab_vx = 0.0
                self._ab_ts_ns = t_ns
                self._ab_ready = True
            return zx, 0.0, False

        if is_new_frame:
            dt_s = (t_ns - self._ab_ts_ns) / 1_000_000_000.0
            if dt_s <= 0.0:
                dt_s = 0.025

            x_pred = self._ab_x + self._ab_vx * dt_s
            residual = zx - x_pred

            if self._ab_locked:
                self._ab_bias_sum += residual
                if abs(residual) > _AB_UNLOCK_JUMP_PX or abs(self._ab_bias_sum) > _AB_UNLOCK_BIAS_PX:
                    self._ab_locked = False
                    self._ab_lock_count = 0
                    self._ab_bias_sum = 0.0
                elif residual * self._ab_bias_sum < 0:
                    self._ab_bias_sum = float(residual)
                    self._ab_x = zx
                    self._ab_vx = 0.0
                    self._ab_ts_ns = t_ns
                    return zx, 0.0, True
                else:
                    self._ab_x = zx
                    self._ab_vx = 0.0
                    self._ab_ts_ns = t_ns
                    return zx, 0.0, True

            self._ab_x = x_pred + _AB_ALPHA * residual
            self._ab_vx = self._ab_vx + (_AB_BETA / dt_s) * residual
            self._ab_ts_ns = t_ns

            speed = abs(self._ab_vx)
            if speed < _AB_V_MIN_LOCK:
                self._ab_lock_count += 1
                if self._ab_lock_count >= _AB_LOCK_FRAMES:
                    self._ab_locked = True
                    self._ab_vx = 0.0
                    self._ab_bias_sum = 0.0
            else:
                self._ab_lock_count = 0

            return self._ab_x, self._ab_vx, self._ab_locked
        else:
            if self._ab_locked:
                return self._ab_x, 0.0, True

            dt_s = (t_ns - self._ab_ts_ns) / 1_000_000_000.0
            if dt_s <= 0.0:
                return self._ab_x, self._ab_vx, self._ab_locked

            x_pred = self._ab_x + self._ab_vx * dt_s
            return x_pred, self._ab_vx, self._ab_locked

    def _build_pendulum_position_payload(self) -> Optional[bytes]:
        data = self._latest_ai
        detections = data.get("detections", [])
        ball = max((d for d in detections if d.class_id == 0), key=lambda d: d.score, default=None)

        is_new_frame = self._ai_new
        self._ai_new = False
        t_ns = time.perf_counter_ns()

        if ball is None:
            self._on_ball_invalid()
            if not self._ball_armed:
                return None
            cx_f, vx, _ = self._ab_ball_filter_apply(0.0, False, t_ns)
            cy_f = self._ball_last_cy if self._ball_last_cy is not None else 0.0
        else:
            cx = ball.x + ball.w / 2
            cy = ball.y + ball.h / 2

            if is_new_frame:
                if self._ball_armed and self._ball_last_cx is not None and self._ball_last_cy is not None:
                    max_disp_sq = self._max_displacement_px ** 2
                    dx = cx - self._ball_last_cx
                    dy = cy - self._ball_last_cy
                    if dx * dx + dy * dy > max_disp_sq:
                        self._on_ball_invalid()
                        return None

                self._ball_hit_count = min(self._ball_hit_count + 1, _BALL_ARM_FRAMES)
                self._ball_miss_count = 0

                if self._ball_hit_count >= _BALL_ARM_FRAMES:
                    self._ball_armed = True

                if not self._ball_armed:
                    return None

                self._ball_last_cx = cx
                self._ball_last_cy = cy

                infer_ts = getattr(self._ai, "infer_timestamp_ns", 0)
                update_ts = infer_ts if infer_ts > 0 else t_ns
                cx_f, vx, _ = self._ab_ball_filter_apply(cx, True, update_ts)
                cy_f = cy
            else:
                if not self._ball_armed:
                    return None
                cx_f, vx, _ = self._ab_ball_filter_apply(0.0, False, t_ns)
                cy_f = self._ball_last_cy if self._ball_last_cy is not None else 0.0

        calib = self._rail_calib
        if calib is not None and calib.calibrated:
            dist_px = calib.project(cx_f, cy_f)
        else:
            dist_px = cx_f - self._frame_width / 2.0
        half = max(self._frame_width, self._frame_height) / 2.0
        pe_x = max(-32768, min(32767, int(((dist_px) / half) * 5000.0)))
        ball_cm = dist_px / self._pixels_per_cm
        ball_cm_scaled = max(-32768, min(32767, int(ball_cm * 100)))

        ball_val = int(round(vx / self._pixels_per_cm * 100))
        ball_val = max(-32768, min(32767, ball_val))

        return (pe_x.to_bytes(2, 'little', signed=True) +
                ball_cm_scaled.to_bytes(2, 'little', signed=True) +
                bytes([0x01, 0x00]) +
                ball_val.to_bytes(2, 'little', signed=True))

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
            "state": self._vision_state.name,
            "state_id": int(self._vision_state),
            "link_active": linked,
            "det_count": len(detections),
            "fps": self._last_fps,
        }
