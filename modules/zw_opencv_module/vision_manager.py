from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference
from framework.slot import Slot
from framework.ui.overlay import Circle, Line, Overlay, Rect, Text
from utils.log_util import log_print

from .pipeline_camera import PipelineCamera
from .performance import profiler

profiler._enabled = False

_module_dir = os.path.dirname(__file__)

# Overlay colours.  These mirror the firmware constants the drawing code used
# before the HUD became primitives, so the device screen looks unchanged.
_COLOR_BLACK = (0, 0, 0)
_COLOR_WHITE = (255, 255, 255)
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)
_COLOR_YELLOW = (255, 255, 0)


class VisionManager:
    _LIST_TEXT_PX = 24
    _LIST_MAX_LINES = 8
    _BOX_THICKNESS = 1
    _CORNER_NUM_PX = 18

    def __init__(self, camera_hub: CameraHub, config_path: str = None,
                 ai: Optional[AIInference] = None,
                 canvas_factory: Optional[Callable[[Any], Any]] = None) -> None:
        self._hub = camera_hub
        self._config_path = config_path or os.path.join(
            _module_dir, "config", "vision_config.yaml"
        )
        self._pipelines: Dict[str, PipelineCamera] = {}
        self._ai = ai
        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []
        self._event_bus = None

        self._fps_data: Dict[str, dict] = {}

        self._any_fresh = False
        self._results_slot: Slot = Slot()
        self._results_seen_gen: int = 0
        self._display_frame = None
        # Rebuild the display only when the sensor delivers a new frame
        # (frame_serial changed), skipping the 1.29MB copy and full redraw on
        # stale iterations.
        self._last_display_serial: int = -1
        # Even on a fresh frame, throttle the rebuild to
        # _DISPLAY_REBUILD_INTERVAL_S. Result delivery via _results_slot does
        # not depend on the display frame, so intermediate frames skip the
        # 10-16ms copy. Consumers of a reused frame only read it.
        self._DISPLAY_REBUILD_INTERVAL_S: float = 0.033  # ~30fps rebuild
        self._DISPLAY_REBUILD_MIN_S: float = 0.033  # floor: never rebuild per frame
        self._last_display_rebuild: float = 0.0
        # With no LCD, nothing reads the composed frame, so rebuilding it is
        # wasted work on the vision thread.
        self._render_enabled: bool = True

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._capture_sink: callable = None
        self._capture_last: float = 0.0
        # Fixed-cadence frame push to the capture sink, independent of vision
        # loop jitter. 0.05s -> 20fps.
        self._CAPTURE_INTERVAL_S: float = 0.05

        # The HUD is described as primitives and rendered by the platform's
        # Canvas, so this class never touches a pixel API and the same list of
        # primitives can be asserted in a test without a camera.
        self._canvas_factory = canvas_factory
        self._overlay = Overlay()
        self._hud_error_logged = False

        # Detection-list overlay (top-left text list).  Off for competition
        # (display.detection_list: false) to avoid per-frame text cost.
        self._detection_list_enabled: bool = True

        self._hdr_prefix: str = ""
        self._time_str_cached: str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self._time_counter: int = 0

        self._aec_enabled: bool = False
        self._aec_cam_id: str = ""
        self._aec_cfg: Dict[str, Any] = {}
        self._aec_counter: int = 0
        self._aec_err_i: float = 0.0
        self._aec_ema: Optional[float] = None

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def set_capture_sink(self, sink: callable) -> None:
        self._capture_sink = sink

    def set_detection_list_enabled(self, enabled: bool) -> None:
        """Enable/disable the top-left detection text list overlay.

        Off for competition (display.detection_list: false) to avoid the
        per-frame draw_string cost; the bbox + corner number remain.
        """
        self._detection_list_enabled = bool(enabled)

    def set_canvas_factory(self, factory: Optional[Callable[[Any], Any]]) -> None:
        """Set (or clear) the factory that turns a frame into a Canvas.

        The platform supplies it, so this module never imports a drawing
        backend.  ``None`` disables HUD rendering without touching the result
        path.
        """
        self._canvas_factory = factory

    def start(self) -> None:
        if self._running:
            return

        if not os.path.exists(self._config_path):
            return

        try:
            with open(self._config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            logger.error("Failed to load vision config: %s", self._config_path)
            return

        pipelines = cfg.get("pipelines", [])
        for pipe_cfg in pipelines:
            pipeline_id = pipe_cfg.get("pipeline_id", "")
            camera_id = pipe_cfg.get("camera_id", "")
            if not pipeline_id or not camera_id:
                continue

            cam = self._hub.get(camera_id)
            if cam is None:
                continue

            focal_length_mm = getattr(cam, 'focal_length_mm', None)
            sensor_width_mm = getattr(cam, 'sensor_width_mm', None)
            sensor_height_mm = getattr(cam, 'sensor_height_mm', None)
            if focal_length_mm is None:
                logger.debug(
                    "Camera '%s' has no intrinsics; distance calculation disabled",
                    camera_id,
                )

            pipe = PipelineCamera(
                pipeline_id=pipeline_id,
                camera=cam,
                task_configs=pipe_cfg.get("tasks", []),
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                image_width=pipe_cfg.get("width", 640),
                image_height=pipe_cfg.get("height", 480),
                ai=self._ai,
            )
            self._pipelines[pipeline_id] = pipe

        for cid in self._hub.list_ids():
            cam = self._hub.get(cid)
            if cam is None:
                continue
            cfg = getattr(cam, "aec_config", None)
            if cfg and cfg.get("enabled"):
                self._aec_enabled = True
                self._aec_cam_id = cid
                self._aec_cfg = cfg
                log_print(f"AEC enabled on '{cid}': target_mean={cfg.get('target_mean')}, interval={cfg.get('adjust_interval_frames')} frames")
                break

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _process_loop(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("vision_processing")

        _frame_count = 0
        while self._running:
            try:
                self._wdt_feed()
                self._wdt_count += 1
                if self._wdt_count % 100 == 0:
                    log_print(f"[WDT] viz feed #{self._wdt_count}")

                profiler.start("total")
                _, all_results, any_fresh = self.process_all()

                self._update_display_frame()

                if self._event_bus:
                    self._results_slot.publish(all_results)

                for cb in self._result_callbacks:
                    try:
                        cb(all_results)
                    except Exception as e:
                        pass

                if self._aec_enabled:
                    self._aec_counter += 1
                    interval = self._aec_cfg.get("adjust_interval_frames", 30)
                    if self._aec_counter >= interval:
                        self._aec_counter = 0
                        self._adjust_exposure()

                profiler.stop("total")
                time.sleep(0)
                if any_fresh:
                    profiler.end_frame()
                else:
                    time.sleep(0.001)

                _frame_count += 1
                if _frame_count % 120 == 0:
                    gc.collect(0)
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

    def process_all(
        self,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Dict], bool]:
        all_results: Dict[str, Dict] = {}
        any_fresh = False

        for pid, pipe in list(self._pipelines.items()):
            cur_fps = self._fps_data.get(pid, {}).get("fps", 0.0)
            frame, results = pipe.process_frame(fps=cur_fps)

            if frame is not None:
                any_fresh = True
                d = self._fps_data.setdefault(
                    pid, {"count": 0, "start": time.time(), "fps": 0.0}
                )
                d["count"] += 1
                elapsed = time.time() - d["start"]
                if elapsed >= 1.0:
                    d["fps"] = d["count"] / elapsed
                    d["count"] = 0
                    d["start"] = time.time()

            all_results[pid] = results

        return None, all_results, any_fresh

    def get_display_frame(self):
        return self._display_frame

    def set_render_enabled(self, enabled: Optional[bool]) -> None:
        """Turn display-frame composition on/off (``display_enabled``).

        Args:
            enabled: ``False`` stops rebuilding the composed frame entirely;
                ``None`` keeps the current value (missing config key).

        Context:
            With the LCD disabled nothing consumes ``get_display_frame()``,
            so the 10-16ms copy+redraw per rebuild window would be spent for
            no reader. Turning it off also drops the capture sink, which
            publishes the composed frame - a headless project that still
            needs frames should read ``camera.last_raw`` directly instead.
        """
        if enabled is None:
            return
        self._render_enabled = bool(enabled)

    def set_display_cadence_ms(self, interval_ms: Optional[float]) -> None:
        """Align the overlay rebuild window with the LCD present interval.

        Args:
            interval_ms: Present interval from ``display_show_interval_ms``.
                ``None``/``0`` keep the built-in ~30fps window (unchanged
                behaviour); a positive value sets the rebuild window to the
                same period, floored at ~30fps.

        Context:
            Drawing at 30fps while the panel only presents 10 times a second
            means two thirds of the copies are never shown. Matching both
            cadences gives "draw one, show one". The floor keeps a very
            small interval (say 16ms) from turning the vision loop into a
            per-frame copy loop.
        """
        if interval_ms is None:
            return
        try:
            interval_s = float(interval_ms) / 1000.0
        except (TypeError, ValueError):
            return
        if interval_s <= 0.0:
            self._DISPLAY_REBUILD_INTERVAL_S = self._DISPLAY_REBUILD_MIN_S
            return
        self._DISPLAY_REBUILD_INTERVAL_S = max(interval_s, self._DISPLAY_REBUILD_MIN_S)

    def _update_display_frame(self) -> None:
        if not self._render_enabled:
            return
        ai_ok = self._ai is not None and self._ai.loaded
        for pid, pipe in list(self._pipelines.items()):
            raw_img = getattr(pipe.camera, "last_raw", None)
            if raw_img is None:
                continue
            fps = self.get_pipeline_fps(pid)
            ai_result = pipe.last_results.get("ai_inference")
            detections = []
            ai_error_code = ""
            if ai_result is not None:
                if ai_result.success:
                    detections = ai_result.result_data.get("detections", [])
                else:
                    # A failed inference must be visible on the display: an
                    # empty detection list is indistinguishable from an
                    # empty scene.
                    ai_error_code = getattr(ai_result, "error_code", "") or ""

            serial = getattr(pipe.camera, "frame_serial", None)
            if serial is not None and self._display_frame is not None \
                    and serial == self._last_display_serial:
                # No new sensor frame this iteration: skip the 1.29MB copy and
                # the full redraw.  The display thread already skips the
                # unchanged object by identity.  Still honour the capture
                # cadence using the previous display frame.
                self._maybe_push_capture(self._display_frame)
                return

            if self._display_frame is not None:
                now = time.monotonic()
                if now - self._last_display_rebuild < self._DISPLAY_REBUILD_INTERVAL_S:
                    # Fresh frame inside the rebuild window: reuse the previous
                    # display frame. Result delivery does not depend on it, so
                    # the 10-16ms copy stays off the hot path. The reused frame
                    # is only read by the display and capture sink, never mutated.
                    self._last_display_serial = serial if serial is not None else -1
                    self._maybe_push_capture(self._display_frame)
                    return

            self._last_display_serial = serial if serial is not None else -1
            self._last_display_rebuild = time.monotonic()
            disp = raw_img.copy()
            if self._canvas_factory is not None:
                width = self._frame_width(disp)
                self._overlay.clear()
                if ai_ok:
                    self._overlay.add(*self._header_primitives(pid, fps, width))
                    self._overlay.add(*self._detection_primitives(detections))
                    if self._detection_list_enabled:
                        self._overlay.add(
                            *self._detection_list_primitives(detections)
                        )
                if ai_error_code:
                    self._overlay.add(
                        *self._fault_primitives("ai_error", width, ai_error_code)
                    )
                elif not ai_ok:
                    self._overlay.add(*self._fault_primitives("ai_fault", width))
                if getattr(pipe.camera, "is_reconnecting", False):
                    self._overlay.add(*self._fault_primitives("camera", width))
                try:
                    self._overlay.render(self._canvas_factory(disp))
                except Exception:
                    # A broken HUD must not take the vision thread with it: the
                    # loop guard would drop result delivery to 1 Hz.  Log once,
                    # then keep the frames flowing without the overlay.
                    if not self._hud_error_logged:
                        logger.warning("HUD render failed; continuing without it",
                                       exc_info=True)
                        self._hud_error_logged = True
            self._display_frame = disp
            self._maybe_push_capture(disp)
            return

    @staticmethod
    def _frame_width(frame) -> int:
        """The pixel width of a frame, maix Image or ndarray.

        The overlay only needs the width for the header bar, so it is read
        here instead of passing a size object through every primitive helper.
        """
        getter = getattr(frame, "width", None)
        if callable(getter):
            return int(getter())
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
        return 0

    def _maybe_push_capture(self, frame) -> None:
        """Push *frame* to the capture sink on the fixed cadence.

        Additive cadence keeps the average rate exactly _CAPTURE_INTERVAL_S;
        the clamp prevents a burst of catch-up pushes after a long stall.
        """
        if self._capture_sink is None:
            return
        now = time.monotonic()
        if now - self._capture_last >= self._CAPTURE_INTERVAL_S:
            self._capture_last += self._CAPTURE_INTERVAL_S
            if self._capture_last < now - 2.0 * self._CAPTURE_INTERVAL_S:
                self._capture_last = now
            self._capture_sink(frame)

    def _fault_primitives(self, kind: str, width: int, code: str = "") -> List[Any]:
        """The banner for one fault, in frame coordinates.

        A fault has to be visible on the display: an empty detection list
        looks exactly like an empty scene, and a silent AI failure is
        indistinguishable from "nothing there".
        """
        if kind == "camera":
            text = "CAMERA FAULT - RECONNECTING"
            top, background, ink = 40, _COLOR_YELLOW, _COLOR_BLACK
            size = 15
        elif kind == "ai_error":
            text = f"AI FAULT - {code}" if code else "AI FAULT"
            top, background, ink = 0, _COLOR_RED, _COLOR_WHITE
            size = 18
        else:
            text = "AI FAULT - MODEL NOT LOADED"
            top, background, ink = 0, _COLOR_RED, _COLOR_WHITE
            size = 18
        return [
            Rect(0, top, width, 32, background, fill=True),
            Text(6, top + 2, text, size, ink),
        ]

    def _adjust_exposure(self) -> None:
        if not self._aec_enabled:
            return
        cam = self._hub.get(self._aec_cam_id)
        if cam is None:
            log_print(f"AEC skipped: camera '{self._aec_cam_id}' not found")
            return
        # last_frame is a cached BGR ndarray from read_raw(); 3-channel required
        frame = getattr(cam, "last_frame", None)
        if frame is None:
            log_print(f"AEC skipped: no frame from camera '{self._aec_cam_id}'")
            return
        try:
            h, w = frame.shape[:2]
            # ROI: [y0_frac, y1_frac, x0_frac, x1_frac] as fractions of height/width
            # e.g. [0.4, 0.7, 0.0, 1.0] = middle 30% height, full width
            roi = self._aec_cfg.get("roi", [0.4, 0.7, 0.0, 1.0])
            if len(roi) != 4:
                logger.warning("AEC: roi must have 4 elements, got %d", len(roi))
                return
            ry0 = max(0, min(h - 1, int(h * roi[0])))
            ry1 = max(0, min(h, int(h * roi[1])))
            rx0 = max(0, min(w - 1, int(w * roi[2])))
            rx1 = max(0, min(w, int(w * roi[3])))
            if ry1 <= ry0 or rx1 <= rx0:
                log_print(f"AEC skipped: empty ROI slice [{ry0}:{ry1}, {rx0}:{rx1}]")
                return
            patch = frame[ry0:ry1, rx0:rx1]
            # frame is BGR (see read_raw); COLOR_BGR2GRAY is correct
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))

            alpha = self._aec_cfg.get("ema_alpha", 0.1)
            if self._aec_ema is None:
                self._aec_ema = mean_val
            else:
                self._aec_ema = (1.0 - alpha) * self._aec_ema + alpha * mean_val

            target = self._aec_cfg.get("target_mean", 80)
            deadband = self._aec_cfg.get("deadband", 8)
            err = target - self._aec_ema
            if abs(err) <= deadband:
                return

            kp = self._aec_cfg.get("kp", 0.5)
            ki = self._aec_cfg.get("ki", 0.05)

            # clamp candidate gain first to decide if integrator should run
            last_gain = getattr(cam, "last_gain", None)
            if last_gain is None:
                log_print(f"AEC skipped: last_gain unknown on '{self._aec_cam_id}'")
                return
            gain_min = self._aec_cfg.get("gain_min", 50)
            gain_max = self._aec_cfg.get("gain_max", 600)
            delta = kp * err
            new_gain = int(last_gain + delta)
            new_gain = max(gain_min, min(gain_max, new_gain))

            # conditional integration: only accumulate when actuator is not saturated
            if gain_min < new_gain < gain_max:
                max_i = self._aec_cfg.get("max_i", 100)
                self._aec_err_i += err
                self._aec_err_i = max(-max_i, min(max_i, self._aec_err_i))
                delta = kp * err + ki * self._aec_err_i
                new_gain = int(last_gain + delta)
                new_gain = max(gain_min, min(gain_max, new_gain))

            if new_gain != last_gain:
                setter = getattr(cam, "set_gain", None)
                if setter is not None:
                    ok = setter(new_gain)
                    log_print(f"AEC: gain {last_gain}->{new_gain}  mean={mean_val:.1f}  ema={self._aec_ema:.1f}  err={err:.1f}  delta={delta:.1f}  ok={ok}")
        except Exception:
            logger.warning("AEC adjust_exposure failed", exc_info=True)

    def _header_primitives(self, pipeline_id: str, fps: float, width: int) -> List[Any]:
        """The top bar: pipeline id + fps on the left, clock centred.

        The clock string is refreshed once per second, not once per frame:
        reading the clock is the cost, drawing three strings is not.
        """
        if not self._hdr_prefix:
            self._hdr_prefix = f"{pipeline_id}  FPS:"
        self._time_counter += 1
        if self._time_counter == 1 or self._time_counter % 60 == 0:
            self._time_str_cached = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return [
            Rect(0, 0, width, 32, _COLOR_BLACK, fill=True),
            Text(6, 2, f"{self._hdr_prefix}{fps:.1f}", 18, _COLOR_WHITE),
            Text(width // 2, 2, self._time_str_cached, 18, _COLOR_WHITE,
                 align="center"),
        ]

    def _detection_list_primitives(self, detections) -> List[Any]:
        """The optional top-left list of detections.

        The line height is fixed because the font size is fixed; measuring
        each line would only buy an alignment nobody can see.
        """
        if not detections:
            return []
        labels = self._ai.labels if self._ai is not None else []
        line_spacing = self._LIST_TEXT_PX + 4
        items: List[Any] = []
        for index, det in enumerate(detections, start=1):
            if index > self._LIST_MAX_LINES:
                items.append(Text(
                    6, 34 + self._LIST_MAX_LINES * line_spacing,
                    f"+{len(detections) - self._LIST_MAX_LINES} more",
                    self._LIST_TEXT_PX, _COLOR_BLUE,
                ))
                break
            label = (
                labels[det.class_id]
                if 0 <= det.class_id < len(labels)
                else str(det.class_id)
            )
            items.append(Text(
                6, 34 + (index - 1) * line_spacing,
                f"{index}: {label} {det.score:.2f}",
                self._LIST_TEXT_PX, _COLOR_BLUE,
            ))
        return items

    def _detection_primitives(self, detections) -> List[Any]:
        """One box plus its corner index, and the mask marker when present."""
        items: List[Any] = []
        for index, det in enumerate(detections, start=1):
            items.append(Rect(
                det.x, det.y, det.w, det.h, _COLOR_GREEN,
                width=self._BOX_THICKNESS,
            ))
            # Clamp the index back into the box: a one-digit label in a 20 px
            # box would otherwise be drawn outside it, which is what the old
            # implementation guarded against with max(x1, ...).
            label = str(index)
            label_w = len(label) * int(self._CORNER_NUM_PX * 0.55)
            nx = max(det.x, det.x + det.w - label_w - 2)
            ny = max(det.y, det.y + det.h - self._CORNER_NUM_PX - 2)
            items.append(Text(nx, ny, label, self._CORNER_NUM_PX, _COLOR_BLUE))
            if det.mask_stats is not None:
                items.extend(self._mask_primitives(det))
        return items

    def _mask_primitives(self, det) -> List[Any]:
        """Mask centre dot and area label, for models that emit a mask."""
        stats = det.mask_stats
        if stats is None or stats.area_px == 0:
            return []
        return [
            Circle(int(stats.center_x), int(stats.center_y), 3, _COLOR_RED,
                   fill=True),
            Text(det.x, det.y + det.h + 4, f"area:{stats.area_px}px",
                 22, _COLOR_YELLOW),
        ]

    def drain_results(self):
        """Newest-wins result delivery: one Slot instead of a bounded deque.

        Intermediate result sets are intentionally dropped - the control path
        only ever needed the freshest snapshot (SCHED-03).
        """
        stamped = self._results_slot.load(self._results_seen_gen)
        if stamped is None:
            return []
        self._results_seen_gen = stamped.gen
        return [stamped.value]

    def get_pipeline_fps(self, pipeline_id: str) -> float:
        fd = self._fps_data.get(pipeline_id)
        return fd.get("fps", 0.0) if fd else 0.0

    def release_pipeline(self, pipeline_id: str) -> None:
        pipe = self._pipelines.pop(pipeline_id, None)
        if pipe is not None and hasattr(pipe.camera, 'release'):
            pipe.camera.release()

    def enable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.enable_task(task_name)
        return False

    def disable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.disable_task(task_name)
        return False

    def set_processor_target(self, pipeline_id: str, task_name: str, color) -> None:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            task = pipe.get_task(task_name)
            if task and hasattr(task.processor, "set_target_color"):
                task.processor.set_target_color(color)

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineCamera]:
        return self._pipelines.get(pipeline_id)

    def get_all_results(self) -> Dict[str, Dict]:
        return {
            pid: pipe.last_results
            for pid, pipe in self._pipelines.items()
        }

    def add_result_callback(self, callback: Callable[[Dict], None]) -> None:
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]) -> None:
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def stop(self) -> None:
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def release(self) -> None:
        self.stop()
        self._pipelines.clear()


class _LegacyCameraManagerShim:
    def __init__(self, vision_manager: VisionManager) -> None:
        self._vm = vision_manager

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.enable_task(camera_id, task_name)

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.disable_task(camera_id, task_name)

    def get_all_results(self) -> dict:
        return self._vm.get_all_results()

    def add_result_callback(self, callback) -> None:
        self._vm.add_result_callback(callback)

    def remove_result_callback(self, callback) -> None:
        self._vm.remove_result_callback(callback)

    @property
    def cameras(self) -> dict:
        return {}
