from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from collections import deque
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference
from utils.log_util import log_print

from .pipeline_camera import PipelineCamera
from .performance import profiler

profiler._enabled = False

_module_dir = os.path.dirname(__file__)


class VisionManager:
    _DISPLAY_TEXT_SCALE = 3.0
    _DISPLAY_TEXT_THICKNESS = 2

    def __init__(self, camera_hub: CameraHub, config_path: str = None, ai: Optional[AIInference] = None) -> None:
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
        self._pending_results: deque = deque(maxlen=5)
        self._display_frame = None

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._capture_sink: callable = None
        self._capture_seq: int = 0
        self._CAPTURE_EVERY_N: int = 4

        self._exit_icon = None
        self._exit_icon_size: int = 0
        self._exit_icon_margin: int = 0

        self._calib_button_icon = None
        self._calib_button_size: int = 48
        self._calib_button_margin: int = 12
        self._calib_button_visible: bool = False
        self._calib_flash_until: float = 0.0
        self._calib_flash_rect = None  # (x, y, w, h)

        self._test_id: int = 0
        self._hdr_prefix: str = ""
        self._test_str_cached: str = ""
        self._test_str_width_cached: int = 0
        self._time_str_cached: str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self._time_str_width_cached: int = 0
        self._time_counter: int = 0

        self._aec_enabled: bool = False
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

    def set_exit_icon(self, icon, icon_size: int = 48, margin: int = 12) -> None:
        self._exit_icon = icon
        self._exit_icon_size = icon_size
        self._exit_icon_margin = margin

    def set_calib_button(self, icon, size: int = 48, margin: int = 12) -> None:
        self._calib_button_icon = icon
        self._calib_button_size = size
        self._calib_button_margin = margin

    def set_calib_button_visible(self, v: bool) -> None:
        self._calib_button_visible = v

    def get_calib_button_rect(self) -> Optional[tuple]:
        if self._display_frame is None or not self._calib_button_visible:
            return None
        frame = self._display_frame
        w = frame.width()
        h = frame.height()
        size = self._calib_button_size
        margin = self._calib_button_margin
        bx = w - size - margin
        by = h - size - 8
        return (bx, by, size, size)

    def trigger_calib_flash(self, bbox: tuple) -> None:
        self._calib_flash_until = time.monotonic() + 1.0
        self._calib_flash_rect = bbox

    def set_test_id(self, test_id: int) -> None:
        self._test_id = test_id

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
                self._aec_cfg = cfg
                logger.info("AEC enabled: target_mean=%s, interval=%s frames",
                            cfg.get("target_mean"), cfg.get("adjust_interval_frames"))
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
                    self._pending_results.append(all_results)

                for cb in self._result_callbacks:
                    try:
                        cb(all_results)
                    except Exception as e:
                        pass

                if self._aec_enabled and any_fresh:
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
                if _frame_count % 60 == 0:
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

    def _update_display_frame(self) -> None:
        if not self._ai or not self._ai.loaded:
            return
        for pid, pipe in list(self._pipelines.items()):
            raw_img = getattr(pipe.camera, "last_raw", None)
            if raw_img is None:
                continue
            fps = self.get_pipeline_fps(pid)
            ai_result = pipe.last_results.get("ai_inference")
            detections = []
            if ai_result is not None and ai_result.success:
                detections = ai_result.result_data.get("detections", [])

            self._draw_overlays(raw_img, pid, fps, detections)
            if self._calib_button_visible:
                self._draw_calib_button(raw_img)
            self._draw_calib_flash(raw_img)
            self._draw_exit_icon(raw_img)
            self._display_frame = raw_img
            self._capture_seq += 1
            if self._capture_sink is not None and self._capture_seq % self._CAPTURE_EVERY_N == 0:
                self._capture_sink(raw_img)
            return

    def _adjust_exposure(self) -> None:
        if not self._aec_enabled:
            return
        cam = self._hub.get("main")
        if cam is None:
            return
        frame = getattr(cam, "last_frame", None)
        if frame is None:
            return
        try:
            h, w = frame.shape[:2]
            roi = self._aec_cfg.get("roi", [0.4, 0.7, 0.0, 1.0])
            ry0 = int(h * roi[0])
            ry1 = int(h * roi[1])
            rx0 = int(w * roi[2])
            rx1 = int(w * roi[3])
            if ry1 <= ry0 or rx1 <= rx0:
                return
            patch = frame[ry0:ry1, rx0:rx1]
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

            max_i = self._aec_cfg.get("max_i", 100)
            self._aec_err_i += err
            self._aec_err_i = max(-max_i, min(max_i, self._aec_err_i))

            kp = self._aec_cfg.get("kp", 0.5)
            ki = self._aec_cfg.get("ki", 0.05)
            delta = kp * err + ki * self._aec_err_i

            last_gain = cam.last_gain
            if last_gain is None:
                return
            new_gain = int(last_gain + delta)
            gain_min = self._aec_cfg.get("gain_min", 50)
            gain_max = self._aec_cfg.get("gain_max", 600)
            new_gain = max(gain_min, min(gain_max, new_gain))

            if new_gain != last_gain:
                cam.set_gain(new_gain)
                logger.debug("AEC: gain %s->%s  mean=%.1f  ema=%.1f  err=%.1f  delta=%.1f",
                             last_gain, new_gain, mean_val, self._aec_ema, err, delta)
        except Exception:
            logger.warning("AEC adjust_exposure failed", exc_info=True)

    def _draw_overlays(self, img, pipeline_id: str, fps: float, detections) -> None:
        try:
            import maix.image
        except ImportError:
            return

        self._draw_header(img, pipeline_id, fps)
        self._draw_detections(img, detections, maix.image)

    def _draw_header(self, img, pipeline_id: str, fps: float) -> None:
        try:
            import maix.image
        except ImportError:
            return
        w = img.width()
        bar_h = 32
        try:
            img.draw_rect(0, 0, w, bar_h, color=maix.image.COLOR_BLACK, thickness=-1)
        except Exception:
            pass
        if not self._hdr_prefix:
            self._hdr_prefix = f"{pipeline_id}  FPS:"
        try:
            img.draw_string(6, 2, f"{self._hdr_prefix}{fps:.1f}",
                            color=maix.image.COLOR_WHITE, scale=1.2, thickness=1)
        except Exception:
            pass
        self._time_counter += 1
        if self._time_counter % 60 == 0:
            self._time_str_cached = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if not self._time_str_width_cached:
            self._time_str_width_cached = maix.image.string_size(
                self._time_str_cached, scale=1.2, thickness=1)[0]
        try:
            img.draw_string((w - self._time_str_width_cached) // 2, 2,
                            self._time_str_cached,
                            color=maix.image.COLOR_WHITE, scale=1.2, thickness=1)
        except Exception:
            pass
        if self._test_id > 0:
            try:
                test_str = f"第{self._test_id}次测试"
                if test_str != self._test_str_cached:
                    self._test_str_cached = test_str
                    self._test_str_width_cached = maix.image.string_size(
                        test_str, scale=1.2, thickness=1)[0]
                img.draw_string(w - self._test_str_width_cached - 8, 2,
                                self._test_str_cached,
                                color=maix.image.COLOR_WHITE, scale=1.2, thickness=1)
            except Exception:
                pass

    def _draw_detections(self, img, detections, maix_image) -> None:
        labels = self._ai.labels if hasattr(self._ai, "labels") and self._ai.labels else []

        for det in detections:
            x1, y1 = det.x, det.y
            x2 = x1 + det.w
            y2 = y1 + det.h

            label_text = (
                labels[det.class_id]
                if det.class_id < len(labels)
                else str(det.class_id)
            )
            text = f"{label_text}:{det.score:.2f}"

            try:
                size = maix_image.string_size(text, scale=self._DISPLAY_TEXT_SCALE, thickness=3)
                tw, th = size[0], size[1]
            except Exception:
                tw, th = 80, 24 * 3

            bar_h = th + 6
            label_y = y1 - bar_h
            if label_y < 34:
                label_y = y1 if y1 >= 34 else 34

            try:
                img.draw_rect(x1, label_y, tw + 8, bar_h, color=maix_image.COLOR_BLACK, thickness=-1)
            except Exception:
                pass
            try:
                img.draw_rect(x1, y1, det.w, det.h, color=maix_image.COLOR_GREEN, thickness=3)
            except Exception:
                pass
            try:
                img.draw_string(x1 + 4, label_y + 2, text, color=maix_image.COLOR_WHITE, scale=self._DISPLAY_TEXT_SCALE, thickness=3)
            except Exception:
                pass

            if det.mask_stats is not None:
                self._draw_mask_center(img, det)
                self._draw_area_text(img, det, y2, maix_image)

    @staticmethod
    def _draw_area_text(img, det, y_below, maix_image) -> None:
        stats = det.mask_stats
        if stats is None or stats.area_px == 0:
            return
        text = f"area:{stats.area_px}px"
        try:
            img.draw_string(det.x, y_below + 4, text, color=maix_image.COLOR_YELLOW, scale=1.5, thickness=2)
        except Exception:
            pass

    @staticmethod
    def _draw_mask_center(img, det) -> None:
        try:
            import maix.image
        except ImportError:
            return
        stats = det.mask_stats
        if stats is None or stats.area_px == 0:
            return
        cx = int(stats.center_x)
        cy = int(stats.center_y)
        try:
            img.draw_circle(cx, cy, 3, color=maix.image.COLOR_RED, thickness=-1)
        except Exception:
            try:
                img.draw_rect(cx - 2, cy - 2, 5, 5, color=maix.image.COLOR_RED, thickness=-1)
            except Exception:
                pass

    def _draw_exit_icon(self, img) -> None:
        if self._exit_icon is None:
            return
        try:
            import maix.image as _mi
            h = img.height()
            by = h - self._exit_icon_size - 8
            bx = self._exit_icon_margin
            img.draw_rect(bx - 2, by - 2, self._exit_icon_size + 4,
                          self._exit_icon_size + 4, color=_mi.COLOR_BLACK, thickness=-1)
            img.draw_image(bx, by, self._exit_icon)
        except Exception:
            pass

    def _draw_calib_button(self, img) -> None:
        if self._calib_button_icon is None:
            return
        try:
            import maix.image as _mi
            h = img.height()
            w = img.width()
            size = self._calib_button_size
            margin = self._calib_button_margin
            bx = w - size - margin
            by = h - size - 8
            img.draw_rect(bx - 2, by - 2, size + 4,
                          size + 4, color=_mi.COLOR_BLACK, thickness=-1)
            img.draw_image(bx, by, self._calib_button_icon)
        except Exception:
            from utils.log_util import log_print
            log_print("[CALIB] Button draw exception")

    def _draw_calib_flash(self, img) -> None:
        if time.monotonic() >= self._calib_flash_until:
            return
        if self._calib_flash_rect is None:
            return
        try:
            import maix.image as _mi
            rect = self._calib_flash_rect
            img.draw_rect(int(rect[0]), int(rect[1]),
                          int(rect[2]), int(rect[3]),
                          color=_mi.COLOR_GREEN, thickness=3)
        except Exception:
            pass

    def drain_results(self):
        results = []
        while self._pending_results:
            results.append(self._pending_results.popleft())
        return results

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
