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
from framework.hal.interface import AIInference, frame_for_display, frame_size
from framework.slot import Slot
from framework.ui.composer import HudComposer
from framework.ui.view_state import ViewState
from utils.log_util import log_print

from .pipeline_camera import PipelineCamera
from .performance import profiler

profiler._enabled = False

_module_dir = os.path.dirname(__file__)


class VisionManager:
    def __init__(self, camera_hub: CameraHub, config_path: str = None,
                 ai: Optional[AIInference] = None,
                 canvas_factory: Optional[Callable[[Any], Any]] = None,
                 layout_name: str = "device") -> None:
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
        # When to copy, when to redraw, which buffer to reuse - that is
        # HudComposer's job.  This class only decides what the frame says.
        self._composer: Optional[HudComposer] = None

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._capture_sink: callable = None
        self._capture_last: float = 0.0
        # Fixed-cadence frame push to the capture sink, independent of vision
        # loop jitter. 0.05s -> 20fps.
        self._CAPTURE_INTERVAL_S: float = 0.05

        # The HUD is data: a layout turns ViewState into a Scene, the platform
        # Canvas renders it.  Without a canvas factory there is no HUD at all,
        # and nothing here knows what a pixel is.
        self._canvas_factory = canvas_factory
        if canvas_factory is not None:
            from framework.ui.layout import load as _load_layout

            self._composer = HudComposer(canvas_factory, _load_layout(layout_name))

        # Detection-list overlay (top-left text list).  Off for competition
        # (display.detection_list: false) to avoid the per-frame text cost.
        self._detection_list_enabled: bool = True

        # Refreshed about once a second: reading the clock is the cost,
        # drawing three strings is not.
        self._clock_text: str = ""
        self._clock_at: float = 0.0

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
        if self._composer is not None:
            self._composer.set_canvas_factory(factory)

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
        """The most recently composed frame, or None without a composer."""
        return self._composer.frame if self._composer is not None else None

    def get_composed(self):
        """The latest composition (frame + scene + generation), or None.

        Sinks take this instead of ``get_display_frame()``: the generation is
        what tells them whether the content actually changed.
        """
        return self._composer.current() if self._composer is not None else None

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
        if enabled is None or self._composer is None:
            return
        self._composer.set_enabled(bool(enabled))

    def set_display_cadence_ms(self, interval_ms: Optional[float]) -> None:
        """Match the HUD rebuild window to the LCD present interval.

        Drawing at 30fps while the panel presents 10 times a second means two
        thirds of the copies are never shown; matching both cadences gives
        "draw one, show one".  ``None`` keeps the built-in 30fps window.
        """
        if interval_ms is None or self._composer is None:
            return
        try:
            seconds = float(interval_ms) / 1000.0
        except (TypeError, ValueError):
            return
        self._composer.set_interval_ms(0 if seconds <= 0.0 else interval_ms)

    def _update_display_frame(self) -> None:
        """Compose one frame per pipeline; the composer decides whether to."""
        if self._composer is None:
            return
        ai_ok = self._ai is not None and self._ai.loaded
        for pid, pipe in list(self._pipelines.items()):
            source = frame_for_display(pipe.camera)
            if source is None:
                continue
            composed = self._composer.compose(
                source,
                self._view_state(pid, pipe, ai_ok),
                getattr(pipe.camera, "frame_serial", None),
            )
            if composed is not None:
                self._maybe_push_capture(composed.frame)

    def _view_state(self, pid: str, pipe: PipelineCamera, ai_ok: bool) -> ViewState:
        """Everything a layout needs from this pipeline, and nothing else."""
        result = pipe.last_results.get("ai_inference")
        detections: Tuple[Any, ...] = ()
        error_code = ""
        if result is not None:
            if result.success:
                detections = tuple(result.result_data.get("detections", ()))
            else:
                # A failed inference must look different from an empty scene,
                # or a broken model is indistinguishable from a picture of
                # nothing.
                error_code = getattr(result, "error_code", "") or ""
        return ViewState(
            pipeline_id=pid,
            fps=self.get_pipeline_fps(pid),
            detections=detections,
            labels=tuple(getattr(self._ai, "labels", ()) or ()),
            ai_ok=ai_ok,
            error_code=error_code,
            reconnecting=bool(getattr(pipe.camera, "is_reconnecting", False)),
            list_enabled=self._detection_list_enabled,
            clock=self._clock_now(),
        )

    def _clock_now(self) -> str:
        """The wall clock, refreshed about once a second."""
        now = time.monotonic()
        if not self._clock_text or now - self._clock_at >= 1.0:
            self._clock_at = now
            self._clock_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return self._clock_text

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
