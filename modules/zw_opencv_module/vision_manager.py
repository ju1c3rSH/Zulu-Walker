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
    _LIST_TEXT_SCALE = 1.6
    _LIST_TEXT_THICKNESS = 2
    _LIST_MAX_LINES = 8
    _BOX_THICKNESS = 1
    _CORNER_NUM_SCALE = 1.2
    _CORNER_NUM_THICKNESS = 1

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
        # Frame-freshness gate: display is rebuilt only when a NEW sensor frame
        # arrived (frame_serial changed), skipping the 1.29MB raw_img.copy()
        # + full redraw on stale iterations.
        self._last_display_serial: int = -1

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._capture_sink: callable = None
        self._capture_last: float = 0.0
        # Fixed-cadence JPEG push (independent of vision-loop jitter) so the
        # PC records evenly-timed frames. 0.1s -> 10fps.
        self._CAPTURE_INTERVAL_S: float = 0.1

        self._exit_icon = None
        self._exit_icon_size: int = 0
        self._exit_icon_margin: int = 0

        self._calib_button_icon = None
        self._calib_button_size: int = 48
        self._calib_button_margin: int = 12
        self._calib_button_visible: bool = False
        self._calib_flash_until: float = 0.0
        self._calib_flash_rect = None  # (x, y, w, h)

        self._fill_light_on: bool = False
        self._fill_light_icon_on = None
        self._fill_light_icon_off = None
        self._fill_light_size: int = 48
        self._fill_light_controller = None  # callable(bool) -> bool

        self._rail_draw_enabled: bool = False
        self._rail_provider = None  # callable() -> RailCalibration or None
        self._rail_ppc: float = 50.0
        self._rail_cm_interval: float = 1.0

        self._test_id: int = 0
        self._hdr_prefix: str = ""
        self._test_str_cached: str = ""
        self._time_str_cached: str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self._time_counter: int = 0

        # Persistent header buffer + per-fragment single-slot sprites.
        # The 32px bar (black bg + FPS + time + test id) is composed by
        # blitting pre-rendered RGBA8888 text sprites onto ONE persistent
        # RGB888 buffer; the buffer is allocated once (or on width change)
        # and never re-allocated per rebuild.  Each fragment uses a single
        # cache slot that is re-rasterised only when its string changes,
        # keeping both per-frame and per-second font work minimal.
        self._header_bmp = None
        self._header_dirty = True
        self._hdr_fps_last: Optional[float] = None
        self._hdr_width: int = 0
        self._hdr_fps_sprite = None
        self._hdr_fps_text: str = ""
        self._hdr_time_sprite = None
        self._hdr_time_text: str = ""
        self._hdr_test_sprite = None
        self._hdr_test_text: str = ""

        # cm-ruler label sprites: each numeric label pre-rendered once into a
        # transparent RGBA8888 sprite and blitted each frame.
        self._cm_label_sprites: Dict[str, Any] = {}

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

    def set_rail_draw(self, enabled: bool, provider=None,
                      pixels_per_cm: float = 50.0,
                      cm_interval: float = 1.0) -> None:
        """Enable/disable test-time rail overlay drawing.

        *provider* is a callable returning a ``RailCalibration`` (or None),
        evaluated lazily each frame so Phase-2 origin updates are reflected.
        ``pixels_per_cm`` / ``cm_interval`` drive the simulated cm ruler ticks
        drawn along the rail axis from the origin (for verifying the physical
        pixel-per-cm figure against the real ruler).
        """
        self._rail_draw_enabled = bool(enabled)
        self._rail_provider = provider
        self._rail_ppc = float(pixels_per_cm) if pixels_per_cm > 0 else 50.0
        self._rail_cm_interval = float(cm_interval) if cm_interval > 0 else 1.0

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

    def set_fill_light_button(self, icon_on, icon_off, size: int = 48) -> None:
        self._fill_light_icon_on = icon_on
        self._fill_light_icon_off = icon_off
        self._fill_light_size = size

    def set_fill_light_controller(self, controller) -> None:
        self._fill_light_controller = controller

    def set_fill_light_state(self, on: bool) -> None:
        self._fill_light_on = bool(on)

    def get_fill_light_state(self) -> bool:
        return self._fill_light_on

    def get_fill_light_button_rect(self) -> Optional[tuple]:
        if self._display_frame is None:
            return None
        if self._fill_light_icon_on is None or self._fill_light_icon_off is None:
            return None
        frame = self._display_frame
        w = frame.width()
        h = frame.height()
        size = self._fill_light_size
        bx = (w - size) // 2
        by = h - size - 8
        return (bx, by, size, size)

    def toggle_fill_light(self) -> Optional[bool]:
        """Toggle the fill light. Returns the new state on success, None on failure."""
        target = not self._fill_light_on
        if self._fill_light_controller is None:
            return None
        try:
            ok = self._fill_light_controller(target)
        except Exception:
            return None
        if not ok:
            return None
        self._fill_light_on = target
        return self._fill_light_on

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
                    self._pending_results.append(all_results)

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

    def _update_display_frame(self) -> None:
        ai_ok = self._ai is not None and self._ai.loaded
        for pid, pipe in list(self._pipelines.items()):
            raw_img = getattr(pipe.camera, "last_raw", None)
            if raw_img is None:
                continue
            fps = self.get_pipeline_fps(pid)
            ai_result = pipe.last_results.get("ai_inference")
            detections = []
            if ai_result is not None and ai_result.success:
                detections = ai_result.result_data.get("detections", [])

            serial = getattr(pipe.camera, "frame_serial", None)
            if serial is not None and self._display_frame is not None \
                    and serial == self._last_display_serial:
                # No new sensor frame this iteration: skip the 1.29MB copy and
                # the full redraw.  The display thread already skips the
                # unchanged object by identity.  Still honour the capture
                # cadence using the previous display frame.
                if self._capture_sink is not None:
                    now = time.monotonic()
                    if now - self._capture_last >= self._CAPTURE_INTERVAL_S:
                        self._capture_last += self._CAPTURE_INTERVAL_S
                        if self._capture_last < now - 2.0 * self._CAPTURE_INTERVAL_S:
                            self._capture_last = now
                        self._capture_sink(self._display_frame)
                return

            self._last_display_serial = serial if serial is not None else -1
            disp = raw_img.copy()
            if self._rail_draw_enabled:
                self._draw_rail(disp, detections)
            if ai_ok:
                self._draw_overlays(disp, pid, fps, detections)
            else:
                self._draw_ai_fault_banner(disp)
            if getattr(pipe.camera, "is_reconnecting", False):
                self._draw_camera_fault_banner(disp)
            if self._calib_button_visible:
                self._draw_calib_button(disp)
            self._draw_calib_flash(disp)
            self._draw_fill_light_button(disp)
            self._draw_exit_icon(disp)
            self._display_frame = disp
            if self._capture_sink is not None:
                now = time.monotonic()
                if now - self._capture_last >= self._CAPTURE_INTERVAL_S:
                    # Additive cadence keeps the average rate exactly
                    # _CAPTURE_INTERVAL_S; the clamp prevents a burst of
                    # catch-up pushes after a long vision-loop stall.
                    self._capture_last += self._CAPTURE_INTERVAL_S
                    if self._capture_last < now - 2.0 * self._CAPTURE_INTERVAL_S:
                        self._capture_last = now
                    self._capture_sink(disp)
            return

    def _draw_rail(self, img, detections) -> None:
        """Test overlay: draw the calibrated rail axis, origin, ball projection,
        and simulated cm ruler ticks (for verifying pixels_per_cm)."""
        if not self._rail_draw_enabled:
            return
        provider = self._rail_provider
        if provider is None:
            return
        try:
            import maix.image
        except ImportError:
            return
        try:
            calib = provider()
            if calib is None or not getattr(calib, "calibrated", False):
                return
            w = img.width()
            h = img.height()
            ox = float(calib.origin_x)
            oy = float(calib.origin_y)
            dx = float(calib.dir_cos)
            dy = float(calib.dir_sin)
            L = int((w * w + h * h) ** 0.5)
            x0 = int(ox - dx * L)
            y0 = int(oy - dy * L)
            x1 = int(ox + dx * L)
            y1 = int(oy + dy * L)
            img.draw_line(x0, y0, x1, y1,
                          color=maix.image.COLOR_GREEN, thickness=2)
            img.draw_circle(int(ox), int(oy), 5,
                            color=maix.image.COLOR_RED, thickness=2)

            # Simulated cm ruler: from the origin, along +/- rail direction,
            # draw a perpendicular tick every pixels_per_cm * cm_interval.
            self._draw_cm_ruler(img, ox, oy, dx, dy, w, h)

            if detections:
                ball = max((d for d in detections if d.class_id == 0),
                           key=lambda d: d.score, default=None)
                if ball is not None:
                    cx = ball.x + ball.w / 2
                    cy = ball.y + ball.h / 2
                    dist = calib.project(cx, cy)
                    px = int(ox + dist * dx)
                    py = int(oy + dist * dy)
                    if 0 <= px < w and 0 <= py < h:
                        img.draw_circle(px, py, 4,
                                        color=maix.image.COLOR_YELLOW, thickness=-1)
        except Exception:
            pass

    def _draw_cm_ruler(self, img, ox, oy, dx, dy, w, h) -> None:
        """Draw vertical (axis-perpendicular) cm ticks with labels.

        Ticks span roughly half the rail thickness on either side of the axis.
        Coordinates outside the frame are skipped.  Labels are pre-rendered
        once into transparent RGBA8888 sprites and blitted each frame.
        """
        import maix.image
        step = self._rail_ppc * self._rail_cm_interval
        if step <= 0:
            return
        tick_half = 20  # ~half the ruler tick length in px
        px0 = -dy * tick_half  # perpendicular (up) component
        py0 = dx * tick_half
        n_max = int((w + h) / max(step, 1.0)) + 1
        n_max = min(n_max, 60)  # hard cap against degenerate configs
        for k in range(1, n_max + 1):
            for sign in (1, -1):
                t = sign * k * step
                cx = int(ox + t * dx)
                cy = int(oy + t * dy)
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                img.draw_line(cx - int(px0), cy - int(py0),
                              cx + int(px0), cy + int(py0),
                              color=maix.image.COLOR_WHITE, thickness=2)
                label = str(int(round(sign * k * self._rail_cm_interval)))
                sprite = self._get_cm_label_sprite(label, maix.image)
                if sprite is not None:
                    try:
                        # Text sits at sprite-local (1,1); blit so the glyph
                        # lands at the same spot as the old draw_string.
                        img.draw_image(cx + 1, cy - 5, sprite)
                    except Exception:
                        pass

    def _make_rgb_text_sprite(self, text: str, maix_image,
                              scale=1, thickness=1) -> Optional[Any]:
        """Rasterise *text* onto an opaque black-background RGB888 sprite.

        Used for the header fragments: the header bar is itself black, so an
        opaque black sprite with white glyphs blits seamlessly onto it.  This
        avoids the RGBA8888 alpha path (draw_string on an RGBA canvas is
        unreliable on this platform) while keeping the same single-slot cache
        behaviour.
        """
        try:
            size = maix_image.string_size(text, scale=scale, thickness=thickness)
            lw = size[0] + 4
            lh = size[1] + 2
            sprite = maix_image.Image(lw, lh, maix_image.Format.FMT_RGB888,
                                      bg=maix_image.COLOR_BLACK)
            sprite.draw_string(1, 1, text,
                               color=maix_image.COLOR_WHITE, scale=scale, thickness=thickness)
            return sprite
        except Exception:
            return None

    def _make_rgba_text_sprite(self, text: str, maix_image,
                               scale=1, thickness=1) -> Optional[Any]:
        """Rasterise *text* onto a transparent RGBA8888 sprite, or None.

        Transparent background (alpha=0) with opaque white glyphs (alpha=0xFF),
        per the MaixPy draw-transparent-image pattern.  Used by the cm ruler
        labels; callers cache the result.
        """
        try:
            size = maix_image.string_size(text, scale=scale, thickness=thickness)
            lw = size[0] + 4
            lh = size[1] + 2
            sprite = maix_image.Image(lw, lh, maix_image.Format.FMT_RGBA8888)
            sprite.draw_string(1, 1, text,
                               color=maix_image.COLOR_WHITE, scale=scale, thickness=thickness)
            for y in range(lh):
                for x in range(lw):
                    pix = sprite.get_pixel(x, y)
                    val = pix[0] & 0x00ffffff
                    if val != 0:
                        val = val | 0xff000000
                    sprite.set_pixel(x, y, [val])
            return sprite
        except Exception:
            return None

    def _get_cm_label_sprite(self, label: str, maix_image) -> Optional[Any]:
        """Return a cached transparent RGBA8888 sprite for a cm label, or None.

        On first use the label is rasterised onto a transparent background
        (per the MaixPy draw-transparent-image pattern), then cached; later
        frames only blit.  Any failure returns None so the caller can degrade
        gracefully (label simply not drawn).
        """
        cached = self._cm_label_sprites.get(label)
        if cached is not None:
            return cached
        sprite = self._make_rgba_text_sprite(label, maix_image, scale=1, thickness=1)
        if sprite is not None:
            self._cm_label_sprites[label] = sprite
        return sprite

    def _draw_camera_fault_banner(self, img) -> None:
        try:
            import maix.image
        except ImportError:
            return
        w = img.width()
        bar_h = 32
        y = 40
        try:
            img.draw_rect(0, y, w, bar_h, color=maix.image.COLOR_YELLOW, thickness=-1)
            img.draw_string(6, y + 2, "CAMERA FAULT - RECONNECTING",
                            color=maix.image.COLOR_BLACK, scale=1.0, thickness=1)
        except Exception:
            pass

    def _draw_ai_fault_banner(self, img) -> None:
        try:
            import maix.image
        except ImportError:
            return
        w = img.width()
        bar_h = 32
        try:
            img.draw_rect(0, 0, w, bar_h, color=maix.image.COLOR_RED, thickness=-1)
            img.draw_string(6, 2, "AI FAULT - MODEL NOT LOADED",
                            color=maix.image.COLOR_WHITE, scale=1.2, thickness=1)
        except Exception:
            pass

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

    def _draw_overlays(self, img, pipeline_id: str, fps: float, detections) -> None:
        try:
            import maix.image
        except ImportError:
            return

        self._draw_header(img, pipeline_id, fps)
        self._draw_detections(img, detections, maix.image)
        self._draw_detection_list(img, detections, maix.image)

    def _draw_header(self, img, pipeline_id: str, fps: float) -> None:
        try:
            import maix.image
        except ImportError:
            return
        w = img.width()
        bar_h = 32
        if not self._hdr_prefix:
            self._hdr_prefix = f"{pipeline_id}  FPS:"

        # Update cached strings on a 1 Hz cadence (same as before). The time
        # change must also mark the header dirty, otherwise the clock could go
        # stale/frozen if fps happens to be stable.
        self._time_counter += 1
        if self._time_counter % 60 == 0:
            self._time_str_cached = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self._header_dirty = True

        # Rebuild the header bitmap only when its content changed.
        if self._header_bmp is None or w != self._hdr_width:
            self._header_dirty = True
        if fps != self._hdr_fps_last:
            self._hdr_fps_last = fps
            self._header_dirty = True
        if self._test_id > 0:
            test_str = f"Record Round: {self._test_id}"
            if test_str != self._test_str_cached:
                self._test_str_cached = test_str
                self._header_dirty = True

        if self._header_dirty:
            try:
                # Allocate the persistent buffer once (or on width change);
                # reuse it across rebuilds to avoid per-second allocation.
                if self._header_bmp is None or w != self._hdr_width:
                    self._header_bmp = maix.image.Image(
                        w, bar_h, maix.image.Format.FMT_RGB888,
                        bg=maix.image.COLOR_BLACK)
                    self._hdr_width = w
                bmp = self._header_bmp
                try:
                    bmp.draw_rect(0, 0, w, bar_h, color=maix.image.COLOR_BLACK, thickness=-1)
                except Exception:
                    pass

                # FPS fragment (single-slot cache).
                fps_text = f"{self._hdr_prefix}{fps:.1f}"
                if fps_text != self._hdr_fps_text:
                    self._hdr_fps_text = fps_text
                    self._hdr_fps_sprite = self._make_rgb_text_sprite(
                        fps_text, maix.image, scale=1.2, thickness=1)
                if self._hdr_fps_sprite is not None:
                    try:
                        # Sprite text sits at local (1,1); blit at (5,1) so the
                        # glyph lands at the old draw_string(6,2) position.
                        bmp.draw_image(5, 1, self._hdr_fps_sprite)
                    except Exception:
                        pass

                # Time fragment (single-slot cache).
                if self._time_str_cached != self._hdr_time_text:
                    self._hdr_time_text = self._time_str_cached
                    self._hdr_time_sprite = self._make_rgb_text_sprite(
                        self._time_str_cached, maix.image, scale=1.2, thickness=1)
                if self._hdr_time_sprite is not None:
                    try:
                        tw = self._hdr_time_sprite.width() - 4  # text width
                        bmp.draw_image((w - tw) // 2, 1, self._hdr_time_sprite)
                    except Exception:
                        pass

                # Test-id fragment (single-slot cache).
                if self._test_id > 0:
                    test_str = f"Record Round: {self._test_id}"
                    if test_str != self._hdr_test_text:
                        self._hdr_test_text = test_str
                        self._hdr_test_sprite = self._make_rgb_text_sprite(
                            test_str, maix.image, scale=1.2, thickness=1)
                    if self._hdr_test_sprite is not None:
                        try:
                            tw = self._hdr_test_sprite.width() - 4  # text width
                            bmp.draw_image(w - tw - 8, 1, self._hdr_test_sprite)
                        except Exception:
                            pass

                self._header_dirty = False
            except Exception:
                self._header_bmp = None
                self._header_dirty = True
                return

        if self._header_bmp is not None:
            try:
                img.draw_image(0, 0, self._header_bmp)
            except Exception:
                pass

    def _draw_detection_list(self, img, detections, maix_image) -> None:
        if not detections:
            return
        labels = self._ai.labels if hasattr(self._ai, "labels") and self._ai.labels else []

        lines = []
        line_h = 0
        for i, det in enumerate(detections, start=1):
            label_text = (
                labels[det.class_id]
                if 0 <= det.class_id < len(labels)
                else str(det.class_id)
            )
            if i > self._LIST_MAX_LINES:
                text = f"+{len(detections) - self._LIST_MAX_LINES} more"
                lines.append(text)
                break
            else:
                text = f"{i}: {label_text} {det.score:.2f}"
                lines.append(text)
            try:
                size = maix_image.string_size(
                    text, scale=self._LIST_TEXT_SCALE, thickness=self._LIST_TEXT_THICKNESS)
                line_h = max(line_h, size[1])
            except Exception:
                line_h = max(line_h, 20)

        line_spacing = line_h + 4
        list_x = 6
        list_y = 34
        for idx, text in enumerate(lines):
            try:
                img.draw_string(list_x, list_y + idx * line_spacing, text,
                                color=maix_image.COLOR_BLUE,
                                scale=self._LIST_TEXT_SCALE,
                                thickness=self._LIST_TEXT_THICKNESS)
            except Exception:
                pass

    def _draw_detections(self, img, detections, maix_image) -> None:
        for i, det in enumerate(detections, start=1):
            x1, y1 = det.x, det.y
            x2 = x1 + det.w
            y2 = y1 + det.h

            try:
                img.draw_rect(x1, y1, det.w, det.h,
                              color=maix_image.COLOR_GREEN, thickness=self._BOX_THICKNESS)
            except Exception:
                pass

            num_text = str(i)
            try:
                num_size = maix_image.string_size(
                    num_text, scale=self._CORNER_NUM_SCALE, thickness=self._CORNER_NUM_THICKNESS)
                num_w, num_h = num_size[0], num_size[1]
            except Exception:
                num_w, num_h = 10, 10
            pad = 2
            nx = x2 - num_w - pad
            ny = y2 - num_h - pad
            nx = max(x1, nx)
            ny = max(y1, ny)
            try:
                img.draw_string(nx, ny, num_text, color=maix_image.COLOR_BLUE,
                                scale=self._CORNER_NUM_SCALE, thickness=self._CORNER_NUM_THICKNESS)
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

    def _draw_fill_light_button(self, img) -> None:
        if self._fill_light_icon_on is None or self._fill_light_icon_off is None:
            return
        try:
            import maix.image as _mi
            h = img.height()
            w = img.width()
            size = self._fill_light_size
            bx = (w - size) // 2
            by = h - size - 8
            img.draw_rect(bx - 2, by - 2, size + 4,
                          size + 4, color=_mi.COLOR_BLACK, thickness=-1)
            icon = self._fill_light_icon_on if self._fill_light_on else self._fill_light_icon_off
            img.draw_image(bx, by, icon)
        except Exception:
            from utils.log_util import log_print
            log_print("[FILL_LIGHT] Button draw exception")

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
