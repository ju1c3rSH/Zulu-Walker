# -*- coding: utf-8 -*-
"""
EOD2026 align processor — one processor, one mode at a time.

The MCU owns the mission state machine and drives this processor through the
single-active-subscription protocol (panzer docs/design.md §2.7/§2.8):
CMD_REQUEST(data_type) -> coordinator maps data_type to a mode here.
Exactly one of BOMB / TARGET / HOSTAGE / BUCKET / HOME is active; the
coordinator enables/disables this task and the QR task accordingly.

Latch semantics implemented here (design §3 rules):
  - bit0 FOUND / bit1 MATCH / bit2 READY latch after N consecutive stable
    frames; brief dropouts do NOT clear the latch.
  - the FOUND latch decays only after lost_hold_ms of continuous dropout
    (error stream then reports 0/0 — no stale-error servoing).
  - latches are cleared ONLY by set_mode() (new REQUEST) or reset() (STOP).

HIT_CONFIRMED (TARGET mode): after a stable lock on the target, a long
dropout or an area collapse of the tracked region latches the hit — the
mannequin-fall heuristic. Disabled by default until calibrated on site;
a custom YOLO "mannequin down" model is the robust upgrade path.

Pure cv2/numpy + injectable clock — runs on device AND on PC (pytest).
"""
import os
import time
from typing import Optional

import cv2
import numpy as np
import yaml

from .registry import register_processor
from .base import Processor, VisionResult
from ..detectors.eod_color import EodColorDetector, EodBlobResult
from ..detectors.eod_color import maix_accel
from ..detectors.eod_shape import classify_contour
from modules.zw_uart_module.protocol import (
    DEV_FLAG_FOUND, DEV_FLAG_MATCH, DEV_FLAG_READY,
    RING_NONE,
)

_MODE_BOMB = "BOMB"
_MODE_TARGET = "TARGET"
_MODE_HOSTAGE = "HOSTAGE"
_MODE_BUCKET = "BUCKET"
_MODE_HOME = "HOME"
_ALL_MODES = (_MODE_BOMB, _MODE_TARGET, _MODE_HOSTAGE, _MODE_BUCKET, _MODE_HOME)

_DEFAULT_CONFIG = {
    "colors": [
        {"id": 1, "name": "red",   "lab": [0, 80, 40, 80, 10, 80]},
        {"id": 2, "name": "green", "lab": [0, 80, -120, -10, 0, 30]},
        {"id": 3, "name": "blue",  "lab": [0, 80, 30, 100, -120, -60]},
    ],
    "hostage_colors": [],   # [] -> fallback to colors
    "bucket": {"id": 1, "name": "black", "lab": [0, 45, -128, 128, -128, 128]},
    "home_colors": [],      # [] -> fallback to colors
    "blob": {"min_area_px": 400, "max_area_px": 300000,
             "edge_margin_px": 4, "morph_open": 3, "morph_close": 7},
    "zones": {"grab": [0.35, 0.45, 0.65, 0.85], "place": [0.35, 0.40, 0.65, 0.85]},
    "latch": {"frames": 5, "lost_hold_ms": 300},
    "shape": {"waist_ratio_max": 0.88, "cone_small_end_max": 0.70,
              "cone_slope_min": 0.30, "cylinder_end_min": 0.82,
              "min_confidence": 0.25},
    "target": {"ref_target_width_px": 200,
               "ring_radii_px": {"r10": 220, "r9": 150, "r8": 85},
               "hit_detect_enabled": False,
               "lock_frames": 15, "hit_lost_frames": 12, "hit_area_ratio": 0.45},
}


def load_eod_config(config_path: Optional[str] = None) -> dict:
    """Load eod_config.yaml over the built-in defaults (shallow per-section)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v))
           for k, v in _DEFAULT_CONFIG.items()}
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config", "eod_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in (data.get("eod") or {}).items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    except (OSError, yaml.YAMLError):
        pass
    return cfg


class EodAlignProcessor(Processor):

    def __init__(self, name: str = "eod_align", config: Optional[dict] = None,
                 now_fn=time.monotonic):
        super().__init__(name)
        self._cfg = config or load_eod_config()
        self._now = now_fn

        blob = self._cfg["blob"]
        colors = self._cfg["colors"]
        hostage_colors = self._cfg["hostage_colors"] or colors
        home_colors = self._cfg["home_colors"] or colors
        zone_grab = tuple(self._cfg["zones"]["grab"])
        zone_place = tuple(self._cfg["zones"]["place"])

        def _detector(cands, zone):
            return EodColorDetector(
                candidates=cands, grab_zone=zone,
                min_area_px=blob["min_area_px"], max_area_px=blob["max_area_px"],
                edge_margin_px=blob["edge_margin_px"],
                morph_open=blob["morph_open"], morph_close=blob["morph_close"])

        self._detectors = {
            _MODE_BOMB:    _detector(colors, zone_grab),
            _MODE_TARGET:  _detector(colors, zone_place),
            _MODE_HOSTAGE: _detector(hostage_colors, zone_grab),
            _MODE_BUCKET:  _detector([self._cfg["bucket"]], zone_place),
            _MODE_HOME:    _detector(home_colors, zone_place),
        }
        self._color_fast = bool(self._cfg.get("accel", {}).get("color_fast_path", True))
        self._shape_cfg = self._cfg["shape"]
        self._target_cfg = self._cfg["target"]
        self._latch_frames = int(self._cfg["latch"]["frames"])
        self._lost_hold_ms = float(self._cfg["latch"]["lost_hold_ms"])

        self._mode: Optional[str] = None
        self._expected_id = 0
        self._reset_state()

    # ---- lifecycle / gating ----

    def set_mode(self, mode: str, expected_id: int = 0) -> None:
        """Called by the coordinator on every CMD_REQUEST (subscription change)."""
        if mode not in _ALL_MODES:
            raise ValueError(f"unknown EOD mode: {mode!r}")
        self._mode = mode
        self._expected_id = int(expected_id)
        self._reset_state()

    def reset(self) -> None:
        """Called on CMD_STOP / master timeout: drop latches, keep mode."""
        self._reset_state()

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    def _reset_state(self) -> None:
        self._found_cnt = 0
        self._ready_cnt = 0
        self._match_cnt = 0
        self._found_latch = False
        self._ready_latch = False
        self._match_latch = False
        self._last_raw_seen_ms = -1e9
        self._last_error = (0, 0)
        self._last_id = 0
        # hit detection state (TARGET mode)
        self._lock_cnt = 0
        self._locked = False
        self._locked_area = 0
        self._lost_cnt = 0
        self._hit_latched = False
        self._last_features: dict = {}
        self._frame_shape = None   # (h, w) of the last processed frame

    # ---- processing ----

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        now_ms = self._now() * 1000.0
        raw = EodBlobResult()
        shape_id, ring_id = 0, RING_NONE

        # MaixPy fast path: raw maix Image via the camera HAL (C++ find_blobs,
        # zero numpy conversion); None on PC -> pure cv2 path.
        maix_img = maix_accel.maix_image_from_context(context) \
            if self._color_fast else None

        if self._mode is not None and frame is not None:
            self._frame_shape = frame.shape[:2]   # (h, w) for ring estimation
            detector = self._detectors[self._mode]
            need_contour = (self._mode == _MODE_HOSTAGE)
            if self._mode == _MODE_BOMB:
                raw = detector.detect(frame, expected_id=self._expected_id,
                                      zone_anchor="bottom", maix_img=maix_img,
                                      need_contour=need_contour)
            elif self._mode == _MODE_TARGET:
                raw = detector.detect(frame, expected_id=self._expected_id,
                                      zone_anchor="center", maix_img=maix_img,
                                      need_contour=need_contour)
                ring_id = self._estimate_ring(raw)
            elif self._mode == _MODE_HOSTAGE:
                raw = detector.detect(frame, zone_anchor="bottom",
                                      maix_img=maix_img,
                                      need_contour=need_contour)
                if raw.contour is not None:
                    shape_id, _, self._last_features = classify_contour(
                        raw.contour, **{k: v for k, v in self._shape_cfg.items()})
            elif self._mode == _MODE_BUCKET:
                raw = detector.detect(frame, zone_anchor="center",
                                      maix_img=maix_img,
                                      need_contour=need_contour)
            elif self._mode == _MODE_HOME:
                raw = detector.detect(frame, zone_anchor="center",
                                      maix_img=maix_img,
                                      need_contour=need_contour)

        self._update_latches(raw, now_ms)

        # per-mode flag semantics
        if self._mode == _MODE_TARGET:
            ready_bit = self._hit_latched
        elif self._mode in (_MODE_BUCKET, _MODE_HOME):
            ready_bit = self._ready_latch
        elif self._mode == _MODE_BOMB:
            ready_bit = self._ready_latch and self._match_latch
        else:   # HOSTAGE: no READY bit in the protocol table
            ready_bit = False

        flags = 0
        if self._found_latch:
            flags |= DEV_FLAG_FOUND
        if self._match_latch:
            flags |= DEV_FLAG_MATCH
        if ready_bit:
            flags |= DEV_FLAG_READY

        if self._found_latch:
            ex, ey = self._last_error if not raw.target_found else (raw.error_x, raw.error_y)
            dev_id = self._last_id
        else:
            ex = ey = dev_id = 0

        if self._mode == _MODE_HOSTAGE:
            dev_id = self._last_id   # shape_id
        data = {
            "mode": self._mode,
            "raw_found": raw.target_found,
            "target_found": self._found_latch,
            "error_x": ex,
            "error_y": ey,
            "id": dev_id,
            "flags": flags,
            "match": self._match_latch,
            "ready": ready_bit,
            "hit": self._hit_latched,
            "shape_id": shape_id,
            "ring_id": ring_id,
            "bbox": raw.bbox,
            "center": raw.center,
            "area_px": raw.area_px,
            "features": self._last_features,
            "candidates": raw.all_candidates,
        }
        return VisionResult(task_name=self.name, success=bool(flags), result_data=data)

    def _update_latches(self, raw: EodBlobResult, now_ms: float) -> None:
        if raw.target_found:
            self._last_raw_seen_ms = now_ms
            self._found_cnt += 1
            self._last_error = (raw.error_x, raw.error_y)
            self._last_id = raw.dev_id
            if self._found_cnt >= self._latch_frames:
                self._found_latch = True
        elif now_ms - self._last_raw_seen_ms > self._lost_hold_ms:
            # long dropout: clear the latch so the error stream goes silent
            self._found_latch = False
            self._found_cnt = 0
            self._last_id = 0

        match_raw = self._raw_match(raw)
        if match_raw:
            self._match_cnt += 1
            if self._match_cnt >= self._latch_frames:
                self._match_latch = True
        # MATCH/READY latches are NOT cleared by dropouts — only set_mode/reset

        ready_raw = self._raw_ready(raw)
        if ready_raw:
            self._ready_cnt += 1
            if self._ready_cnt >= self._latch_frames:
                self._ready_latch = True

        if self._mode == _MODE_TARGET:
            self._update_hit(raw)

    def _raw_match(self, raw: EodBlobResult) -> bool:
        if not raw.target_found:
            return False
        if self._mode in (_MODE_BOMB, _MODE_TARGET):
            return (raw.dev_id == self._expected_id) if self._expected_id else True
        if self._mode == _MODE_HOSTAGE:
            sid = classify_contour(raw.contour, **dict(self._shape_cfg))[0] \
                if raw.contour is not None else 0
            return (sid == self._expected_id) if self._expected_id else sid != 0
        return False

    def _raw_ready(self, raw: EodBlobResult) -> bool:
        if not raw.target_found:
            return False
        if self._mode == _MODE_BOMB:
            return raw.ready and (self._match_latch or not self._expected_id)
        if self._mode in (_MODE_BUCKET, _MODE_HOME):
            return raw.ready
        return False

    def _update_hit(self, raw: EodBlobResult) -> None:
        cfg = self._target_cfg
        if not cfg.get("hit_detect_enabled", False):
            return
        if raw.target_found:
            self._lost_cnt = 0
            if not self._locked:
                self._lock_cnt += 1
                if self._lock_cnt >= int(cfg["lock_frames"]):
                    self._locked = True
                    self._locked_area = max(1, raw.area_px)
            elif raw.area_px < float(cfg["hit_area_ratio"]) * self._locked_area:
                self._hit_latched = True
        else:
            self._lost_cnt += 1
            if self._locked and self._lost_cnt >= int(cfg["hit_lost_frames"]):
                self._hit_latched = True

    def _estimate_ring(self, raw: EodBlobResult) -> int:
        """Which ring the boreline (image center) currently falls on."""
        if not raw.target_found or raw.center is None or raw.bbox is None:
            return RING_NONE
        ref_w = float(self._target_cfg["ref_target_width_px"])
        if ref_w <= 0 or raw.bbox[2] <= 0:
            return RING_NONE
        frame_shape = getattr(self, "_frame_shape", None)
        if not frame_shape:
            return RING_NONE
        h, w = frame_shape
        cx, cy = raw.center
        dist_px = float(np.hypot(w / 2.0 - cx, h / 2.0 - cy))
        scale = raw.bbox[2] / ref_w
        radii = self._target_cfg["ring_radii_px"]
        for ring, key in ((10, "r10"), (9, "r9"), (8, "r8")):
            if dist_px <= float(radii[key]) * scale:
                return ring
        return RING_NONE

    # ---- display ----

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame
        data = result.result_data
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
        if data.get("bbox"):
            x, y, bw, bh = data["bbox"]
            color = (0, 255, 0) if data.get("target_found") else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            if data.get("ready"):
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        tag = (f"{data.get('mode') or '-'} "
               f"ex={data.get('error_x', 0):+d} ey={data.get('error_y', 0):+d} "
               f"id={data.get('id', 0)} f={data.get('flags', 0):02b}")
        cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        return frame

    def release(self) -> None:
        self._detectors.clear()
