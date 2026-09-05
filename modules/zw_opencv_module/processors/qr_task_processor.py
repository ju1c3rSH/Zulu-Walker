# -*- coding: utf-8 -*-
"""
EOD2026 QR task-code processor.

Decodes the field QR code giving the 3-digit task code:
  digit1 = bomb color (1红/2绿/3蓝)
  digit2 = target color (1红/2绿/3蓝)
  digit3 = hostage shape (1圆柱/2圆锥/3腰鼓)

Engines (eod_config.yaml `accel.qr_engine`, default auto):
  - npu : maix.image.QRCodeDetector — NPU-accelerated, 60+ fps @320x224
          (MaixPy >= 4.7.9). Uses the same raw maix Image as the AI path;
          NPU contention is a non-issue because QR and YOLO never run in
          the same subscription phase (single-active rule).
  - zbar: img.find_qrcodes(ZBAR) — MaixCV CPU decoder.
  - cv2 : cv2.QRCodeDetector + CLAHE — the portable fallback (PC pytest,
          USB-camera debug). Also the automatic fallback whenever the
          maix engines are unavailable or raise.

Payload acceptance follows the 2025 provincial-first-prize practice:
require N consecutive identical decodes before latching (lighting flicker
produces single-frame garbage).

Protocol semantics (panzer docs/design.md §3 rule 1): QR is a one-shot event
streamed as DATA_STREAM(0x01) — push once on new decode, stay silent, dedup
by content. The coordinator owns per-subscription dedup; this processor only
reports (code, changed, latched).
"""
import logging
from typing import Optional

import cv2
import numpy as np

from .registry import register_processor
from .base import Processor, VisionResult
from ..detectors.eod_color.maix_accel import (
    maix_image_from_context, qr_detect_npu, qr_detect_zbar,
)
from .eod_align_processor import load_eod_config

logger = logging.getLogger(__name__)

_VALID_DIGITS = {"1", "2", "3"}


def code_domain_valid(code: str) -> bool:
    """True when code is exactly 3 digits, each within the rule domain 1..3."""
    return len(code) == 3 and all(c in _VALID_DIGITS for c in code)


@register_processor("QrTaskProcessor")
class QrTaskProcessor(Processor):
    def __init__(self, name: str = "qr_task", min_valid_frames: int = 2,
                 engine: Optional[str] = None):
        super().__init__(name)
        self._detector = cv2.QRCodeDetector()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cfg = load_eod_config().get("accel", {})
        self._engine = engine or cfg.get("qr_engine", "auto")   # auto|npu|zbar|cv2
        if min_valid_frames is None:
            min_valid_frames = int(cfg.get("qr_min_valid_frames", 2))
        self._min_valid_frames = max(1, int(min_valid_frames))
        self._last_code = ""
        self._stable_count = 0
        self._changed = False

    def _decode_maix(self, frame: np.ndarray, context: dict):
        """Try the maix engines; returns (code, corners, ran).

        ran=True means a maix engine executed (even with empty result) —
        caller must not double-spend on cv2. ran=False means unavailable.
        """
        if self._engine == "cv2":
            return "", None, False
        img = maix_image_from_context(context)
        if img is None:
            return "", None, False

        results = None
        if self._engine in ("auto", "npu"):
            results = qr_detect_npu(img)
            if results is not None:
                return self._best_payload(results), True
        if self._engine in ("auto", "zbar") and results is None:
            results = qr_detect_zbar(img)
            if results is not None:
                return self._best_payload(results), True
        return "", None, False

    @staticmethod
    def _best_payload(results) -> tuple:
        for r in results:
            payload = (r.get("payload") or "").strip()
            if payload:
                return payload, r.get("corners")
        return "", None

    def _decode_cv2(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        try:
            decoded, corners, _ = self._detector.detectAndDecode(gray)
            return (decoded or "").strip(), None if corners is None \
                else np.asarray(corners, dtype=float).reshape(-1, 2)
        except cv2.error:
            return "", None

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        decoded_code, corners = "", None
        if frame is not None:
            code, corners, ran = self._decode_maix(frame, context)
            decoded_code = code
            if not ran and (self._engine in ("auto", "cv2")):
                decoded_code, corners = self._decode_cv2(frame)

        if decoded_code and decoded_code == self._last_code:
            self._stable_count += 1
            self._changed = False
        else:
            self._changed = bool(decoded_code)
            self._last_code = decoded_code
            self._stable_count = 1 if decoded_code else 0

        decoded = bool(decoded_code)
        return VisionResult(
            task_name=self.name,
            success=decoded,
            result_data={
                "decoded": decoded,
                "code": decoded_code,
                "code_domain_valid": code_domain_valid(decoded_code) if decoded else False,
                "latched": decoded and self._stable_count >= self._min_valid_frames,
                "changed": self._changed,
                "engine": self._engine,
                "corners": corners,
            },
        )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame
        data = result.result_data
        corners = data.get("corners")
        if corners is not None and len(corners) >= 4:
            pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        code = data.get("code", "")
        color = (0, 255, 0) if data.get("code_domain_valid") else (0, 0, 255)
        tag = f"QR[{data.get('engine', '?')}] {code}" + \
              (" OK" if data.get("code_domain_valid") else " ?")
        cv2.putText(frame, tag, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def release(self) -> None:
        self._detector = None
