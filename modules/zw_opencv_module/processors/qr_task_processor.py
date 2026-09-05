# -*- coding: utf-8 -*-
"""
EOD2026 QR task-code processor.

Decodes the field QR code giving the 3-digit task code:
  digit1 = bomb color (1红/2绿/3蓝)
  digit2 = target color (1红/2绿/3蓝)
  digit3 = hostage shape (1圆柱/2圆锥/3腰鼓)

Device notes (docs/eod/method_selection.md):
  - CPU cv2.QRCodeDetector is used instead of the NPU image.QRCodeDetector so
    it never contends with YOLO for NPU time; QR reading is a dedicated mission
    phase anyway (single active subscription).
  - Payload acceptance follows the 2025 provincial-first-prize practice:
    require N consecutive identical decodes before latching (lighting flicker
    produces single-frame garbage).

Protocol semantics (panzer docs/design.md §3 rule 1): QR is a one-shot event
streamed as DATA_STREAM(0x01) — push once on new decode, stay silent, dedup
by content. The coordinator owns per-subscription dedup; this processor only
reports (code, changed, latched).
"""
from typing import Optional

import cv2
import numpy as np

from .registry import register_processor
from .base import Processor, VisionResult

_VALID_DIGITS = {"1", "2", "3"}


def code_domain_valid(code: str) -> bool:
    """True when code is exactly 3 digits, each within the rule domain 1..3."""
    return len(code) == 3 and all(c in _VALID_DIGITS for c in code)


@register_processor("QrTaskProcessor")
class QrTaskProcessor(Processor):
    def __init__(self, name: str = "qr_task", min_valid_frames: int = 2):
        super().__init__(name)
        self._detector = cv2.QRCodeDetector()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._min_valid_frames = max(1, min_valid_frames)
        self._last_code = ""
        self._stable_count = 0
        self._changed = False

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        decoded_code = ""
        corners = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)
            try:
                decoded_code, corners, _ = self._detector.detectAndDecode(gray)
                decoded_code = (decoded_code or "").strip()
            except cv2.error:
                decoded_code = ""

        if decoded_code and decoded_code == self._last_code:
            self._stable_count += 1
            self._changed = False
        else:
            if decoded_code:
                self._changed = True
            else:
                self._changed = False
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
                "corners": None if corners is None else np.asarray(corners, dtype=float).reshape(-1, 2),
            },
        )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame
        data = result.result_data
        corners = data.get("corners")
        if corners is not None and len(corners) >= 4:
            pts = corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        code = data.get("code", "")
        color = (0, 255, 0) if data.get("code_domain_valid") else (0, 0, 255)
        tag = f"QR {code}" + (" OK" if data.get("code_domain_valid") else " ?")
        cv2.putText(frame, tag, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def release(self) -> None:
        self._detector = None
