import time
from typing import Optional

import cv2
import numpy as np

from ....models.color import Color
from .. import CargoDetector
from .config import CargoConfig, CARGO_PARAM_DEFS
from .window import CargoDebugWindow


_COLORS = [Color.RED, Color.GREEN, Color.BLUE]


class CargoDebugRunner:
    def __init__(self, camera_source: int = 0, width: int = 640, height: int = 480):
        self.camera_source = camera_source
        self.width = width
        self.height = height

        self.config = CargoConfig()
        self.detector = CargoDetector()
        self.window = CargoDebugWindow(
            title="Cargo Debug",
            param_defs=CARGO_PARAM_DEFS,
            on_change=self._on_param_changed,
        )
        self.stream: Optional["CameraStream"] = None

        self._running = False
        self._save_pending = False
        self._last_save_time = 0.0

        self._load_config()

    def _load_config(self):
        params = self.config.load()
        for pdef in CARGO_PARAM_DEFS:
            if pdef.name in params:
                raw = params[pdef.name]
                actual = raw * pdef.scale
                if pdef.scale == 1.0:
                    actual = int(actual)
                setattr(self.detector, pdef.name, actual)
                self.window.set_param(pdef.name, raw)

    def _on_param_changed(self, name: str, raw_value: int):
        for p in CARGO_PARAM_DEFS:
            if p.name == name:
                actual = raw_value * p.scale
                if p.scale == 1.0:
                    actual = int(actual)
                setattr(self.detector, name, actual)
                break
        self._save_pending = True

    def run(self):
        from ....camera_stream import CameraStream

        self.stream = CameraStream(self.camera_source, self.width, self.height)
        self.window.setup()
        self._running = True

        while self._running:
            frame = self.stream.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self._process_frame(frame)
            self.window.update(
                frame=frame,
                mask=self.detector._last_morphed,
                result=result,
            )
            self.window.refresh()

            if self._save_pending and time.time() - self._last_save_time > 0.5:
                self.config.save(self.window.get_raw_params())
                self._save_pending = False
                self._last_save_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        self._cleanup()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        for color in _COLORS:
            item = self.detector.detect_cargo(frame, color)
            if item is not None:
                cx, cy = item.coordinate
                color_bgr = {
                    Color.RED: (0, 0, 255),
                    Color.GREEN: (0, 255, 0),
                    Color.BLUE: (255, 0, 0),
                }[color]
                cv2.circle(display, (cx, cy), 6, color_bgr, 2)
                cv2.putText(
                    display, f"{color.name} ({cx},{cy})",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1,
                )
        return display

    def _cleanup(self):
        self.window.close()
        if self.stream:
            self.stream.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
