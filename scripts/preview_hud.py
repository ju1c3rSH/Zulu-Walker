"""Render the device HUD offscreen, without a board and without a camera.

The desktop panel draws its own HUD from ``framework/ui/hud.py`` and never
reaches the vision pipeline's overlay, so running the app on a laptop cannot
show what the device screen looks like.  This script drives the real
``_update_display_frame`` with a synthetic frame and a ``CvCanvas``: the
primitive list is the one the device renders, only the backend differs.

Usage:
    python scripts/preview_hud.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.hal.platforms.linux.canvas import CvCanvas
from modules.zw_opencv_module.vision_manager import VisionManager

WIDTH, HEIGHT = 640, 480
LABELS = ["ball", "obstacle"]


def _frame() -> np.ndarray:
    """A stand-in camera frame: a lit floor, two objects and a rail edge."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    ramp = np.linspace(35, 110, HEIGHT, dtype=np.uint8)
    image[:, :, 0] = ramp[:, None]
    image[:, :, 1] = (ramp[:, None] * 0.82).astype(np.uint8)
    image[:, :, 2] = (ramp[:, None] * 0.62).astype(np.uint8)
    cv2.circle(image, (215, 300), 26, (70, 80, 205), -1)
    cv2.circle(image, (430, 205), 17, (205, 125, 60), -1)
    cv2.line(image, (0, 386), (WIDTH, 340), (105, 105, 105), 2)
    return image


def _det(x, y, w, h, class_id, score, area=None):
    stats = None
    if area is not None:
        stats = SimpleNamespace(
            area_px=area, center_x=x + w / 2, center_y=y + h / 2
        )
    return SimpleNamespace(
        x=x, y=y, w=w, h=h, class_id=class_id, score=score, mask_stats=stats
    )


class _Camera:
    def __init__(self, frame, reconnecting=False):
        self.last_raw = frame
        self.frame_serial = 1
        self.is_reconnecting = reconnecting


class _Pipe:
    def __init__(self, camera, results):
        self.camera = camera
        self.last_results = results


class _AI:
    loaded = True
    labels = LABELS


def _live_frame(source=0):
    """Grab one frame from a local camera, or None when there is no camera."""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def render(detections, *, ai_ok=True, error="", reconnecting=False,
           show_list=True, frame=None) -> np.ndarray:
    """Drive the real display step and return the composed BGR frame."""
    if error:
        result = SimpleNamespace(success=False, result_data={}, error_code=error)
    elif detections is not None:
        result = SimpleNamespace(
            success=True, result_data={"detections": detections}, error_code=""
        )
    else:
        result = None

    manager = VisionManager(camera_hub=object(), canvas_factory=CvCanvas)
    manager._ai = _AI() if ai_ok else None
    manager._detection_list_enabled = show_list
    manager._pipelines["cam_main"] = _Pipe(
        _Camera(_frame() if frame is None else frame, reconnecting=reconnecting),
        {"ai_inference": result} if result is not None else {},
    )
    manager._update_display_frame()
    return manager.get_display_frame()


def _caption(image, text):
    """Bottom-left caption with a dark halo so it reads on any background."""
    org = (10, HEIGHT - 12)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> int:
    live = "--live" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = Path(args[0]) if args else ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    detections = [
        _det(189, 274, 52, 52, 0, 0.91, area=2124),
        _det(413, 188, 34, 34, 1, 0.64),
    ]

    if live:
        frame = _live_frame()
        if frame is None:
            print("no camera available; falling back to the synthetic frame")
        else:
            composed = render(detections, frame=frame)
            target = out_dir / "hud_live.png"
            cv2.imwrite(str(target), _caption(
                composed, "live camera frame, synthetic boxes"
            ))
            print(f"wrote {target}  ({composed.shape[1]}x{composed.shape[0]})")
            return 0

    panels = [
        _caption(render(detections), "normal: header + boxes + list"),
        _caption(render(detections, show_list=False), "list disabled"),
        _caption(render(None, error="E_INFER_FAILED"),
                 "ai error: banner carries the code"),
        _caption(render(None, ai_ok=False), "model not loaded"),
        _caption(render(detections, reconnecting=True), "camera reconnecting"),
    ]

    sheet = np.vstack([
        np.hstack(panels[0:2]),
        np.hstack(panels[2:4]),
        np.hstack([panels[4], np.zeros_like(panels[4])]),
    ])

    target = out_dir / "hud_preview.png"
    cv2.imwrite(str(target), sheet)
    print(f"wrote {target}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
