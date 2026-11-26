#!/usr/bin/env python
"""Desktop port of the MaixCAM main HUD + threshold page.

Geometry comes from ``feat/maixcam1-port``: ``display/hud.py`` for the main
page and ``debug/threshold_tuner.py`` for the threshold page. Only the pixel
backend differs (OpenCV instead of ``maix.image``), so the desktop screen and
the LCD screen are recognisably the same product.

No model, no serial, no MaixPy. Local camera, HSV threshold + contours.

    python scripts/desktop_ui_demo.py --source camera          # live
    python scripts/desktop_ui_demo.py --source synthetic       # no camera
    python scripts/desktop_ui_demo.py --png out.png --page MAIN
    python scripts/desktop_ui_demo.py --seconds 5              # smoke test

Main page: TUNE / EXIT stand in for the device's KEY_OK page cycling and its
exit icon. Threshold page: < Exit / SAVE / RESET, at the device's coordinates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from framework.cmd_queue import Cmd, CmdQueue
from framework.hal.platforms.linux.canvas import CvCanvas
from framework.hal.platforms.linux.window import CvMouseSource, CvWindow, CvWindowSink
from framework.ui import hud, tuner
from framework.ui.panel import Panel, Rect
from framework.ui.widgets import Button, find_button, find_slider

PANEL_W, PANEL_H = 640, 480
CV_INTERVAL_S = 1.0 / 15.0

MAIN_BUTTONS = [Button("TUNE", Rect(8, 434, 88, 38)),
                Button("EXIT", Rect(104, 434, 88, 38))]

GROUP_DEFAULTS = {
    "GREEN": {"H_lo": 35, "H_hi": 85, "S_lo": 80, "S_hi": 255,
              "V_lo": 60, "V_hi": 255},
    "RED": {"H_lo": 0, "H_hi": 10, "S_lo": 90, "S_hi": 255,
            "V_lo": 70, "V_hi": 255},
}
GROUPS = tuple(GROUP_DEFAULTS)
SLIDER_SPEC = (("H_lo", 0, 180), ("H_hi", 0, 180), ("S_lo", 0, 255),
               ("S_hi", 0, 255), ("V_lo", 0, 255), ("V_hi", 0, 255))


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

class Source:
    """Camera or synthetic pattern behind one ``read()``."""

    def __init__(self, kind: str, camera_index: int = 0) -> None:
        self.kind = kind
        self._cap = None
        self._t0 = time.monotonic()
        if kind != "camera":
            return
        cap = cv2.VideoCapture(camera_index,
                               cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ui] camera unavailable, falling back to synthetic")
            self.kind = "synthetic"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap

    def read(self):
        if self.kind == "camera":
            ok, frame = self._cap.read()
            return frame if ok else None
        return synthetic(time.monotonic() - self._t0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def synthetic(t: float):
    """A green block and a red ball on a dark table."""
    img = np.full((480, 640, 3), (38, 36, 34), np.uint8)
    img[80:, :] = (52, 50, 48)
    x = int(60 + (t * 90) % 440)
    cv2.rectangle(img, (x, 300), (x + 90, 390), (60, 180, 90), -1)
    by = int(180 + 40 * np.sin(t * 1.6))
    cv2.circle(img, (200, by), 26, (70, 70, 210), -1)
    return img


# --------------------------------------------------------------------------
# a small CV instance: HSV threshold -> contours -> boxes
# --------------------------------------------------------------------------

class ColorBlobs:
    """Threshold + contour boxes. Half resolution, area filter, top-N."""

    def __init__(self) -> None:
        self.params = {g: dict(v) for g, v in GROUP_DEFAULTS.items()}
        self.min_area = 400
        self.scale = 0.5
        self.top_n = 8
        self.count = 0
        self.hit_px = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame, group: str):
        p = self.params[group]
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (p["H_lo"], p["S_lo"], p["V_lo"]),
                           (p["H_hi"], p["S_hi"], p["V_hi"]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        self.hit_px = int((mask > 0).sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / self.scale
        boxes = []
        for c in contours:
            area = cv2.contourArea(c) * inv * inv
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv), area))
        boxes.sort(key=lambda b: b[4], reverse=True)
        self.count = len(boxes)
        return boxes[: self.top_n]


# --------------------------------------------------------------------------
# composition: frame -> panel buffer
# --------------------------------------------------------------------------

class Composer:
    """Letterbox into the panel, reusing buffers so no frame allocates."""

    def __init__(self, panel: Panel) -> None:
        self.panel = panel
        self.out = np.zeros((panel.height, panel.width, 3), np.uint8)
        self._tmp = None
        self._tmp_size = (0, 0)

    def compose(self, frame):
        self.out[:] = (0, 0, 0)
        r = self.panel.fit(frame.shape[1], frame.shape[0])
        if (r.w, r.h) == (frame.shape[1], frame.shape[0]):
            self.out[r.y:r.bottom, r.x:r.right] = frame
        else:
            if self._tmp_size != (r.w, r.h):
                self._tmp = np.empty((r.h, r.w, 3), np.uint8)
                self._tmp_size = (r.w, r.h)
            cv2.resize(frame, (r.w, r.h), dst=self._tmp, interpolation=cv2.INTER_AREA)
            self.out[r.y:r.bottom, r.x:r.right] = self._tmp
        return self.out, r


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

class App:
    def __init__(self, args) -> None:
        self.frame_budget = 1.0 / args.fps_cap if args.fps_cap > 0 else 0.0
        self.panel = Panel(PANEL_W, PANEL_H)
        self.composer = Composer(self.panel)
        self.source = Source(args.source, args.camera)
        self.blobs = ColorBlobs()
        self.window = CvWindow(PANEL_W, PANEL_H, scale=args.scale,
                               headless=bool(args.png), topmost=args.topmost)
        self.sink = CvWindowSink(self.window, interval_ms=args.interval_ms)
        self.input = CvMouseSource(self.window)
        self.cmds = CmdQueue()
        self.page = "MAIN"
        self.group = GROUPS[0]
        self.pressed = ""
        self.drag = None
        self.actions = tuner.group_actions(PANEL_W)
        self.sliders = self._build_sliders()
        self.fps = 0.0
        self.frames, self.fps_at = 0, time.monotonic()
        self.boxes = []
        self.last_cv = 0.0

    # ---- sliders ----

    def _build_sliders(self):
        geoms = tuner.slider_rects(PANEL_W, PANEL_H)
        return [tuner.slider_for(key, geom, lo, hi,
                                 self.blobs.params[self.group][key])
                for geom, (key, lo, hi) in zip(geoms, SLIDER_SPEC)]

    def _store_group(self) -> None:
        for s in self.sliders:
            self.blobs.params[self.group][s.key] = int(s.value)

    # ---- commands ----

    def _handle(self, cmds) -> bool:
        for cmd in cmds:
            if cmd.kind == "EXIT":
                return False
            if cmd.kind == "UI_PRESS":
                self._press(cmd.payload["x"], cmd.payload["y"])
            elif cmd.kind == "UI_MOVE" and self.drag is not None:
                self.drag.drag(cmd.payload["x"])
            elif cmd.kind == "UI_RELEASE":
                self._release(cmd.payload["x"], cmd.payload["y"])
        return True

    def _press(self, x: int, y: int) -> None:
        if self.page == "MAIN":
            btn = find_button(MAIN_BUTTONS, x, y)
            self.pressed = btn.label if btn else ""
            return
        for act in self.actions:
            if Rect(act.x, act.y, act.w, act.h).contains(x, y):
                self.pressed = act.label
                return
        s = find_slider(self.sliders, x, y)
        if s is not None and s.begin(x, y):
            self.drag = s

    def _release(self, x: int, y: int) -> None:
        if self.drag is not None:
            self._store_group()
            self.drag.end()
            self.drag = None
            return
        label, self.pressed = self.pressed, ""
        if self.page == "MAIN":
            if label == "TUNE":
                self.page = "TUNER"
            elif label == "EXIT":
                self.cmds.put(Cmd("EXIT"))
            return
        if label == "< Exit":
            self._store_group()
            self.page = "MAIN"
        elif label == "SAVE":
            print(f"[ui] SAVE {self.group} {self.blobs.params[self.group]}")
        elif label == "RESET":
            for s in self.sliders:
                s.value = GROUP_DEFAULTS[self.group][s.key]
            self._store_group()
        else:
            self._select_group(x, y)

    def _select_group(self, x: int, y: int) -> None:
        bw = PANEL_W // len(GROUPS)
        for i in range(len(GROUPS)):
            if Rect(i * bw + 2, 2, bw - 4, tuner.GROUP_H - 6).contains(x, y):
                self._store_group()
                self.group = GROUPS[i]
                self.sliders = self._build_sliders()
                return

    # ---- frame ----

    def render(self) -> np.ndarray:
        frame = self.source.read()
        if frame is None:
            frame = synthetic(time.monotonic())
        buf, rect = self.composer.compose(frame)
        canvas = CvCanvas(buf)

        now = time.monotonic()
        if now - self.last_cv >= CV_INTERVAL_S:
            self.last_cv = now
            self.boxes = self.blobs.detect(frame, self.group)

        if self.page == "MAIN":
            hits = [hud.Hit(rect.x + x, rect.y + y, w, h, "blob")
                    for (x, y, w, h, _a) in self.boxes]
            state = hud.HudState(
                mode="TRACK", link_ok=True, found=bool(hits), fps=self.fps,
                line2=f"CODES:[{len(hits)}]  BLOBS:{len(hits)}  "
                      f"HIT:{self.blobs.hit_px}px",
                hits=hits)
            hud.draw(canvas, state, MAIN_BUTTONS, self.pressed)
        else:
            boxes = [(rect.x + x, rect.y + y, w, h)
                     for (x, y, w, h, _a) in self.boxes]
            groups = [(g, g == self.group) for g in GROUPS]
            info = (f"hit_px:{self.blobs.hit_px} blobs:{len(boxes)} "
                    f"s:{self.blobs.scale:g}x")
            tuner.draw(canvas, groups, self.sliders, self.actions, info, boxes)
        return buf

    def run(self, seconds: float = 0.0) -> None:
        t_start = time.monotonic()
        drawn = 0
        present_ms = 0.0
        deadline = t_start + seconds if seconds else None
        try:
            while True:
                if deadline and time.monotonic() > deadline:
                    break
                self.input.poll(self.cmds)
                if not self._handle(self.cmds.drain()):
                    break
                frame = self.render()
                t0 = time.monotonic()
                self.sink.push(frame)
                self.sink.flush()
                present_ms += (time.monotonic() - t0) * 1000.0
                drawn += 1
                if not self.window.has_gui:
                    break
                if self.frame_budget:
                    slack = self.frame_budget - (time.monotonic() - t0)
                    if slack > 0:
                        time.sleep(slack)
        finally:
            elapsed = max(1e-6, time.monotonic() - t_start)
            print(f"[ui] frames={drawn} rate={drawn / elapsed:.1f}/s "
                  f"present_avg={present_ms / max(1, drawn):.2f}ms "
                  f"gui={self.window.has_gui}")
            self.sink.close()
            self.source.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="camera", choices=("camera", "synthetic"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--interval-ms", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--fps-cap", type=float, default=60.0,
                    help="idle pacing; 0 = run as fast as possible")
    ap.add_argument("--png", metavar="PATH", help="render one frame and exit")
    ap.add_argument("--page", default="MAIN", choices=("MAIN", "TUNER"))
    ap.add_argument("--topmost", action="store_true",
                    help="keep the window above other apps (demo aid)")
    args = ap.parse_args()

    if args.png:
        app = App(args)
        app.page = args.page
        cv2.imwrite(args.png, app.render())
        app.source.close()
        print(f"wrote {args.png}")
        return 0

    App(args).run(args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
