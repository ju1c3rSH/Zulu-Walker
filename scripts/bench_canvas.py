#!/usr/bin/env python
"""Micro-benchmark for the panel drawing path (mean / p95, milliseconds).

Numbers are the reason the canvas is written the way it is: fills go through
``cv2.rectangle`` rather than a numpy slice, text is not cached, and the window
pre-scales instead of letting the GUI backend do it per frame. Run it after any
change to the canvas or the HUD and compare against these baselines.

    python scripts/bench_canvas.py
    python scripts/bench_canvas.py --window --scale 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from framework.hal.platforms.linux.canvas import CvCanvas
from framework.ui import hud

W, H = 640, 480


def _time(fn, n: int = 200):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return statistics.mean(samples), samples[int(len(samples) * 0.95) - 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", action="store_true", help="also measure present()")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    panel = np.zeros((H, W, 3), np.uint8)
    frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    hi = cv2.resize(frame, (1280, 720))
    canvas = CvCanvas(panel)

    rows = []

    def add(name, fn, n=200):
        mean, p95 = _time(fn, n)
        rows.append((name, mean, p95))

    add("copy 640x480 -> panel", lambda: panel.__setitem__(slice(None), frame))
    add("resize 1280x720 -> 640x360",
        lambda: cv2.resize(hi, (640, 360), interpolation=cv2.INTER_AREA))
    add("20 fills", lambda: [canvas.rect(i % W, 10, 40, 20, (0, 200, 0), fill=True)
                             for i in range(0, 800, 40)], 100)
    add("20 strokes", lambda: [canvas.rect(i % W, 10, 40, 20, (0, 200, 0), width=2)
                               for i in range(0, 800, 40)], 100)
    add("50 lines", lambda: [canvas.line(0, i, W, i, (60, 60, 60)) for i in range(50)],
        100)
    add("10 texts", lambda: [canvas.text(8, 8 + i * 16, f"FPS:{i:4.1f}", 15,
                                         (255, 255, 255)) for i in range(10)], 100)

    state = hud.HudState(mode="TRACK", found=True, fps=28.4,
                         line2="CODES:[2]  BLOBS:2  HIT:2116px",
                         hits=[hud.Hit(100, 200, 90, 90, "blob"),
                               hud.Hit(300, 120, 60, 60, "blob")])
    add("hud.draw (2 hits)", lambda: hud.draw(canvas, state), 100)

    if args.window:
        from framework.hal.platforms.linux.window import CvWindow

        win = CvWindow(W, H, scale=args.scale)
        add("present (window %dx)" % args.scale, lambda: win.present(panel), 60)
        win.close()

    print(f"{'op':32s} {'mean':>7s} {'p95':>7s}")
    for name, mean, p95 in rows:
        print(f"{name:32s} {mean:7.3f} {p95:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
