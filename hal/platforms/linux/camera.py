from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.log_util import log_print


class LinuxCamera:
    def __init__(
        self,
        camera_id: str,
        source,
        width: int = 640,
        height: int = 480,
        fps: float = 120,
        queue_size: int = 2,
        focal_length_mm: Optional[float] = None,
        sensor_width_mm: Optional[float] = None,
        sensor_height_mm: Optional[float] = None,
    ) -> None:
        self._camera_id = camera_id
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._queue_size = queue_size
        self._focal_length_mm = focal_length_mm
        self._sensor_width_mm = sensor_width_mm
        self._sensor_height_mm = sensor_height_mm
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._capture_thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps = fps
        self._actual_width = width
        self._actual_height = height

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def fps(self) -> float:
        return self._actual_fps

    @property
    def focal_length_mm(self) -> Optional[float]:
        return self._focal_length_mm

    @property
    def sensor_width_mm(self) -> Optional[float]:
        return self._sensor_width_mm

    @property
    def sensor_height_mm(self) -> Optional[float]:
        return self._sensor_height_mm

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @property
    def actual_height(self) -> int:
        return self._actual_height

    def _open_once(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
        if not cap.isOpened():
            log_print(f"[WARN] [Camera:{self._camera_id}] V4L2 open failed for {self._source}, trying default backend")
            cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {self._camera_id}: device not accessible")
        return cap

    def _subprocess_warmup(self, timeout: float) -> bool:
        script = Path(__file__).parent / "camera_warmup.py"
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script), str(self._source)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait(timeout=timeout)
            ok = proc.returncode == 0
            if ok:
                log_print(f"[INFO] [Camera:{self._camera_id}] warmup subprocess succeeded")
            else:
                log_print(f"[WARN] [Camera:{self._camera_id}] warmup subprocess exited with code {proc.returncode}")
            return ok
        except KeyboardInterrupt:
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_print(f"[ERROR] [Camera:{self._camera_id}] warmup subprocess did not die after kill")
            raise
        except subprocess.TimeoutExpired:
            if proc is not None and proc.poll() is None:
                proc.kill()
                log_print(f"[WARN] [Camera:{self._camera_id}] warmup subprocess killed after {timeout:.0f}s timeout")
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_print(f"[ERROR] [Camera:{self._camera_id}] warmup subprocess did not die after kill")
            return False
        except Exception as e:
            if proc is not None and proc.poll() is None:
                proc.kill()
                log_print(f"[WARN] [Camera:{self._camera_id}] warmup subprocess killed due to error: {e}")
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    log_print(f"[ERROR] [Camera:{self._camera_id}] warmup subprocess did not die after kill")
            return False

    def _open_once_direct(self, timeout: float):
        # Known limitation: on timeout the worker daemon thread may remain
        # alive inside a blocking cv2.VideoCapture call until process exit.
        # This is acceptable only as a last-resort fallback.
        log_print(f"[INFO] [Camera:{self._camera_id}] trying direct open, timeout={timeout:.0f}s")
        result: list = []
        exc: list = []

        def worker():
            try:
                result.append(self._open_once())
            except BaseException as e:
                exc.append(e)

        t = threading.Thread(
            target=worker,
            daemon=True,
            name=f"cam-open-{self._camera_id}",
        )
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            log_print(f"[WARN] [Camera:{self._camera_id}] direct open timed out after {timeout:.0f}s")
            return None
        if exc:
            log_print(f"[WARN] [Camera:{self._camera_id}] direct open thread raised {type(exc[0]).__name__}: {exc[0]}")
            e = exc[0]
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise e
            raise RuntimeError(
                f"Camera {self._camera_id}: open failed"
            ) from e
        cap = result[0] if result else None
        if cap is not None:
            log_print(f"[INFO] [Camera:{self._camera_id}] direct open succeeded")
        return cap

    def start(self) -> None:
        for attempt, warmup_timeout in enumerate((12.0, 20.0, 35.0), 1):
            log_print(f"[INFO] [Camera:{self._camera_id}] warmup attempt {attempt}/3, timeout={warmup_timeout:.0f}s")
            if self._subprocess_warmup(warmup_timeout):
                try:
                    self._cap = self._open_once()
                    if self._cap.isOpened():
                        log_print(f"[INFO] [Camera:{self._camera_id}] opened after warmup round {attempt}")
                        break
                    else:
                        log_print(f"[WARN] [Camera:{self._camera_id}] warmup succeeded but cap not open (attempt {attempt})")
                except RuntimeError as e:
                    log_print(f"[WARN] [Camera:{self._camera_id}] open after warmup failed (attempt {attempt}): {e}")
            else:
                log_print(f"[WARN] [Camera:{self._camera_id}] warmup timed out (attempt {attempt}/3)")
            backoff = 2.0 * attempt
            log_print(f"[INFO] [Camera:{self._camera_id}] backoff {backoff:.0f}s before retry")
            time.sleep(backoff)
        else:
            log_print(f"[WARN] [Camera:{self._camera_id}] all warmup rounds failed, trying direct open as last resort")
            self._cap = self._open_once_direct(10.0)
            if self._cap is None:
                raise RuntimeError(
                    f"Camera {self._camera_id}: all open attempts failed"
                )

        if not self._cap.isOpened():
            raise RuntimeError(f"Camera {self._camera_id}: opened but not accessible")

        if not self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')):
            log_print(f"[DEBUG] [Camera:{self._camera_id}] failed to set FOURCC=MJPG")

        for prop, name in [
            (cv2.CAP_PROP_FRAME_WIDTH, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
            (cv2.CAP_PROP_FPS, "fps"),
        ]:
            value = {"width": self._width, "height": self._height, "fps": self._fps}[name]
            if not self._cap.set(prop, value):
                log_print(f"[DEBUG] [Camera:{self._camera_id}] failed to set {name}={value}")

        actual = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = actual if actual > 0 else self._fps
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (self._actual_width, self._actual_height) != (self._width, self._height):
            log_print(
                f"[WARN] [Camera:{self._camera_id}] requested resolution {self._width}x{self._height}, "
                f"got {self._actual_width}x{self._actual_height}"
            )
        if abs(self._actual_fps - self._fps) > 1:
            log_print(
                f"[WARN] [Camera:{self._camera_id}] requested fps {self._fps:.1f}, "
                f"got {self._actual_fps:.1f}"
            )

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-capture-{self._camera_id}",
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        self._bind_capture_cores()
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                log_print(f"[ERROR] [Camera:{self._camera_id}] read failed")

    def _bind_capture_cores(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("camera_capture")

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def set(self, prop_id: int, value) -> bool:
        if self._cap is None:
            return False
        return self._cap.set(prop_id, value)

    def release(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1)
        if self._cap:
            self._cap.release()
            self._cap = None
