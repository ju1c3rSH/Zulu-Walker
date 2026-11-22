# -*- coding: utf-8 -*-
import time
from queue import Queue
from threading import Thread
import numpy as np
import cv2
import sys
from utils.log_util import log_print



class CameraStream:
    """Capture frames from a camera and serve the newest frame to readers.

    Capture uses the V4L2 backend when available and falls back to the default
    backend otherwise. Unsupported property settings are ignored. A daemon
    thread reads frames into a bounded queue and drops the oldest frame when
    full, so readers never block. Call release() to stop the thread and free
    the camera.
    """

    def __init__(self, source=0, width=640, height=480, camera_id="", queue_size=2):
        try:
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        except AttributeError:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source}")

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 120)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except:
            pass

        self.queue = Queue(maxsize=queue_size)
        self.running = True

        # Capture diagnostics.
        self._cap_fps_count = 0
        self._cap_fps_start = time.time()
        self._cap_fps_actual = 0.0
        self._frames_dropped = 0
        self._frames_produced = 0
        self.camera_id = camera_id
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    @property
    def fps(self) -> float:
        """Actual capture FPS, updated every ~2 seconds."""
        return self._cap_fps_actual

    @property
    def queue_depth(self) -> int:
        """Current frame queue depth."""
        return self.queue.qsize()

    @property
    def dropped_count(self) -> int:
        """Total frames dropped due to queue overflow."""
        return self._frames_dropped

    def _bind_capture_cores(self):
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("camera_capture")

    def _update(self):
        self._bind_capture_cores()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                log_print("Can't receive frame (stream end?). Exiting ...")
                self.running = False
                break

            # Measure the actual capture FPS over a two-second window.
            self._cap_fps_count += 1
            self._frames_produced += 1
            elapsed = time.time() - self._cap_fps_start
            if elapsed >= 2.0:
                self._cap_fps_actual = self._cap_fps_count / elapsed
                self._cap_fps_count = 0
                self._cap_fps_start = time.time()

            """
            这里使用非阻塞式队列来更新每一帧的数据
            如果队列已满，则先尝试移除旧的帧以腾出空间，然后再将新的帧放入队列中。这种方式确保了读取最新帧时不会被旧帧阻塞，同时也避免了内存占用过多的问题。
            """
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self._frames_dropped += 1
                except Exception as e:
                    log_print(f"Error dropping frame from queue: {e}")
                    pass


            try:
                self.queue.put_nowait(frame)
            except Exception as e:
                log_print(f"Error putting frame into queue: {e}")
                pass

    def read_frame(self):
        """Return the newest queued frame, or None when the queue is empty."""
        from .performance import profiler
        with profiler.timer("frame_capture"):
            try:
                return self.queue.get_nowait()
            except:
                return None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()


# Usage: camera = CameraStream(0)
