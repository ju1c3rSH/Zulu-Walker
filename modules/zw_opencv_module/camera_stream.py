# -*- coding: utf-8 -*-
import os
import time
from queue import Queue
from threading import Thread
import numpy as np
import cv2
import sys
from utils.log_util import log_print



class CameraStream:
    """
    摄像头初始化的时候，会先检测是否支持V4L2接口（Linux专用），如果支持则使用V4L2接口进行视频捕获，这通常会提供更好的性能和更低的延迟。如果不支持V4L2接口，则回退到默认的视频捕获方式。摄像头参数设置部分也进行了错误处理，以确保在某些参数无法设置时不会导致程序崩溃。帧的读取和更新通过一个独立的线程进行，使用队列来存储最新的帧，确保读取时不会阻塞。
    设置完捕获方式后，会开始设置相关摄像头参数，如分辨率、帧率、缓冲区大小和视频编码格式。最后，摄像头捕获线程会持续运行，直到调用`release`方法来停止线程并释放摄像头资源。

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

        # 诊断计数器
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

    def _set_thread_affinity(self, cores):
        """设置当前线程的 CPU 亲和性（小核心）"""
        try:
            os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, PermissionError):
            pass  # Windows 或权限不足时忽略

    def _update(self):
        # 绑定到小核心 (RK3588: 0-3 是小核心 A55)
        self._set_thread_affinity([0, 1, 2, 3])

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                log_print("Can't receive frame (stream end?). Exiting ...")
                self.running = False
                break

            # 实际摄像头 FPS 诊断
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
        """读取帧（从队列获取）"""
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


# usage :camera = CameraStream(0)
