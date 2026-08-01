from __future__ import annotations

import sys
import time
import threading
from collections import deque
from typing import Optional

_QUEUE_MAX = 12
_EMPTY_SLEEP = 0.015
# JPEG encode quality for the WiFi stream.  Lower quality -> smaller frames
# -> higher effective frame rate on WiFi links.
_JPEG_QUALITY = 90
# Proportional downscale factor applied to the streamed frame before JPEG
# encoding (e.g. 0.5 -> half width/height).  Smaller frames -> much lower
# encode cost and bandwidth -> higher effective frame rate on WiFi links.
_STREAM_SCALE = 0.5


class JpegStreamer:
    def __init__(self) -> None:
        self._server: Optional[object] = None
        self._is_running: bool = False
        self._url: str = ""
        self._thread: Optional[threading.Thread] = None
        self._queue: deque = deque(maxlen=_QUEUE_MAX)
        self._lock: threading.Lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def url(self) -> str:
        return self._url

    def start_async(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            from maix import http
            html = '<html><body><img src="/stream" /></body></html>'
            srv = http.JpegStreamer()
            srv.set_html(html)
            srv.start()
            self._server = srv
            self._url = "http://{}:{}/".format(srv.host(), srv.port())
            self._is_running = True
            print("[HTTP] %s" % self._url)
            sys.stdout.flush()
        except Exception as e:
            print("[HTTP] start failed: %s" % e)
            sys.stdout.flush()
            return

        while self._is_running:
            with self._lock:
                try:
                    img = self._queue.pop()
                except IndexError:
                    img = None
            if img is not None:
                try:
                    if _STREAM_SCALE < 1.0:
                        w = max(1, int(img.width() * _STREAM_SCALE))
                        h = max(1, int(img.height() * _STREAM_SCALE))
                        img = img.resize(w, h)
                    jpg = img.to_jpeg(_JPEG_QUALITY)
                    self._server.write(jpg)
                except Exception:
                    try:
                        self._server.write(img)
                    except Exception:
                        pass
            else:
                time.sleep(_EMPTY_SLEEP)

    def push_frame(self, img) -> None:
        if not self._is_running or self._server is None:
            return
        with self._lock:
            self._queue.appendleft(img)

    def stop(self) -> None:
        self._is_running = False
        if self._server is not None:
            try:
                del self._server
            except Exception:
                pass
            self._server = None
        self._url = ""
        print("[HTTP] stopped")
        sys.stdout.flush()
