from __future__ import annotations

import sys
import threading
from collections import deque
from typing import Optional

_QUEUE_MAX = 2


class JpegStreamer:
    def __init__(self) -> None:
        self._server: Optional[object] = None
        self._is_running: bool = False
        self._url: str = ""
        self._thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
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
        self._thread = threading.Thread(target=self._do_start, daemon=True)
        self._thread.start()

    def _do_start(self) -> None:
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
            self._server = None
            self._is_running = False
            return

        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def push_frame(self, img) -> None:
        if not self._is_running or self._server is None:
            return
        with self._lock:
            self._queue.appendleft(img)

    def _send_loop(self) -> None:
        while self._is_running:
            with self._lock:
                try:
                    img = self._queue.pop()
                except IndexError:
                    img = None
            if img is not None:
                try:
                    self._server.write(img)
                except Exception:
                    pass
            else:
                import time
                time.sleep(0.005)

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
