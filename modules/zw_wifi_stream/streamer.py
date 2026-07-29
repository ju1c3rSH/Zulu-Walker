from __future__ import annotations

import sys
import threading
from typing import Optional


class RtspStreamer:
    def __init__(self) -> None:
        self._raw_cam: Optional[object] = None
        self._server: Optional[object] = None
        self._is_running: bool = False
        self._url: str = ""
        self._thread: Optional[threading.Thread] = None

    def setup_camera(self, raw_cam) -> None:
        self._raw_cam = raw_cam
        print("[RTSP] camera set (%dx%d)" % (raw_cam.width(), raw_cam.height()))
        sys.stdout.flush()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def url(self) -> str:
        return self._url

    def start_async(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            print("[RTSP] already starting")
            return
        self._thread = threading.Thread(target=self._do_start, daemon=True)
        self._thread.start()

    def _do_start(self) -> None:
        if self._raw_cam is None:
            print("[RTSP] no camera, abandoned")
            sys.stdout.flush()
            return

        try:
            from maix import rtsp
            self._server = rtsp.Rtsp()
            self._server.bind_camera(self._raw_cam)
            self._server.start()
            self._url = self._server.get_url()
            self._is_running = True
            print("[RTSP] %s" % self._url)
            sys.stdout.flush()
        except Exception as e:
            print("[RTSP] start failed: %s" % e)
            sys.stdout.flush()
            self._server = None
            self._url = ""
            self._is_running = False

    def stop(self) -> None:
        self._is_running = False
        if self._server is not None:
            try:
                del self._server
            except Exception:
                pass
            self._server = None
        self._url = ""
        self._raw_cam = None
        print("[RTSP] stopped")
        sys.stdout.flush()
