from __future__ import annotations

import sys
import time
import threading
from collections import deque
from typing import Optional

_QUEUE_MAX = 1
_EMPTY_SLEEP = 0.015


def _local_ip() -> str:
    """探测本机对外 IP（仅用于日志展示）。

    UDP connect 不会发出任何数据包，只让内核选一次路由源地址；
    没有任何网络（WiFi 未连）时抛异常，返回空串。
    """
    import socket
    for probe in ("8.8.8.8", "192.168.1.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
            finally:
                s.close()
        except Exception:
            continue
    return ""


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
            ip = _local_ip()
            if ip:
                print("[HTTP] %s  <- 浏览器访问用 http://%s:%s/" % (
                    self._url, ip, srv.port()))
            else:
                # 0.0.0.0 监听成功但没有对外 IP = WiFi 没连，外部必访问不了
                print("[HTTP] %s  <- 未探测到本机 IP（WiFi 未连?），"
                      "外部访问不了; SSH 上设备跑 ifconfig 确认" % self._url)
            sys.stdout.flush()
        except Exception as e:
            print("[HTTP] start failed: %s" % e)
            sys.stdout.flush()
            return

        from utils.log_util import perf_note, perf_t0
        while self._is_running:
            with self._lock:
                try:
                    img = self._queue.pop()
                except IndexError:
                    img = None
            if img is not None:
                # [PERF] jpeg: 编码在 write() 内部的 C++ to_jpeg 里, 这里只能量总代价。
                _t0 = perf_t0()
                try:
                    self._server.write(img)
                except Exception:
                    pass
                else:
                    if _t0:
                        perf_note("jpeg", (time.perf_counter() - _t0) * 1000.0)
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
