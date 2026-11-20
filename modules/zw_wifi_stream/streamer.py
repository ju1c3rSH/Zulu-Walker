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
    """HTTP JPEG 推流。仅 maix 平台可用（依赖 maix.http + Image.to_jpeg）。"""

    def __init__(self, fps: float = 5.0, quality: int = 80,
                 max_width: int = 320) -> None:
        self._server: Optional[object] = None
        self._is_running: bool = False
        self._url: str = ""
        self._thread: Optional[threading.Thread] = None
        self._queue: deque = deque(maxlen=_QUEUE_MAX)
        self._lock: threading.Lock = threading.Lock()
        # 推流节奏与编码开销限制（不受显示频率影响）
        self._min_interval_s = (1.0 / fps) if fps and fps > 0 else 0.0
        self._last_push_ts = 0.0
        self._quality = int(quality)
        self._max_width = int(max_width)
        # 统计：编码耗时 / 丢帧 / 失败（浏览器端看不到这些，只能靠日志）
        self._enc_ms_sum = 0.0
        self._enc_n = 0
        self._enc_t0 = 0.0
        self._dropped = 0
        self._failed = 0

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

        while self._is_running:
            with self._lock:
                try:
                    img = self._queue.pop()
                except IndexError:
                    img = None
            if img is not None:
                self._write_one(img)
            else:
                time.sleep(_EMPTY_SLEEP)
            self._flush_stats()

    def _write_one(self, img) -> None:
        """编码一帧并写入。write() 要的是 JPEG; 首次失败只打一次日志。"""
        t0 = time.perf_counter()
        try:
            self._server.write(self._encode(img))
        except Exception as e:
            self._failed += 1
            if self._failed == 1:
                from utils.log_util import log_print
                log_print(f"[HTTP] encode/write FAILED: {type(e).__name__}: {e} "
                          f"(only logged once)")
            return
        self._enc_ms_sum += (time.perf_counter() - t0) * 1000.0
        self._enc_n += 1

    def _encode(self, img):
        """原生 Image -> 降采样 -> JPEG。降采样走平台 resize(C++ 内完成)。"""
        if self._max_width > 0:
            try:
                w = img.width()
                if w > self._max_width:
                    h = max(1, img.height() * self._max_width // w)
                    img = img.resize(self._max_width, h)
            except Exception:
                pass
        to_jpeg = getattr(img, "to_jpeg", None)
        if to_jpeg is None:
            raise TypeError(f"frame has no to_jpeg(): {type(img).__name__}")
        try:
            return to_jpeg(quality=self._quality)
        except TypeError:
            return to_jpeg()

    def _flush_stats(self) -> None:
        from utils.log_util import perf_enabled, perf_note
        if not perf_enabled():
            self._enc_t0 = 0.0
            return
        now = time.monotonic()
        if self._enc_t0 == 0.0:
            self._enc_t0 = now
            return
        if now - self._enc_t0 < 1.0:
            return
        n = self._enc_n
        avg = (self._enc_ms_sum / n) if n else 0.0
        self._enc_t0 = now
        self._enc_ms_sum = 0.0
        self._enc_n = 0
        if n:
            perf_note("jpeg", avg)
        # 推流健康度: 这几项是"页面空白"时唯一能定位问题的依据
        from utils.log_util import log_print
        log_print(f"[HTTP] stream enc={n}/s avg={avg:.1f}ms "
                  f"drop={self._dropped} fail={self._failed}")
        self._dropped = 0

    def push_frame(self, img) -> None:
        """喂一帧。按 fps 时间节流；deque(maxlen=1) 丢旧，永不阻塞生产者。

        判断就绪的条件用 `_server`（而非 `_is_running`）: 后者在 srv.start() 之后才置位,
        启动早期送达的帧会被静默丢弃, 表现为浏览器永远空白。
        """
        if self._server is None:
            return
        if self._min_interval_s > 0.0:
            now = time.monotonic()
            if now - self._last_push_ts < self._min_interval_s:
                return
            self._last_push_ts = now
        with self._lock:
            if len(self._queue) >= _QUEUE_MAX:
                self._dropped += 1
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
