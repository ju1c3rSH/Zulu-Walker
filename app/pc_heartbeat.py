import json
import socket
import threading
import time
from typing import Optional


class PcHeartbeatDetector:
    """UDP heartbeat listener to detect if PC monitoring app is connected.

    PC sends {"type":"heartbeat"} JSON every 2s to UDP port 5001.
    If no heartbeat received for 5s, PC is considered disconnected.
    """

    _PORT = 5001
    _TIMEOUT = 5.0

    def __init__(self):
        self._last_heartbeat = 0.0
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self._pc_ip: Optional[str] = None
        self._on_connected: Optional[callable] = None
        self._was_connected = False

    @property
    def is_connected(self) -> bool:
        return time.monotonic() - self._last_heartbeat < self._TIMEOUT

    @property
    def pc_ip(self) -> Optional[str]:
        return self._pc_ip

    def set_on_connected(self, cb: callable) -> None:
        self._on_connected = cb

    def start(self) -> None:
        """Start the UDP listener daemon thread (idempotent)."""
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self._PORT))
        except OSError as e:
            sock.close()
            raise RuntimeError(f"PcHeartbeatDetector: cannot bind to port {self._PORT}: {e}") from e

        self._sock = sock
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listener and release resources (idempotent)."""
        self._stop_event.set()
        self._was_connected = False
        sock = self._sock
        if sock is not None:
            self._sock = None
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _listen(self) -> None:
        """Background loop: recv JSON heartbeats, update _last_heartbeat."""
        sock = self._sock
        if sock is None:
            return
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(1024)
            except OSError:
                if self._stop_event.is_set():
                    break
                continue
            except Exception:
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
                if isinstance(msg, dict) and msg.get("type") == "heartbeat":
                    self._last_heartbeat = time.monotonic()
                    self._pc_ip = addr[0]
                    if not self._was_connected and self._on_connected:
                        try:
                            self._on_connected()
                        except Exception:
                            pass
                    self._was_connected = True
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
