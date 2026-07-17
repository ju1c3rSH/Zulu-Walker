import re
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional


_VISIBLE_LOG_LINES = 30

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import SIMPLE


_UPDATE_INTERVAL = 0.1
_MAX_LOG_LINES = 500
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class DebugConsole:
    _instance: Optional["DebugConsole"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    cls._instance = obj
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._status: Dict[str, str] = {}
        self._log_lines: deque = deque(maxlen=_MAX_LOG_LINES)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._error_count = 0
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._console = Console(stderr=True)

    # ---- public API ----

    def set(self, key: str, value) -> None:
        with self._status_lock:
            self._status[key] = str(value)

    def get(self, key: str, default="") -> str:
        with self._status_lock:
            return self._status.get(key, default)

    def log(self, msg: str) -> None:
        cleaned = _ANSI_RE.sub("", msg)
        ts = time.strftime("%H:%M:%S", time.localtime())
        with self._log_lock:
            self._log_lines.append(f"[{ts}] {cleaned}")

    def incr_error(self) -> None:
        self._error_count += 1

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self._key_thread = threading.Thread(target=self._key_listener, daemon=True)
        self._key_thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        from utils.console_capture import ConsoleCapture
        ConsoleCapture.uninstall()

    # ---- render loop ----

    def _render_loop(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("debug_console")

        layout = self._build_layout()
        try:
            with Live(layout, console=self._console, screen=True,
                       refresh_per_second=10, auto_refresh=False) as live:
                # Restore capture — Live(screen=True) may have replaced sys.stdout
                from utils.console_capture import ConsoleCapture
                if ConsoleCapture._instance is not None:
                    import sys
                    sys.stdout = ConsoleCapture._instance
                while self._running:
                    try:
                        self._update_layout(layout)
                        live.update(layout, refresh=True)
                    except Exception:
                        pass
                    self._stop_event.wait(timeout=_UPDATE_INTERVAL)
        except Exception:
            pass

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=1),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        layout["body"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="logs", ratio=2),
        )
        return layout

    def _update_layout(self, layout: Layout) -> None:
        layout["header"].update(
            Panel("Zulu-Walker Debug Console", style="bold white on blue",
                  box=SIMPLE)
        )
        layout["status"].update(self._render_status())
        layout["logs"].update(self._render_logs())
        layout["footer"].update(self._render_footer())

    def _render_status(self) -> Panel:
        with self._status_lock:
            s = dict(self._status)
        table = Table(show_header=False, box=SIMPLE, padding=(0, 1))
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value", no_wrap=True)

        rows = [
            ("Mission", s.get("mission_state", "-"), "white"),
            ("Visual",  s.get("visual_state", "-"), "white"),
            ("Link",    self._link_str(s), self._link_color(s)),
        ]

        # Dynamic camera rows (keys: <cam_id>_fps, <cam_id>_queue, <cam_id>_drop)
        for fps_key in sorted(k for k in s if k.endswith("_fps")):
            prefix = fps_key[:-4]
            if not prefix:
                continue
            fps_val = s.get(fps_key, "-")
            q_val   = s.get(f"{prefix}_queue", "-")
            d_val   = s.get(f"{prefix}_drop", "-")
            rows.append((f"Cam {prefix}", f"{fps_val} FPS | Q:{q_val} | D:{d_val}", "white"))

        rows.extend([
            ("Cargo",   s.get("cargo_count", "-"), "white"),
            ("Batch",   s.get("batch", "-"), "white"),
            ("Step",    s.get("step", "-"), "white"),
            ("Task",    s.get("active_task", "-"), "white"),
            ("Color",   s.get("target_color", "-") or "-", "white"),
            ("Batch1 Seq", s.get("batch1_order", "-"), "white"),
            ("Batch2 Seq", s.get("batch2_order", "-"), "white"),
            ("UART TX", s.get("uart_tx", "-"), "white"),
            ("UART RX", s.get("uart_rx", "-"), "white"),
            ("Frame",   s.get("perf_frame", "-"), "white"),
            ("Detect",  s.get("perf_detect", "-"), "white"),
            ("Error",   f"{self._error_count}", "red" if self._error_count > 0 else "green"),
        ])

        for key, value, style in rows:
            table.add_row(key, Text(value, style=style))

        return Panel(table, title="Status", border_style="green")

    def _render_logs(self) -> Panel:
        with self._log_lock:
            lines = list(self._log_lines)
        if not lines:
            return Panel("(no output)", title="Logs", border_style="blue")
        visible = lines[-_VISIBLE_LOG_LINES:]
        title = f"Logs (last {len(visible)}/{len(lines)})"
        text = Text("\n".join(visible), no_wrap=True, overflow="ellipsis")
        return Panel(text, title=title, border_style="blue")

    def _render_footer(self) -> Panel:
        uptime = time.monotonic() - self._start_time
        uptime_str = f"{int(uptime // 60):02d}:{int(uptime % 60):02d}"
        text = (
            f" Running | Errors: {self._error_count}"
            f" | Uptime: {uptime_str}"
            f" | [q] quit"
        )
        return Panel(text, style="white on grey23", box=SIMPLE)

    # ---- key listener ----

    def _key_listener(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("debug_console")

        try:
            import msvcrt
            while self._running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"q", b"Q"):
                        self.stop()
                        import os
                        os._exit(0)
                        break
                time.sleep(0.1)
        except ImportError:
            pass

    # ---- helpers ----

    def _link_str(self, s: Dict[str, str]) -> str:
        raw = s.get("link_active", "false")
        return "✓ Active" if raw.lower() == "true" else "✗ Lost"

    def _link_color(self, s: Dict[str, str]) -> str:
        raw = s.get("link_active", "false")
        return "green" if raw.lower() == "true" else "red"
