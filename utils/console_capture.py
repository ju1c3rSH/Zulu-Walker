import os
import sys
import threading
from collections import deque
from typing import Optional


class ConsoleCapture:
    _instance: Optional["ConsoleCapture"] = None

    def __init__(self, real_stdout, log_dir="logs"):
        self._real_stdout = real_stdout
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "debug.log")
        self._file = open(self._log_path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._installed = True
        self._history = deque(maxlen=200)
        ConsoleCapture._instance = self

    def write(self, text: str) -> None:
        with self._lock:
            if not self._installed:
                return
            self._file.write(text)
            self._file.flush()
        if text.strip():
            from utils.debug_console import DebugConsole
            dc = DebugConsole()
            if dc._running:
                for line in text.splitlines():
                    stripped = line.rstrip()
                    if stripped:
                        dc.log(stripped)
        with self._lock:
            self._history.append(text)

    def flush(self) -> None:
        with self._lock:
            if self._installed:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._installed = False
            self._file.close()

    def replay_history(self) -> None:
        for line in self._history:
            self._real_stdout.write(line)
        self._real_stdout.flush()

    @classmethod
    def install(cls, log_dir="logs") -> "ConsoleCapture":
        if cls._instance is not None:
            return cls._instance
        real = sys.__stdout__
        capture = cls(real, log_dir)
        sys.stdout = capture
        return capture

    @classmethod
    def reinstall(cls) -> None:
        """Re-assert sys.stdout = capture (undo external replacement)."""
        if cls._instance is not None:
            sys.stdout = cls._instance

    @classmethod
    def uninstall(cls) -> None:
        if cls._instance is not None:
            cls._instance.replay_history()
            sys.stdout = cls._instance._real_stdout
            cls._instance.close()
            cls._instance = None
