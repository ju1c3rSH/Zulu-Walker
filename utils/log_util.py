import logging
import logging.handlers
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "debug.log")
_LOG_QUEUE: queue.Queue = queue.Queue(maxsize=2048)
_WRITER_THREAD: Optional[threading.Thread] = None
_SENTINEL: object = object()
_DROP_COUNT: int = 0  # cumulative drops when the queue is full; GIL-protected, no extra lock

_CONSOLE_MODULE = "utils.debug_console"

# High-frequency perf-line gates. Names match by substring; [PERF] lines are
# gated by the two-level observability switch below, other names by the mute set.
_DEFAULT_PERF_NAMES = (
    "[PERF] vision",
    "[PERF] mainloop",
    "[PERF] lcd show",
    "[UART TX] DATA_STREAM",
    "[UART TX] CMD_ACK",
    "[WDT] ",
    "[MEM] ",
)
_PERF_LOG_NAMES: set = set()

# Two-level observability switch. The trailing space in "[PERF] " is required:
# startswith("[PERF") would also swallow [PERFLOG] self-check lines.
_PERF_TAG = "[PERF] "
_PERF_COARSE = ("[PERF] vision", "[PERF] mainloop", "[PERF] lcd show")
_PERF_GATE = True
_PERF_DETAIL = True
_PERF_GATE_SET = False
_PERF_NAMES_MUTE_ALL = False
_PERF_DEFAULT_SET = frozenset(_DEFAULT_PERF_NAMES)

# Aggregation window in seconds for periodic [PERF] summaries.
DEFAULT_PERF_INTERVAL_S = 10.0
_PERF_INTERVAL_S = DEFAULT_PERF_INTERVAL_S

try:
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
except Exception:
    pass


def _writer_loop() -> None:
    global _DROP_COUNT
    while True:
        item = _LOG_QUEUE.get()
        if item is _SENTINEL:
            _LOG_QUEUE.task_done()
            break

        line, msg = item

        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass

        try:
            mod = sys.modules.get(_CONSOLE_MODULE)
            if mod is None:
                # Not imported: writing to stdout without pulling in rich.
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
            else:
                dc = mod.DebugConsole()
                if not mod.DebugConsole._global_enabled:
                    sys.__stdout__.write(line)
                    sys.__stdout__.flush()
                else:
                    dc.log(msg)
                    if not dc._running:
                        sys.__stdout__.write(line)
                        sys.__stdout__.flush()
        except Exception:
            pass

        # Report any drops once the queue drains, from the writer thread only.
        # This keeps log_print callers free of I/O and never blocks the
        # MaixCAM2 vision pipeline.
        if _DROP_COUNT > 0 and _LOG_QUEUE.empty():
            drops = _DROP_COUNT
            _DROP_COUNT = 0
            warn_ts = datetime.now().strftime("[%H:%M:%S] ")
            warn = f"{warn_ts}[WARN] log_print queue full, dropped {drops} messages\n"
            try:
                with open(_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(warn)
                    f.flush()
            except Exception:
                pass
            sys.__stdout__.write(warn)
            sys.__stdout__.flush()
            try:
                mod = sys.modules.get(_CONSOLE_MODULE)
                if mod is not None:
                    dc = mod.DebugConsole()
                    dc.log(warn.rstrip("\n"))
                    dc.set("log_drop", str(drops))
            except Exception:
                pass

        _LOG_QUEUE.task_done()


def start_log_writer() -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD is not None and _WRITER_THREAD.is_alive():
        return
    _WRITER_THREAD = threading.Thread(target=_writer_loop, daemon=True, name="log-writer")
    _WRITER_THREAD.start()


def stop_log_writer(timeout: float = 2.0) -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD is None:
        return
    try:
        _LOG_QUEUE.put(_SENTINEL, timeout=1.0)
    except queue.Full:
        pass
    _WRITER_THREAD.join(timeout=timeout)
    _WRITER_THREAD = None


def _refresh_mute_all() -> None:
    """Set the mute-all flag when the default group is wholly muted.

    Uses superset, not equality: adding names must not let ``[PERF] page*``
    leak through once the default group is muted.
    """
    global _PERF_NAMES_MUTE_ALL
    _PERF_NAMES_MUTE_ALL = bool(_PERF_LOG_NAMES) and _PERF_DEFAULT_SET <= _PERF_LOG_NAMES


def set_perf_log(enabled: bool = True, names=None) -> None:
    """Mute or allow named high-frequency lines.

    Args:
        enabled: False mutes ``names``; True allows all.
        names: Substrings to control; None selects the default group.

    Context:
        Does not touch the observability gate, so it cannot re-enable a
        switch that observability explicitly turned off.
    """
    global _PERF_LOG_NAMES
    if enabled:
        _PERF_LOG_NAMES = set()
    else:
        _PERF_LOG_NAMES = (set(_DEFAULT_PERF_NAMES) if not names
                           else {str(n) for n in names})
    _refresh_mute_all()


def set_perf_gate(perf=None, detail=None) -> None:
    """Apply the observability switch: ``perf`` for all [PERF] lines, ``detail`` for probes.

    Args:
        perf: Master switch for [PERF] lines; None keeps the current value.
        detail: Per-item probe switch; None keeps the current value.
    """
    global _PERF_GATE, _PERF_DETAIL, _PERF_GATE_SET
    if perf is not None:
        _PERF_GATE = bool(perf)
        _PERF_GATE_SET = True
    if detail is not None:
        _PERF_DETAIL = bool(detail)


def perf_log_enabled() -> bool:
    """Return whether performance lines are fully allowed (for tests/self-check)."""
    return not _PERF_LOG_NAMES


def set_perf_interval(seconds=None) -> None:
    """Set the [PERF] summary window in seconds.

    Args:
        seconds: Positive finite number; None/bool/NaN/inf/<=0 keep current.

    Context:
        Clamps to a 0.5s floor to avoid re-entering log-flood territory.
    """
    global _PERF_INTERVAL_S
    if seconds is None or isinstance(seconds, bool):
        return
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return
    if not math.isfinite(v) or v <= 0:
        return
    _PERF_INTERVAL_S = max(0.5, v)


def perf_interval_s() -> float:
    """Return the shared [PERF] summary window."""
    return _PERF_INTERVAL_S


def perf_log_summary() -> str:
    """Return a three-state description of the mute set (for [PERFLOG])."""
    if not _PERF_LOG_NAMES:
        return "ON"
    if _PERF_NAMES_MUTE_ALL:
        return "MUTED"
    return "custom(" + ",".join(sorted(_PERF_LOG_NAMES)) + ")"


def perf_enabled() -> bool:
    """Return the master [PERF] switch; explicit config wins over name inference."""
    if _PERF_GATE_SET:
        return _PERF_GATE
    return not _PERF_NAMES_MUTE_ALL


def perf_detail_enabled() -> bool:
    """Return whether per-item probes are on (always False when master is off)."""
    return _PERF_DETAIL and perf_enabled()


def perf_t0() -> float:
    """Return a probe start time, or 0.0 when gated so callers skip timing."""
    if _PERF_DETAIL and perf_enabled():
        return time.perf_counter()
    return 0.0


def _is_coarse_perf(msg: str) -> bool:
    return any(msg.startswith(name) for name in _PERF_COARSE)


def _is_perf_log(msg: str) -> bool:
    """Return True when a message is currently muted. Empty set is zero-cost."""
    if msg.startswith("[PERFLOG]"):
        return False
    if msg.startswith(_PERF_TAG):
        if not perf_enabled():
            return True
        if not _PERF_DETAIL and not _is_coarse_perf(msg):
            return True
    if not _PERF_LOG_NAMES:
        return False
    for name in _PERF_LOG_NAMES:
        if name in msg:
            return True
    return False


class PerfAgg:
    """Emit one ``[PERF] <label>`` summary per window; zero formatting when gated."""

    __slots__ = ("_label", "_tag", "_t0", "_n", "_ms")

    def __init__(self, label: str, tag: Optional[str] = None) -> None:
        self._label = label
        self._tag = tag
        self._t0 = 0.0
        self._n = 0
        self._ms = 0.0

    def add(self, ms: float, tag: Optional[str] = None) -> None:
        """Accumulate one sample; flush a summary when the window elapses."""
        if not perf_detail_enabled():
            return
        self._n += 1
        self._ms += ms
        if tag is not None:
            self._tag = tag
        now = time.monotonic()
        if self._t0 == 0.0:
            self._t0 = now
            return
        elapsed = now - self._t0
        if elapsed < _PERF_INTERVAL_S:
            return
        n = self._n
        avg = self._ms / n
        extra = " " + self._tag if self._tag else ""
        log_print(f"[PERF] {self._label} {n / elapsed:.1f}Hz "
                  f"avg={avg:.2f}ms{extra}")
        self._t0 = now
        self._n = 0
        self._ms = 0.0


_AGGS: Dict[str, PerfAgg] = {}


def perf_note(label: str, ms: float, tag: Optional[str] = None) -> None:
    """Probe entry: one ``[PERF] <label>`` summary per window; gated costs one bool.

    Args:
        label: Aggregation key (one PerfAgg per label).
        ms: Measured milliseconds for this sample.
        tag: Optional suffix carried into the summary line.
    """
    if not perf_detail_enabled():
        return
    agg = _AGGS.get(label)
    if agg is None:
        agg = _AGGS[label] = PerfAgg(label)
    agg.add(ms, tag)


def debug_console_active() -> bool:
    """Return whether the debug console TUI is actually running.

    Uses a ``sys.modules`` lookup so ``rich`` is never imported just to ask
    this question (CAM1 has 128MB user memory).
    """
    mod = sys.modules.get(_CONSOLE_MODULE)
    if mod is None:
        return False
    return bool(getattr(getattr(mod, "DebugConsole", None), "_global_enabled", False))


def log_print(msg: str = "", *args, **kwargs) -> None:
    """Emit a line through the non-blocking queue.

    Args:
        msg: Message, with optional printf-style positional args.

    Context:
        Caller never touches I/O. Muted lines are dropped before enqueue;
        a full queue only increments a counter.
    """
    global _DROP_COUNT
    ts = datetime.now().strftime("[%H:%M:%S] ")
    if args:
        parts = [str(msg)] + [str(a) for a in args]
        msg = " ".join(parts)
    else:
        msg = str(msg)
    if _is_perf_log(msg):
        return
    line = f"{ts}{msg}\n"
    try:
        _LOG_QUEUE.put_nowait((line, msg))
    except queue.Full:
        _DROP_COUNT += 1


class LoggerFactory:
    """Factory that creates and caches dynamically configured loggers."""
    
    _instances: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, 

                    name: str,
                    level: int = logging.INFO,
                    log_file: Optional[str] = None,
                    max_bytes: int = 10 * 1024 * 1024,  # 10MB
                    backup_count: int = 5,
                    console_output: bool = True,
                    format_str: Optional[str] = None) -> logging.Logger:
        """
        Return a configured logger, creating it if needed.
        
        Args:
            name: Logger name.
            level: Logging level.
            log_file: Optional path to a log file; when set, also log to file.
            max_bytes: Maximum size of a single log file in bytes.
            backup_count: Number of rotated backup files to keep.
            console_output: Whether to emit to the console.
            format_str: Custom log format string.
        """
        if name in cls._instances:
            return cls._instances[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        

        if logger.handlers:
            return logger
        
        format_str = format_str or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        
        # Console handler (stdout).
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        

        # File handler (rotating file output).
        if log_file:
            # Ensure the log directory exists.
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._instances[name] = logger
        return logger
    
    @classmethod
    def clear_instances(cls):
        """Clear all cached logger instances (mainly for tests)."""
        cls._instances.clear()


class ConfigurableLogger:
    """Thin configurable wrapper around a standard logger."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a configurable logger.
        
        Args:
            config: Configuration dict; for example::
                {
                    'name': 'myapp',
                    'level': 'INFO',
                    'log_file': 'logs/app.log',
                    'max_bytes': 10485760,
                    'backup_count': 5,
                    'console_output': True,
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'extra_fields': ['user_id', 'request_id']
                }
        """
        self.config = config
        self.name = config.get('name', 'default')
        self.level = getattr(logging, config.get('level', 'INFO').upper())
        
        self.logger = LoggerFactory.get_logger(
            name=self.name,
            level=self.level,
            log_file=config.get('log_file'),
            max_bytes=config.get('max_bytes', 10 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            console_output=config.get('console_output', True),
            format_str=config.get('format')
        )
        
        self.extra_fields = config.get('extra_fields', [])
    
    def _add_extra_fields(self, extra: Optional[Dict] = None) -> Dict:
        """Add extra context fields to the given dict."""
        if not extra:
            extra = {}
        for field in self.extra_fields:
            if field not in extra:
                extra[field] = getattr(self, field, None)
        return extra
    
    def info(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.info(msg, extra=extra if extra else None)
    
    def debug(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.debug(msg, extra=extra if extra else None)
    
    def warning(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.warning(msg, extra=extra if extra else None)
    
    def error(self, msg: str, exc_info: bool = True, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.error(msg, exc_info=exc_info, extra=extra if extra else None)
    
    def set_extra_field(self, key: str, value: Any):
        """Set an extra context field."""
        setattr(self, key, value)


class UARTModuleLogger(ConfigurableLogger):
    """Logger for a UART module, with module-level context."""
    
    def __init__(self, module_name: str):
        config = {
            'name': f'module_{module_name}',
            'level': 'DEBUG',
            'log_file': f'logs/modules/{module_name}.log',
            'max_bytes': 50 * 1024 * 1024,
            'backup_count': 10,
            'console_output': True,
            'extra_fields': ['module_name']
        }
        super().__init__(config)
        self.module_name = module_name
        
    def log_send(self, data: str):
        self.info(f"发送数据: {data}")
        
    def log_receive(self, data: str):
        self.info(f"接收数据: {data}")
        
    def log_info(self, message: str):
        self.info(message)
    def log_error(self, error: Exception):
        self.error(f"模块错误: {error}")        


class DataPipelineLogger(ConfigurableLogger):
    """Logger for a data pipeline, with batch and step context."""
    
    def __init__(self, pipeline_name: str):
        config = {
            'name': f'pipeline_{pipeline_name}',
            'level': 'DEBUG',
            'log_file': f'logs/pipelines/{pipeline_name}.log',
            'max_bytes': 100 * 1024 * 1024,
            'backup_count': 20,
            'console_output': False,  # file-only
            'extra_fields': ['batch_id', 'step_name']
        }
        super().__init__(config)
        self.pipeline_name = pipeline_name
    
    def log_step_start(self, batch_id: str, step_name: str):
        self.set_extra_field('batch_id', batch_id)
        self.set_extra_field('step_name', step_name)
        self.info(f"开始处理步骤: {step_name}")
    
    def log_step_end(self, batch_id: str, step_name: str, records_processed: int):
        self.set_extra_field('batch_id', batch_id)
        self.set_extra_field('step_name', step_name)
        self.info(f"完成处理步骤: {step_name}, 处理记录数: {records_processed}")
    
    def log_data_quality(self, batch_id: str, issues: list):
        self.set_extra_field('batch_id', batch_id)
        self.warning(f"数据质量问题: {len(issues)}个问题 - {issues[:3]}")