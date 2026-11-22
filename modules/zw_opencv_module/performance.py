# -*- coding: utf-8 -*-
"""Provide fine-grained timing and sliding-window statistics for the vision pipeline."""
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics

from utils.log_util import LoggerFactory, log_print


@dataclass
class TimingRecord:
    """A single timed interval."""

    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingStats:
    """Aggregated statistics for one timer."""
    name: str
    avg_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = float('inf')
    count: int = 0
    total_ms: float = 0.0

    def update(self, duration_ms: float):
        """Fold one duration into the aggregate."""
        self.count += 1
        self.total_ms += duration_ms
        self.avg_ms = self.total_ms / self.count
        self.max_ms = max(self.max_ms, duration_ms)
        self.min_ms = min(self.min_ms, duration_ms)


class PerformanceProfiler:
    """Record named timings and emit periodic sliding-window statistics."""

    def __init__(self, window_size: int = 60, log_file: str = "logs/performance.log"):
        """Initialize the profiler.

        Args:
            window_size: Number of recent records retained per timer.
            log_file: Path passed to the logger factory.
        """
        self._enabled = True
        self._window_size = window_size
        self._log_file = log_file

        # Timer state.
        self._start_times: Dict[str, float] = {}
        self._records: Dict[str, deque] = {}  # name -> deque of TimingRecord
        self._stats: Dict[str, TimingStats] = {}

        # Parent-child relationships used when formatting output.
        self._hierarchy: Dict[str, List[str]] = {}

        # Logger.
        self._logger = LoggerFactory.get_logger(
            name="performance",
            level=20,  # INFO
            log_file=log_file,
            console_output=True,
            format_str='%(asctime)s - %(message)s'
        )

        # Frame counter.
        self._frame_count = 0
        self._last_report_time = time.time()
        self._report_interval = 1.0  # Emit a report once per second.

    def enable(self, enabled: bool = True):
        """Enable or disable profiling."""
        self._enabled = enabled

    def set_window_size(self, size: int):
        """Set the sliding window size."""
        self._window_size = size
        for name in self._records:
            if len(self._records[name]) > size:
                # Keep only the most recent records.
                self._records[name] = deque(list(self._records[name])[-size:], maxlen=size)

    def set_report_interval(self, interval: float):
        """Set the report interval in seconds."""
        self._report_interval = interval

    def set_hierarchy(self, parent: str, children: List[str]):
        """Set the children of a timer for hierarchical report output.

        Args:
            parent: Name of the parent timer.
            children: Names of the child timers.
        """
        self._hierarchy[parent] = children

    def start(self, name: str):
        """Start timing under the given name."""
        if not self._enabled:
            return

        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop a timer, record it, and return the elapsed milliseconds."""
        if not self._enabled:
            return 0.0

        if name not in self._start_times:
            return 0.0

        elapsed = (time.perf_counter() - self._start_times[name]) * 1000  # Seconds to ms.
        del self._start_times[name]

        self._record(name, elapsed)
        return elapsed

    def _record(self, name: str, duration_ms: float):
        """Append a record and update the aggregate statistics."""
        if name not in self._records:
            self._records[name] = deque(maxlen=self._window_size)
            self._stats[name] = TimingStats(name=name)

        self._records[name].append(TimingRecord(name=name, duration_ms=duration_ms))
        self._stats[name].update(duration_ms)

    @contextmanager
    def timer(self, name: str):
        """Time a block of code.

        Usage:
            with profiler.timer("frame_capture"):
                frame = camera.read()
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def end_frame(self):
        """Mark the end of a frame and emit a report when the interval elapses."""
        if not self._enabled:
            return

        self._frame_count += 1
        current_time = time.time()

        if current_time - self._last_report_time >= self._report_interval:
            self.print_report()
            self._last_report_time = current_time

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Return aggregate statistics for a named timer."""
        return self._stats.get(name)

    def get_window_stats(self, name: str) -> Optional[TimingStats]:
        """Compute statistics from the most recent records, not all-time totals."""
        if name not in self._records or not self._records[name]:
            return None

        records = list(self._records[name])
        durations = [r.duration_ms for r in records]

        return TimingStats(
            name=name,
            avg_ms=statistics.mean(durations),
            max_ms=max(durations),
            min_ms=min(durations),
            count=len(durations),
            total_ms=sum(durations)
        )

    def print_report(self):
        """Build the report and publish frame and detection times to the debug console."""
        if not self._records:
            return

        lines = []
        lines.append("=" * 60)
        lines.append(f"[Performance] Last {self._window_size} frames summary")
        lines.append("=" * 60)

        # Primary timers.
        main_timers = ["frame_capture", "processing", "total"]
        for timer_name in main_timers:
            stats = self.get_window_stats(timer_name)
            if stats and stats.count > 0:
                lines.append(
                    f"{timer_name:20s}: avg={stats.avg_ms:6.2f}ms  "
                    f"max={stats.max_ms:6.2f}ms  min={stats.min_ms:6.2f}ms"
                )

        # Preprocess stage timers.
        preprocess_timers = [
            "preprocess_color_convert",
            "preprocess_mask",
            "preprocess_morphology"
        ]
        preprocess_lines = []
        has_preprocess_data = False
        for timer_name in preprocess_timers:
            stats = self.get_window_stats(timer_name)
            if stats and stats.count > 0:
                has_preprocess_data = True
                display_name = timer_name.replace("preprocess_", "")
                preprocess_lines.append(
                    f"  ├─ {display_name:25s}: avg={stats.avg_ms:6.2f}ms"
                )

        if has_preprocess_data:
            lines.append("-" * 40)
            lines.append("Preprocess stage:")
            lines.extend(preprocess_lines)

        # Detection stage timers.
        detect_timers = ["detect_contours", "detect_ellipse"]
        detect_lines = []
        has_detect_data = False
        for timer_name in detect_timers:
            stats = self.get_window_stats(timer_name)
            if stats and stats.count > 0:
                has_detect_data = True
                detect_lines.append(
                    f"  ├─ {timer_name:25s}: avg={stats.avg_ms:6.2f}ms"
                )

        if has_detect_data:
            lines.append("-" * 40)
            lines.append("Detection stage:")
            lines.extend(detect_lines)

        # Task-level timers.
        task_timers = [name for name in self._records.keys() if name.startswith("task_")]
        if task_timers:
            lines.append("-" * 40)
            lines.append("Task breakdown:")
            for timer_name in sorted(task_timers):
                stats = self.get_window_stats(timer_name)
                if stats and stats.count > 0:
                    # Strip the prefix for display.
                    display_name = timer_name.replace("task_", "")
                    lines.append(
                        f"  ├─ {display_name:25s}: avg={stats.avg_ms:6.2f}ms"
                    )

        # Pipeline-stage timers.
        pipeline_timers = ["process_all", "handle_detection", "state_machine",
                          "ffmpeg_push", "local_display", "callbacks"]
        pipeline_lines = []
        has_pipeline_data = False
        for timer_name in pipeline_timers:
            stats = self.get_window_stats(timer_name)
            if stats and stats.count > 0:
                has_pipeline_data = True
                pipeline_lines.append(
                    f"  ├─ {timer_name:25s}: avg={stats.avg_ms:6.2f}ms"
                )

        if has_pipeline_data:
            lines.append("-" * 40)
            lines.append("Pipeline breakdown:")
            lines.extend(pipeline_lines)

        # Derive FPS from the sliding-window average.
        total_stats = self.get_window_stats("total")
        if total_stats and total_stats.avg_ms > 0:
            fps = 1000.0 / total_stats.avg_ms
            lines.append("-" * 40)
            lines.append(f"Average frame time: {total_stats.avg_ms:.2f}ms")

        from utils.debug_console import DebugConsole

        total_stats = self.get_window_stats("total")
        if total_stats and total_stats.avg_ms > 0:
            DebugConsole().set("perf_frame", f"{total_stats.avg_ms:.1f}ms")
        detect_stats = self.get_window_stats("detect_contours")
        if detect_stats and detect_stats.avg_ms > 0:
            DebugConsole().set("perf_detect", f"{detect_stats.avg_ms:.1f}ms")

        lines.append("=" * 60)

        report = "\n".join(lines)
        # log_print(report)
        # self._logger.info(f"\n{report}")

    def reset(self):
        """Clear all recorded timings and statistics."""
        self._records.clear()
        self._stats.clear()
        self._start_times.clear()
        self._frame_count = 0
        self._last_report_time = time.time()


profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Return the shared profiler instance."""
    return profiler
