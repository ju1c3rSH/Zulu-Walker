# -*- coding: utf-8 -*-
"""
性能分析器模块

提供细粒度的性能计时和统计功能，帮助定位视觉识别流程中的瓶颈。
"""
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics

from utils.log_util import LoggerFactory


@dataclass
class TimingRecord:
    """单次计时记录"""
    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingStats:
    """计时统计数据"""
    name: str
    avg_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = float('inf')
    count: int = 0
    total_ms: float = 0.0

    def update(self, duration_ms: float):
        """更新统计数据"""
        self.count += 1
        self.total_ms += duration_ms
        self.avg_ms = self.total_ms / self.count
        self.max_ms = max(self.max_ms, duration_ms)
        self.min_ms = min(self.min_ms, duration_ms)


class PerformanceProfiler:
    """
    性能分析器

    支持多阶段计时、滑动窗口统计、控制台和日志文件输出。
    """

    def __init__(self, window_size: int = 60, log_file: str = "logs/performance.log"):
        """
        初始化性能分析器

        Args:
            window_size: 滑动窗口大小（帧数）
            log_file: 日志文件路径
        """
        self._enabled = True
        self._window_size = window_size
        self._log_file = log_file

        # 计时器状态
        self._start_times: Dict[str, float] = {}
        self._records: Dict[str, deque] = {}  # name -> deque of TimingRecord
        self._stats: Dict[str, TimingStats] = {}

        # 层级关系（用于格式化输出）
        self._hierarchy: Dict[str, List[str]] = {}

        # 日志记录器
        self._logger = LoggerFactory.get_logger(
            name="performance",
            level=20,  # INFO
            log_file=log_file,
            console_output=True,
            format_str='%(asctime)s - %(message)s'
        )

        # 帧计数器
        self._frame_count = 0
        self._last_report_time = time.time()
        self._report_interval = 1.0  # 每秒输出一次报告

    def enable(self, enabled: bool = True):
        """启用/禁用性能分析"""
        self._enabled = enabled

    def set_window_size(self, size: int):
        """设置滑动窗口大小"""
        self._window_size = size
        for name in self._records:
            if len(self._records[name]) > size:
                # 保留最新的记录
                self._records[name] = deque(list(self._records[name])[-size:], maxlen=size)

    def set_report_interval(self, interval: float):
        """设置报告输出间隔（秒）"""
        self._report_interval = interval

    def set_hierarchy(self, parent: str, children: List[str]):
        """
        设置层级关系

        Args:
            parent: 父节点名称
            children: 子节点名称列表
        """
        self._hierarchy[parent] = children

    def start(self, name: str):
        """
        开始计时

        Args:
            name: 计时器名称
        """
        if not self._enabled:
            return

        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        停止计时并记录

        Args:
            name: 计时器名称

        Returns:
            耗时（毫秒）
        """
        if not self._enabled:
            return 0.0

        if name not in self._start_times:
            return 0.0

        elapsed = (time.perf_counter() - self._start_times[name]) * 1000  # 转换为毫秒
        del self._start_times[name]

        self._record(name, elapsed)
        return elapsed

    def _record(self, name: str, duration_ms: float):
        """记录计时数据"""
        if name not in self._records:
            self._records[name] = deque(maxlen=self._window_size)
            self._stats[name] = TimingStats(name=name)

        self._records[name].append(TimingRecord(name=name, duration_ms=duration_ms))
        self._stats[name].update(duration_ms)

    @contextmanager
    def timer(self, name: str):
        """
        计时上下文管理器

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
        """
        结束一帧的处理，检查是否需要输出报告
        """
        if not self._enabled:
            return

        self._frame_count += 1
        current_time = time.time()

        if current_time - self._last_report_time >= self._report_interval:
            self.print_report()
            self._last_report_time = current_time

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """获取指定计时器的统计数据"""
        return self._stats.get(name)

    def get_window_stats(self, name: str) -> Optional[TimingStats]:
        """
        获取滑动窗口内的统计数据

        基于最近N帧的数据计算，而非全量累计。
        """
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
        """打印性能报告"""
        if not self._records:
            return

        lines = []
        lines.append("=" * 60)
        lines.append(f"[Performance] Last {self._window_size} frames summary")
        lines.append("=" * 60)

        # 主要计时点
        main_timers = ["frame_capture", "processing", "total"]
        for timer_name in main_timers:
            stats = self.get_window_stats(timer_name)
            if stats and stats.count > 0:
                lines.append(
                    f"{timer_name:20s}: avg={stats.avg_ms:6.2f}ms  "
                    f"max={stats.max_ms:6.2f}ms  min={stats.min_ms:6.2f}ms"
                )

        # 预处理阶段计时
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

        # 检测阶段计时
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

        # 任务级别计时
        task_timers = [name for name in self._records.keys() if name.startswith("task_")]
        if task_timers:
            lines.append("-" * 40)
            lines.append("Task breakdown:")
            for timer_name in sorted(task_timers):
                stats = self.get_window_stats(timer_name)
                if stats and stats.count > 0:
                    # 简化名称显示
                    display_name = timer_name.replace("task_", "")
                    lines.append(
                        f"  ├─ {display_name:25s}: avg={stats.avg_ms:6.2f}ms"
                    )

        # 流程细分计时
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

        # 计算FPS（基于滑动窗口）
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
        # print(report)
        # self._logger.info(f"\n{report}")

    def reset(self):
        """重置所有统计数据"""
        self._records.clear()
        self._stats.clear()
        self._start_times.clear()
        self._frame_count = 0
        self._last_report_time = time.time()


profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """获取全局性能分析器实例"""
    return profiler
