"""Per-thread CPU-time sampling for wall-vs-CPU cost attribution.

``perf_counter`` measures wall time, which includes GIL waits and scheduler
suspension; a thread doing no work can still appear busy. Report both so a
reader can tell CPU-bound work from blocking.
"""
from __future__ import annotations

import time


def thread_cpu() -> float:
    """Return the current thread's cumulative CPU time in seconds.

    Return:
        CPU seconds consumed by this thread; falls back to process CPU time
        on platforms without ``time.thread_time``.
    """
    try:
        return time.thread_time()
    except AttributeError:
        return time.process_time()
