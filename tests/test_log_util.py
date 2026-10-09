# -*- coding: utf-8 -*-
"""log_util 日志写线程回归测试"""
import queue
import time

import utils.log_util as log_util


def test_writer_survives_and_reports_drops(tmp_path, monkeypatch):
    """_writer_loop 对 _DROP_COUNT 赋值需 global 声明，否则线程读到它即抛
    UnboundLocalError 死亡，日志写线程随后静默丢日志。"""
    monkeypatch.setattr(log_util, "_LOG_FILE", str(tmp_path / "debug.log"))
    monkeypatch.setattr(log_util, "_DROP_COUNT", 3)
    q: queue.Queue = queue.Queue()
    monkeypatch.setattr(log_util, "_LOG_QUEUE", q)

    try:
        log_util.start_log_writer()
        log_util.log_print("hello")

        deadline = time.monotonic() + 5.0
        text = ""
        while time.monotonic() < deadline:
            p = tmp_path / "debug.log"
            text = p.read_text(encoding="utf-8") if p.exists() else ""
            if "dropped 3 messages" in text:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("writer 未在超时前输出丢包警告")

        assert log_util._DROP_COUNT == 0
        assert "hello" in text
        assert log_util._WRITER_THREAD.is_alive()
    finally:
        log_util.stop_log_writer()


def test_log_print_survives_full_queue(monkeypatch):
    """队列满时 log_print 只能丢包计数, **绝不能抛**。

    缺 `global _DROP_COUNT` 时这里抛 UnboundLocalError, 而调用点在
    coordinator 主循环里（apply_gate_task / _on_cmd_request 的 ACK 日志）——
    异常会穿出 coordinator.loop() 打到 module_manager 外层兜底, 整轮主循环
    被 sleep(1.0) 拖停, 期间零 DATA_STREAM（2026-10-09 台架实际表现）。
    """
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put(("占位\n", "占位"))            # 塞满 -> 下一次 put_nowait 必抛 Full
    monkeypatch.setattr(log_util, "_LOG_QUEUE", q)
    monkeypatch.setattr(log_util, "_DROP_COUNT", 0)

    for _ in range(3):
        log_util.log_print("boom")       # 不得抛

    assert log_util._DROP_COUNT == 3
