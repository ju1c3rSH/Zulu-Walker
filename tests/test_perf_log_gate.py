# -*- coding: utf-8 -*-
"""高频性能日志闸门（`perf_log:` 配置）—— 静音性能行, 保留事件行。

背景（2026-10-09 台架）:
  调试时 `[PERF] vision/mainloop/lcd` × 3 + `[UART TX] DATA_STREAM`
  + `CMD_ACK` + `[WDT] feed` + `[MEM]` **每秒各刷一条**, 把真正想看的信息淹掉。
  它们全是性能观测/心跳痕迹, 与协议语义无关。

钉住的不变量（缺一条都会让台架重新刷屏, 或让链路状态变瞎）:
  1. **只少打印, 不改变行为**: 推流帧照发、`log_print` 仍非阻塞（queue 行为不变）;
  2. 默认组被吞, **事件类必须保留**（`[UART ACTION]` / `[UART RX]` /
     `[UART TX] CMD_NACK` / `[FPS]`）—— 关了就看不出"订没订上 / 该不该 NACK";
  3. `perf_log: true`（旧行为）与映射写法都能被 main 的 `_apply_perf_log_config` 消化;
  4. 兼容旧布尔配置: 老 `perf_log: false` 不得因升级配置结构而崩。
"""
import queue

import pytest

import utils.log_util as L


@pytest.fixture(autouse=True)
def _restore_gate():
    """每条用例后恢复全开 —— 闸门是模块级全局状态, 泄漏会污染其它测试。"""
    L.set_perf_log(True)
    yield
    L.set_perf_log(True)


def _drain() -> list:
    """把队列里排队的 (line, msg) 全取走, 返回 msg 列表。"""
    out = []
    while True:
        try:
            line, msg = L._LOG_QUEUE.get_nowait()
        except queue.Empty:
            break
        out.append(msg)
    return out


# ---------------- 闸门本身 ----------------

def test_default_is_all_on():
    """默认（未配置 / enabled: true）= 全开, 不得因引入闸门而悄悄少打。"""
    assert L.perf_log_enabled() is True
    assert L._PERF_LOG_NAMES == set()


def test_default_group_muted_when_disabled():
    L.set_perf_log(False)
    assert L.perf_log_enabled() is False

    _drain()
    for line in (
        "[PERF] vision 20.0Hz loop=50ms",
        "[PERF] mainloop 300Hz body=0.01ms",
        "[PERF] lcd show 1.0Hz avg=4.4ms",
        "[UART TX] DATA_STREAM 32.6Hz seq=1",
        "[UART TX] CMD_ACK type=TARGET freq=60Hz",
        "[WDT] main feed #600",
        "[MEM] user=100/128MB CMM=10/300MB",
    ):
        L.log_print(line)
    assert _drain() == [], "默认性能组必须被完整吞掉"


def test_event_lines_survive_muting():
    """静音的是性能/心跳痕迹, 不是事件 —— 这是"关了还能调试"的前提。"""
    L.set_perf_log(False)
    _drain()
    keep = [
        "[UART ACTION] gate=TARGET codes=[0, 2, 0] latch=cleared",
        "[UART RX] CMD_REQUEST type=EXPLOSIVE",
        "[UART TX] CMD_NACK type=0x10 reason=UNSUPPORTED_TYPE",
        "[FPS] pipeline: 21.5",
    ]
    for line in keep:
        L.log_print(line)
    assert _drain() == keep, "事件类日志不得被性能闸门误吞"


def test_re_enable_restores_all_lines():
    L.set_perf_log(False)
    L.set_perf_log(True)
    _drain()
    L.log_print("[PERF] vision back")
    assert _drain() == ["[PERF] vision back"]


def test_custom_names_override_default_group():
    """names 自定义时, 默认组不再自动生效 —— 只吞列出的标签。"""
    L.set_perf_log(False, names=["[MEM] "])
    _drain()
    L.log_print("[MEM] user=100/128MB")
    L.log_print("[PERF] vision 20Hz")
    assert _drain() == ["[PERF] vision 20Hz"]


def test_muting_is_non_blocking_and_queue_unchanged():
    """闸门必须在入队**之前**短路: queue 大小不变（这是"不改变行为"的硬证据）。"""
    L.set_perf_log(False)
    _drain()
    before = L._LOG_QUEUE.qsize()
    for _ in range(50):
        L.log_print("[PERF] vision 20Hz")
    assert L._LOG_QUEUE.qsize() == before
    assert L._DROP_COUNT == 0


def test_empty_gate_is_zero_overhead_path():
    """全开时 `_is_perf_log` 恒 False（不遍历集合）—— 热路径不得被闸门拖慢。"""
    assert L._is_perf_log("[PERF] vision 20Hz") is False


# ---------------- main 的配置应用（兼容布尔 / 映射 / 缺键）----------------

def _apply(cfg):
    from app.main import _apply_perf_log_config
    _apply_perf_log_config(cfg)


def test_apply_bool_false_mutes_default_group():
    _apply(False)
    assert L.perf_log_enabled() is False


def test_apply_bool_true_is_all_on():
    _apply(False)
    _apply(True)
    assert L.perf_log_enabled() is True


def test_apply_mapping_with_names():
    _apply({"enabled": False, "names": ["[WDT] ", "[MEM] "]})
    _drain()
    L.log_print("[WDT] main feed #600")
    L.log_print("[MEM] user=100/128MB")
    L.log_print("[PERF] vision 20Hz")     # 不在自定义 names 里 -> 放行
    assert _drain() == ["[PERF] vision 20Hz"]


def test_apply_mapping_missing_enabled_defaults_on():
    """`perf_log: {names: [...]}` 但没写 enabled -> 视为全开（不误静音）。"""
    _apply({"names": ["[MEM] "]})
    assert L.perf_log_enabled() is True


def test_apply_none_keeps_default():
    _apply(None)
    assert L.perf_log_enabled() is True


def test_apply_does_not_raise_on_garbage():
    """配置乱写不能让启动挂掉 —— 异常路径保持全开。"""
    _apply(object())
    assert L.perf_log_enabled() is True
