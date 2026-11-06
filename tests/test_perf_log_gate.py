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
    """每条用例后恢复全开 —— 闸门/窗口/聚合器都是模块级全局状态, 泄漏会污染其它测试。

    ⚠ 三样都要复位:
      · `perf_log` 名单（`set_perf_log(True)`）;
      · 汇总窗口（`set_perf_interval`）—— 某例设成 0.5 会让后续用例的节拍全错;
      · `_AGGS` —— 跨用例残留 _t0/_n 会让"未到窗口不打印"的用例假失败。
    `_PERF_DETAIL` / `_PERF_GATE_SET` 没有 setter 能还原成"未配置", 直接写回默认
    （`set_perf_gate(None, None)` 是 no-op, 还原不了）。
    """
    L.set_perf_log(True)
    L.set_perf_interval(L.DEFAULT_PERF_INTERVAL_S)
    L._PERF_DETAIL = True
    L._PERF_GATE = True
    L._PERF_GATE_SET = False
    L._AGGS.clear()
    yield
    L.set_perf_log(True)
    L.set_perf_interval(L.DEFAULT_PERF_INTERVAL_S)
    L._PERF_DETAIL = True
    L._PERF_GATE = True
    L._PERF_GATE_SET = False
    L._AGGS.clear()


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


# ---------------- 汇总窗口（observability.perf_interval_s）----------------

def test_perf_interval_default_is_ten_seconds():
    """默认 10s: 配置缺键时也必须保持"已降频"（1s 一条才是刷屏的根因）。"""
    assert L.DEFAULT_PERF_INTERVAL_S == 10.0
    assert L.perf_interval_s() == 10.0


def test_set_perf_interval_ignores_garbage(monkeypatch):
    """非法值不得改变现值。

    ⚠ NaN 必须挡掉: `elapsed < nan` 恒 False -> 每次 add 都打印（洪泛）;
      inf 则永远等不到窗口 -> `[PERF]` 全静默（失明）。True 也不能当 1 秒用。
    """
    monkeypatch.setattr(L, "_PERF_INTERVAL_S", 5.0)
    for bad in (0, -1, "x", None, float("nan"), float("inf"), True, object()):
        L.set_perf_interval(bad)
        assert L.perf_interval_s() == 5.0, f"{bad!r} 不该改变窗口"
    L.set_perf_interval(0.1)          # 下限 0.5, 再快就回到刷屏区间
    assert L.perf_interval_s() == 0.5


class _Clock:
    """只替换 log_util 的模块引用 —— patch 全局 time 会冻住 writer 线程。"""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t


def test_perf_agg_waits_full_window(monkeypatch):
    """窗口未到不打印, 到窗口只打一条, 之后重新起窗口 —— "降频"的硬证据。"""
    clk = _Clock()
    monkeypatch.setattr(L, "time", clk)
    monkeypatch.setattr(L, "_PERF_INTERVAL_S", 10.0)
    L._AGGS.clear()
    _drain()

    L.perf_note("unit", 1.0)          # 首拍只记 t0
    clk.t += 9.0
    L.perf_note("unit", 1.0)          # 9s: 窗口未满
    assert _drain() == [], "未到 10s 窗口不得打印"

    clk.t += 1.5
    L.perf_note("unit", 1.0)          # 10.5s: 该打一条
    lines = _drain()
    assert len(lines) == 1, f"一个窗口只应打一条, 实际 {len(lines)}: {lines}"
    assert lines[0].startswith("[PERF] unit ")
    assert "Hz" in lines[0] and "avg=" in lines[0]

    L.perf_note("unit", 1.0)          # 刚 flush 过 -> 不得立刻再打
    assert _drain() == []


def test_detail_off_makes_probes_free(monkeypatch):
    """perf_detail=false: 探针不产出任何行, 连计时入口都返回 0（零开销）。"""
    monkeypatch.setattr(L, "_PERF_DETAIL", False)
    L._AGGS.clear()
    _drain()
    for _ in range(20):
        L.perf_note("hud", 1.0)
    assert _drain() == [], "detail 关闭时探针行必须完全消失"
    assert L.perf_t0() == 0.0, "关 detail 后计时起点必须是 0, 调用方据此跳过计时"


def test_wdt_only_mute_keeps_perf_and_heartbeats():
    """只静音 `[WDT] ` 时, `[PERF]` / `[MEM]` / `DATA_STREAM` 必须照打。

    这是当前 project_config.yaml 的形态（enabled: false + names 只列 WDT）:
    `_PERF_NAMES_MUTE_ALL` 判据是"超集", 列一条不会连带放开或静音别的东西。
    """
    L.set_perf_log(False, names=["[WDT] "])
    assert L.perf_log_enabled() is False
    assert L._PERF_NAMES_MUTE_ALL is False
    _drain()
    keep = [
        "[PERF] vision 6.0Hz loop=50ms",
        "[MEM] user=103/128MB CMM=10/300MB",
        "[UART TX] DATA_STREAM 32.6Hz seq=1",
    ]
    L.log_print("[WDT] main feed #600")
    for line in keep:
        L.log_print(line)
    assert _drain() == keep, "只静音 WDT, 其余高频行不得被连带吞掉"


def test_perf_log_summary_is_three_state():
    """`[PERFLOG]` 自检行不能把"只静音 WDT"打成 MUTED（会让人以为全关了）。"""
    assert L.perf_log_summary() == "ON"
    L.set_perf_log(False, names=["[WDT] "])
    assert L.perf_log_summary() == "custom([WDT] )"
    L.set_perf_log(False)             # 默认组整组静音
    assert L.perf_log_summary() == "MUTED"


def test_perflog_line_is_never_swallowed():
    """自检行必须免疫于它自己描述的闸门。

    回归: 自定义名单 "custom([WDT] )" 的原文里就含被静音子串 "[WDT] ",
    子串匹配会把 `[PERFLOG]` 行一起吃下去 —— 改完配置连"改成了什么"都看不到。
    """
    L.set_perf_log(False, names=["[WDT] "])
    _drain()
    line = ("[PERFLOG] perf_log=custom([WDT] ) observability.perf=ON "
            "perf_detail=OFF interval=10.0s")
    L.log_print(line)
    assert _drain() == [line], "[PERFLOG] 行不得被自己的静音名单吞掉"


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


def _apply_obs(obs):
    from app.main import _apply_perf_log_config
    _apply_perf_log_config(None, obs)


def test_apply_observability_interval_transparent():
    """`observability.perf_interval_s` 必须透传到 log_util（否则配置改了等于没改）。"""
    _apply_obs({"perf": True, "perf_detail": False, "perf_interval_s": 3})
    assert L.perf_interval_s() == 3.0
    assert L.perf_detail_enabled() is False


def test_apply_missing_interval_keeps_default():
    """只写 `perf` 时窗口保持默认 10s —— 空配置不得把节拍重置回 1s。"""
    _apply_obs({"perf": True})
    assert L.perf_interval_s() == L.DEFAULT_PERF_INTERVAL_S
