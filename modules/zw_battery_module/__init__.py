# -*- coding: utf-8 -*-
"""电量采集: 主循环里的慢节拍，不开线程。

为什么是主循环: 显示线程只在 MAIN 页跑，采集放那里会让数据源跟着页面状态走。
为什么用 monotonic 不用计数器: 主循环实测 89Hz（11.2ms/拍，不是 2ms），拍长还随
负载变 —— 计数器不是时间单位。详见 docs/architecture/battery_hal_hud.md。
采样那一拍 1.02ms（含 is_charging 约 1.30ms），落在主循环里，推流推迟 ≤1ms、不丢帧。

日志策略: **快照发布就打印**。`BatterySampler.poll()` 返回新快照或 None，这里只做
`if snap is not None: log_print(...)` —— 不记"上次是多少"、不判"值变没变"、不限流。
所以模块里没有任何日志状态，也不会在读失败时把旧值当新值重打一遍。
`log_print` 一次调用同时进控制台与 logs/debug.log，存档与控制台因此天然一致。
"""
from __future__ import annotations

import time
from typing import Optional

from framework.hal.battery_sampler import BatterySampler, BatterySnapshot
from utils.log_util import log_print

_SLOW_READ_MS = 50.0            # 单次读超过这个值 = 总线不正常，周期翻倍
_MAX_INTERVAL_S = 60.0

_sampler: Optional[BatterySampler] = None
_base_interval = 10.0           # 配置值; 退避只往上翻, 不会低于它
_interval = 0.0
_next_t = 0.0                   # 0 = 第一拍就采，不要等满一个周期
_driver = "-"


def init(machine, **kwargs) -> None:
    global _sampler, _base_interval, _interval, _next_t, _driver
    _sampler, _interval, _next_t = None, 0.0, 0.0

    cfg = (kwargs.get("config") or {}).get("battery") or {}
    if not cfg.get("enabled", True):
        log_print("[BAT] 未启用（battery.enabled=false）")
        return
    battery = getattr(machine, "battery", None)
    if battery is None:
        log_print("[BAT] 平台未提供电量 HAL，不显示")
        return

    _base_interval = max(float(cfg.get("poll_interval_ms", 10000)) / 1000.0,
                         float(getattr(battery, "min_poll_s", 0.0) or 0.0))
    _interval = _base_interval
    _driver = type(battery).__name__
    _sampler = BatterySampler(battery, low=cfg.get("low_percent", 20))
    log_print(f"[BAT] poll={_interval:.0f}s 每拍一行（首拍即采）driver={_driver}")


def loop() -> None:
    if _sampler is None:
        return
    _poll_if_due(time.monotonic())


def _poll_if_due(now: float) -> None:
    """时间可注入 —— 测试直接喂时间戳，不需要 sleep。"""
    global _next_t, _interval
    if _sampler is None or now < _next_t:
        return

    snap = _sampler.poll()
    if _sampler.last_ms > _SLOW_READ_MS:
        _interval = min(_interval * 2, _MAX_INTERVAL_S)
        log_print(f"[BAT] 单次读 {_sampler.last_ms:.0f}ms，周期退避到 {_interval:.0f}s")
    elif _interval > _base_interval:
        # 一次好读就还一半 —— 否则一次 50ms 抖动会把 60s 永久吃下去。
        _interval = max(_base_interval, _interval / 2)

    if snap is not None:                       # 新快照 = 一行; 没有新快照 = 不说话
        log_print(_format(snap, _sampler.last_ms, _driver))

    # 排下一次要等多久，必须在读完/退避之后再算 —— 否则退避要等一个周期才生效
    _next_t = now + _interval


def _format(snap: BatterySnapshot, read_ms: float, driver: str) -> str:
    """一行一个事实。失败原因随快照过来（`snap.reason`），HAL/采样器不再各说一句。"""
    if snap.percent < 0:
        return (f"[BAT] 无读数（{snap.reason or '原因未知'}）"
                f" read={read_ms:.2f}ms driver={driver}")
    return (f"[BAT] {snap.percent}% {snap.millivolts}mV {snap.level} "
            f"read={read_ms:.2f}ms")


def get_sampler() -> Optional[BatterySampler]:
    """给 HUD 接线用；未启用 / 无 HAL 时返回 None。"""
    return _sampler
