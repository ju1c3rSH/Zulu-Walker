# -*- coding: utf-8 -*-
"""电量通道探针 —— 在设备上量"封装的库到底要多久"。

## 为什么要这个脚本

`framework/hal/platforms/maixcam1/battery.py` 的设计依赖一个数：一次
`get_bat_percent()` 是毫秒级。这个数是从 I2C 时钟周期推出来的（100kHz 下一次寄存器读
≈ 36~38 个时钟 ≈ 0.37ms），**没有在板上量过**。总线时钟由适配器默认值决定，代码里
没人设过它，所以真值可能是推断值的 1/4（400kHz）。

这个脚本给三样东西:
  1. 构造 `pmu.PMU()` 要多久（一次芯片 ID 校验的价格）；
  2. `get_bat_percent()` / `get_bat_vol()` 各 20 次的 min/中位/p90/max；
  3. `voltage_now` 到底是不是 0（调研说 CAM2 的 `get_bat_vol()` 恒 0，Pro 未知）。

## 用法

    # 设备上、仓库根目录
    python scripts/battery_probe.py             # 当前平台该走哪条通道就走哪条
    python scripts/battery_probe.py sysfs       # 强制读 sysfs（CAM2 用）
    python scripts/battery_probe.py pmu         # 强制走官方 pmu（Pro 用）

PC 上跑会直接告诉你"没有 maix"，不报错。整个脚本只读，不改任何东西。
"""
from __future__ import annotations

import statistics
import sys
import time

_DIR = "/sys/class/power_supply/cw2015-battery"
_N = 20


def _stats(samples: list[float]) -> str:
    if not samples:
        return "-"
    p90 = sorted(samples)[max(0, int(len(samples) * 0.9) - 1)]
    return (f"min={min(samples):.3f} 中位={statistics.median(samples):.3f} "
            f"p90={p90:.3f} max={max(samples):.3f} ms")


def _bench(label: str, fn) -> None:
    """跑 N 次，打印值与耗时分布。"""
    samples, last = [], None
    for _ in range(_N):
        t0 = time.perf_counter()
        last = fn()
        dt = (time.perf_counter() - t0) * 1000.0
        samples.append(dt)
    print(f"  {label:<22} value={last!r:<10} {_stats(samples)}")


def probe_pmu() -> int:
    try:
        from maix.ext_dev import pmu
    except ImportError as e:
        print(f"[!] 没有 maix.ext_dev: {e}")
        return 2

    t0 = time.perf_counter()
    try:
        p = pmu.PMU()
    except Exception as e:
        print(f"[!] pmu.PMU() 构造失败（板上没有这颗 PMIC？）: {type(e).__name__}: {e}")
        return 2
    print(f"  {'pmu.PMU() 构造':<22} {'':<18} {(time.perf_counter() - t0) * 1000:.3f} ms")

    _bench("get_bat_percent()", lambda: p.get_bat_percent())
    _bench("get_bat_vol()", lambda: p.get_bat_vol())
    if hasattr(p, "is_charging"):
        _bench("is_charging()", lambda: p.is_charging())
    return 0


def probe_sysfs() -> int:
    import os
    if not os.path.isdir(_DIR):
        print(f"[!] 没有 {_DIR}（没装电池 / 内核没起驱动？）")
        return 2

    def read(name):
        with open(f"{_DIR}/{name}") as f:
            return f.read().strip()

    _bench("capacity", lambda: read("capacity"))
    _bench("voltage_now", lambda: read("voltage_now"))
    try:
        cap = int(read("capacity"))
        print(f"  放大到 0~100: {min(100, cap * 100 // 90)}%（原始 {cap}/90）")
        print(f"  status={read('status')}")
    except (OSError, ValueError) as e:
        print(f"[!] 读数解析失败: {e}")
    return 0


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(f"电量探针: {_N} 次/项，耗时含 Python 与内核开销（单位 ms）")
    if which == "pmu":
        return probe_pmu()
    if which == "sysfs":
        return probe_sysfs()
    try:
        from maix import sys as maix_sys
        device = maix_sys.device_id()
    except ImportError:
        print("[!] 没有 maix（在 PC 上？）—— 这个脚本要在设备上跑。")
        return 2
    print(f"device_id={device}")
    return probe_sysfs() if device == "maixcam2" else probe_pmu()


if __name__ == "__main__":
    sys.exit(main())
