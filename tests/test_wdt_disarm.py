# -*- coding: utf-8 -*-
"""退出路径与硬件看门狗 —— 钉住"退出不再莫名重启"这件事。

## 现象（2026-10-09 台架）

MaixCAM-Pro 上**运行期一切正常**（帧率/推流/协议全绿），但**只要一退出**
（MaixVision 停止键 / 屏幕退出图标 / Ctrl+C），设备就在约 10 秒后自己重启。
日志最后一行却是干净的 `program exited` —— 看不出任何异常。

## 根因（MaixCDK 源码实锤，`port/linux_common/maix_wdt.cpp`）

    WDT::WDT(id, feed_ms) → open("/dev/watchdog") + WDIOC_SETTIMEOUT + close
                            ⇒ **构造即启动**，硬件倒计时立刻开始
    WDT::feed()           → open + WDIOC_KEEPALIVE + close（喂一次 = 重置倒计时）
    WDT::stop()           → log::warn("this operation is not supported"); return 0;
    WDT::restart()        → 同上，也是**空实现**

所以：

  · `wdt.stop()` 返回 0（=ERR_NONE）却**什么都没做**，调用方无法从返回值分辨；
  · Linux watchdog 在 close 时若不写魔法字符 'V' 会保持运行（内核还会 print
    `watchdog0: watchdog did not stop!`）⇒ **进程退出不会让它停**；
  · 喂狗只来自三个 Python 线程（主循环/协调器/视觉），退出时全部消失
    ⇒ 窗口（本项目 10s）到点 ⇒ SoC 硬复位。

第一版修复只加了 `wdt.stop()` + 一句 "stopped" 日志，台架反馈**照样重启** ——
就是被这个空实现骗了。因此现在的核心不变量是：

  1. **`wdt_enabled: false` 时绝不构造 WDT**（构造了就关不掉了！顺序有语义）；
  2. `_wdt_stop()` **不相信** `stop()` 的返回值，必须用 sysfs `timeleft` 递减验证；
  3. 验证出"还在跑" ⇒ 走 raw `WDIOS_DISABLECARD`，再验证一次；
  4. 仍然在跑 ⇒ 日志必须**明确告警**并给出规避办法，绝不输出"已解除"这种假结论；
  5. 退出图标路径（`os._exit(0)`，跳过 finally）必须先 `_wdt_stop()`。
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.main as M


class FakeWDT:
    """`stop()` 忠实模仿 MaixCDK: 返回 0 但什么都不做。"""

    def __init__(self, id_=0, feed_ms=0):
        self.id = id_
        self.feed_ms = feed_ms
        self.feeds = 0
        self.stops = 0

    def feed(self):
        self.feeds += 1

    def stop(self):
        self.stops += 1
        return 0          # 永远是 0 —— 正是它骗了第一版修复


@pytest.fixture()
def wdt_state():
    """接管模块级 `_WDT_SHARED_STATE`（测试后原样恢复）。"""
    sentinel = object()
    saved = M.__dict__.get("_WDT_SHARED_STATE", sentinel)
    yield
    if saved is sentinel:
        M.__dict__.pop("_WDT_SHARED_STATE", None)
    else:
        M.__dict__["_WDT_SHARED_STATE"] = saved


@pytest.fixture()
def fake_maix_wdt(monkeypatch):
    """给 `from maix import wdt` 打假模块（PC 端没有 maix）。"""
    created = {}

    class _Mod:
        @staticmethod
        def WDT(id_, feed_ms):
            w = FakeWDT(id_, feed_ms)
            created["wdt"] = w
            return w

    pkg = types.ModuleType("maix")
    pkg.wdt = _Mod
    monkeypatch.setitem(sys.modules, "maix", pkg)
    monkeypatch.setitem(sys.modules, "maix.wdt", _Mod)
    return created


# ---------------- 最关键的一条：不武装，就永远不会被它重启 ----------------


def test_disabled_never_constructs_wdt(monkeypatch, wdt_state):
    """`wdt_enabled: false` ⇒ **绝不构造 WDT**。

    这是本 bug 的根治手段。必须钉在"构造之前就返回"—— 一旦 `maix_wdt.WDT(...)`
    被构造出来，倒计时就开始跑，而 `stop()` 是空实现 ⇒ 之后没有任何用户态办法
    叫停它。所以顺序错一步，开关就失效了。
    """
    constructed = []

    class _Mod:
        @staticmethod
        def WDT(id_, feed_ms):
            constructed.append((id_, feed_ms))
            return FakeWDT(id_, feed_ms)

    pkg = types.ModuleType("maix")
    pkg.wdt = _Mod
    monkeypatch.setitem(sys.modules, "maix", pkg)
    monkeypatch.setitem(sys.modules, "maix.wdt", _Mod)
    M.__dict__.pop("_WDT_SHARED_STATE", None)

    feed = M._make_wdt_feed(enabled=False)
    assert constructed == [], "禁用时构造 WDT 就等于把自己锁进'退出必重启'"
    assert "_WDT_SHARED_STATE" not in M.__dict__, "禁用时不该留下共享状态"
    feed()          # 也要是个安全的空操作
    feed()


def test_default_is_disabled():
    """CAM1 默认 **不武装** —— 默认值是安全开关, 改动要连注释与清单一起改。"""
    assert M._WDT_ENABLED_DEFAULT is False


def test_enabled_still_arms_documented_window(fake_maix_wdt, wdt_state):
    """`wdt_enabled: true` 时才武装, 且窗口/ID 与文档一致。"""
    M.__dict__.pop("_WDT_SHARED_STATE", None)
    feed = M._make_wdt_feed(enabled=True)
    w = fake_maix_wdt["wdt"]
    assert (w.id, w.feed_ms) == (0, M._WDT_WINDOW_MS)
    assert M._WDT_WINDOW_MS == 10000
    M._WDT_SHARED_STATE["last"] = 0.0
    feed()
    assert w.feeds == 1


# ---------------- disarm 必须"验证", 不能相信 stop() 的返回值 ----------------


def test_stop_is_not_trusted_and_warns_when_still_armed(monkeypatch, wdt_state):
    """stop() 返回 0 但倒计时仍在走 ⇒ 必须**告警**（不能再说"已解除"）。

    这是台架第二次反馈的核心: 日志打了 "stopped" 却照样重启。所以断言里
    既要有告警关键词, 也要**明确禁止**出现"已解除"这种会被当成结论的说法。
    """
    w = FakeWDT()
    M._WDT_SHARED_STATE = {"last": 0.0, "fails": 0, "wdt": w, "stopped": False}

    monkeypatch.setattr(M, "_wdt_armed", lambda *a, **k: True)     # 永远"还在跑"
    monkeypatch.setattr(M, "_wdt_disarm_raw",
                        lambda: "DISABLECARD 失败 errno=16 EBUSY")

    logs = []
    monkeypatch.setattr(M, "log_print", lambda msg: logs.append(msg))

    M._wdt_stop()

    assert w.stops == 1, "仍应尝试调用 maix 的 stop()"
    blob = "\n".join(logs)
    assert "disarm 失败" in blob
    assert "重启" in blob and str(M._WDT_WINDOW_MS // 1000) in blob
    assert "wdt_enabled: false" in blob, "告警必须给出规避办法"
    assert "已解除" not in blob, "不能再输出假的成功结论"


def test_success_is_verified_via_timeleft(monkeypatch, wdt_state):
    """能验证到"倒计时已停"时才允许报成功。"""
    w = FakeWDT()
    M._WDT_SHARED_STATE = {"last": 0.0, "fails": 0, "wdt": w, "stopped": False}
    monkeypatch.setattr(M, "_wdt_armed", lambda *a, **k: False)     # 已验证停下

    logs = []
    monkeypatch.setattr(M, "log_print", lambda msg: logs.append(msg))
    M._wdt_stop()
    assert "已验证" in "\n".join(logs)


def test_raw_disable_attempted_when_stop_insufficient(monkeypatch, wdt_state):
    """maix stop() 不够时, 必须尝试绕过 MaixPy 的 DISABLECARD。"""
    w = FakeWDT()
    M._WDT_SHARED_STATE = {"last": 0.0, "fails": 0, "wdt": w, "stopped": False}

    seq = iter([True, False])          # 第一次采样: 还在跑; raw 之后: 停了
    monkeypatch.setattr(M, "_wdt_armed", lambda *a, **k: next(seq))
    calls = {"raw": 0}

    def _raw():
        calls["raw"] += 1
        return "DISABLECARD OK"

    monkeypatch.setattr(M, "_wdt_disarm_raw", _raw)
    logs = []
    monkeypatch.setattr(M, "log_print", lambda msg: logs.append(msg))

    M._wdt_stop()
    assert calls["raw"] == 1, "stop() 无效时必须走 raw ioctl"
    assert "真正关掉了" in "\n".join(logs)


def test_stop_is_idempotent(monkeypatch, wdt_state):
    """多条退出路径会重复调用, 重复 stop 必须无害。"""
    w = FakeWDT()
    M._WDT_SHARED_STATE = {"last": 0.0, "fails": 0, "wdt": w, "stopped": False}
    monkeypatch.setattr(M, "_wdt_armed", lambda *a, **k: False)
    M._wdt_stop()
    M._wdt_stop()
    assert w.stops == 1


def test_feed_disabled_after_stop(fake_maix_wdt, wdt_state, monkeypatch):
    """disarm 之后一次都不该再 feed（避免把停掉的狗重新武装）。"""
    M.__dict__.pop("_WDT_SHARED_STATE", None)
    feed = M._make_wdt_feed(enabled=True)
    w = fake_maix_wdt["wdt"]

    monkeypatch.setattr(M, "_wdt_armed", lambda *a, **k: False)
    M._WDT_SHARED_STATE["last"] = 0.0
    feed()
    assert w.feeds == 1

    M._wdt_stop()
    M._WDT_SHARED_STATE["last"] = 0.0
    feed()
    feed()
    assert w.feeds == 1, "disarm 后任何入口都不该再 feed"


def test_stop_without_wdt_is_silent(wdt_state):
    """PC 端 / 未武装 / 未初始化: `_wdt_stop()` 不得抛。"""
    M._WDT_SHARED_STATE = None
    M._wdt_stop()
    M.__dict__.pop("_WDT_SHARED_STATE", None)
    M._wdt_stop()
    M._WDT_SHARED_STATE = {"last": 0.0, "fails": 0, "wdt": None, "stopped": False}
    M._wdt_stop()


# ---------------- 退出图标路径（os._exit 前的最后一道） ----------------


class _StubDisplay:
    def width(self):
        return 640

    def height(self):
        return 480

    def show(self, _frame):
        return True


class _StubMachine:
    def __init__(self):
        self.display = _StubDisplay()


class _StubFrame:
    def width(self):
        return 640

    def height(self):
        return 480


class _StubVM:
    def get_display_frame(self):
        return _StubFrame()


class _Touch:
    def __init__(self, pressed):
        self._pressed = pressed

    def read(self):
        # 退出图标在左下角（见 _in_exit_button: bx=12, by=frame.h-56）
        return 20, 440, self._pressed


def test_exit_icon_disarms_watchdog_before_os_exit(monkeypatch, wdt_state):
    """回归: 触摸退出图标 → 必须先关激光/_wdt_stop() 再 `os._exit(0)`。

    `os._exit(0)` 不走 finally/atexit, 这是唯一能挡住"按退出=设备重启"的地方,
    也是唯一能挡住"退出后激光常亮"的地方。
    """
    order = []
    monkeypatch.setattr(M, "_wdt_stop", lambda: order.append("wdt_stop"))
    exited = []
    monkeypatch.setattr(os, "_exit", lambda code: exited.append(code))

    class _StubFactory:
        def send_laser_off(self):
            order.append("laser_off")

    ui = M.MainUiShell(machine=_StubMachine(), pages=None,
                       factory=_StubFactory(), vision_manager=_StubVM())
    ui._touch = _Touch(pressed=True)
    ui._consume_main_touch()
    ui._touch = _Touch(pressed=False)
    ui._consume_main_touch()

    assert exited == [0]
    assert order == ["laser_off", "wdt_stop"], \
        "os._exit(0) 之前必须先关激光再 disarm"


# ---------------- 配置接线 ----------------


def test_config_key_selects_arming(monkeypatch):
    """`main()` 的接线: project_config.yaml 的 `wdt_enabled` 决定是否武装。"""
    seen = {}

    def _fake_make(enabled=True):
        seen["enabled"] = enabled
        return lambda: None

    monkeypatch.setattr(M, "_make_wdt_feed", _fake_make)

    for cfg, expect in (({"wdt_enabled": True}, True),
                        ({"wdt_enabled": False}, False),
                        ({}, False)):
        monkeypatch.setattr(M, "_project_config", lambda c=cfg: c)
        M._make_wdt_feed(bool(M._project_config().get(
            "wdt_enabled", M._WDT_ENABLED_DEFAULT)))
        assert seen["enabled"] is expect, f"cfg={cfg} 应解析为 {expect}"
