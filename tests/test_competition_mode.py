# -*- coding: utf-8 -*-
"""比赛模式渲染开关（VisionManager.set_render_enabled）测试。

背景（2026-10-07）: 打靶比去年 main.py 明显滞后, 定位到 HUD 绘制与 wifi JPEG
编码串在视觉线程"取帧→处理→推流"的同一拍里。`competition_mode: true` 关掉
这两项, 但**日志与协议推流必须照常** —— 这是本测试钉住的不变量。

PC 端无 maix, HUD.draw 全部静默跳过, 所以这里用 spy 对象替代真 HUD,
断言"调用次数"而非像素结果; 相机侧用一个只有 last_raw 的最小 stub。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.zw_opencv_module.vision_manager import VisionManager


class _StubCamera:
    """最小相机 stub: VisionManager._update_display_frame 只读 last_raw。"""

    def __init__(self, raw):
        self.last_raw = raw
        self.focal_length_mm = None
        self.sensor_width_mm = None
        self.sensor_height_mm = None


class _StubPipe:
    def __init__(self, raw):
        self.camera = _StubCamera(raw)
        self.last_results = {"gks_task": object()}


class _SpyHud:
    def __init__(self):
        self.draw_calls = 0

    def draw(self, img, fps, results=None):
        self.draw_calls += 1


class _StubDisplay:
    """替代 VisionManager._draw_exit_icon 依赖的 maix Image 接口。"""

    def __init__(self):
        self.icon_draws = 0

    def width(self):
        return 640

    def height(self):
        return 480


@pytest.fixture()
def vm_with_pipe():
    vm = VisionManager(camera_hub=None)
    vm._pipelines["cam_main"] = _StubPipe(raw="RAW_FRAME")
    vm.set_hud(_SpyHud())
    vm._exit_icon = object()          # 非 None -> _draw_exit_icon 才会动作
    vm._exit_icon_size = 48
    vm._exit_icon_margin = 12
    return vm


def test_render_enabled_draws_hud_and_keeps_frame(vm_with_pipe):
    """默认（台架行为）: HUD 照画, _display_frame 照常更新。"""
    vm = vm_with_pipe
    vm._update_display_frame()
    assert vm._hud.draw_calls == 1
    assert vm.get_display_frame() == "RAW_FRAME"


def test_competition_mode_skips_hud_but_keeps_frame(vm_with_pipe):
    """比赛模式: HUD 一次都不画, 但 _display_frame 必须仍更新。

    _display_frame 是显示线程上屏的唯一来源, 也是触摸退出判定读宽高的对象 ——
    跳过它会让画面黑屏、退出按钮失灵, 那是回归而不是优化。
    """
    vm = vm_with_pipe
    vm.set_render_enabled(False)
    vm._update_display_frame()
    assert vm._hud.draw_calls == 0
    assert vm.get_display_frame() == "RAW_FRAME"


def test_vision_thread_has_no_capture_sink(vm_with_pipe):
    """C4: 推流已迁出视觉线程 —— VisionManager 不再持有/调用任何 capture sink。

    ⚠ 本用例此前钉的是已被删除的旧设计（`set_capture_sink` + 每帧喂帧）。
      那个钩子正是 C4 要移除的东西（编码在视觉线程里串行, 与取帧抢同一拍）,
      所以这里改为钉新不变量: 钩子不存在, 且推流只由 presenter 的 JpegTap 发起。
    """
    vm = vm_with_pipe
    assert not hasattr(vm, "set_capture_sink"), "视觉线程不得再持有推流钩子"
    assert not hasattr(vm, "_capture_sink"), "视觉线程不得再持有推流钩子"

    # 渲染开/关都只更新 HUD 与 _display_frame, 不产生任何编码副作用
    for render in (True, False):
        vm.set_render_enabled(render)
        for _ in range(4):
            vm._update_display_frame()
        assert vm.get_display_frame() == "RAW_FRAME"


def test_render_toggle_is_reversible(vm_with_pipe):
    """开关必须可反复切 —— 台架↔比赛来回切是常态, 且不能残留上一次的相位。"""
    vm = vm_with_pipe
    for expected in (False, True, False, True):
        vm.set_render_enabled(expected)
        before = vm._hud.draw_calls
        vm._update_display_frame()
        assert vm._hud.draw_calls == before + (1 if expected else 0)


# ---------------- HUD 节流期间的"发布"策略（2026-10-09 闪烁回归）----------------

def test_hud_throttle_must_not_publish_undrawn_frames(vm_with_pipe):
    """HUD 节流跳过的帧**不得**更新 _display_frame —— 否则 show 到的大多是裸帧。

    真机现象: display_show_interval_ms=100（HUD 10Hz 画）而视觉 46Hz 发布,
    显示线程 show 到的 ~78% 没有 HUD -> "顶部调试信息时有时无"。
    不变量: 渲染开启时, _display_frame 只落在画过 HUD 的帧上。
    """
    vm = vm_with_pipe
    vm.set_hud_draw_interval(100)
    cam = vm._pipelines["cam_main"].camera

    cam.last_raw = "RAW_1"
    vm._update_display_frame()
    assert vm._hud.draw_calls == 1
    assert vm.get_display_frame() == "RAW_1"       # 画了 HUD -> 发布

    cam.last_raw = "RAW_2"                          # 下一帧, HUD 节流未到
    vm._update_display_frame()
    assert vm._hud.draw_calls == 1                  # 没画
    assert vm.get_display_frame() == "RAW_1"        # 关键: 裸帧不发布, 保持带 HUD 的帧


def test_hud_interval_zero_publishes_every_frame(vm_with_pipe):
    """interval=0（每帧都画）时发布也回到每帧 —— 等价旧行为, 不得丢帧。"""
    vm = vm_with_pipe
    vm.set_hud_draw_interval(0)
    cam = vm._pipelines["cam_main"].camera
    cam.last_raw = "RAW_1"
    vm._update_display_frame()
    cam.last_raw = "RAW_2"
    vm._update_display_frame()
    assert vm._hud.draw_calls == 2
    assert vm.get_display_frame() == "RAW_2"


def test_competition_mode_still_publishes_every_frame(vm_with_pipe):
    """比赛模式无 HUD 可画, 发布必须保持每帧（"最新原始帧"语义不变）。"""
    vm = vm_with_pipe
    vm.set_render_enabled(False)
    cam = vm._pipelines["cam_main"].camera
    cam.last_raw = "RAW_1"
    vm._update_display_frame()
    cam.last_raw = "RAW_2"
    vm._update_display_frame()
    assert vm.get_display_frame() == "RAW_2"


def test_competition_mode_does_not_stop_processing(vm_with_pipe):
    """关键不变量: 关渲染只停显示, 取帧/处理/结果产出不受影响。

    视觉线程里 process_all() 与 _update_display_frame() 是并列的两步,
    跳过后者不得让前者短路 —— 否则推流会因"没有新结果"退化成心跳,
    表现为"开了比赛模式反而更慢"的诡异现象。
    """
    vm = vm_with_pipe
    vm.set_render_enabled(False)
    pipe = vm._pipelines["cam_main"]
    pipe.last_results = {"gks_task": "NEW_RESULT"}
    vm._update_display_frame()
    # 结果仍挂在管线上, 供 coordinator.drain_results -> _latest_task 消费
    assert pipe.last_results == {"gks_task": "NEW_RESULT"}


def test_set_render_enabled_logs_state(vm_with_pipe, caplog):
    """开关切换必须留日志, 否则台架无法确认比赛模式到底生效没有。"""
    import logging
    vm = vm_with_pipe
    with caplog.at_level(logging.INFO, logger="modules.zw_opencv_module.vision_manager"):
        vm.set_render_enabled(False)
    # log_print 走自己的写盘队列, 不进 caplog; 这里只断言不抛异常,
    # 真正的台架判据是设备日志里的 [RENDER] 行（见 _apply_render_policy）。
    assert vm._render_enabled is False


def test_apply_render_policy_reads_config():
    """main._apply_render_policy: competition_mode 缺失/false -> 渲染开; true -> 关。"""
    from app.main import _apply_render_policy

    vm = VisionManager(camera_hub=None)
    _apply_render_policy(vm, {})
    assert vm._render_enabled is True

    _apply_render_policy(vm, {"competition_mode": False})
    assert vm._render_enabled is True

    _apply_render_policy(vm, {"competition_mode": True})
    assert vm._render_enabled is False
