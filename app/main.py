import logging
import os
import sys
import threading
import time
from utils.log_util import log_print, start_log_writer

_DISPLAY_EVERY_N = 2
_ICON_SIZE = 48
_ICON_MARGIN = 12

_KEY_HOLD = {"obj": None}  # Key 实例必须持引用: 回调线程挂在其上, 被 GC 后回调失效


class AppPageState:
    """页面状态单一事实来源，决定输入/显示路由。

    消费方: 按键切页(写) / tick 渲染 / 显示线程 / main_callback 触摸处理(读)。
    非主页面时 main_callback 短路: 调参页有自己的触摸消费渠道（tuner.step）,
    这里不做坐标映射也不做退出图标判定 — 否则底部左列滑条热区会与
    退出图标重叠, 误触直接 os._exit 且阈值未保存。
    线程模型: 名字字符串单字段读写（key 回调线程写, 主循环/显示线程读）。
    """
    MAIN = "main"
    TUNER = "tuner"

    def __init__(self) -> None:
        self.name = self.MAIN

    def is_main(self) -> bool:
        return self.name == self.MAIN


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    filename="logs/app.log",
    filemode="a",
)

from framework.hal import Machine
from framework.module_manager import ModuleManager


try:
    from maix import sys as maix_sys
    _HAVE_MAIX_SYS = True
except ImportError:
    maix_sys = None
    _HAVE_MAIX_SYS = False


def _check_cmm_pressure(skip_threshold: float = 0.80) -> bool:
    if not _HAVE_MAIX_SYS:
        return False
    try:
        info = maix_sys.memory_info()
        cmm_used = info.get("cmm_used", 0)
        cmm_total = info.get("cmm_total", 256 * 1024 * 1024)
        return cmm_used > int(cmm_total * skip_threshold)
    except Exception:
        return False


def _push_coordinator_status(coordinator) -> None:
    from utils.debug_console import DebugConsole
    dc = DebugConsole()
    info = coordinator.get_info()
    dc.set("state", info.get("state", "-"))
    dc.set("link_active", str(info.get("link_active", False)))
    dc.set("codes", str(info.get("codes", [0, 0, 0])))
    dc.set("pending", str(info.get("pending_count", 0)))
    dc.set("fps", f"{info.get('fps', 0.0):.1f}")


def _make_key_switch(vm, pages) -> None:
    """KEY_OK：主页面 ↔ 阈值调参页切换。

    注册自定义 key 回调后, MaixPy 默认"OK 键 = 退出程序"行为自动失效
    （官方 peripheral/key.md）。回调在独立线程执行, 这里只做置位/暂停。
    进调参页时暂停视觉采集（相机让位为唯一读取者）,
    结果 >1s 陈旧由 coordinator 归零心跳自动兜底、UART 不断流。
    """
    def on_key(key_id, state) -> None:
        prev = _KEY_HOLD.get("last_state")
        _KEY_HOLD["last_state"] = state
        try:
            log_print(f"[KEY] event id={key_id} state={state}")
        except Exception:
            pass
        if state == prev:
            return
        # 切沿按事件语义判定（str 化, 不依赖枚举对象）: 按下沿不切换,
        # 释放沿/未知状态才切换 — 防止 press+release 双翻转"闪进闪出"。
        # 运行久了事件投递间隔会拉大, 防抖窗口只作连发兜底, 不作主判据。
        s = str(state).upper()
        if "PRESS" in s:
            edge = "press"
        elif "RELEASE" in s:
            edge = "release"
        else:
            edge = "unknown"
        if edge == "press":
            return
        import time as _time
        now = _time.monotonic()
        if now - _KEY_HOLD.get("last_toggle", 0.0) < 0.5:
            return
        try:
            _KEY_HOLD["last_toggle"] = now
            if pages.is_main():
                if vm.get_pipeline_camera() is None:
                    log_print("[KEY] no pipeline camera, enter tuner refused")
                    return
                vm.set_paused(True)
                pages.name = pages.TUNER
                log_print("[KEY] toggle: edge=release, page -> tuner")
            else:
                pages.name = pages.MAIN
                vm.set_paused(False)
                log_print(f"[KEY] toggle: edge={edge}, page -> main, vision resume")
        except Exception as e:
            log_print(f"[KEY] toggle fail: {e}")

    try:
        from maix import key as _mkey
        # Key 实例必须保存引用：回调线程挂在其上, 被 GC 后按键即为无响应
        _KEY_HOLD["obj"] = _mkey.Key(on_key)
        log_print("[KEY] KEY_OK toggles tuner page (default exit-app disabled)")
    except Exception as e:
        log_print(f"[KEY] init fail: {e}")


def _make_tick(machine, vm, pages) -> None:
    """主循环 tick：主页面推 coordinator 状态; 调参页渲染调参器单帧。"""
    from modules.zw_opencv_module.debug.threshold_tuner import GksThresholdTuner

    _tick_state = {"tuner": None, "ts": None}

    def _tick(coordinator) -> None:
        if pages.name == pages.TUNER:
            tuner = _tick_state["tuner"]
            ts = _tick_state["ts"]
            if tuner is None:
                from modules.zw_opencv_module.debug.threshold_tuner import (
                    GksThresholdTuner as _T,
                )
                tuner = _T()
                tuner.begin()
                _tick_state["tuner"] = tuner
            if ts is None:
                try:
                    from maix import touchscreen
                    ts = touchscreen.TouchScreen()
                    _tick_state["ts"] = ts
                except Exception:
                    ts = None
                    _tick_state["ts"] = None
            cam = vm.get_pipeline_camera()
            if cam is not None and ts is not None:
                frame_bgr = cam.read()
                if frame_bgr is not None:
                    tuner.step(machine.display, ts, frame_bgr)
                    if tuner.exit_requested():
                        pages.name = pages.MAIN
                        vm.set_paused(False)
                        log_print("[KEY] tuner Exit -> main page, vision resume")
        else:
            _push_coordinator_status(coordinator)

    return _tick


def _load_exit_icon():
    try:
        from maix import image as _mi
        icon = _mi.load("/maixapp/share/icon/ret.png")
        if icon is not None:
            w = icon.width() * 48 // icon.height()
            if w % 2:
                w += 1
            return icon.resize(w, 48)
    except Exception:
        return None


def _build_callbacks(manager, machine, pages):
    """Returns (main_callback, start_display_thread).

    main_callback: touch/input handling (~0ms), runs in main loop.
    display thread: display.show() only, runs independently.
    exit icon is drawn by VisionManager (via set_exit_icon).
    """
    _last_seen_frame = None

    _touch = None
    try:
        from maix import touchscreen
        _touch = touchscreen.TouchScreen()
    except Exception:
        pass

    _touch_down = False
    _touch_x = _touch_y = 0

    def _in_btn(x, y, bx, by):
        return bx - 4 <= x <= bx + _ICON_SIZE + 4 and by - 4 <= y <= by + _ICON_SIZE + 4

    def main_callback():
        """Touch handling only — fast (~0ms), no JPEG encoding, no GIL hog.

        页面路由: 调参页整体短路 — 该页触摸由 tuner.step 自己消费,
        这里既不读触摸屏（避免多读者状态互相干扰）也不做
        退出图标判定（旧帧坐标映射失义 + 滑条热区重叠 os._exit 误触）。
        """
        nonlocal _touch_down, _touch_x, _touch_y

        if not pages.is_main():
            return

        if _touch is not None:
            try:
                x, y, pressed = _touch.read()
                if pressed:
                    _touch_down = True
                    try:
                        vision_mod = manager.modules.get("zw_opencv_module")
                        if vision_mod:
                            vm2 = getattr(vision_mod, "get_vision_manager", lambda: None)()
                            if vm2:
                                frame2 = vm2.get_display_frame()
                                if frame2 is not None and machine is not None:
                                    import maix.image as _mi2
                                    pt = _mi2.resize_map_pos_reverse(
                                        frame2.width(), frame2.height(),
                                        machine.display.width(), machine.display.height(),
                                        _mi2.Fit.FIT_CONTAIN, x, y,
                                    )
                                    _touch_x, _touch_y = int(pt[0]), int(pt[1])
                                else:
                                    _touch_x, _touch_y = x, y
                            else:
                                _touch_x, _touch_y = x, y
                        else:
                            _touch_x, _touch_y = x, y
                    except Exception:
                        _touch_x, _touch_y = x, y
                else:
                    if _touch_down:
                        vision_mod = manager.modules.get("zw_opencv_module")
                        if vision_mod:
                            vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
                            if vm:
                                frame = vm.get_display_frame()
                                if frame is not None:
                                    h = frame.height()
                                    by = h - _ICON_SIZE - 8
                                    bx = _ICON_MARGIN
                                    if _in_btn(_touch_x, _touch_y, bx, by):
                                        import os
                                        os._exit(0)
                    _touch_down = False
            except Exception:
                pass

    def _display_loop():
        nonlocal _last_seen_frame
        _tick = 0
        while True:
            if pages.name == pages.TUNER:
                time.sleep(0.01)
                continue
            vision_mod = manager.modules.get("zw_opencv_module")
            if vision_mod and machine and machine.display:
                vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
                if vm:
                    frame = vm.get_display_frame()
                    if frame is not None and frame is not _last_seen_frame:
                        _last_seen_frame = frame
                        _tick += 1
                        if _tick % _DISPLAY_EVERY_N != 0:
                            continue
                        if _check_cmm_pressure():
                            continue
                        try:
                            machine.display.show(frame)
                        except Exception:
                            pass
            time.sleep(0.001)

    def start_display_thread():
        t = threading.Thread(target=_display_loop, daemon=True)
        t.start()
        return t

    return main_callback, start_display_thread


def _make_wdt_feed():
    try:
        from maix.peripheral import wdt as _mwdt
        _w = _mwdt.WDT(0, 10000)
        import time
        _last = [0.0]
        _fail_count = [0]

        def _feed():
            now = time.monotonic()
            if now - _last[0] < 1.0:
                return
            _last[0] = now
            try:
                _w.feed()
            except Exception as e:
                _fail_count[0] += 1
                if _fail_count[0] % 50 == 1:
                    log_print(f"[WDT] feed FAIL x{_fail_count[0]}: {e}")
        return _feed
    except Exception as e:
        log_print(f"[WDT] init FAIL: {e}")
        return lambda: None


def _init_streamer(vm) -> None:
    from modules.zw_wifi_stream import get_streamer
    s = get_streamer()
    if s is not None and vm is not None:
        vm.set_capture_sink(s.push_frame)
        s.start_async()


def main():
    log_print("0xfb709394")

    import yaml
    try:
        with open("project_config.yaml") as f:
            _cfg = yaml.safe_load(f) or {}
    except Exception:
        _cfg = {}
    from utils.debug_console import DebugConsole
    DebugConsole.set_global_enabled(_cfg.get("debug_console_enabled", True))
    start_log_writer()

    wdt_feed = _make_wdt_feed()

    from framework.event_bus import EventBus
    from app.coordinator import GksCoordinator
    from modules.zw_opencv_module.display import DisplayHUD
    bus = EventBus()
    coordinator = GksCoordinator(bus)
    coordinator.set_wdt_feed(wdt_feed)

    machine = Machine.create("project_config.yaml")

    try:
        from maix import display
        display.set_trans_image_quality(10)
    except Exception:
        pass

    manager = ModuleManager(machine, event_bus=bus, wdt_feed=wdt_feed)
    manager.register_many(["zw_opencv_module", "zw_uart_module", "zw_wifi_stream"])

    from modules.zw_opencv_module import get_vision_manager
    from modules.zw_uart_module import get_interface

    vm = get_vision_manager()
    if vm:
        coordinator.connect_vision(vm)
        vm.set_hud(DisplayHUD(status_fn=coordinator.get_info))

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.set_ai(machine.ai)

    _init_streamer(vm)

    exit_icon = _load_exit_icon()
    if exit_icon is not None and vm is not None:
        vm.set_exit_icon(exit_icon, _ICON_SIZE, _ICON_MARGIN)

    cam_cfg = _cfg.get("cameras", [{}])[0]
    cam_w = cam_cfg.get("width", 640)
    cam_h = cam_cfg.get("height", 480)
    log_print(f"cam w:{cam_w} h:{cam_h}")
    coordinator.set_frame_size(cam_w, cam_h)

    pages = AppPageState()
    if vm:
        _make_key_switch(vm, pages)

    main_callback, start_display_thread = _build_callbacks(manager, machine, pages)

    coordinator.start()
    start_display_thread()
    manager.run_main_loop(coordinator, tick_callback=_make_tick(machine, vm, pages), display_callback=main_callback)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
