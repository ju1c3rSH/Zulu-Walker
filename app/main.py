import logging
import os
import sys
from utils.log_util import log_print


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
    dc.set("det_count", str(info.get("det_count", 0)))
    dc.set("fps", f"{info.get('fps', 0.0):.1f}")


def _build_display_callback(manager, machine):
    _last_seen_frame = None

    _icon_exit = None
    _touch = None
    try:
        from maix import image as _mi
        from maix import touchscreen
        _touch = touchscreen.TouchScreen()
        _icon_exit = _mi.load("/maixapp/share/icon/ret.png")
        if _icon_exit is not None:
            w = _icon_exit.width() * 48 // _icon_exit.height()
            if w % 2:
                w += 1
            _icon_exit = _icon_exit.resize(w, 48)
    except Exception:
        pass

    _ICON_SIZE = 48
    _MARGIN = 12
    _touch_down = False
    _touch_x = _touch_y = 0

    def _in_btn(x, y, bx, by):
        return bx - 4 <= x <= bx + _ICON_SIZE + 4 and by - 4 <= y <= by + _ICON_SIZE + 4

    def display_fn():
        nonlocal _last_seen_frame, _touch_down, _touch_x, _touch_y

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
                                    bx = _MARGIN
                                    if _in_btn(_touch_x, _touch_y, bx, by):
                                        import os
                                        os._exit(0)
                    _touch_down = False
            except Exception:
                pass

        vision_mod = manager.modules.get("zw_opencv_module")
        if vision_mod and machine and machine.display:
            vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
            if vm:
                frame = vm.get_display_frame()
                if frame is not None and frame is not _last_seen_frame:
                    _last_seen_frame = frame
                    try:
                        import maix.image as _mi3
                        h = frame.height()
                        by = h - _ICON_SIZE - 8
                        bx = _MARGIN
                        if _icon_exit is not None:
                            frame.draw_rect(bx - 2, by - 2, _ICON_SIZE + 4, _ICON_SIZE + 4, color=_mi3.COLOR_BLACK, thickness=-1)
                            frame.draw_image(bx, by, _icon_exit)
                    except Exception:
                        pass
                    if _check_cmm_pressure():
                        return
                    try:
                        machine.display.show(frame)
                    except Exception:
                        pass
    return display_fn


def _make_wdt_feed():
    try:
        from maix.peripheral import wdt as _mwdt
        _w = _mwdt.WDT(0, 5000)
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

    wdt_feed = _make_wdt_feed()

    from framework.event_bus import EventBus
    from app.coordinator import Ti2026Coordinator
    bus = EventBus()
    coordinator = Ti2026Coordinator(bus)
    coordinator.set_wdt_feed(wdt_feed)

    machine = Machine.create("project_config.yaml")
    manager = ModuleManager(machine, event_bus=bus, wdt_feed=wdt_feed)
    manager.register_many(["zw_opencv_module", "zw_uart_module"])

    from modules.zw_opencv_module import get_vision_manager
    from modules.zw_uart_module import get_interface

    vm = get_vision_manager()
    if vm:
        coordinator.connect_vision(vm)

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.set_ai(machine.ai)

    display_callback = _build_display_callback(manager, machine)

    coordinator.start()
    manager.run_main_loop(coordinator, tick_callback=_push_coordinator_status, display_callback=display_callback)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
