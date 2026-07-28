import logging
import os
import sys
from utils.log_util import log_print


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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

    def display_fn():
        nonlocal _last_seen_frame
        vision_mod = manager.modules.get("zw_opencv_module")
        if vision_mod and machine and machine.display:
            vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
            if vm:
                frame = vm.get_display_frame()
                if frame is not None and frame is not _last_seen_frame:
                    _last_seen_frame = frame
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
        _w = _mwdt.WDT(timeout=3000)

        def _feed():
            try:
                _w.feed()
            except Exception:
                pass
        return _feed
    except Exception:
        return lambda: None


def main():
    log_print("0xfb709394")

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
