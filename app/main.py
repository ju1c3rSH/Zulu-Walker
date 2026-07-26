import os
import sys
from utils.log_util import log_print


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from framework.hal import Machine
from framework.module_manager import ModuleManager


def _push_coordinator_status(coordinator) -> None:
    from utils.debug_console import DebugConsole
    dc = DebugConsole()
    info = coordinator.get_info()
    dc.set("state", info.get("state", "-"))
    dc.set("link_active", str(info.get("link_active", False)))
    dc.set("det_count", str(info.get("det_count", 0)))
    dc.set("fps", f"{info.get('fps', 0.0):.1f}")


def _build_display_callback(manager, machine):
    """Build display callback for the module manager."""
    def display_fn():
        vision_mod = manager.modules.get("zw_opencv_module")
        if vision_mod and machine and machine.display:
            vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
            if vm:
                frame = vm.compose_frame()
                if frame is not None:
                    if not machine.display.show(frame):
                        manager._running = False
    return display_fn


def main():
    log_print("0xfb709394")

    from framework.event_bus import EventBus
    from app.coordinator import LineFollowCoordinator
    bus = EventBus()
    coordinator = LineFollowCoordinator(bus)

    machine = Machine.create("project_config.yaml")
    manager = ModuleManager(machine, event_bus=bus)
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
