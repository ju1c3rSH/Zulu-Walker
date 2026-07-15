import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from hal import Machine
from utils.module_manager import ModuleManager


def _push_coordinator_status(coordinator) -> None:
    from utils.debug_console import DebugConsole
    dc = DebugConsole()
    info = coordinator.get_info()
    sm_info = info.get("mission", {})
    dc.set("mission_state", sm_info.get("state", "-"))
    dc.set("visual_state", info.get("visual_state", "-"))
    dc.set("link_active", str(coordinator.is_link_active()))
    dc.set("active_task", info.get("active_task", "-"))
    dc.set("cargo_count", str(sm_info.get("cargo_count", "-")))
    dc.set("batch", str(sm_info.get("batch", "-")))
    dc.set("step", str(sm_info.get("step", "-")))
    dc.set("target_color", sm_info.get("target_color", "-"))
    dc.set("batch1_order", ",".join(sm_info.get("first_batch_order", [])) or "-")
    dc.set("batch2_order", ",".join(sm_info.get("second_batch_order", [])) or "-")


def main():
    print("0xfb709394")

    from context import EventBus, MissionCoordinator
    bus = EventBus()
    coordinator = MissionCoordinator(bus)

    machine = Machine.create("project_config.yaml")
    manager = ModuleManager(machine, event_bus=bus)
    manager.start_all()

    from modules.zw_opencv_module import get_vision_manager
    from modules.zw_uart_module import get_interface

    vm = get_vision_manager()
    if vm:
        coordinator.connect_vision(vm)

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.start()
    manager.run_main_loop(coordinator, tick_callback=_push_coordinator_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
