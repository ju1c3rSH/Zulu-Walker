"""Application skeleton — the starting point for a new project.

Copy this layer and put your orchestration into app/coordinator.py (the
designated per-project "dirty code" slot). A complete worked example —
UART master-slave bridge, two-phase calibration UI, WiFi streaming, record
signaling — lives on branch ``release/2026H``.

Entry points:
  desktop/linux : ``python run.py main``  (platform from project_config.yaml)
  maixcam2      : firmware runs the deploy/maixcam/main.py shell -> here

Iron rules (violating these bricks the board or hides bugs):
1. Watchdog first: construct-and-feed before any slow step; feed between
   every step (WiFi, model load, calibration loops).
2. Anything created BEFORE Machine.create must receive platform
   capabilities via late-injection setters (see coordinator.set_sys_info),
   never through its constructor.
3. No display thread: overlay composition runs inside the vision thread;
   the main loop pumps sinks every tick
   (docs/architecture/thread_tick_topology.md).
"""

from __future__ import annotations


def _make_wdt_feed():
    """1Hz-throttled feed over the platform watchdog; noop where absent."""
    try:
        from framework.hal.platforms.maixcam2 import create_watchdog

        wdt = create_watchdog()
    except Exception:
        wdt = None
    if wdt is None:
        noop = lambda: None
        noop.disable = lambda: None
        return noop

    import time

    last = [0.0]
    fails = [0]

    def _feed():
        now = time.monotonic()
        if now - last[0] < 1.0:
            return
        last[0] = now
        try:
            wdt.feed()
        except Exception as e:
            fails[0] += 1
            print(f"[WDT] feed FAIL x{fails[0]}: {e}")

    # Countdown starts at construction; close the boot gap immediately.
    try:
        wdt.feed()
        last[0] = time.monotonic()
    except Exception:
        pass
    _feed.disable = wdt.disable
    return _feed


def _build_tick(manager, machine):
    """Per-tick callback: push the newest composed frame, then flush sinks."""
    from framework.hal.interface.sink import SinkGroup

    sinks = SinkGroup()
    try:
        from framework.hal.platforms.maixcam2.sink import MaixLcdSink

        if getattr(machine, "display", None) is not None:
            sinks.add(MaixLcdSink(machine.display))
    except ImportError:
        pass
    # TODO(project): add more sinks / input sources here (CvWindowSink,
    # JpegStreamSink, touchscreen -> CmdQueue).

    def tick() -> None:
        mod = manager.modules.get("zw_opencv_module")
        vm = getattr(mod, "get_vision_manager", lambda: None)() if mod else None
        frame = vm.get_display_frame() if vm else None
        if frame is not None:
            sinks.push(frame)
        sinks.flush()

    return tick


def main() -> None:
    import yaml

    from utils.debug_console import DebugConsole

    try:
        with open("project_config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    DebugConsole.set_global_enabled(cfg.get("debug_console_enabled", True))

    wdt_feed = _make_wdt_feed()  # rule 1
    wdt_feed()

    from framework.hal import Machine

    machine = Machine.create("project_config.yaml")
    wdt_feed()  # AI model load can take seconds

    # TODO(project): platform bring-up that needs `machine` goes here
    # (cameras for calibration, WiFi AP, beacon, ...), feeding wdt_feed()
    # after each step.

    from framework.event_bus import EventBus
    from framework.module_manager import ModuleManager

    bus = EventBus()

    from app.coordinator import AppCoordinator

    coordinator = AppCoordinator(bus)  # <-- per-project logic slot (rule 2)
    coordinator.set_sys_info(getattr(machine, "sys_info", None))

    manager = ModuleManager(
        machine,
        event_bus=bus,
        wdt_feed=wdt_feed,
        exit_check=getattr(machine, "exit_check", None),
    )
    # TODO(project): add modules per need ("zw_uart_module", "zw_wifi_stream").
    manager.register_many(["zw_opencv_module"])

    manager.run_main_loop(
        coordinator=coordinator,
        display_callback=_build_tick(manager, machine),
    )


if __name__ == "__main__":
    main()
