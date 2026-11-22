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


def _make_wdt_feed(platform: str = "maixcam2"):
    """1Hz-throttled feed over the platform watchdog; noop where absent.

    The watchdog capability is probed on the *active* platform module.
    A hardcoded maixcam2 import burns an exception on every boot for boards
    whose firmware has no ``nn.YOLO11`` (MaixCAM-Pro) -- and whose watchdog
    is a firmware stub anyway.
    """
    try:
        import importlib

        platform_mod = importlib.import_module(f"framework.hal.platforms.{platform}")
        create_watchdog = getattr(platform_mod, "create_watchdog", None)
        wdt = create_watchdog() if callable(create_watchdog) else None
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


def _apply_observability(cfg: dict) -> None:
    """Apply ``observability:``/``perf_log:`` config to the log gates.

    Args:
        cfg: Parsed project config; missing keys keep current gate values.

    Context:
        Call once at startup, before modules emit perf lines.
    """
    from utils import log_util

    obs = cfg.get("observability") or {}
    perf = cfg.get("perf_log")
    if isinstance(perf, bool):
        log_util.set_perf_log(enabled=perf)
    if isinstance(obs, dict):
        log_util.set_perf_gate(
            perf=obs.get("perf"),
            detail=obs.get("perf_detail"),
        )
        log_util.set_perf_interval(obs.get("perf_interval_s"))
    log_util.log_print(
        f"[PERFLOG] perf={'ON' if log_util.perf_enabled() else 'OFF'} "
        f"detail={'ON' if log_util.perf_detail_enabled() else 'OFF'} "
        f"window={log_util.perf_interval_s():.1f}s"
    )


def _vision_manager_of(manager):
    """Return the vision manager exposed by the vision module, or None."""
    mod = manager.modules.get("zw_opencv_module")
    if mod is None:
        return None
    getter = getattr(mod, "get_vision_manager", None)
    return getter() if callable(getter) else None


def _build_tick(manager, machine, cfg: dict = None):
    """Per-tick callback: push the newest composed frame, then flush sinks.

    Args:
        manager: ModuleManager owning the vision module.
        machine: Assembled Machine (``display`` may be None when headless).
        cfg: Parsed project config. ``display_show_interval_ms`` maps to the
            LCD present interval; absent or 0 means "no throttle".

    Context:
        Both the present cadence and the overlay rebuild window read the same
        key, so a slow panel draws exactly as often as it shows. With
        ``display_enabled: false`` the vision side stops composing frames at
        all — nothing would consume them.
    """
    from framework.hal.interface.sink import SinkGroup
    from framework.hal.sink import MaixLcdSink

    cfg = cfg or {}
    interval_ms = cfg.get("display_show_interval_ms")
    sinks = SinkGroup()
    if getattr(machine, "display", None) is not None:
        sinks.add(MaixLcdSink(machine.display, interval_ms=interval_ms))
    # TODO(project): add more sinks / input sources here (CvWindowSink,
    # JpegStreamSink, touchscreen -> CmdQueue).

    vm = _vision_manager_of(manager)
    if vm is not None:
        vm.set_render_enabled(cfg.get("display_enabled", True))
        vm.set_display_cadence_ms(interval_ms)

    def tick() -> None:
        live_vm = vm or _vision_manager_of(manager)
        frame = live_vm.get_display_frame() if live_vm else None
        if frame is not None:
            sinks.push(frame)
        sinks.flush()

    return tick


def main() -> None:
    import os

    import yaml

    from utils.debug_console import DebugConsole

    # 平台配置路径: 默认读工作区的 project_config.yaml; ZWULU_CONFIG 可指向
    # 其它配置 (如 configs/project_config.maixcam1.yaml), 避免为了切平台去
    # 覆盖工作区文件。
    config_path = os.environ.get("ZWULU_CONFIG", "project_config.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    DebugConsole.set_global_enabled(cfg.get("debug_console_enabled", True))
    _apply_observability(cfg)

    wdt_feed = _make_wdt_feed(cfg.get("platform", "maixcam2"))  # rule 1
    wdt_feed()

    from framework.hal import Machine

    machine = Machine.create(config_path)
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
        display_callback=_build_tick(manager, machine, cfg),
    )


if __name__ == "__main__":
    main()
