"""Application skeleton and entry point for a new project.

Copy this layer and place orchestration in ``app.coordinator``. Entry points:
desktop/linux via ``python run.py main`` (platform from project_config.yaml);
MaixCAM2 firmware via the ``deploy/maixcam/main.py`` shell.

Startup rules:
1. Construct and feed the watchdog before any slow step; feed between steps.
2. Inject platform capabilities into objects created before ``Machine.create``
   through setters, never their constructors.
3. Overlay composition runs in the vision thread, not a display thread; the
   main loop pumps sinks each tick.
"""

from __future__ import annotations


def _make_wdt_feed(platform: str = "maixcam2"):
    """Build a 1 Hz-throttled watchdog feed, or a no-op where unsupported.

    Probe the watchdog on the active platform module: a hardcoded ``maixcam2``
    import raises on every boot for boards without ``nn.YOLO11`` (MaixCAM-Pro),
    whose watchdog is a firmware stub regardless.
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
    """Apply ``observability`` / ``perf_log`` config to the log gates.

    Args:
        cfg: Parsed project config; missing keys leave the current gates unchanged.

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


def _attr(obj, name, default=""):
    """Read an attribute that may be a property or a method."""
    if obj is None:
        return default
    value = getattr(obj, name, default)
    if callable(value):
        value = value()
    return value


def _panel_of(machine, cfg: dict):
    """The desktop panel, when the platform offers one.

    The device has no panel window - it shows the same HUD on its LCD - so
    the capability hook is absent there and the LCD path stays untouched.
    """
    import importlib

    from utils.log_util import log_print

    platform = getattr(machine, "platform", "maixcam2")
    try:
        platform_mod = importlib.import_module(
            f"framework.hal.platforms.{platform}")
    except Exception:
        return None
    factory = getattr(platform_mod, "create_panel", None)
    if not callable(factory):
        return None
    disp = cfg.get("display") or {}
    if not isinstance(disp, dict):
        disp = {}
    try:
        panel = factory(
            width=int(disp.get("width", 640)),
            height=int(disp.get("height", 480)),
            scale=int(disp.get("scale", 2)),
            title=str(disp.get("title", "Zulu-Walker")),
            headless=bool(disp.get("headless", False)),
            interval_ms=disp.get("interval_ms"),
        )
    except Exception as exc:
        log_print(f"[Panel] desktop panel creation failed: {exc}")
        return None
    if panel is not None:
        size = panel.size
        log_print(f"[Panel] desktop panel {size[0]}x{size[1]} ready")
    else:
        log_print("[Panel] no desktop window available (headless host)")
    return panel


def _camera_frame(machine, vm, pipeline_id: str):
    """The newest camera frame, without touching the inference queue.

    The camera is asked for through its pipeline: a pipeline id is not a
    camera id (the vision module names them independently, "cam_main" versus
    "main"), so the hub lookup would miss and the panel would show black.

    The device camera keeps a native image buffer; the desktop camera keeps
    the frame it just captured.  Either way the display reads it and the
    pipeline keeps its own copy, so a panel never starves the detector.
    """
    getter = getattr(vm, "get_pipeline", None) if vm is not None else None
    pipeline = getter(pipeline_id) if callable(getter) else None
    camera = getattr(pipeline, "camera", None)
    if camera is None:
        hub = getattr(machine, "camera_hub", None)
        camera = hub.get(pipeline_id) if hub is not None else None
    if camera is None:
        return None
    return getattr(camera, "last_frame", None)


def _hits_of(detections, labels):
    """The detection boxes as HUD hits, in frame pixels."""
    from framework.ui import hud

    hits = []
    for det in detections or []:
        class_id = int(getattr(det, "class_id", 0))
        label = str(getattr(det, "label", "") or "")
        if not label:
            label = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
        score = float(getattr(det, "score", 0.0))
        hits.append(
            hud.Hit(
                x=int(det.x), y=int(det.y),
                w=int(det.w), h=int(det.h),
                text=f"{label}:{score:.2f}",
            )
        )
    return hits


def _panel_content(machine, vm, cfg: dict):
    """The camera frame, the frame-space boxes and the HUD content."""
    from framework.ui import hud

    cfg = cfg or {}
    results = vm.get_all_results() if vm is not None else {}
    pipeline_id = next(iter(results), None)
    if pipeline_id is None:
        cameras = cfg.get("cameras") or []
        pipeline_id = str(cameras[0].get("camera_id", "main")) if cameras else "main"

    ai_result = (results.get(pipeline_id) or {}).get("ai_inference")
    detections = []
    error_code = ""
    if ai_result is not None:
        if getattr(ai_result, "success", False):
            detections = (ai_result.result_data or {}).get("detections", []) or []
        else:
            error_code = getattr(ai_result, "error_code", "") or ""

    ai = getattr(machine, "ai", None)
    labels = _attr(ai, "labels", []) or []
    engine = str(_attr(ai, "_engine_id", "") or "ai")
    model = str(_attr(ai, "active_model", "") or "")
    fps = vm.get_pipeline_fps(pipeline_id) if vm is not None else 0.0
    loaded = bool(_attr(ai, "loaded", False))

    state = hud.HudState(
        mode="DESKTOP",
        link_ok=loaded,
        found=bool(detections),
        fps=fps,
        line2=f"{engine} {model} | det {len(detections)}".strip(),
        hits=_hits_of(detections, labels),
        alert=error_code,
    )
    return _camera_frame(machine, vm, pipeline_id), state.hits, state


def _exit_check(machine, panel):
    """Stop the main loop when the platform or the panel says so."""
    platform_check = getattr(machine, "exit_check", None)

    def check() -> bool:
        if panel is not None and panel.exit_requested:
            return True
        if callable(platform_check):
            try:
                return bool(platform_check())
            except Exception:
                return False
        return False

    return check


def _build_tick(manager, machine, cfg: dict = None, panel=None):
    """Build a per-tick callback that pushes the newest frame and flushes sinks.

    Args:
        manager: ModuleManager owning the vision module.
        machine: Assembled Machine; ``display`` may be None when headless.
        cfg: Parsed config; ``display_show_interval_ms`` is the LCD present
            interval, with absent or 0 meaning no throttle.
        panel: The desktop panel, when the platform offers one.  It presents
            the composed HUD itself, so the LCD sink stays out of the way.

    The present cadence and overlay rebuild window share that key, so a slow
    panel draws as often as it shows. With ``display_enabled: false`` the vision
    side stops composing frames because nothing would consume them.
    """
    from framework.hal.interface.sink import SinkGroup
    from framework.hal.sink import MaixLcdSink

    cfg = cfg or {}
    interval_ms = cfg.get("display_show_interval_ms")
    sinks = SinkGroup()
    if panel is None and getattr(machine, "display", None) is not None:
        sinks.add(MaixLcdSink(machine.display, interval_ms=interval_ms))
    # TODO(project): add more sinks / input sources here (CvWindowSink,
    # JpegStreamSink, touchscreen -> CmdQueue).

    vm = _vision_manager_of(manager)
    if vm is not None:
        vm.set_render_enabled(cfg.get("display_enabled", True))
        vm.set_display_cadence_ms(interval_ms)

    def tick() -> None:
        live_vm = vm or _vision_manager_of(manager)
        if panel is not None:
            frame, hits, state = _panel_content(machine, live_vm, cfg)
            panel.compose(frame, hits, state)
            panel.present()
            panel.pump()
            return
        frame = live_vm.get_display_frame() if live_vm else None
        if frame is not None:
            sinks.push(frame)
        sinks.flush()

    return tick


def main() -> None:
    import os

    import yaml

    from utils.debug_console import DebugConsole
    from utils.log_util import start_log_writer, stop_log_writer

    # Config path: default to the workspace project_config.yaml; ZWULU_CONFIG
    # can point elsewhere (e.g. configs/project_config.maixcam1.yaml) so
    # switching platforms does not overwrite the workspace file.
    config_path = os.environ.get("ZWULU_CONFIG", "project_config.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    DebugConsole.set_global_enabled(cfg.get("debug_console_enabled", True))
    start_log_writer()
    _apply_observability(cfg)

    wdt_feed = _make_wdt_feed(cfg.get("platform", "maixcam2"))  # rule 1: watchdog first
    wdt_feed()

    from framework.hal import Machine

    machine = Machine.create(config_path)
    wdt_feed()  # AI model load can take seconds

    # The desktop panel is a platform capability: the device has none and
    # keeps its LCD path, the desktop opens a window with the same HUD.
    panel = _panel_of(machine, cfg)

    # TODO(project): platform bring-up that needs `machine` goes here
    # (cameras for calibration, WiFi AP, beacon, ...), feeding wdt_feed()
    # after each step.

    from framework.event_bus import EventBus
    from framework.module_manager import ModuleManager

    bus = EventBus()

    from app.coordinator import AppCoordinator

    coordinator = AppCoordinator(bus)  # per-project logic slot (rule 2)
    coordinator.set_sys_info(getattr(machine, "sys_info", None))

    manager = ModuleManager(
        machine,
        event_bus=bus,
        wdt_feed=wdt_feed,
        exit_check=_exit_check(machine, panel),
        config=cfg,
    )
    # TODO(project): add modules per need ("zw_uart_module", "zw_wifi_stream").
    manager.register_many(["zw_opencv_module", "zw_battery_module"])

    try:
        manager.run_main_loop(
            coordinator=coordinator,
            display_callback=_build_tick(manager, machine, cfg, panel),
        )
    finally:
        if panel is not None:
            panel.close()
        stop_log_writer()


if __name__ == "__main__":
    main()
