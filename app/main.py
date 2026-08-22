import logging
import os
import sys
import threading
import time
from utils.log_util import log_print
from app.vision_state import VisionState
from app.pc_heartbeat import PcHeartbeatDetector

_DISPLAY_EVERY_N = 2
_ICON_SIZE = 48
_ICON_MARGIN = 12


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.makedirs("logs", exist_ok=True)
from utils.log_util import start_log_writer
start_log_writer()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    filename="logs/app.log",
    filemode="a",
)

from framework.hal import Machine
from framework.module_manager import ModuleManager


def _check_cmm_pressure(machine, skip_threshold: float = 0.80) -> bool:
    """Skip a display frame when the CMM pool is under pressure (ARCH-07).

    Reads the platform-provided SysInfo snapshot; returns False (no skip) on
    platforms without memory stats.
    """
    sysinfo = getattr(machine, "sys_info", None)
    if sysinfo is None:
        return False
    try:
        info = sysinfo.memory_snapshot()
    except Exception:
        return False
    if not info:
        return False
    cmm_used = info.get("cmm_used", 0)
    cmm_total = info.get("cmm_total", 256 * 1024 * 1024)
    return cmm_used > int(cmm_total * skip_threshold)


def _push_coordinator_status(coordinator) -> None:
    from utils.debug_console import DebugConsole
    dc = DebugConsole()
    info = coordinator.get_info()
    dc.set("state", info.get("state", "-"))
    dc.set("link_active", str(info.get("link_active", False)))
    dc.set("det_count", str(info.get("det_count", 0)))
    dc.set("fps", f"{info.get('fps', 0.0):.1f}")


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


def _build_callbacks(manager, machine, coordinator, wdt_feed=None):
    """Returns (main_callback, start_display_thread).

    main_callback: touch/input handling (~0ms), runs in main loop.
    display thread: display.show() only, runs independently.
    exit icon is drawn by VisionManager (via set_exit_icon).
    calibrate button is drawn by VisionManager and handled here.
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
    _last_flash_toggle = 0.0

    def _in_btn(x, y, bx, by, size=None):
        if size is None:
            size = _ICON_SIZE
        return bx - 4 <= x <= bx + size + 4 and by - 4 <= y <= by + size + 4

    def main_callback():
        """Touch handling only — fast (~0ms), no JPEG encoding, no GIL hog."""
        nonlocal _touch_down, _touch_x, _touch_y, _last_flash_toggle

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
                                        if wdt_feed is not None:
                                            try:
                                                wdt_feed.disable()
                                            except Exception:
                                                pass
                                        os._exit(0)
                                    # Check calibrate button
                                    calib_rect = vm.get_calib_button_rect()
                                    if calib_rect is not None:
                                        cbx, cby, cbw, _ = calib_rect
                                        if _in_btn(_touch_x, _touch_y, cbx, cby, size=cbw):
                                                coordinator.change_state(VisionState.CALIB)
                                                ok = coordinator.calibrate_origin_from_ball()
                                                if ok:
                                                    log_print("[CALIB] Phase2 done: origin set from ball position")
                                                    bbox = coordinator.get_last_ball_bbox()
                                                    if bbox:
                                                        vm.trigger_calib_flash(bbox)
                                                    vm.set_calib_button_visible(False)
                                                    _persist_calibration(coordinator.get_rail_calibration())
                                                else:
                                                    log_print("[CALIB] Phase2 FAILED: no ball detected")
                                                coordinator.change_state(VisionState.IDLE)
                                    # Check fill light button
                                    fl_rect = vm.get_fill_light_button_rect()
                                    if fl_rect is not None:
                                        fx, fy, fw, _ = fl_rect
                                        if _in_btn(_touch_x, _touch_y, fx, fy, size=fw):
                                            now = time.monotonic()
                                            if now - _last_flash_toggle >= 0.3:
                                                _last_flash_toggle = now
                                                new_state = vm.toggle_fill_light()
                                                if new_state is not None:
                                                    _persist_fill_light_state(new_state)
                                                    log_print(f"[LED] fill light -> {'ON' if new_state else 'OFF'}")
                                                else:
                                                    log_print("[LED] fill light toggle FAILED (GPIO)")
                    _touch_down = False
            except Exception:
                pass

    def _display_loop():
        nonlocal _last_seen_frame
        _tick = 0
        while True:
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
                        if _check_cmm_pressure(machine):
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
    """Boot-sequence watchdog policy over the platform's hardware WDT.

    Rate-limits hardware feeds to >=1s apart, tolerates transient failures,
    and feeds once immediately (the WDT countdown starts at construction
    while WiFi AP start / model load can delay the first loop feed).
    The maix-specific implementation lives in the platform package (ARCH-02);
    app/ may reference it directly because the dog must outlive pre-Machine
    boot steps.
    """
    try:
        from framework.hal.platforms.maixcam2 import create_watchdog

        wdt = create_watchdog()
    except Exception as e:
        log_print(f"[WDT] init FAIL: {e}")
        wdt = None
    if wdt is None:
        _noop = lambda: None
        _noop.disable = lambda: None
        return _noop

    last = [0.0]
    fail_count = [0]

    def _feed():
        now = time.monotonic()
        if now - last[0] < 1.0:
            return
        last[0] = now
        try:
            wdt.feed()
        except Exception as e:
            fail_count[0] += 1
            if fail_count[0] % 50 == 1:
                log_print(f"[WDT] feed FAIL x{fail_count[0]}: {e}")

    # Immediate first feed closes the construction->loop gap.
    try:
        wdt.feed()
        last[0] = time.monotonic()
    except Exception as e:
        log_print(f"[WDT] initial feed FAIL: {e}")

    _feed.disable = wdt.disable
    return _feed


def _init_streamer(vm) -> None:
    from modules.zw_wifi_stream import get_streamer
    s = get_streamer()
    if s is not None and vm is not None:
        vm.set_capture_sink(s.push_frame)
        s.start_async()


def _init_wifi(cfg: dict):
    streaming = cfg.get("streaming", {})
    mode = streaming.get("wifi_mode", "off")
    if mode not in ("ap", "sta"):
        return None
    try:
        from maix.network import wifi as _mw
        w = _mw.Wifi()
        if mode == "ap":
            if w.is_connected():
                log_print("[WiFi] Disconnecting existing STA before AP...")
                e = w.disconnect()
                if e != 0:
                    log_print(f"[WiFi] Disconnect failed, err={e}")
            # Credentials must come from project_config.yaml — no hardcoded
            # fallback (a missing key now skips AP start instead of bringing
            # up a well-known-password network).
            ssid = streaming.get("ap_ssid")
            password = streaming.get("ap_password")
            if not ssid or not password:
                log_print("[WiFi] ap_ssid/ap_password missing in config, skip AP start")
                return None
            e = w.start_ap(ssid, password, ip="192.168.1.1")
            if e != 0:
                log_print(f"[WiFi] AP start failed, err={e}")
                return None
            time.sleep(0.5)
            if not w.is_ap_mode():
                log_print("[WiFi] AP mode not confirmed after start_ap")
                return None
            ip = w.get_ip()
            log_print(f"[WiFi] AP mode started, SSID={ssid}, IP={ip}")
            return ip
        elif mode == "sta":
            ip = w.get_ip()
            if ip:
                log_print(f"[WiFi] STA mode, system-managed, IP={ip}")
            else:
                log_print("[WiFi] STA mode, no IP yet (waiting for system connection)")
            return ip
    except Exception as e:
        log_print(f"[WiFi] init FAIL: {e}")
        return None


def _send_record_cmd(cmd: str, test_id: int, target_ip: str = None) -> None:
    try:
        import socket
        import json
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            msg = json.dumps({"cmd": cmd, "test_id": test_id})
            encoded = msg.encode()
            if target_ip:
                for _ in range(3):
                    s.sendto(encoded, (target_ip, 5000))
                    time.sleep(0.05)
            else:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                for _ in range(3):
                    s.sendto(encoded, ("255.255.255.255", 5000))
                    time.sleep(0.05)
        finally:
            s.close()
    except Exception:
        pass


def _setup_record_signaling(coordinator, cfg) -> None:
    streaming = cfg.get("streaming", {})
    mode = streaming.get("wifi_mode", "off")
    if mode in ("ap", "sta"):
        coordinator.set_record_cmd_sender(
            lambda cmd, tid, target_ip=None: _send_record_cmd(cmd, tid, target_ip)
        )


def _start_beacon() -> None:
    def _beacon_loop():
        import socket
        import json
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception as e:
            log_print(f"[Beacon] socket init FAIL: {e}")
            return
        while True:
            try:
                msg = json.dumps({"type": "beacon", "service": "zulu-walker", "port": 8000})
                sock.sendto(msg.encode(), ("255.255.255.255", 9999))
            except Exception:
                pass
            time.sleep(2)

    t = threading.Thread(target=_beacon_loop, daemon=True)
    t.start()


def _load_calib_icon():
    """Load calibrate button icon from assets/calibrate.png."""
    base = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(os.path.dirname(base), "assets", "calibrate.png")
    try:
        from maix import image as _mi
        icon = _mi.load(icon_path)
        if icon is None:
            log_print(f"[CALIB] Icon load FAILED, path={icon_path}")
            return None
        w = icon.width() * 48 // icon.height()
        if w % 2:
            w += 1
        log_print(f"[CALIB] Icon loaded OK, size={w}x48")
        return icon.resize(w, 48)
    except Exception as e:
        log_print(f"[CALIB] Icon load exception: {e}")
        return None


def _load_persisted_calibration(cfg: dict):
    """Load RailCalibration from project_config.yaml if persisted."""
    from modules.zw_opencv_module.detectors.pendulum_calibrator import RailCalibration
    data = cfg.get("pendulum", {}).get("rail_calibration", None)
    if data and data.get("calibrated"):
        try:
            return RailCalibration.from_dict(data)
        except Exception:
            pass
    return None


_CALIB_PARAM_KEYS = frozenset({
    'binary_threshold', 'min_contour_area_ratio', 'min_aspect_ratio',
    'canny_low', 'canny_high', 'hough_threshold', 'hough_min_line_len',
    'edge_angle_max_deg', 'column_threshold', 'max_contour_area_ratio',
})


def _run_phase1_calibration(camera, calib_params=None):
    """Phase 1: grab frames until calibration succeeds or max retries.
    Returns RailCalibration or None on failure.
    Camera may need a few frames for auto-exposure to settle."""
    import time
    from modules.zw_opencv_module.detectors.pendulum_calibrator import PendulumCalibrator

    kwargs = {}
    if calib_params:
        kwargs = {k: calib_params[k] for k in _CALIB_PARAM_KEYS if k in calib_params}

    for attempt in range(15):
        try:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            calib = PendulumCalibrator(frame_w=frame.shape[1], frame_h=frame.shape[0], **kwargs)
            result = calib.calibrate(frame)
            if result.calibrated:
                try:
                    diag = calib.get_last_diagnostics()
                    log_print("[CALIB] Phase1 method=%s pts=%s angle=%.4f" % (
                        diag.get('method', '?'),
                        diag.get('column_points', '-'),
                        result.angle_rad))
                except Exception:
                    pass
                return result
        except Exception:
            pass
        time.sleep(0.05)
    return None


_CONFIG_WRITE_LOCK = threading.Lock()


def _persist_calibration(calib):
    """Write RailCalibration to project_config.yaml atomically, preserving existing content."""
    import tempfile

    import yaml
    with _CONFIG_WRITE_LOCK:
        try:
            with open("project_config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        if "pendulum" not in cfg:
            cfg["pendulum"] = {}
        cfg["pendulum"]["rail_calibration"] = calib.to_dict()
        # Same-directory temp + os.replace: a power cut mid-write leaves the
        # old config intact instead of a truncated file losing WiFi/AEC/model
        # settings along with the calibration.
        fd, tmp_path = tempfile.mkstemp(dir=".", suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            os.replace(tmp_path, "project_config.yaml")
        except Exception as e:
            log_print(f"[CALIB] persist FAILED: {e}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _load_icon(path, size: int = 48):
    """Load an icon, resized to `size`x`size`. Absolute or project-root-relative path."""
    if not path:
        return None
    try:
        from maix import image as _mi
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, path)
        icon = _mi.load(path)
        if icon is None:
            log_print(f"[ICON] load FAILED, path={path}")
            return None
        return icon.resize(size, size)
    except Exception as e:
        log_print(f"[ICON] load exception: {e}")
        return None


_FILL_LIGHT_STATE_PATH = os.path.join("logs", "fill_light.state")
_fill_light_persist_q = None
_fill_light_persist_thread = None


def _load_fill_light_state():
    """Read persisted fill-light state. Returns bool, or None if no state file."""
    try:
        with open(_FILL_LIGHT_STATE_PATH, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return v == "1"
    except Exception:
        return None


def _write_fill_light_state(on: bool) -> None:
    """Write fill_light state to logs/fill_light.state (atomic, tiny).
    Runs in the background writer thread; must not block the main loop."""
    import tempfile
    if _load_fill_light_state() == bool(on):
        return
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir="logs", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("1" if on else "0")
        os.replace(tmp_path, _FILL_LIGHT_STATE_PATH)
    except Exception as e:
        log_print(f"[LED] fill_light state persist FAILED: {e}")
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _ensure_fill_light_persist():
    """Start the single background state-writer thread on first use."""
    global _fill_light_persist_q, _fill_light_persist_thread
    if _fill_light_persist_q is None:
        import queue
        _fill_light_persist_q = queue.Queue()

        def _writer_loop():
            while True:
                on = _fill_light_persist_q.get()
                try:
                    while True:
                        on = _fill_light_persist_q.get_nowait()
                except Exception:
                    pass
                _write_fill_light_state(on)

        _fill_light_persist_thread = threading.Thread(target=_writer_loop, daemon=True)
        _fill_light_persist_thread.start()


def _persist_fill_light_state(on: bool) -> None:
    """Queue a fill_light state persist; non-blocking, coalesced."""
    _ensure_fill_light_persist()
    _fill_light_persist_q.put(bool(on))


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
    wdt_feed()  # pre-WiFi feed: WDT countdown starts at construction

    _init_wifi(_cfg)

    wdt_feed()  # post-WiFi feed: AP/STA start can take >1s

    _start_beacon()

    fill_light_cfg = _cfg.get("fill_light") or {}
    if not isinstance(fill_light_cfg, dict):
        fill_light_cfg = {}
    fill_start_on = bool(fill_light_cfg.get("start_on", True))
    _persisted_state = _load_fill_light_state()
    if _persisted_state is not None:
        fill_start_on = _persisted_state
    fill_icon_on = fill_light_cfg.get("icon_on")
    fill_icon_off = fill_light_cfg.get("icon_off")
    try:
        from framework.hal.platforms.maixcam2 import set_fill_light as _set_fill_light
        _HAS_FILL_LIGHT = True
    except Exception as e:
        _set_fill_light = None
        _HAS_FILL_LIGHT = False
        log_print(f"[LED] fill light platform import failed: {e}")

    # actual hardware state; falls back to OFF if GPIO init fails so the
    # displayed icon always matches the real light
    _fill_light_actual = fill_start_on
    if _HAS_FILL_LIGHT:
        try:
            _fill_light_actual = bool(_set_fill_light(fill_start_on))
            log_print(f"[LED] fill light {'ON' if _fill_light_actual else 'OFF'} (start_on={fill_start_on})")
        except Exception as e:
            _fill_light_actual = False
            log_print(f"[LED] fill light init exception: {e}")

    from framework.event_bus import EventBus
    from app.coordinator import Ti2026Coordinator
    from framework.hal.camera_hub import CameraHub
    bus = EventBus()
    coordinator = Ti2026Coordinator(bus, sys_info=getattr(machine, "sys_info", None))
    coordinator.set_wdt_feed(wdt_feed)

    pc_heartbeat = PcHeartbeatDetector()
    try:
        pc_heartbeat.start()
        coordinator.set_pc_heartbeat(pc_heartbeat)
    except RuntimeError as e:
        log_print(f"[Heartbeat] init FAIL: {e}")

    _setup_record_signaling(coordinator, _cfg)

    machine = Machine.create("project_config.yaml")
    wdt_feed()  # post-model-load feed: AI model load from flash can take seconds

    # --- Pendulum rail calibration (Phase-1) ---
    # Runs here, BEFORE the vision module starts, so the camera is read by a
    # single thread at startup (no concurrent read()/read_raw() window).
    rail_calib = _load_persisted_calibration(_cfg)
    phase1_result = None
    try:
        cam = CameraHub.instance().get("main")
        if cam is not None:
            # Let the camera settle after open (initial exposure/gain are
            # fixed until software-AEC runs in the vision thread); feed the
            # WDT during the wait so a slow boot never trips it.
            for _ in range(3):
                time.sleep(0.5)
                wdt_feed()
            calib_params = _cfg.get("pendulum", {}).get("calib_params", None)
            phase1_result = _run_phase1_calibration(cam, calib_params)
        else:
            log_print("[CALIB] No camera available, skip Phase1")
    except Exception as e:
        log_print(f"[CALIB] Phase1 error: {e}")

    uart_cfg = _cfg.get("uart_defaults", {})
    uart_connected = machine.uart.is_connected if machine.uart else False
    log_print(f"[UART] port={uart_cfg.get('port')} baud={uart_cfg.get('baudrate')} connected={uart_connected}")

    try:
        from maix import display
        display.set_trans_image_quality(10)
    except Exception:
        pass

    manager = ModuleManager(
        machine,
        event_bus=bus,
        wdt_feed=wdt_feed,
        exit_check=getattr(machine, "exit_check", None),
    )
    manager.register_many(["zw_opencv_module", "zw_uart_module", "zw_wifi_stream"])

    from modules.zw_opencv_module import get_vision_manager
    from modules.zw_uart_module import get_interface

    # Single source of truth for pixels_per_cm (used by both the measurement
    # path in coordinator and the cm-ruler overlay in vision_manager).
    pixels_per_cm = _cfg.get("pendulum", {}).get("pixels_per_cm", 50.0)

    vm = get_vision_manager()
    if vm:
        coordinator.connect_vision(vm)
        draw_rail = _cfg.get("pendulum", {}).get("draw_rail", False)
        rail_cm_int = _cfg.get("pendulum", {}).get("cm_interval", 1.0)
        vm.set_rail_draw(draw_rail, lambda: coordinator.get_rail_calibration(),
                         pixels_per_cm=pixels_per_cm, cm_interval=rail_cm_int)
        log_print(f"[RAIL] draw_rail={'ON' if draw_rail else 'OFF'} ppc={pixels_per_cm} cm_interval={rail_cm_int}")
        det_list = _cfg.get("display", {}).get("detection_list", False)
        vm.set_detection_list_enabled(det_list)
        log_print(f"[DISPLAY] detection_list={'ON' if det_list else 'OFF'}")

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.set_ai(machine.ai)

    _init_streamer(vm)

    exit_icon = _load_exit_icon()
    if exit_icon is not None and vm is not None:
        vm.set_exit_icon(exit_icon, _ICON_SIZE, _ICON_MARGIN)

    if vm is not None and _HAS_FILL_LIGHT and fill_icon_on and fill_icon_off:
        fl_icon_on = _load_icon(fill_icon_on, _ICON_SIZE)
        fl_icon_off = _load_icon(fill_icon_off, _ICON_SIZE)
        if fl_icon_on is not None and fl_icon_off is not None:
            vm.set_fill_light_button(fl_icon_on, fl_icon_off, size=_ICON_SIZE)
            vm.set_fill_light_controller(_set_fill_light)
            vm.set_fill_light_state(_fill_light_actual)
            log_print("[LED] fill light button enabled")
        else:
            log_print("[LED] fill light button NOT shown: icon load failed")

    cam_cfg = _cfg.get("cameras", [{}])[0]
    cam_w = cam_cfg.get("width", 640)
    cam_h = cam_cfg.get("height", 640)
    log_print(f"ppc:{pixels_per_cm} w:{cam_w} h:{cam_h}")
    coordinator.set_pendulum_calibration(pixels_per_cm, cam_w, cam_h)

    # --- Pendulum rail calibration ---
    if phase1_result is not None:
        if rail_calib is not None:
            from modules.zw_opencv_module.detectors.pendulum_calibrator import RailCalibration
            rail_calib = RailCalibration(
                origin_x=rail_calib.origin_x,
                origin_y=rail_calib.origin_y,
                angle_rad=phase1_result.angle_rad,
                calibrated=True,
            )
            log_print(f"[CALIB] Phase1 angle={rail_calib.angle_rad:.4f} "
                      f"origin=(kept: {rail_calib.origin_x:.0f},{rail_calib.origin_y:.0f})")
        else:
            rail_calib = phase1_result
            log_print(f"[CALIB] Phase1 done: angle={rail_calib.angle_rad:.4f} "
                      f"origin=({rail_calib.origin_x:.0f},{rail_calib.origin_y:.0f})")
        coordinator.set_rail_calibration(rail_calib)
    elif rail_calib is not None:
        coordinator.set_rail_calibration(rail_calib)
        log_print(f"[CALIB] Phase1 FAILED, using persisted: angle={rail_calib.angle_rad:.4f} "
                  f"origin=({rail_calib.origin_x:.0f},{rail_calib.origin_y:.0f})")
    else:
        log_print("[CALIB] Phase1 FAILED, no calibration available")

    if rail_calib is not None:
        calib_icon = _load_calib_icon()
        if vm is not None and calib_icon is not None:
            vm.set_calib_button(calib_icon, _ICON_SIZE, _ICON_MARGIN)
            vm.set_calib_button_visible(True)
            log_print("[CALIB] Calibrate button visible (bottom-right)")
        elif calib_icon is None:
            log_print("[CALIB] Button NOT shown: icon load failed")
        else:
            log_print("[CALIB] Button NOT shown: VisionManager is None")

    main_callback, start_display_thread = _build_callbacks(manager, machine, coordinator, wdt_feed=wdt_feed)

    coordinator.start()
    start_display_thread()
    manager.run_main_loop(coordinator, tick_callback=_push_coordinator_status, display_callback=main_callback)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
