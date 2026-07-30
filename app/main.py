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


def _build_callbacks(manager, machine, coordinator):
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

    def _in_btn(x, y, bx, by, size=None):
        if size is None:
            size = _ICON_SIZE
        return bx - 4 <= x <= bx + size + 4 and by - 4 <= y <= by + size + 4

    def main_callback():
        """Touch handling only — fast (~0ms), no JPEG encoding, no GIL hog."""
        nonlocal _touch_down, _touch_x, _touch_y

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
            ssid = streaming.get("ap_ssid", "Zulu-Walker")
            password = streaming.get("ap_password", "88888888")
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
    'edge_angle_max_deg',
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
                return result
        except Exception:
            pass
        time.sleep(0.05)
    return None


def _persist_calibration(calib):
    """Write RailCalibration to project_config.yaml, preserving existing content."""
    import yaml
    try:
        with open("project_config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    if "pendulum" not in cfg:
        cfg["pendulum"] = {}
    cfg["pendulum"]["rail_calibration"] = calib.to_dict()
    try:
        with open("project_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception:
        pass


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

    _init_wifi(_cfg)

    _start_beacon()

    from framework.event_bus import EventBus
    from app.coordinator import Ti2026Coordinator
    from framework.hal.camera_hub import CameraHub
    bus = EventBus()
    coordinator = Ti2026Coordinator(bus)
    coordinator.set_wdt_feed(wdt_feed)

    pc_heartbeat = PcHeartbeatDetector()
    try:
        pc_heartbeat.start()
        coordinator.set_pc_heartbeat(pc_heartbeat)
    except RuntimeError as e:
        log_print(f"[Heartbeat] init FAIL: {e}")

    _setup_record_signaling(coordinator, _cfg)

    machine = Machine.create("project_config.yaml")

    uart_cfg = _cfg.get("uart_defaults", {})
    uart_connected = machine.uart.is_connected if machine.uart else False
    log_print(f"[UART] port={uart_cfg.get('port')} baud={uart_cfg.get('baudrate')} connected={uart_connected}")

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

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.set_ai(machine.ai)

    _init_streamer(vm)

    exit_icon = _load_exit_icon()
    if exit_icon is not None and vm is not None:
        vm.set_exit_icon(exit_icon, _ICON_SIZE, _ICON_MARGIN)

    pixels_per_cm = _cfg.get("pendulum", {}).get("pixels_per_cm", 25.6)
    cam_cfg = _cfg.get("cameras", [{}])[0]
    cam_w = cam_cfg.get("width", 640)
    cam_h = cam_cfg.get("height", 640)
    log_print(f"ppc:{pixels_per_cm} w:{cam_w} h:{cam_h}")
    coordinator.set_pendulum_calibration(pixels_per_cm, cam_w, cam_h)

    # --- Pendulum rail calibration ---
    rail_calib = _load_persisted_calibration(_cfg)
    phase1_result = None
    try:
        cam = CameraHub.instance().get("main")
        if cam is not None:
            calib_params = _cfg.get("pendulum", {}).get("calib_params", None)
            phase1_result = _run_phase1_calibration(cam, calib_params)
        else:
            log_print("[CALIB] No camera available, skip Phase1")
    except Exception as e:
        log_print(f"[CALIB] Phase1 error: {e}")

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

    main_callback, start_display_thread = _build_callbacks(manager, machine, coordinator)

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
