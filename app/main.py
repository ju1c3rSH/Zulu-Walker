# -*- coding: utf-8 -*-
"""工科赛近端视觉主程序 — 装配 + 页面路由 + 显示线程。

页面四态由 AppPageState（framework/state_machine 页面状态机）驱动:
MAIN 正常运行 / EXPLOSIVE / TARGET / RESCUE 三调试页, KEY_OK 循环切页。
调试页独占相机的 set_paused 由页面状态机 enter/exit 托管; 显示线程对
非 MAIN 页整体让位（页面自身 disp.show 全权接管）。退出图标由
VisionManager 画入显示帧, 仅 MAIN 页消费触屏判定点击退出。
"""
import logging
import os
import sys
import threading
import time

from utils.log_util import log_print, start_log_writer

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

_DISPLAY_SHOW_SKIP = 2
_ICON_SIZE = 48
_ICON_MARGIN = 12
_ICON_HIT_PAD = 4
_CMM_PRESSURE_THRESHOLD = 0.80
_CMM_TOTAL_FALLBACK = 256 * 1024 * 1024
_WDT_WINDOW_MS = 10000
_WDT_FEED_PERIOD_S = 1.0

_EXPLOSIVE_GROUPS = (0, 1, 2)
_TARGET_GROUPS = (3, 4, 5)

_PIPELINE_ID = "cam_main"
_TASK_GKS = "gks_task"
_VISION_MODULE = "zw_opencv_module"

from app.page_sm import STATE_MAIN


def _check_cmm_pressure() -> bool:
    if not _HAVE_MAIX_SYS:
        return False
    try:
        info = maix_sys.memory_info()
        cmm_used = info.get("cmm_used", 0)
        cmm_total = info.get("cmm_total", _CMM_TOTAL_FALLBACK)
        return cmm_used > int(cmm_total * _CMM_PRESSURE_THRESHOLD)
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
    dc.set("tx_hz", f"{info.get('tx_hz', 0.0):.1f}")


class DebugPageFactory:
    """调试页懒创建 + 会话保持: 页面实例跨按键循环连续存在。

    首次进入 create(), 每次切入 begin()（重置交互态, 不丢共享会话数据）;
    页面 exit_requested() 由 step_active 检测并回 MAIN。
    """

    def __init__(self, machine, vision_manager, pages):
        self._machine = machine
        self._vm = vision_manager
        self._pages = pages
        self._touchscreen = None
        self._instances = {}
        self._active = None

    def ensure_entered(self, page_name: str) -> None:
        if self._active == page_name:
            return
        page = self._instances.get(page_name)
        if page is None:
            page = self._create(page_name)
            if page is None:
                log_print(f"[PAGE] {page_name} build failed -> main")
                self._pages.select(STATE_MAIN)
                return
            self._instances[page_name] = page
            log_print(f"[PAGE] built debug page: {page_name}")
        page.begin()
        self._active = page_name

    def step_active(self, page_name: str) -> None:
        page = self._instances.get(page_name)
        ts = self._ensure_touchscreen()
        cam = self._vm.get_pipeline_camera() if self._vm else None
        if page is None or cam is None or ts is None:
            log_print(f"[PAGE] {page_name} unavailable, "
                      f"touch={ts is not None} cam={cam is not None} -> main")
            self._pages.select(STATE_MAIN)
            self._active = None
            return
        frame_bgr = cam.read()
        if frame_bgr is None:
            return
        page.step(self._machine.display, ts, frame_bgr)
        if getattr(page, "exit_requested", lambda: False)():
            self._pages.select(STATE_MAIN)
            self._active = None
            log_print("[KEY] debug page Exit -> main, vision resume")

    def page_active(self) -> bool:
        return self._active is not None

    def _ensure_touchscreen(self):
        if self._touchscreen is None:
            try:
                from maix import touchscreen
                self._touchscreen = touchscreen.TouchScreen()
            except Exception:
                return None
        return self._touchscreen

    def _create(self, page_name: str):
        if page_name in ("EXPLOSIVE", "TARGET"):
            from modules.zw_opencv_module.debug.threshold_tuner import GksThresholdTuner
            groups = _EXPLOSIVE_GROUPS if page_name == "EXPLOSIVE" else _TARGET_GROUPS
            return GksThresholdTuner(group_indices=groups,
                                     get_processor=self._get_gks_processor)
        if page_name == "RESCUE":
            from modules.zw_opencv_module.debug.rescue_tuner import GksRescueTuner
            return GksRescueTuner(
                get_processor=self._get_gks_processor,
                get_ai=lambda: self._machine.ai,
            )
        return None

    def _get_gks_processor(self):
        if not self._vm:
            return None
        pipe = self._vm.get_pipeline(_PIPELINE_ID)
        if pipe is None:
            return None
        task = pipe.task_manager.get_task(_TASK_GKS)
        return task.processor if task else None


class MainUiShell:
    """主循环 tick / MAIN 页触摸退出判定 / 显示线程。

    MAIN 页 tick 只推 DebugConsole 状态; display_callback 读触摸判定
    退出图标; 显示线程每 _DISPLAY_SHOW_SKIP 帧 show 一次（CMM 压力跳帧）。
    非 MAIN 页所有路径短路, 与调试页触摸/显示互不干扰。
    """

    def __init__(self, machine, pages, factory: DebugPageFactory, vision_manager):
        self._machine = machine
        self._pages = pages
        self._factory = factory
        self._vm = vision_manager
        self._touch = self._open_touch()
        self._touch_down = False
        self._touch_x = 0
        self._touch_y = 0
        self._last_seen_frame = None
        self._display_counter = 0

    def _open_touch(self):
        try:
            from maix import touchscreen
            return touchscreen.TouchScreen()
        except Exception:
            return None

    def tick(self, coordinator) -> None:
        page_name = self._pages.name
        if page_name == STATE_MAIN:
            self._factory.reset()
            _push_coordinator_status(coordinator)
            return
        self._factory.ensure_entered(page_name)
        if self._factory.page_active():
            self._factory.step_active(page_name)

    def display_callback(self) -> None:
        if not self._pages.is_main():
            return
        self._consume_main_touch()

    def start_display_thread(self):
        thread = threading.Thread(target=self._display_loop, daemon=True)
        thread.start()
        return thread

    def _display_loop(self) -> None:
        while True:
            if not self._pages.is_main():
                time.sleep(0.01)
                continue
            frame = self._vm.get_display_frame() if self._vm else None
            if frame is not None and frame is not self._last_seen_frame:
                self._last_seen_frame = frame
                self._display_counter += 1
                if self._display_counter % _DISPLAY_SHOW_SKIP == 0 \
                        and not _check_cmm_pressure():
                    try:
                        self._machine.display.show(frame)
                    except Exception:
                        pass
            time.sleep(0.001)

    def _consume_main_touch(self) -> None:
        if self._touch is None:
            return
        try:
            x, y, pressed = self._touch.read()
            if pressed:
                self._touch_down = True
                self._touch_x, self._touch_y = self._map_screen_to_frame(x, y)
            elif self._touch_down:
                self._touch_down = False
                if self._in_exit_button(self._touch_x, self._touch_y):
                    os._exit(0)
        except Exception:
            pass

    def _map_screen_to_frame(self, x, y):
        frame = self._vm.get_display_frame() if self._vm else None
        if frame is None:
            return int(x), int(y)
        try:
            import maix.image as maix_image
            point = maix_image.resize_map_pos_reverse(
                frame.width(), frame.height(),
                self._machine.display.width(), self._machine.display.height(),
                maix_image.Fit.FIT_CONTAIN, x, y,
            )
            return int(point[0]), int(point[1])
        except Exception:
            return int(x), int(y)

    def _in_exit_button(self, x, y) -> bool:
        frame = self._vm.get_display_frame() if self._vm else None
        if frame is None:
            return False
        bx = _ICON_MARGIN
        by = frame.height() - _ICON_SIZE - 8
        return (bx - _ICON_HIT_PAD <= x <= bx + _ICON_SIZE + _ICON_HIT_PAD
                and by - _ICON_HIT_PAD <= y <= by + _ICON_SIZE + _ICON_HIT_PAD)


def _make_wdt_feed():
    try:
        from maix.peripheral import wdt as maix_wdt
        watchdog = maix_wdt.WDT(0, _WDT_WINDOW_MS)
        import time
        # 多处 feed 入口(main/coord/viz)共享同一节流时钟, 全局最多 1 次/周期,
        # 避免各自独立计时导致 C 层 WDT::feed() 刷屏 (窗口 10s 足够余量)
        if "_WDT_SHARED_STATE" not in globals():
            globals()["_WDT_SHARED_STATE"] = {"last": 0.0, "fails": 0, "wdt": watchdog}
        _state = globals()["_WDT_SHARED_STATE"]
        _state["wdt"] = watchdog

        def _feed():
            now = time.monotonic()
            if now - _state["last"] < _WDT_FEED_PERIOD_S:
                return
            _state["last"] = now
            try:
                watchdog.feed()
            except Exception as e:
                _state["fails"] += 1
                if _state["fails"] % 50 == 1:
                    log_print(f"[WDT] feed FAIL x{_state['fails']}: {e}")
        return _feed
    except Exception as e:
        log_print(f"[WDT] init FAIL: {e}")
        return lambda: None


def _load_exit_icon():
    try:
        from maix import image as maix_image
        icon = maix_image.load("/maixapp/share/icon/ret.png")
        if icon is None:
            return None
        width = icon.width() * _ICON_SIZE // icon.height()
        if width % 2:
            width += 1
        return icon.resize(width, _ICON_SIZE)
    except Exception:
        return None


def _apply_stream_policy(coordinator, config: dict) -> None:
    """把 project_config.yaml `gks_stream:` 段注入 coordinator（缺段即用类内默认）。

    默认即结果驱动（每帧结果立即推）; 现场若发现主站吃不下这个速率，改配置即可
    回退到旧的 interval 限速，不必重新烧固件。
    """
    section = config.get("gks_stream") or {}
    coordinator.set_stream_policy(
        push_on_new_result=section.get("push_on_new_result"),
        min_push_interval_ms=section.get("min_push_interval_ms"),
    )


def _init_streamer(vision_manager) -> None:
    from modules.zw_wifi_stream import get_streamer
    streamer = get_streamer()
    if streamer is not None and vision_manager is not None:
        vision_manager.set_capture_sink(streamer.push_frame)
        streamer.start_async()


def _project_config() -> dict:
    import yaml
    try:
        with open("project_config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main():
    log_print("0xfb709394")

    from utils.debug_console import DebugConsole
    config = _project_config()
    DebugConsole.set_global_enabled(config.get("debug_console_enabled", True))
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
    manager.register_many([_VISION_MODULE, "zw_uart_module", "zw_wifi_stream"])

    from modules.zw_opencv_module import get_vision_manager
    from modules.zw_uart_module import get_interface

    vision_manager = get_vision_manager()
    if vision_manager:
        coordinator.connect_vision(vision_manager)
        vision_manager.set_hud(DisplayHUD(status_fn=coordinator.get_info))

    uart_interface = get_interface()
    if uart_interface:
        coordinator.set_uart_sender(uart_interface.send_raw)

    coordinator.set_ai(machine.ai)
    _apply_stream_policy(coordinator, config)
    _init_streamer(vision_manager)

    exit_icon = _load_exit_icon()
    if exit_icon is not None and vision_manager is not None:
        vision_manager.set_exit_icon(exit_icon, _ICON_SIZE, _ICON_MARGIN)

    camera_config = config.get("cameras", [{}])[0]
    coordinator.set_frame_size(
        camera_config.get("width", 640), camera_config.get("height", 480))
    log_print(f"cam w:{camera_config.get('width', 640)} h:{camera_config.get('height', 480)}")

    from app.page_router import AppPageState, KeyCycleRouter
    pages = AppPageState(vision_manager)
    pages.set_camera_gate(
        lambda: bool(vision_manager and vision_manager.get_pipeline_camera() is not None))
    if vision_manager:
        KeyCycleRouter(pages)

    factory = DebugPageFactory(machine, vision_manager, pages)
    ui = MainUiShell(machine, pages, factory, vision_manager)

    coordinator.start()
    ui.start_display_thread()
    manager.run_main_loop(
        coordinator,
        tick_callback=ui.tick,
        display_callback=ui.display_callback,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
