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

from utils.log_util import log_print, start_log_writer, debug_console_active
from utils.boot_timing import thread_cpu as _thread_cpu

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

# ⚠⚠ **是否武装硬件看门狗**（2026-10-09 重大修正，默认 **False**）。
#
# 起因: 台架现象"运行期正常, 一退出设备就重启"。查 MaixCDK 源码
#   `components/peripheral/port/linux_common/maix_wdt.cpp` 得到决定性事实:
#
#     WDT::WDT(id, feed_ms) → open("/dev/watchdog") + WDIOC_SETTIMEOUT(feed_ms/1000)
#                             + close ⇒ **构造即启动**, 硬件倒计时立刻开始
#     WDT::feed()           → open + WDIOC_KEEPALIVE + close（喂一次 = 重置倒计时）
#     WDT::stop()           → 只打一行 "this operation is not supported", **return 0**
#     WDT::restart()        → 同上, 也是空实现
#
# 也就是说: **MaixCAM(MaixPy) 上的 `wdt.stop()` 是个空壳** —— 它返回 0(=ERR_NONE) 却
# 什么都没做, 调用方**无法从返回值分辨**"真关了"还是"没关"。而 Linux watchdog 的
# 语义是: close 时不写魔法字符 'V' 就保持运行, 内核还会明确 print
# `watchdog0: watchdog did not stop!` —— 所以进程退出**不会**让它停。
#
# 于是逻辑闭环了: 喂狗只来自三个 Python 线程（主循环/协调器/视觉），进程一退出
# 三者全没 ⇒ 窗口(10s)到点 ⇒ SoC 硬复位。**这就是"退出即重启"的全部原因。**
# （第一版修复只加了 `stop()` 调用 + 日志, 所以日志打了 "[WDT] stopped" 却照样重启 ——
#  被那个空实现骗了。现在的 `_wdt_stop` 会用 sysfs 倒计时与 raw ioctl 双重验证。）
#
# 规避手段只有两条（见 _wdt_stop）:
#   ① 退出前绕过 MaixPy, 直接对 /dev/watchdog 下 `WDIOS_DISABLECARD`；
#      若驱动带 nowayout（内核禁止停止看门狗）则返回 EBUSY —— 此路不通；
#   ② **压根别武装它**。这是 CAM1 上的默认选择: 代价是失去"线程死锁自动复位"兜底,
#      收益是**退出不再莫名重启**（台架每次收工都省一次 10s 掉线, 且 MaixVision/
#      SSH 不会莫名断线）。需要死锁兜底时把 `project_config.yaml` 的
#      `wdt_enabled` 置 true —— 但必须接受"每次退出后 ~10s 设备重启"。
_WDT_ENABLED_DEFAULT = False

# Linux watchdog uapi（绕过 MaixPy 空实现时用）
# ⚠ ioctl 码要按**有符号 32 位**传: 0x80045704 作为 C int 是负数(-2147199228)。
#   2026-10-09 设备实测: 传正数时内核回 errno=14 EFAULT(Bad address) —— 请求码不认。
# ⚠ 打开 /dev/watchdog 必须带 O_NONBLOCK: 看门狗已启动时, 不带 nonblock 的 open
#   会在 watchdog_open() 里**阻塞等标志位**, 表现为进程挂死。
_WATCHDOG_DEV = "/dev/watchdog"
_WATCHDOG_SYSFS = "/sys/class/watchdog/watchdog0"
# sysfs 属性名因内核而异（多为 `<name>_timeleft`, 不是 `timeleft`）——全试一遍
_WATCHDOG_TIMELEFT_FILES = ("timeleft", "watchdog0_timeleft", "wd_timeleft")
_WDIOC_SETOPTIONS = 0x80045704      # _IOR('W', 4, int)
_WDIOS_DISABLECARD = 0x0001


def _i32(value: int) -> int:
    """ioctl 码转有符号 32 位（C int）——fcntl/ioctl 需要这个形式。"""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value

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
    """把协调器状态推给调试台（调试台没在跑就整段跳过）。

    ⚠ 这里原本**无条件** `from utils.debug_console import DebugConsole`, 两个问题:
      ① 每次调用一次模块导入（sys.modules 查找, 便宜但不是零）;
      ② 更糟的是: 若调试台从未被启动（启动器按 `debug_console_enabled: false`
         禁用了它）, 这句导入会把 **rich 整个模块拉进内存**, 只为往一个不存在的
         TUI 里塞几个键值 —— 而 CAM1 只有 128MB 用户内存（实测已占 80%）。
    先用 `debug_console_active()`（纯 sys.modules 判定）挡一道。
    """
    if not debug_console_active():
        return
    from utils.debug_console import DebugConsole
    dc = DebugConsole()
    info = coordinator.get_info()
    dc.set("state", info.get("state", "-"))
    dc.set("link_active", str(info.get("link_active", False)))
    dc.set("codes", str(info.get("codes", [0, 0, 0])))
    dc.set("pending", str(info.get("pending_count", 0)))
    dc.set("fps", f"{info.get('fps', 0.0):.1f}")
    dc.set("tx_hz", f"{info.get('tx_hz', 0.0):.1f}")


def _frame_tag(frame) -> str:
    """把一帧的"真实格式/内存来源"压缩成短标记, 用于 [PERF] show 归因。

    为什么需要（2026-10-15, MaixCDK 源码证实）:
      MaixCAM1 的 `Display::show()` 只白名单接受 FMT_RGB888 / YVU420SP /
      YUV420SP / BGRA8888 / GRAYSCALE。其它格式会先在 show() 内部走一次
      `to_format(FMT_RGB888)` **全帧分配 + 逐像素通道交换**, 而且结果落在
      CPU 堆而不是 CMM/VIPU, VO 再拷一次。
      而 `maix.image.cv2image(bgr, bgr=True)` 产出的正是 **FMT_BGR888** ——
      于是调试页每一帧都白付这两次全帧搬运。
    这就是"调试页图更小(552x368)却比 MAIN(640x480)慢 3~8 倍"的全部原因:
    成本在**格式/内存来源**, 不在像素数量。

    返回值: "RGB888" / "BGR888" / "numpy" / 类型名 —— 出现非 RGB888 即是病。
    """
    if frame is None:
        return "none"
    try:
        import maix.image as _mi
        fmt_fn = getattr(frame, "format", None)
        if callable(fmt_fn):
            fmt = fmt_fn()
            names = getattr(_mi, "Format", None)
            for attr in dir(names) if names is not None else []:
                if attr.startswith("FMT_") and getattr(names, attr) == fmt:
                    return attr[4:]
            return str(fmt)
    except Exception:
        pass
    shape = getattr(frame, "shape", None)
    if shape is not None:
        return f"numpy{tuple(shape[2:]) or ''}"
    return type(frame).__name__


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
        self._step_fail_logged = set()
        self._laser_sender = None
        # ---- 调试页上屏节流（2026-10-15）----
        # 为什么必须在这里节流: 调试页的 disp.show() 由页面自己调（见 step_active）,
        # 而 MainUiShell._maybe_show 的 display_show_interval_ms 节流**只在 MAIN 页生效**
        #（_display_loop 对非 MAIN 页直接 continue 让位）。于是调试页上 show() 退化为
        # "每个相机帧一次" ≈ 58Hz, 比 MAIN 页的 10Hz 高约 6 倍, 且每次都走一遍 MaixCAM1
        # 的 VPU/VENC JPEG 编码（show 同时要压缩一份发给 MaixVision）。
        # 这正是「激光页 fps 低 + CVI_VENC_GetStream failed 0xc0078012 /
        # compress image failed 刷屏」的应用层成因 —— 官方 FAQ 明确: 拿不到 venc 资源
        # 就报该错, 并建议"关掉 maixvision 预览视图, 或者切换到 png 流"。
        # 语义与 MAIN 页完全一致: 同一组配置键 display_enabled / display_show_interval_ms;
        # interval == 0 回退到 _DISPLAY_SHOW_SKIP 帧跳 = 改前行为。
        self._display_enabled = True
        self._show_interval_s = 0.0
        self._last_show_time = 0.0
        self._display_counter = 0
        self._show_ms_sum = 0.0
        self._show_cpu_sum = 0.0
        self._show_n = 0
        self._show_t0 = 0.0
        self._last_show_tag = "-"
        # ---- 调试页迭代归因（2026-10-15）----
        # 为什么需要: 台架实测调试页"每轮 ~150ms, 其中 show 只占 18ms", 剩下 ~130ms
        # 完全无法归因 —— 而 MAIN 页的视觉线程有 [PERF] vision 的 read/conv/task/disp
        # 四项打点, 调试页这条路径却是黑盒。没有归因就只能靠猜, 所以先补打点再谈重构。
        # 分项语义（全部是"墙钟", 单位 ms）:
        #   read = cam.read()（含 image2cv 拷贝与负步长视图生成）
        #   prep = 构造上屏代理
        #   step = page.step() 全程（不含 show, show 由代理单独计时到 show 项）
        #   show = 真正进入 display.show() 的那一段（编码 + 上屏）
        #   iter = step_active 整体; 与 1/Hz 的真实周期对照, 差额就是"iter 之外"的开销
        self._iter = {"t0": 0.0, "n": 0, "read": 0.0, "prep": 0.0,
                      "step": 0.0, "show": 0.0, "iter": 0.0, "none": 0}

    def _note_page_iter(self, key: str, dt_s: float) -> None:
        self._iter[key] += dt_s * 1000.0

    def _flush_page_iter(self, page_name: str) -> None:
        """每秒一条 [PERF] page: 把调试页一轮的耗时拆到 read/prep/step/show。

        读法: iter 是 step_active 自身耗时, 1/Hz 是真实周期,
        `iter外 = 周期 - iter` = 主循环里**非调试页**的开销
        （coordinator.loop + 视觉线程抢 GIL + display_callback + MAIN_LOOP_DELAY）——
        这一项才是"每轮还有 130ms"的答案。
        """
        it = self._iter
        if it["t0"] == 0.0:
            it["t0"] = time.monotonic()
            return
        elapsed = time.monotonic() - it["t0"]
        if elapsed < 1.0 or it["n"] == 0:
            return
        n = it["n"]
        iter_ms = it["iter"] / n
        period_ms = elapsed / n * 1000.0
        show_ms = it["show"] / n
        log_print(
            f"[PERF] page[{page_name}] {n / elapsed:.1f}Hz period={period_ms:.1f}ms | "
            f"read={it['read'] / n:.1f} prep={it['prep'] / n:.1f} "
            f"step={it['step'] / n:.1f}(含show {show_ms:.1f}) "
            f"iter={iter_ms:.1f} none={it['none']} | "
            f"iter外={period_ms - iter_ms:.1f}ms(主循环+视觉抢GIL)")
        self._iter = {"t0": time.monotonic(), "n": 0, "read": 0.0, "prep": 0.0,
                      "step": 0.0, "show": 0.0, "iter": 0.0, "none": 0}

    def set_display_policy(self, enabled=None, show_interval_ms=None) -> None:
        """调试页上屏策略（与 MainUiShell 同名同语义, 由 main() 注入同一组配置键）。

        为什么两个类各持一份状态而不共享: MAIN 页与调试页的调用方线程和节奏不同,
        共享 _last_show_time 会让"上一页的节流相位"漏到下一页, 表现为进页第一帧被吞。
        语义必须一致, 状态不能共享。
        """
        if enabled is not None:
            self._display_enabled = bool(enabled)
        if show_interval_ms is not None:
            self._show_interval_s = max(0.0, float(show_interval_ms) / 1000.0)

    def _page_display(self, ts=None):
        """返回给 page.step() 用的 display 代理（show 被节流, 其余透传）。

        每次现建一个极轻的代理对象（无状态, 状态都在 factory 上）, 避免
        display 实例被页面长期持有而在切页后继续用旧策略上屏。
        """
        return _ThrottledPageDisplay(self, self._machine.display)

    def _reset_show_clock(self) -> None:
        """进页/回 MAIN 时清节流相位: 新页首帧必须上屏（用户据此判断"进没进去"）。"""
        self._last_show_time = 0.0
        self._display_counter = 0

    def _maybe_show(self, show_fn, frame=None) -> bool:
        """按策略决定这一拍要不要真的上屏; 返回是否执行了 show_fn。

        与 MainUiShell._maybe_show 的四条语义逐一对应（见 tests/test_display_policy.py）:
          1. display_enabled=False -> 一次都不上屏;
          2. interval > 0 -> 按**时间**限流, 与帧率无关;
          3. interval == 0 -> 回退"每 _DISPLAY_SHOW_SKIP 帧一次"帧跳;
          4. "有无新帧"由调用方判（调试页每次都是刚读到的帧）。
        另按秒汇总 show 的墙上/CPU 开销 —— 判断 VENC 编码是否成为瓶颈的唯一依据。
        """
        if not self._display_enabled:
            return False
        now = time.monotonic()
        if self._show_interval_s > 0.0:
            if now - self._last_show_time < self._show_interval_s:
                return False
        else:
            self._display_counter += 1
            if self._display_counter % _DISPLAY_SHOW_SKIP != 0:
                return False
        t0 = time.perf_counter()
        cpu0 = _thread_cpu()
        try:
            show_fn()
        except Exception as e:
            if "show" not in self._step_fail_logged:
                self._step_fail_logged.add("show")
                log_print(f"[PAGE] display.show FAIL: {type(e).__name__}: {e}")
        self._last_show_time = now
        _wall_ms = (time.perf_counter() - t0) * 1000.0
        self._iter["show"] += _wall_ms
        self._last_show_tag = _frame_tag(frame)
        self._note_show_perf(_wall_ms, (_thread_cpu() - cpu0) * 1000.0)
        return True

    def _note_show_perf(self, wall_ms: float, cpu_ms: float = 0.0) -> None:
        """调试页每秒一条 [PERF] page show（与 MAIN 页 [PERF] lcd show 同口径）。"""
        self._show_ms_sum += wall_ms
        self._show_cpu_sum += cpu_ms
        self._show_n += 1
        now = time.monotonic()
        if self._show_t0 == 0.0:
            self._show_t0 = now
        elapsed = now - self._show_t0
        if elapsed < 1.0:
            return
        n = max(1, self._show_n)
        log_print(f"[PERF] page show {n / elapsed:.1f}Hz "
                  f"avg={self._show_ms_sum / n:.1f}ms "
                  f"(cpu {self._show_cpu_sum / n:.1f}ms) "
                  f"fmt={self._last_show_tag} "
                  f"-> 每秒占墙上 {self._show_ms_sum / elapsed:.0f}ms / "
                  f"CPU {self._show_cpu_sum / elapsed:.0f}ms")
        self._show_t0 = now
        self._show_ms_sum = 0.0
        self._show_cpu_sum = 0.0
        self._show_n = 0

    def set_laser_sender(self, sender) -> None:
        """注入激光帧发送回调（main() 装配 UART 后传入, LASER 页专用）。"""
        self._laser_sender = sender

    def _leave_page(self, page_name: str) -> None:
        """页面被切走时通知其收尾（LASER 页据此自动关激光）。"""
        page = self._instances.get(page_name)
        if page is not None and hasattr(page, "on_leave"):
            try:
                page.on_leave()
            except Exception as e:
                log_print(f"[PAGE] {page_name} on_leave FAIL: {e}")

    def send_laser_off(self) -> None:
        """程序退出兜底: LASER 页开过激光才发关帧（幂等, os._exit 前手动调）。"""
        page = self._instances.get("LASER")
        if page is None or not getattr(page, "laser_is_on", lambda: False)():
            return
        self._leave_page("LASER")

    def ensure_entered(self, page_name: str) -> None:
        if self._active == page_name:
            return
        if self._active is not None:
            self._leave_page(self._active)
            self._active = None
        page = self._instances.get(page_name)
        if page is None:
            page = self._create(page_name)
            if page is None:
                log_print(f"[PAGE] {page_name} build failed -> main")
                self._pages.select(STATE_MAIN)
                return
            self._instances[page_name] = page
            log_print(f"[PAGE] built debug page: {page_name}")
        # 进页清节流相位: 上一页残留的 _last_show_time 会把新页首帧吞掉,
        # 而"进页第一帧"正是用户确认页面切过来了的依据。
        self._reset_show_clock()
        # begin() 与 step() 同一条静默链路（tick 兜底只写 app.log）: 抛异常时
        # 同样留痕 + 回 MAIN, 避免页面卡在"建好了但永远渲染不出来"的冻屏态。
        try:
            page.begin()
        except Exception as e:
            self._instances.pop(page_name, None)   # 下次切入允许重建
            log_print(f"[PAGE] {page_name} begin FAIL: "
                      f"{type(e).__name__}: {e} -> 回 MAIN（详见 logs/app.log）")
            self._pages.select(STATE_MAIN)
            return
        self._active = page_name

    def step_active(self, page_name: str) -> None:
        _t_iter = time.perf_counter()
        try:
            self._step_active_inner(page_name)
        finally:
            self._iter["iter"] += (time.perf_counter() - _t_iter) * 1000.0
            self._iter["n"] += 1
            self._flush_page_iter(page_name)

    def _step_active_inner(self, page_name: str) -> None:
        page = self._instances.get(page_name)
        ts = self._ensure_touchscreen()
        cam = self._vm.get_pipeline_camera() if self._vm else None
        if page is None or cam is None or ts is None:
            log_print(f"[PAGE] {page_name} unavailable, "
                      f"touch={ts is not None} cam={cam is not None} -> main")
            self._pages.select(STATE_MAIN)
            self._leave_page(page_name)
            self._active = None
            return
        _t_read = time.perf_counter()
        frame_bgr = cam.read()
        self._note_page_iter("read", time.perf_counter() - _t_read)
        if frame_bgr is None:
            # 停帧/取帧失败：**必须退让**, 否则这里会以主循环上限 500Hz 空转
            # （MAIN_LOOP_DELAY=2ms 是唯一节拍）, 白吃 CPU 且与视觉线程抢 GIL,
            # 而且控制台一行提示都没有 —— 用户只看到 LCD 冻住。
            # 节拍取 5ms: 远快于任何相机帧率, 又比 2ms 自旋少一半唤醒。
            self._iter["none"] += 1
            time.sleep(0.005)
            return
        # ⚠ 页面渲染抛异常绝不能静默: run_main_loop 的 tick_callback 兜底只把
        # traceback 写进 logs/app.log（控制台看不见）, 而显示线程对非 MAIN 页
        # 整体让位 —— 结果就是"切到调试页 LCD 直接冻住, 控制台毫无声息"
        # （2026-10-09 真机现象, 根因是 LazyFrame 缺 .copy()）。这里改为:
        # 控制台留痕一次 + 自动回 MAIN 页恢复显示, 别让用户对着一帧死画面猜。
        try:
            # 上屏节流: 页面 step() 的最后一句是 disp.show()。这里交给一个
            # **只拦 show 的代理**, 而不是把整个 step() 包进节流里 ——
            # step() 里还有 ts.read() 与识别，必须每帧照跑（否则触摸/退出失效，
            # 页面直接变成死的）。代理把 show 按 display_enabled /
            # display_show_interval_ms 与 MAIN 页同口径限流, 顺带按秒统计开销。
            _t_pre = time.perf_counter()
            _disp = self._page_display(ts)
            self._note_page_iter("prep", time.perf_counter() - _t_pre)
            _t_step = time.perf_counter()
            try:
                page.step(_disp, ts, frame_bgr)
            finally:
                self._iter["step"] += (time.perf_counter() - _t_step) * 1000.0
        except Exception as e:
            if page_name not in self._step_fail_logged:
                self._step_fail_logged.add(page_name)
                log_print(f"[PAGE] {page_name} step FAIL: "
                          f"{type(e).__name__}: {e} -> 回 MAIN（详见 logs/app.log）")
            self._pages.select(STATE_MAIN)
            self._leave_page(page_name)
            self._active = None
            return
        if getattr(page, "exit_requested", lambda: False)():
            self._pages.select(STATE_MAIN)
            self._leave_page(page_name)
            self._active = None
            log_print("[KEY] debug page Exit -> main, vision resume")

    def page_active(self) -> bool:
        return self._active is not None

    def reset(self) -> None:
        """回到 MAIN 页时清掉"当前激活的调试页"标记（install 见 MainUiShell.tick）。

        语义: 页面**实例**留在 _instances 里（共享会话数据不丢, 见类 docstring）,
        只清激活标记, 于是下次切入同一页时 ensure_entered 会重新走一遍 begin()
        （重置滑条等交互态, 不重建页面）。

        ⚠⚠ 这个方法此前**根本不存在**, 而 MainUiShell.tick() 在 MAIN 页每一拍都在
        调它 —— 于是每一拍都抛 AttributeError, 被 ModuleManager.run_main_loop 的
        `except Exception: logger.exception("tick_callback() failed")` 吞掉。
        两个后果都不报错, 只能靠打点发现:

          ① 每拍格式化一次**完整 traceback** 再写盘。台架实测 `[PERF] mainloop`
             里 tick=27.7ms, 而 coord=0.17 / touch=0.17 / wdt=0.04ms —— 主循环
             99% 的耗时都堆在这个回调上, 主因就是它（traceback 格式化是纯 Python
             字符串工作, 正好对上"CPU 不高但墙上很高"）。
          ② 紧随其后的 `_push_coordinator_status()` **永远执行不到** ——
             DebugConsole 状态栏一直是 "-", 台架在控制台上看不到任何状态。
             也就是说: 这个 bug 让"给台架看状态"的功能整个失效, 同时还在拖慢主循环。

        修法取最小语义(清激活标记), 与 step_active 里页面自请求退出时的
        `self._active = None` 一致; 重复清是幂等的。
        2026-10-15: 清标记前先 _leave_page 通知旧页收尾（LASER 页关激光）。
        """
        if self._active is not None:
            self._leave_page(self._active)
        self._active = None
        self._reset_show_clock()

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
        if page_name == "LASER":
            from modules.zw_opencv_module.debug.laser_calib_tuner import GksLaserCalibTuner
            return GksLaserCalibTuner(
                get_processor=self._get_gks_processor,
                send_laser=self._laser_sender or (lambda on: False),
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


class _ThrottledPageDisplay:
    """调试页专用的 display 代理: **只拦 show()**, 其余接口原样转发。

    为什么不能直接节流整个 step(): 调试页的 step() 里既有识别也有 ts.read(),
    触摸退出判定就在 show 之前 —— 一票否决 step 会让页面既看不出画面也没法退出。
    """

    def __init__(self, factory, real):
        self._factory = factory
        self._real = real

    def show(self, frame, *args, **kwargs):
        return self._factory._maybe_show(
            lambda: self._real.show(frame, *args, **kwargs), frame=frame)

    def width(self):
        return self._real.width()

    def height(self):
        return self._real.height()

    def __getattr__(self, name):
        # 其余接口（close/set_backlight/...）一律透传给真实 display
        return getattr(self._real, name)


class MainUiShell:
    """主循环 tick / MAIN 页触摸退出判定 / 显示线程。

    MAIN 页 tick 只推 DebugConsole 状态; display_callback 读触摸判定
    退出图标; 显示线程每 _DISPLAY_SHOW_SKIP 帧 show 一次（CMM 压力跳帧）。
    非 MAIN 页所有路径短路, 与调试页触摸/显示互不干扰。
    """

    # 向 DebugConsole 灌状态的周期（秒）。
    # ⚠ 必须限频: 台架真机实测 `[PERF] mainloop ... tick=27.7ms`（对比
    #   coord=0.17ms / touch=0.17ms / wdt=0.04ms）—— 主循环 99% 的阻塞都发生在
    #   这个 tick 回调里, 而整个主循环的 CPU 只有 205ms/s, 说明它不是在算, 是在**等**
    #   （被 show() 与 rich 重绘持着 GIL）。功能上它只是喂一个 10Hz 的文字控制台,
    #   按主循环 2ms 的节奏喂=每秒喂几百次, 没有任何收益。
    #   0.5s => 2Hz, 肉眼对控制台读数无感, 争用点少一个数量级。
    _STATUS_PUSH_INTERVAL_S = 0.5

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
        self._last_status_push = 0.0

        # ---- 显示输出策略（只影响 LCD, 不影响日志与协议推流）----
        # `project_config.yaml: display_enabled` 此前是个**死配置**: 全仓库只有
        # 定义那一行, 没有任何代码读它 —— 于是"关了显示"只是心理安慰, 显示线程
        # 照旧每 2 帧 show 一次。2026-10-08 台架实测 CAM1 `display.show()` 单次
        # **23.4ms**（C++ 编码期间全程持 GIL）, 每 2 帧一次 ≈ 每秒 117ms 的 GIL
        # 独占, 与视觉线程直接串行 —— 这是"识别链路单独跑 59.7Hz, 带显示的真机
        # 只有 20.3Hz"的首位嫌疑。
        self._display_enabled = True
        self._show_interval_s = 0.0        # 0 = 沿用 _DISPLAY_SHOW_SKIP 帧跳
        self._last_show_time = 0.0
        self._cmm_check_time = 0.0
        self._cmm_pressure = False
        self._show_ms_sum = 0.0
        self._show_cpu_sum = 0.0
        self._show_n = 0
        self._show_t0 = 0.0
        self._last_show_tag = "-"

    def set_display_policy(self, enabled=None, show_interval_ms=None) -> None:
        """注入显示策略（main() 从 project_config.yaml 读取）。

        *show_interval_ms*: 两次 show 的最小间隔。0 = 保留原来的"每
        _DISPLAY_SHOW_SKIP 帧一次"帧跳语义（等价改前行为）。帧跳按**帧数**跳,
        帧率一变节流比例就变; 按时间限流则与帧率无关, 台架更好预测。
        """
        if enabled is not None:
            self._display_enabled = bool(enabled)
        if show_interval_ms is not None:
            self._show_interval_s = max(0.0, float(show_interval_ms) / 1000.0)
        if self._show_interval_s > 0.0:
            throttle = f"{self._show_interval_s * 1000:.0f}ms 限流"
        else:
            throttle = f"每 {_DISPLAY_SHOW_SKIP} 帧一次(帧跳)"
        log_print(f"[DISPLAY] lcd={'ON' if self._display_enabled else 'OFF'}, {throttle}")
        # HUD 绘制与 show 用同一间隔 —— 否则视觉线程每帧画、只 show 其中一部分,
        # 中间那些绘制是白做（CAM1 纯软件渲染, TARGET 还多 6 个误差刻度标签）。
        if self._vm is not None and hasattr(self._vm, "set_hud_draw_interval"):
            self._vm.set_hud_draw_interval(show_interval_ms)

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
            # 限频: 见 _STATUS_PUSH_INTERVAL_S 的说明（这是主循环最大的阻塞源）
            now = time.monotonic()
            if now - self._last_status_push >= self._STATUS_PUSH_INTERVAL_S:
                self._last_status_push = now
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
                self._maybe_show(frame)
            # 原为 1ms 自旋: 每秒 1000 次唤醒抢 GIL, 对视觉线程是纯干扰。
            # 显示侧只需要"有新帧就上屏", 5ms 轮询(200Hz) 远快于任何帧率。
            time.sleep(0.005)

    def _maybe_show(self, frame) -> None:
        """按显示策略决定是否真的上屏, 并统计 show() 的 GIL 开销。"""
        if not self._display_enabled:
            return
        now = time.monotonic()
        if self._show_interval_s > 0.0:
            if now - self._last_show_time < self._show_interval_s:
                return
        elif self._display_counter % _DISPLAY_SHOW_SKIP != 0:
            return
        if self._cmm_pressure_throttled():
            return
        t0 = time.perf_counter()
        cpu0 = _thread_cpu()
        try:
            self._machine.display.show(frame)
        except Exception:
            pass
        self._last_show_time = now
        self._last_show_tag = _frame_tag(frame)
        self._note_show_perf((time.perf_counter() - t0) * 1000.0,
                            (_thread_cpu() - cpu0) * 1000.0)

    def _cmm_pressure_throttled(self) -> bool:
        """CMM 压力查询降频到 1s 一次。

        原实现每 2 帧调一次 maix_sys.memory_info()（一次系统调用 + 建 dict）,
        在 60fps 下就是每秒 30 次无谓开销; 内存压力本来也不是秒级突变。
        """
        now = time.monotonic()
        if now - self._cmm_check_time >= 1.0:
            self._cmm_check_time = now
            self._cmm_pressure = _check_cmm_pressure()
        return self._cmm_pressure

    def _note_show_perf(self, wall_ms: float, cpu_ms: float = 0.0) -> None:
        """每秒汇总一条 show() 开销 —— 直接量化"显示吃掉多少 GIL / CPU"。

        ⚠ 两个数都必须报, 且**单位别乘错**:
        · 墙上 ms/s = 本秒内 show() 占用的墙上时间合计（含被 GIL 挡住的等待）
        · CPU ms/s  = 本秒内 show() 真正消耗的 CPU（thread_time 差值）
        首版实现把已经是"毫秒"的量又乘了 1000, 打出 `每秒占用 164818ms`,
        比一秒还长 —— 读者会直接当成 bug 而忽略整行。真实值是 ~164ms/s。
        """
        self._show_ms_sum += wall_ms
        self._show_cpu_sum += cpu_ms
        self._show_n += 1
        now = time.monotonic()
        if self._show_t0 == 0.0:
            self._show_t0 = now
        elapsed = now - self._show_t0
        if elapsed < 1.0:
            return
        n = max(1, self._show_n)
        log_print(f"[PERF] lcd show {n / elapsed:.1f}Hz "
                  f"avg={self._show_ms_sum / n:.1f}ms "
                  f"(cpu {self._show_cpu_sum / n:.1f}ms) "
                  f"fmt={self._last_show_tag} "
                  f"-> 每秒占墙上 {self._show_ms_sum / elapsed:.0f}ms / "
                  f"CPU {self._show_cpu_sum / elapsed:.0f}ms")
        self._show_t0 = now
        self._show_ms_sum = 0.0
        self._show_cpu_sum = 0.0
        self._show_n = 0

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
                    # ⚠ `os._exit(0)` 跳过所有 finally/atexit —— 必须**先**手动解除
                    # 硬件看门狗, 否则用户按"退出"看到的是一次设备重启
                    if self._factory is not None:
                        self._factory.send_laser_off()   # 同理跳过 finally, 激光先关
                    _wdt_stop()
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


def _make_wdt_feed(enabled: bool = True):
    """返回喂狗回调；*enabled*=False 时**完全不武装**（CAM1 默认, 见 _WDT_ENABLED_DEFAULT）。

    ⚠ 顺序很重要: `enabled=False` 必须在**构造 WDT 之前**返回 —— 一旦
    `maix_wdt.WDT(...)` 被构造出来, 硬件倒计时就开始跑, 而 MaixPy 的 `stop()` 是空实现
    （见 _WDT_ENABLED_DEFAULT 的注释）, 之后**没有任何用户态办法叫停它**（nowayout 时
    连 raw ioctl 都会被内核拒绝）。所以"不重启"这件事只能在"别武装"这一步做。

    # CAM1(MaixCAM-Pro) 的模块布局是 `maix.wdt`，CAM2 才是 `maix.peripheral.wdt`
    # （docs/doc/zh/peripheral/wdt.md:15 官方示例 `from maix import wdt`）。
    # 两条路径都试：只写 CAM2 那条会让 CAM1 上 ImportError 被下面的 except 吞掉，
    # 看门狗**静默失效** —— 应用照常跑、协议照常通，只有日志一行不起眼的 FAIL，
    # 视觉/串口线程死锁时没有硬件复位兜底，现场表现为"设备卡死需断电"。
    """
    if not enabled:
        log_print("[WDT] 未武装（wdt_enabled=false）—— 退出不会触发硬件复位；"
                  "代价: 线程死锁时没有自动复位兜底（需兜底请在 project_config.yaml "
                  "打开 wdt_enabled，但退出后 ~10s 设备会重启）")
        return lambda: None

    maix_wdt = None
    try:
        from maix import wdt as _wdt_mod
        maix_wdt = _wdt_mod
    except (ImportError, ModuleNotFoundError):
        try:
            from maix.peripheral import wdt as _wdt_mod
            maix_wdt = _wdt_mod
        except (ImportError, ModuleNotFoundError):
            maix_wdt = None

    try:
        if maix_wdt is None:
            raise ImportError("neither maix.wdt nor maix.peripheral.wdt available")
        watchdog = maix_wdt.WDT(0, _WDT_WINDOW_MS)
        import time
        # 多处 feed 入口(main/coord/viz)共享同一节流时钟, 全局最多 1 次/周期,
        # 避免各自独立计时导致 C 层 WDT::feed() 刷屏 (窗口 10s 足够余量)
        if "_WDT_SHARED_STATE" not in globals():
            globals()["_WDT_SHARED_STATE"] = {
                "last": 0.0, "fails": 0, "wdt": watchdog, "stopped": False}
        _state = globals()["_WDT_SHARED_STATE"]
        _state["wdt"] = watchdog
        _state["stopped"] = False

        def _feed():
            if _state.get("stopped"):
                return          # 已 disarm（退出中），再 feed 会把停掉的狗重新武装
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
        # 看门狗失效属于"带病上台架"级别，措辞上与普通降级区分开
        log_print(f"[WDT] init FAIL（硬件看门狗未生效，线程死锁将无自动复位）: {e}")
        return lambda: None


def _wdt_timeleft_s():
    """内核报告的看门狗剩余秒数；读不到（PC 端 / 无该 sysfs）返回 None。

    ⚠ 属性名并非总是 `timeleft`（2026-10-09 设备上就是读不到的）, 所以逐个试。
    """
    for name in _WATCHDOG_TIMELEFT_FILES:
        try:
            with open(f"{_WATCHDOG_SYSFS}/{name}") as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None


def _wdt_armed(sample_gap_s: float = 0.4):
    """采样两次 timeleft: 递减 ⇒ 倒计时仍在跑(= 没能关掉)。None = 判不了。

    这是绕过 MaixPy 那个**假 stop()** 的唯一验证手段: stop() 返回 0 却什么都不做,
    只有内核自己的倒计时读数能说明真相。
    """
    a = _wdt_timeleft_s()
    if a is None:
        return None
    time.sleep(sample_gap_s)
    b = _wdt_timeleft_s()
    if b is None:
        return None
    return b < a


def _wdt_disarm_raw() -> str:
    """绕过 MaixPy, 直接对 /dev/watchdog 下 `WDIOS_DISABLECARD`（stop() 本该做的事）。

    返回人类可读结果串（只进日志, 不抛）。带 nowayout 的驱动会回 EBUSY(16) ——
    那是内核层面禁止停止看门狗, 此时**没有任何用户态办法能关**, 只能"别武装"。

    ⚠ 两个实测坑（2026-10-09）: ① open 必须 O_NONBLOCK, 否则已启动时挂死;
    ② ioctl 码必须传有符号 32 位, 否则 EFAULT(14)。
    """
    try:
        import fcntl
        import struct
        flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(_WATCHDOG_DEV, flags)
    except Exception as e:
        return f"open {_WATCHDOG_DEV} 失败: {e!r}"
    try:
        buf = bytearray(struct.pack("i", 0))
        fcntl.ioctl(fd, _i32(_WDIOC_SETOPTIONS), buf, True)
        return "DISABLECARD OK"
    except OSError as e:
        extra = "（nowayout: 内核禁止停止看门狗）" if e.errno == 16 else ""
        return f"DISABLECARD 失败 errno={e.errno} {e.strerror}{extra}"
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _wdt_stop() -> None:
    """退出前尽力解除硬件看门狗 —— 并**如实回报**到底解开了没有。

    ⚠⚠ 2026-10-09 二次修正。第一版只调了 `wdt.stop()` 就打印 "stopped", 台架
    反馈**照样重启**。原因是 MaixCDK 的实现
    （`components/peripheral/port/linux_common/maix_wdt.cpp`）:

        int WDT::stop() { log::warn("this operation is not supported"); return 0; }
        int WDT::restart() { 同上, 也是空实现 }

    **空实现**: 什么都不做却 `return 0`(=ERR_NONE)。调用方从返回值分辨不出,
    所以第一版日志是一句善意的谎话。现在的流程是"调用 + **验证**":

      1. 调 `maix stop()`（记下 rc, 但不再信它）;
      2. 采样 `/sys/class/watchdog/watchdog0/timeleft` 两次: 递减 ⇒ 还武装着;
      3. 还武装着 ⇒ 绕过 MaixPy 直接 raw ioctl `WDIOS_DISABLECARD`;
      4. 再采样一次: 仍递减 ⇒ **明确告警**（写明 ~Ns 后设备会重启 + 规避办法）,
         绝不再输出"已解除"这种会被当成结论的日志。

    根治办法见 `_WDT_ENABLED_DEFAULT` 的注释: **CAM1 上默认不武装**。本函数保留
    是因为 `wdt_enabled: true` 时若固件支持 DISABLECARD, 退出仍能干净收场; 同时它
    把"这台机器到底能不能关看门狗"的答案直接打进日志, 台架一眼可见。
    """
    state = globals().get("_WDT_SHARED_STATE")
    if not state or state.get("stopped"):
        return
    state["stopped"] = True     # 先置位: 让所有 _feed 入口立刻失效

    wdt = state.get("wdt")
    rc = None
    if wdt is not None:
        try:
            rc = wdt.stop()
        except Exception as e:
            log_print(f"[WDT] maix stop() 抛异常: {e!r}")

    if _wdt_armed() is False:
        log_print(f"[WDT] disarm 已验证: 倒计时已停（maix stop() rc={rc}）")
        return

    detail = _wdt_disarm_raw()
    armed = _wdt_armed()
    if armed is False:
        log_print(f"[WDT] disarm 已验证: maix stop() 是空实现（rc={rc}）, "
                  f"改用 /dev/watchdog 的 {detail} 真正关掉了")
        return
    if armed is None:
        log_print(f"[WDT] disarm 结论未知: maix stop() rc={rc}（空实现, 不可信）; "
                  f"raw {detail}; 本机读不到 {_WATCHDOG_SYSFS}/timeleft, 无法验证。"
                  f"wdt_enabled=true 时请用 scripts/wdt_probe.py 判定")
        return
    log_print(
        f"[WDT] ⚠⚠ disarm 失败: 看门狗仍在跑, **退出后约 {_WDT_WINDOW_MS // 1000}s "
        f"设备会重启**。原因: MaixPy/MaixCDK 的 WDT::stop() 是空实现（打 "
        f"'this operation is not supported' 却 return 0）; raw {detail}。"
        f"规避: project_config.yaml 设 wdt_enabled: false")



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


def _apply_render_policy(vision_manager, config: dict) -> None:
    """把 `competition_mode:` 注入 VisionManager（缺省即渲染全开 = 台架行为）。

    比赛模式关掉的是**显示侧**每帧开销: HUD 约 20 次 maix draw_string/draw_rect
    （CAM1 纯软件渲染, TARGET 模式额外 6 个误差刻度标签）+ wifi JPEG 编码, 都在
    视觉线程取帧→处理→推流的同一拍里串行跑。去年 main.py task2 只画 1 个框 +
    1 行字, 当前实现多出的这些直接表现为打靶误差滞后。

    **日志与协议推流完全不受影响**: log_print 走独立写盘线程, coordinator 的
    UART DATA_STREAM 走主循环, 两者都不经过渲染路径。台架验收看
    `[FPS] pipeline` / `[UART TX] DATA_STREAM` 的 Hz 是否上抬, 日志照旧可查。
    """
    if vision_manager is None:
        return
    mode = bool((config.get("competition_mode")) or {})
    vision_manager.set_render_enabled(not mode)


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


def _apply_perf_log_config(cfg) -> None:
    """把 `perf_log:` 配置应用到 log_util 层的闸门。

    两种写法都认（老配置不改也不炸）:
        perf_log: true / false                     # 布尔: 全开 / 按默认组静音
        perf_log: {enabled: false, names: [...]}   # 映射: 可自定义静音标签
    缺键/解析异常 -> 保持默认（全开, 即改前行为）。
    """
    from utils.log_util import set_perf_log
    try:
        if isinstance(cfg, bool):
            set_perf_log(cfg)
            enabled, names = cfg, None
        elif isinstance(cfg, dict):
            enabled = bool(cfg.get("enabled", True))
            names = cfg.get("names")
            set_perf_log(enabled, names)
        else:
            return                      # 缺键/None -> 用模块默认（全开）
    except Exception as e:
        log_print(f"[PERFLOG] 配置应用失败, 保持全开: {type(e).__name__}: {e}")
        return
    log_print(f"[PERFLOG] perf_log={'ON' if enabled else 'MUTED'} "
              f"names={list(names) if names else 'default'}")


def main():
    log_print("0xfb709394")

    config = _project_config()
    start_log_writer()

    # ---- 高频性能日志闸门（project_config.yaml `perf_log:`）----
    # 必须在 `log_print("0xfb709394")`/DCONSOLE 之外最早生效: 闸门在 log_util 层,
    # 设完之后下面所有线程的 `[PERF]`/`[MEM]`/`[WDT]`/`[UART TX] CMD_ACK` 等
    # 高频行都被吞掉, 台架调试只剩事件类日志。
    # 兼容旧的布尔写法 (`perf_log: true/false`) 与新的映射写法。
    _apply_perf_log_config(config.get("perf_log"))

    # ---- 调试台（rich TUI）----
    # 决策权在 **run.py::_init_debug_console()**: 它在导入 app.main 之前读
    # `debug_console_enabled`, 禁用时**连 rich 都不导入**（该模块顶层 import rich,
    # 而 rich 是纯 Python 大包；CAM1 只有 128MB 用户内存, 实测已占 80%）。
    # 这里**不再无条件 `from utils.debug_console import DebugConsole`** ——
    # 那会把 rich 拉回来, 让"禁用"只剩一半效果。
    # 用 log_util.debug_console_active() 廉价判定；只有"配置说关、但它确实已经被
    # 导入并启动过"这种不一致情况才需要在这里补一次真正生效的关闭。
    if config.get("debug_console_enabled", True):
        dc_mod = sys.modules.get("utils.debug_console")
        if dc_mod is not None:
            dc_mod.DebugConsole.set_global_enabled(True)
    else:
        dc_mod = sys.modules.get("utils.debug_console")
        if dc_mod is not None:
            try:
                dc_mod.DebugConsole.set_global_enabled(False)
                dc_mod.DebugConsole().stop()
                log_print("[DCONSOLE] 按配置关闭（已导入过, 这里补一次真正生效的停止）")
            except Exception as e:
                log_print(f"[DCONSOLE] 关闭失败: {type(e).__name__}: {e}")

    # 看门狗是否武装: 由 project_config.yaml `wdt_enabled` 决定（缺键用默认,
    # CAM1 默认 **False** —— MaixPy 的 stop() 是空实现, 武装了就关不掉, 退出即重启。
    # 详见 _WDT_ENABLED_DEFAULT 的注释与 scripts/wdt_probe.py）。
    wdt_feed = _make_wdt_feed(bool(config.get("wdt_enabled", _WDT_ENABLED_DEFAULT)))

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

    # 激光帧发送回调（LASER 调试页专用; 0x25 CMD_LASER, 无应答）。
    def _send_laser(on: bool) -> bool:
        if not uart_interface:
            return False
        from modules.zw_uart_module.protocol import build_cmd_laser_frame
        return uart_interface.send_raw(build_cmd_laser_frame(on))

    coordinator.set_ai(machine.ai)
    _apply_stream_policy(coordinator, config)
    _apply_render_policy(vision_manager, config)
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
    factory.set_laser_sender(_send_laser)
    # 调试页上屏与 MAIN 页同口径（同一组配置键）; 缺键即保持改前行为
    factory.set_display_policy(config.get("display_enabled"),
                               config.get("display_show_interval_ms"))
    ui = MainUiShell(machine, pages, factory, vision_manager)
    # 显示策略与分项计时的配置入口（缺键即保持改前行为, 可一键回退）
    ui.set_display_policy(config.get("display_enabled"),
                          config.get("display_show_interval_ms"))
    if vision_manager is not None:
        # [PERF] vision 的静音统一由 log_util 层的 perf_log 闸门处理（见
        # _apply_perf_log_config）。这里保留 set_perf_log(True) 作为**兜底**:
        # 万一闸门没被配置（缺键）, 视觉线程的分项计时照旧输出, 不会因缺配置变哑。
        vision_manager.set_perf_log(True)

    coordinator.start()
    ui.start_display_thread()
    try:
        manager.run_main_loop(
            coordinator,
            tick_callback=ui.tick,
            display_callback=ui.display_callback,
        )
    finally:
        factory.send_laser_off()   # 激光兜底关（os._exit 路径已单独处理）
        # ⚠ **必须在任何退出路径上 disarm 硬件看门狗**（含异常路径与
        # `app.need_exit()` 正常收工）。漏掉这一步 = 每次退出后 ~10s 设备重启,
        # 详见 _wdt_stop 的说明。
        _wdt_stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        _wdt_stop()
        sys.exit(1)
