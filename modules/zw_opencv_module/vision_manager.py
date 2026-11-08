from __future__ import annotations

import logging
import os
import time
import traceback
from collections import deque
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import yaml

from framework.hal.camera_hub import CameraHub
from framework.display.sources import DisplaySource
from framework.hal.interface import AIInference
from utils.boot_timing import thread_cpu as _thread_cpu
from utils.log_util import log_print, perf_note, perf_t0, perf_interval_s

from .pipeline_camera import PipelineCamera
from .performance import profiler

profiler._enabled = False

_module_dir = os.path.dirname(__file__)


class VisionManager:
    def __init__(self, camera_hub: CameraHub, config_path: str = None, ai: Optional[AIInference] = None) -> None:
        self._hub = camera_hub
        self._config_path = config_path or os.path.join(
            _module_dir, "config", "vision_config.yaml"
        )
        self._pipelines: Dict[str, PipelineCamera] = {}
        self._ai = ai
        self._hud = None
        self._running = False
        self._process_thread: Optional[Thread] = None
        self._result_callbacks: List[Callable[[Dict], None]] = []
        self._event_bus = None

        self._fps_data: Dict[str, dict] = {}

        self._any_fresh = False
        self._pending_results: deque = deque(maxlen=5)
        self._display_src = DisplaySource()   # seq 化发布槽: 显示侧据此判"新帧"

        self._wdt_feed = lambda: None
        self._wdt_count = 0

        self._exit_icon = None
        self._exit_icon_size: int = 0
        self._exit_icon_margin: int = 0

        # 调参页等独占相机的场景: 暂停采集处理, 让出相机成为唯一读取者
        self._paused = False

        # 比赛模式: 跳过每帧的 HUD 叠加, 只保留"把最新原始帧
        # 交给显示线程"。见 _update_display_frame。
        self._render_enabled = True

        # HUD 叠加的**绘制**节流（秒）。0 = 每帧都画（旧行为）。
        # 为什么需要: HUD 画在原始 maix Image 上, 而那张图正是 find_blobs / YOLO
        # 直接吃的 same-frame 数据。每帧画 ~20 次 maix draw_string/draw_rect
        # （CAM1 纯软件渲染, TARGET 还多 6 个刻度标签）都在视觉线程里、与识别
        # 串行 —— 画了却只 show 其中一部分时纯属白做（显示 thread 按
        # display_show_interval_ms 节流, 视觉按帧率跑, 两者不同步）。
        # 与显示侧用同一个间隔后: 画一帧、show 一帧, 不做无用绘制。
        self._hud_draw_interval_s = 0.0
        self._last_hud_draw = 0.0

        # 分项计时汇总（每 `perf_interval_s()` 秒一条 [PERF], 默认 10s）。用途见 _note_perf:
        # 真机帧周期如果明显大于 read+tasks+disp 之和, 差额就是被其它线程的
        # GIL 占用吞掉的时间 —— 该差额必须能被归因, 否则只能靠猜。
        self._perf_log = True
        self._perf = {"t0": 0.0, "n": 0,
                      "read": 0.0, "read_cpu": 0.0, "convert": 0.0,
                      "tasks": 0.0, "tasks_cpu": 0.0,
                      "disp": 0.0, "disp_cpu": 0.0,
                      "body": 0.0, "cpu": 0.0, "none": 0}

    def set_event_bus(self, bus) -> None:
        self._event_bus = bus

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def set_exit_icon(self, icon, icon_size: int = 48, margin: int = 12) -> None:
        self._exit_icon = icon
        self._exit_icon_size = icon_size
        self._exit_icon_margin = margin

    def start(self) -> None:
        if self._running:
            return

        if not os.path.exists(self._config_path):
            return

        try:
            with open(self._config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            logger.error("Failed to load vision config: %s", self._config_path)
            return

        pipelines = cfg.get("pipelines", [])
        for pipe_cfg in pipelines:
            pipeline_id = pipe_cfg.get("pipeline_id", "")
            camera_id = pipe_cfg.get("camera_id", "")
            if not pipeline_id or not camera_id:
                continue

            cam = self._hub.get(camera_id)
            if cam is None:
                continue

            focal_length_mm = getattr(cam, 'focal_length_mm', None)
            sensor_width_mm = getattr(cam, 'sensor_width_mm', None)
            sensor_height_mm = getattr(cam, 'sensor_height_mm', None)
            if focal_length_mm is None:
                logger.debug(
                    "Camera '%s' has no intrinsics; distance calculation disabled",
                    camera_id,
                )

            pipe = PipelineCamera(
                pipeline_id=pipeline_id,
                camera=cam,
                task_configs=pipe_cfg.get("tasks", []),
                focal_length_mm=focal_length_mm,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                image_width=pipe_cfg.get("width", 640),
                image_height=pipe_cfg.get("height", 480),
                ai=self._ai,
            )
            self._pipelines[pipeline_id] = pipe

        self._running = True
        self._process_thread = Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def _process_loop(self) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("vision_processing")

        while self._running:
            try:
                self._wdt_feed()
                self._wdt_count += 1
                if self._wdt_count % 600 == 0:
                    log_print(f"[WDT] viz feed #{self._wdt_count}")

                if self._paused:
                    time.sleep(0.01)
                    continue

                _wall0, _cpu0 = time.perf_counter(), _thread_cpu()
                profiler.start("total")
                _, all_results, any_fresh = self.process_all()

                # ⚠ 结果入队必须在 _update_display_frame() **之前**。
                # 台架模式(render 开)下 HUD 要画二十几次 maix draw_string/draw_rect,
                # 这段绘制原本插在"find_blobs 已出结果"和"主循环能看到结果"之间,
                # 纯粹白吃时延。顺序调换零风险, 结果内容与语义都不变。
                if self._event_bus:
                    self._pending_results.append(all_results)

                _disp_t0, _disp_c0 = time.perf_counter(), _thread_cpu()
                self._update_display_frame()
                _disp_ms = (time.perf_counter() - _disp_t0) * 1000.0
                _disp_cpu = (_thread_cpu() - _disp_c0) * 1000.0

                for cb in self._result_callbacks:
                    try:
                        cb(all_results)
                    except Exception as e:
                        pass

                profiler.stop("total")
                self._note_perf(_disp_ms, _disp_cpu, any_fresh,
                                (time.perf_counter() - _wall0) * 1000.0,
                                (_thread_cpu() - _cpu0) * 1000.0)
                time.sleep(0)
                if any_fresh:
                    profiler.end_frame()
                else:
                    time.sleep(0.001)
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

    def set_perf_log(self, enabled: bool) -> None:
        """开关每帧分项计时的 [PERF] 汇总行（默认开, 每窗口一条, 开销可忽略）。"""
        self._perf_log = bool(enabled)

    def _note_perf(self, disp_ms: float, disp_cpu: float, any_fresh: bool,
                   body_ms: float = 0.0, cpu_ms: float = 0.0) -> None:
        """累计本帧分项耗时, 每 `perf_interval_s()` 秒汇总打一条 [PERF]。

        怎么读这行（五段式, 缺一段就会读错）:
          ① loop = 帧周期（墙上）。loop - body = 视觉线程 sleep 等帧的余量。
          ② read / conv / task / disp = 分项（墙上）。
          ③ 每个分项后面的 `cpu` = 该项真正消耗的 CPU（thread_time）。
             **墙上大而 cpu 小 = 在等（硬件/GIL）; 两者都大 = 真在算。**
             这是回答"到底谁最耗性能"的唯一依据 —— 只看墙上是分不出
             "驱动调用在等硬件"和"框架在烧 CPU"的。
          ④ `自身cpu` = body_cpu - (read_cpu + tasks_cpu + disp_cpu)
             = 视觉线程这一轮里、除取帧/识别/显示之外的 CPU 开销,
             即**框架自身的编排开销**（process 包壳、结果入队、profiler、
              本打点函数自身、sleep(0) 的调度）。
             它接近 0 -> 瓶颈不在框架, 在驱动调用与 GIL, 别去重构框架;
             它很大  -> 才轮到怀疑框架本身。
          ⑤ `none=` 是**本窗口内**无新帧的帧数, 分母看行尾 `win=`（默认 10s）;
             `Hz` / `avg` / `每秒 CPU` 本身已按窗口长度归一, 口径与 1s 时代一致。
        """
        if not self._perf_log:
            return
        p = self._perf
        if p["t0"] == 0.0:
            p["t0"] = time.perf_counter()
        p["n"] += 1
        p["disp"] += disp_ms
        p["disp_cpu"] += disp_cpu
        p["body"] += body_ms
        p["cpu"] += cpu_ms
        if not any_fresh:
            p["none"] += 1
        for pipe in self._pipelines.values():
            st = getattr(pipe, "stage_ms", None)
            if st:
                p["read"] += float(st.get("read", 0.0))
                p["read_cpu"] += float(st.get("read_cpu", 0.0))
                p["convert"] += float(st.get("convert", 0.0))
                p["tasks"] += float(st.get("tasks", 0.0))
                p["tasks_cpu"] += float(st.get("tasks_cpu", 0.0))

        now = time.perf_counter()
        elapsed = now - p["t0"]
        if elapsed < perf_interval_s():
            return
        n = max(1, p["n"])
        loop_ms = elapsed / n * 1000.0
        body_ms = p["body"] / n
        cpu_ms = p["cpu"] / n
        gil_ms = body_ms - cpu_ms
        parts = (p["read"] + p["convert"] + p["tasks"] + p["disp"]) / n
        # 框架自身 CPU = 本轮 CPU 减去三个已计量分项的 CPU。
        # conv 的 CPU 天然包含在 read_cpu 里（转换发生在 read_raw 内部），不重复扣。
        own_cpu = cpu_ms - (p["read_cpu"] + p["tasks_cpu"] + p["disp_cpu"]) / n
        log_print(
            f"[PERF] vision {n / elapsed:.1f}Hz loop={loop_ms:.1f}ms "
            f"body={body_ms:.1f}ms cpu={cpu_ms:.1f}ms gil_wait={gil_ms:.1f}ms | "
            f"read={p['read'] / n:.1f}(cpu{p['read_cpu'] / n:.2f}) "
            f"conv={p['convert'] / n:.1f} "
            f"task={p['tasks'] / n:.1f}(cpu{p['tasks_cpu'] / n:.2f}) "
            f"disp={p['disp'] / n:.1f}(cpu{p['disp_cpu'] / n:.2f}) "
            f"none={p['none']} | idle={loop_ms - body_ms:.1f}ms "
            f"自身cpu={own_cpu:.2f}ms "
            f"未归因={loop_ms - parts - (loop_ms - body_ms):.1f}ms "
            f"win={elapsed:.1f}s")
        p.update(t0=now, n=0, read=0.0, read_cpu=0.0, convert=0.0, tasks=0.0,
                 tasks_cpu=0.0, disp=0.0, disp_cpu=0.0,
                 body=0.0, cpu=0.0, none=0)

    def process_all(
        self,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Dict], bool]:
        all_results: Dict[str, Dict] = {}
        any_fresh = False

        for pid, pipe in list(self._pipelines.items()):
            cur_fps = self._fps_data.get(pid, {}).get("fps", 0.0)
            frame, results = pipe.process_frame(fps=cur_fps)

            if frame is not None:
                any_fresh = True
                d = self._fps_data.setdefault(
                    pid, {"count": 0, "start": time.time(), "fps": 0.0}
                )
                d["count"] += 1
                elapsed = time.time() - d["start"]
                if elapsed >= 1.0:
                    d["fps"] = d["count"] / elapsed
                    d["count"] = 0
                    d["start"] = time.time()

            all_results[pid] = results

        return None, all_results, any_fresh

    def get_display_frame(self):
        return self._display_src.latest()

    def take_display_frame(self):
        """有新帧才返回（同一帧不重复上屏）; 显示线程用这个。"""
        return self._display_src.take_new()

    def display_seq(self) -> int:
        return self._display_src.seq()

    def set_render_enabled(self, enabled: bool) -> None:
        """比赛模式开关: False = 跳过 HUD 叠加与 JPEG 推流（推流已迁到 presenter）。

        **只影响显示, 不影响日志与协议推流。** 视觉线程里被跳过的只有
        `DisplayHUD.draw()`（约 20 次 maix draw_string/draw_rect, CAM1 纯软件
        渲染, TARGET 模式还多 6 个误差刻度标签）。
        `log_print` 走独立写盘线程, coordinator 的 UART 推流走主循环,
        两者都不经这里, 照常工作。

        保留项: ① `_display_frame` 仍每帧更新（显示线程要拿它上屏, 且触摸
        退出判定读它的宽高）; ② 退出图标仍画（2 次 draw, 比赛时误触退出
        的代价远大于这两次绘制）。可现场对照: 关掉后看 `[FPS] pipeline`
        与 `[UART TX] DATA_STREAM` 的 Hz 是否上抬。
        """
        self._render_enabled = bool(enabled)
        log_print(f"[RENDER] hud+stream {'ON' if enabled else 'OFF (competition)'}")

    def set_hud_draw_interval(self, interval_ms) -> None:
        """HUD 绘制节流（毫秒）。0/None = 每帧都画（旧行为）。

        main.py 用 `display_show_interval_ms` 同一个值调用, 保证"画的频率 ==
        show 的频率"——否则视觉线程每帧画、显示线程每 N 帧才 show, 中间那些
        绘制全是白做（CAM1 纯软件渲染, TARGET 模式还多 6 个刻度标签）。
        """
        try:
            ms = float(interval_ms) if interval_ms is not None else 0.0
        except (TypeError, ValueError):
            ms = 0.0
        self._hud_draw_interval_s = max(0.0, ms / 1000.0)

    def _update_display_frame(self) -> None:
        # 显示不依赖 AI 模型加载（色块任务无模型也须可显示）
        for pid, pipe in list(self._pipelines.items()):
            raw_img = getattr(pipe.camera, "last_raw", None)
            if raw_img is None:
                continue
            if self._render_enabled:
                # HUD 绘制按 _hud_draw_interval_s 节流（0 = 每帧）。
                # ⚠ 节流跳过绘制的帧**必须连 _display_frame 一起跳过**（只发布
                # 画过 HUD 的帧）—— 否则会出现"46Hz 发布 vs 10Hz 画 HUD",
                # 显示线程 show 到的大半是**裸帧**, 真机表现就是"顶部信息时有时无"。
                # 发布节流到与绘制同频后, show 到的每一帧都带 HUD;
                # 触摸退出判定只读宽高, 不受影响。
                _now = time.monotonic()
                if (self._hud_draw_interval_s <= 0.0
                        or _now - self._last_hud_draw >= self._hud_draw_interval_s):
                    self._last_hud_draw = _now
                    fps = self.get_pipeline_fps(pid)
                    if self._hud is not None:
                        # [PERF] hud: 原本埋在 disp= 里面, 单独拆出来才看得见绘制成本。
                        _hud_t0 = perf_t0()
                        self._hud.draw(raw_img, fps, pipe.last_results)
                        if _hud_t0:
                            perf_note("hud", (time.perf_counter() - _hud_t0) * 1000.0)
                    self._draw_exit_icon(raw_img)
                    self._display_src.publish(raw_img)
            else:
                # 比赛模式: 无 HUD 可画, 每帧照旧发布（保持"最新原始帧"语义）
                self._display_src.publish(raw_img)
            return

    def set_hud(self, hud) -> None:
        """注入显示 HUD（状态栏 + 检测叠加）; None = 只画退出图标。"""
        self._hud = hud

    def set_paused(self, paused: bool) -> None:
        """暂停/恢复采集处理（调参页独占相机时置 True）。
        暂停期间不读相机、不 produce 新结果, WDT 照常喂;
        结果 >1s 陈旧由 coordinator 的归零心跳逻辑兜底。"""
        self._paused = bool(paused)

    def get_pipeline_camera(self):
        """首个 pipeline 的 HAL 相机（供调参页等独占式页面读帧）。"""
        for _pid, pipe in self._pipelines.items():
            return pipe.camera
        return None

    def _draw_exit_icon(self, img) -> None:
        if self._exit_icon is None:
            return
        try:
            import maix.image as _mi
            h = img.height()
            by = h - self._exit_icon_size - 8
            bx = self._exit_icon_margin
            img.draw_rect(bx - 2, by - 2, self._exit_icon_size + 4,
                          self._exit_icon_size + 4, color=_mi.COLOR_BLACK, thickness=-1)
            img.draw_image(bx, by, self._exit_icon)
        except Exception:
            pass

    def drain_results(self):
        results = []
        while self._pending_results:
            results.append(self._pending_results.popleft())
        return results

    def get_pipeline_fps(self, pipeline_id: str) -> float:
        fd = self._fps_data.get(pipeline_id)
        return fd.get("fps", 0.0) if fd else 0.0

    def release_pipeline(self, pipeline_id: str) -> None:
        pipe = self._pipelines.pop(pipeline_id, None)
        if pipe is not None and hasattr(pipe.camera, 'release'):
            pipe.camera.release()

    def enable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.enable_task(task_name)
        return False

    def disable_task(self, pipeline_id: str, task_name: str) -> bool:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            return pipe.disable_task(task_name)
        return False

    def set_processor_target(self, pipeline_id: str, task_name: str, color) -> None:
        pipe = self._pipelines.get(pipeline_id)
        if pipe:
            task = pipe.get_task(task_name)
            if task and hasattr(task.processor, "set_target_color"):
                task.processor.set_target_color(color)

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineCamera]:
        return self._pipelines.get(pipeline_id)

    def get_all_results(self) -> Dict[str, Dict]:
        return {
            pid: pipe.last_results
            for pid, pipe in self._pipelines.items()
        }

    def add_result_callback(self, callback: Callable[[Dict], None]) -> None:
        self._result_callbacks.append(callback)

    def remove_result_callback(self, callback: Callable[[Dict], None]) -> None:
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)

    def stop(self) -> None:
        self._running = False
        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

    def release(self) -> None:
        self.stop()
        self._pipelines.clear()


class _LegacyCameraManagerShim:
    def __init__(self, vision_manager: VisionManager) -> None:
        self._vm = vision_manager

    def enable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.enable_task(camera_id, task_name)

    def disable_task(self, camera_id: str, task_name: str) -> bool:
        return self._vm.disable_task(camera_id, task_name)

    def get_all_results(self) -> dict:
        return self._vm.get_all_results()

    def add_result_callback(self, callback) -> None:
        self._vm.add_result_callback(callback)

    def remove_result_callback(self, callback) -> None:
        self._vm.remove_result_callback(callback)

    @property
    def cameras(self) -> dict:
        return {}
