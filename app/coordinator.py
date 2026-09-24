"""
工科赛近端视觉协调器（panzer_links_protocol.md v2.1 订阅制从站）。

Brain (panzer_Brain, 主站) 通过 USART1 驱动本端:
  CMD_REQUEST [data_id, min_interval_ms, 爆色, 靶色, 形状] -> ACK [did,60,10] -> 流
  CMD_STOP -> 停推（门控 idle, 处理器清锁存）

不设主机静默自动 idle（与旧上端 main.py 一致）: Brain 的 Comm_Task 看门狗是
"收不到 CAM 数据 >1s 才重订阅", 健康推流期间主站本就静默——若从站按静默判
掉线, 会在每个任务阶段周期性断流并清锁存。Brain 侧异常退出由 STOP 缺失 +
人质/视觉超时兜底, 本端持续推流无害。

v2.1 关键语义（对照 Brain interrupt.c 实际消费逻辑）:
- 订阅期**零值心跳不断流**: 阈值缺失/无结果也按 interval 推全零帧,
  否则 Brain 1s 帧龄 stale 判定 + 1s 看门狗重订阅; 结果陈旧 >1s 同样归零。
- 同 data_id 同任务码的重订阅幂等, 不清视觉锁存（set_task 内部判重）;
  换 data_id / 换任务码才清（对应审查报告 §5.3 锁存语义）。
- ACK [did,60,10] / NACK [did,reason] 照发保协议一致性
  （Brain on_cam1 只消费 0x24 DATA_STREAM, ACK/NACK 不被消费）。
- DATA_STREAM payload 恒 10B = seq + data_id + 8B 子载荷。

本类不解释像素（只消费 VisionResult dict）; 锁存/识别在 GksTaskProcessor。
线程模型: UART RX 回调线程只做事件入队; 门控变更在主循环 _sm_queue 应用,
视觉线程只见原子布尔翻转与单写者 set_task。
"""
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from modules.zw_opencv_module.vision_manager import VisionManager
from utils.log_util import log_print

try:
    from maix import sys as maix_sys
    _HAVE_MAIX_SYS = True
except ImportError:
    maix_sys = None
    _HAVE_MAIX_SYS = False

from app.subscription_sm import GateSignal, SubscriptionStateMachine
from framework.event_bus import EventBus
from modules.zw_uart_module.events import (
    CmdRequestEvent,
    CmdStopEvent,
)
from modules.zw_uart_module.protocol import (
    build_cmd_ack_frame,
    build_cmd_nack_frame,
    build_data_stream_frame,
    build_explosive_sub_payload,
    build_target_sub_payload,
    build_rescue_sub_payload,
    SUPPORTED_DATA_IDS,
    DATA_ID_NAMES,
    DATA_EXPLOSIVE,
    DATA_TARGET,
    DATA_RESCUE,
    NACK_UNSUPPORTED_TYPE,
    NACK_NOT_READY,
    NACK_REASON_NAMES,
    MIN_INTERVAL_MS,
    ACK_MAX_FREQ_HZ,
    ACK_PAYLOAD_SIZE,
)
from modules.zw_opencv_module.processors.base import VisionResult

_TASK_STREAM_STALE_S = 1.0   # 视觉结果陈旧阈值: 超过按零值心跳推送
_PIPELINE_ID = "cam_main"
_TASK_GKS = "gks_task"

# data_id -> GksTaskProcessor mode
_DATA_ID_MODES = {
    DATA_EXPLOSIVE: "EXPLOSIVE",
    DATA_TARGET:    "TARGET",
    DATA_RESCUE:    "RESCUE",
}

# 反查: 模式 -> data_id（锁存清理判据需要以 data_id 比较, 见 _latch_reset_needed）
_MODE_DATA_IDS = {v: k for k, v in _DATA_ID_MODES.items()}

# data_id -> (optional) AI model nickname to hot-switch before ACKing
_DATA_ID_MODEL = {
    DATA_RESCUE: "eod_hostage",
}


class GksCoordinator:
    """订阅门控 + 流控推送 + v2.1 协议应答。"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self._uart_sender: Optional[Callable[[bytes], bool]] = None
        self._vision_manager: Optional[VisionManager] = None
        self._ai = None

        # master-slave streaming state (v2.1)
        self._sub_id = 0                 # 当前订阅 data_id, 0=未订阅
        self._codes = (0, 0, 0)          # (爆色, 靶色, 形状), 随完整订阅帧更新
        self._interval_ms = MIN_INTERVAL_MS
        self._stream_seq = 0
        self._last_cmd_time = 0.0
        self._last_send_time = 0.0
        self._master_linked = False
        self._cmd_lock = threading.Lock()
        self._last_nack = None           # (reason code, monotonic time) HUD 横幅用

        # gating mutations from the RX thread are applied in the main loop
        self._gate_queue: Deque[GateSignal] = deque()
        self._gate_lock = threading.Lock()
        self._gating = SubscriptionStateMachine(self)

        self._stream_log_count = 0
        self._last_streamed_id = 0

        # task gating cache (avoid redundant TaskManager calls)
        self._task_enabled = False

        # 上次订阅门控快照: 锁存清理判据的"上一次"基准。
        # STOP 时 _prev_gate_id 归零(等价 main.py 的 sub_id=0), 使 STOP 之后的
        # 完整帧重订阅必定命中"data_id 变了"而清锁存; 短帧重订阅则保留。
        self._prev_gate_id = 0
        self._prev_gate_codes: tuple = (0, 0, 0)

        self._running = False
        self._wdt_feed: Callable[[], None] = lambda: None
        self._wdt_count = 0
        self._last_fps: float = 0.0
        self._last_fps_time: float = 0.0
        self._mem_log_counter: int = 0

        # vision result cache (written in loop(), read in loop())
        self._latest_task: dict = {}
        self._latest_task_ts: float = 0.0
        self._frame_width: int = 640
        self._frame_height: int = 480

    # ===== wiring =====

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_width = width
        self._frame_height = height

    def set_wdt_feed(self, feed_fn) -> None:
        self._wdt_feed = feed_fn

    def connect_vision(self, vision_manager: VisionManager) -> None:
        self._vision_manager = vision_manager

    def set_uart_sender(self, sender: Callable[[bytes], bool]) -> None:
        self._uart_sender = sender

    def set_ai(self, ai) -> None:
        self._ai = ai

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            try:
                return self._uart_sender(frame)
            except Exception:
                return False
        return False

    def _enqueue(self, signal: GateSignal) -> None:
        with self._gate_lock:
            self._gate_queue.append(signal)

    # ===== main loop =====

    def loop(self) -> None:
        if self._vision_manager:
            for all_results in self._vision_manager.drain_results():
                self._process_vision_results(all_results)

            now = time.monotonic()
            if now - self._last_fps_time >= 5.0:
                self._last_fps = self._vision_manager.get_pipeline_fps(_PIPELINE_ID)
                self._last_fps_time = now
                log_print(f"[FPS] pipeline: {self._last_fps:.1f}")

        with self._gate_lock:
            while self._gate_queue:
                self._gating.dispatch(self._gate_queue.popleft())

        self._wdt_feed()
        self._wdt_count += 1
        if self._wdt_count % 600 == 0:
            log_print(f"[WDT] coord feed #{self._wdt_count}")

        now = time.monotonic()
        with self._cmd_lock:
            sub_id = self._sub_id

        if sub_id != 0:
            payload = self._build_stream_payload(sub_id, now)   # 恒非 None: 零值心跳
            if self._send_allowed(now):
                current_seq = self._stream_seq
                frame = build_data_stream_frame(current_seq, sub_id, payload)
                if self._send(frame):
                    self._stream_seq = (current_seq + 1) & 0xFF
                    self._last_send_time = now
                    self._stream_log_count += 1
                    if sub_id != self._last_streamed_id or self._stream_log_count >= 30:
                        self._stream_log_count = 0
                        self._last_streamed_id = sub_id
                        type_name = DATA_ID_NAMES.get(sub_id, f"0x{sub_id:02X}")
                        log_print(f"[UART TX] DATA_STREAM seq={current_seq} "
                                  f"type={type_name} {self._stream_payload_desc()}")

        self._mem_log_counter += 1
        if self._mem_log_counter >= 900:
            self._mem_log_counter = 0
            self._log_memory()

    def _send_allowed(self, now: float) -> bool:
        """min_interval 节流（main.py push_ok 等价; 协议下限 30ms）。"""
        interval = max(self._interval_ms, MIN_INTERVAL_MS) / 1000.0
        return (now - self._last_send_time) >= interval

    def start(self) -> None:
        self._running = True
        self._wire_events()
        self._last_cmd_time = time.monotonic()

    def stop(self) -> None:
        self._running = False

    def _wire_events(self) -> None:
        self.event_bus.subscribe(CmdRequestEvent, self._on_cmd_request)
        self.event_bus.subscribe(CmdStopEvent, self._on_cmd_stop)

    # ===== vision results =====

    def _process_vision_results(self, all_results: dict) -> None:
        for _pipeline_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if task_name == _TASK_GKS and isinstance(vision_result, VisionResult):
                    self._latest_task = vision_result.result_data or {}
                    self._latest_task_ts = time.monotonic()

    # ===== master events =====

    def _on_cmd_request(self, event: CmdRequestEvent) -> None:
        data_id = event.data_type
        dt_name = DATA_ID_NAMES.get(data_id, f"0x{data_id:02X}")
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._master_linked = True

        if data_id not in SUPPORTED_DATA_IDS:
            log_print(f"[UART TX] CMD_NACK type={dt_name} "
                      f"reason={NACK_REASON_NAMES.get(NACK_UNSUPPORTED_TYPE)} "
                      f"-> 不订阅不推流")
            self._send(build_cmd_nack_frame(data_id, NACK_UNSUPPORTED_TYPE))
            self._last_nack = (NACK_UNSUPPORTED_TYPE, time.monotonic())
            return

        # RESCUE 需要 YOLO 模型: 热切换失败（含 AI 未初始化）回 NACK NOT_READY
        # （不订阅不推流；Brain 收到 NACK 才知道本端不可用，否则 ACK 成功
        #  却永远零值会误导主站时序）
        nick = _DATA_ID_MODEL.get(data_id)
        if nick:
            ai_ready = (
                self._ai is not None
                and getattr(self._ai, "active_model", None) == nick
            ) or (
                self._ai is not None and self._ai.switch(nick)
            )
            if not ai_ready:
                log_print(f"[UART TX] CMD_NACK type={dt_name} "
                          f"reason={NACK_REASON_NAMES.get(NACK_NOT_READY)}")
                self._send(build_cmd_nack_frame(data_id, NACK_NOT_READY))
                self._last_nack = (NACK_NOT_READY, time.monotonic())
                return

        mode = _DATA_ID_MODES[data_id]
        # 换任务码才写码; 短帧重订阅继承原任务码（且处理器判重保锁存）
        codes_action = "换任务码" if event.has_codes else "继承原任务码"
        with self._cmd_lock:
            if event.has_codes:
                self._codes = event.codes
            self._sub_id = data_id
            self._stream_seq = 0
            # main.py 语义: 任何 len>=2 订阅帧都刷新 interval, 下限 30ms
            self._interval_ms = max(event.min_interval_ms, MIN_INTERVAL_MS)
            self._last_send_time = 0.0   # 新订阅立即推一帧
            codes_snapshot = self._codes

        self._enqueue(GateSignal(state_name=mode, codes=codes_snapshot,
                                 has_codes=bool(event.has_codes)))

        log_print(f"[UART TX] CMD_ACK type={dt_name} freq={ACK_MAX_FREQ_HZ}Hz "
                  f"size={ACK_PAYLOAD_SIZE}B interval={self._interval_ms}ms "
                  f"codes={list(self._codes)} ({codes_action})")
        self._send(build_cmd_ack_frame(data_id))

    def _on_cmd_stop(self, event: CmdStopEvent) -> None:
        now = time.monotonic()
        with self._cmd_lock:
            self._last_cmd_time = now
            self._sub_id = 0
            self._stream_seq = 0
        log_print("[UART RX] CMD_STOP -> idle (no ACK: Brain 不消费)")
        self._enqueue(GateSignal(is_stop=True))

    # ===== subscription gating (runs in main loop, via SubscriptionStateMachine) =====

    def _get_gks_processor(self):
        if not self._vision_manager:
            return None
        pipe = self._vision_manager.get_pipeline(_PIPELINE_ID)
        if pipe is None:
            return None
        task = pipe.task_manager.get_task(_TASK_GKS)
        return task.processor if task else None

    def _latch_reset_needed(self, data_id: int, codes: tuple, has_codes: bool) -> bool:
        """本次订阅是否清视觉锁存 —— 判据与基准 main.py reset_task_latch 完全对齐:

            清锁存 ⟺ 完整订阅帧(带任务码) AND (data_id 变了 OR 任务码变了)

        两个细节都来自基准, 缺一就不等价:
          - 短帧重订阅(无任务码)沿用旧码, **不清**锁存, 否则 Brain 1s 看门狗的
            同参重订阅会周期性清掉辛苦锁上的坑位;
          - 比较基准是"上次订阅的 data_id", 而 STOP 会把它归零(见 apply_gate_idle),
            于是 STOP 后的完整帧重订阅必然命中"data_id 变了" → 清。
        """
        if not has_codes:
            return False
        return (data_id != self._prev_gate_id) or (tuple(codes) != tuple(self._prev_gate_codes))

    def apply_gate_task(self, mode: str, codes: tuple, has_codes: bool = False) -> None:
        """启用 gks_task 并切换模式（codes 为订阅帧快照）。

        锁存清理的唯一决策点是 _latch_reset_needed, 且只在**完整订阅帧**上发生;
        处理器 set_task 自身的判重只负责切模式, 不再兼任锁存语义。

        注: set_task 在"mode 真的变了"时仍会自清一次, 那是处理器的下限自保护,
        覆盖 coordinator 判不清的边缘 —— "短帧重订阅却换了 data_id"(无任务码,
        判据直接返回 False)。基准 main.py 在这个边缘会残留上一模式的锁存, 属于
        基准自身的隐患; 这里多清一次幂等无害, 方向与"宁可多清也不跨模式残留"
        一致, 故保留。
        """
        task_enabled_before = self._task_enabled
        data_id = _MODE_DATA_IDS.get(mode, 0)
        reset_latch = self._latch_reset_needed(data_id, codes, has_codes)

        if not task_enabled_before:
            if self._vision_manager:
                self._vision_manager.enable_task(_PIPELINE_ID, _TASK_GKS)
            self._task_enabled = True

        processor = self._get_gks_processor()
        if processor is not None:
            # 先清再切: set_task 对"同模式同码"会提前返回, 若把 reset 放后面,
            # STOP(清过锁存但保留 _mode/_codes) 后的同码重订阅就再也清不掉。
            if reset_latch:
                processor.reset()
            processor.set_task(mode, tuple(codes))

        self._prev_gate_id = data_id
        self._prev_gate_codes = tuple(codes)

        latch_desc = ("cleared(换任务/换码/STOP后重订阅)" if reset_latch
                      else "kept(幂等重订阅)")
        setup = "任务首次启用" if not task_enabled_before else "任务已启用"
        log_print(f"[UART ACTION] gate={mode} codes={list(codes)} "
                  f"has_codes={int(bool(has_codes))} "
                  f"latch={latch_desc} ({setup})")

    def apply_gate_idle(self) -> None:
        """停推: 关任务门 + 清结果缓存, **不清处理器锁存**(对齐 main.py handle_stop)。

        基准的 handle_stop 只置 sub_id=0 / seq=0, 不碰 reset_task_latch; 锁存是否
        作废由**下次订阅**的判据决定。这里同样只把 _prev_gate_id 归零 —— 于是
        STOP 后再订阅的行为与基准逐条一致:
            完整帧重订阅 -> data_id 与 0 不同 -> 清锁存
            短帧重订阅   -> has_codes=False   -> 保留锁存继续累积
        结果缓存仍要清: 已停推, 缓存里的旧结果不该在下一次订阅的首帧被透出。
        """
        was_active = self._task_enabled
        if self._task_enabled:
            if self._vision_manager:
                self._vision_manager.disable_task(_PIPELINE_ID, _TASK_GKS)
            self._task_enabled = False
        self._prev_gate_id = 0        # 等价 main.py handle_stop 的 sub_id = 0
        self._latest_task = {}
        self._latest_task_ts = 0.0
        setup = "任务已停用" if was_active else "已是空闲"
        log_print(f"[UART ACTION] gate=IDLE latch=kept(是否清理由下次订阅判定) ({setup})")

    # ===== streaming payload builders =====

    def _active_result(self, data_id: int, now: Optional[float] = None) -> dict:
        """本拍可上报的视觉结果; 返回 {} 表示推零值心跳。

        两道判据集中在这一个方法里, 推送与日志共用同一份, 免得两处判据分叉:

        1) 陈旧: 超过 _TASK_STREAM_STALE_S 未刷新 -> 不透传陈旧的 found=1。
        2) 归属: result_data["mode"] 必须等于"产生它的模式"。

        第 2 条是切任务脏帧的根治手段。视觉结果由后台线程异步产出, RX 线程却在
        收到订阅帧的那一刻就改了 _sub_id —— 于是"正在处理的那一帧"仍带旧 mode,
        且会在主循环切完模式之后才落进 _latest_task。若按新 data_id 打包, 切到
        TARGET 会推出 err=0 且 found=1(底盘恰好据此判"已收敛"并刹停), 切到
        RESCUE 会推出 found=1/pos=0。

        选择在**数据消费点**验归属, 而不是"切换瞬间清空缓存"那种时序手段:
        后者对"清空之后才到达的异步残留帧"无效, 且要靠清空时机碰运气。校验 mode
        则对任意到达时刻的残留帧都成立, 同时天然放行同模式的幂等重订阅。
        """
        expected = _DATA_ID_MODES.get(data_id)
        if expected is None or not self._task_enabled:
            return {}
        data = self._latest_task
        if not data:
            return {}
        if now is not None and (now - self._latest_task_ts) > _TASK_STREAM_STALE_S:
            return {}
        if data.get("mode") != expected:
            return {}
        return data

    def _build_stream_payload(self, data_id: int, now: Optional[float] = None) -> bytes:
        """恒返回 8B 子载荷: 无结果/陈旧/模式不匹配都推零值心跳（Brain 1s stale 判定）。"""
        data = self._active_result(data_id, now)
        if data_id == DATA_EXPLOSIVE:
            return build_explosive_sub_payload(
                int(data.get("cx", 0)), int(data.get("pit", 0)),
                bool(data.get("found", False)))
        if data_id == DATA_TARGET:
            return build_target_sub_payload(
                int(data.get("err_fixed", 0)),
                bool(data.get("found", False)))
        if data_id == DATA_RESCUE:
            return build_rescue_sub_payload(
                int(data.get("pos", 0)),
                bool(data.get("found", False)))
        return b"\x00" * 8

    def _stream_payload_desc(self) -> str:
        data = self._latest_task
        return (f"mode={data.get('mode', '-')} cx={data.get('cx', 0)} "
                f"pit={data.get('pit', 0)} pos={data.get('pos', 0)} "
                f"err={data.get('err_fixed', 0):+d} found={int(bool(data.get('found')))}")

    # ===== memory logging =====

    def _log_memory(self) -> None:
        if not _HAVE_MAIX_SYS:
            return
        try:
            info = maix_sys.memory_info()
            user = info.get("used", 0) / 1048576
            user_total = info.get("total", 0) / 1048576
            cmm = info.get("cmm_used", 0) / 1048576
            cmm_total = info.get("cmm_total", 0) / 1048576
            log_print(
                f"[MEM] user={user:.0f}/{user_total:.0f}MB "
                f"CMM={cmm:.0f}/{cmm_total:.0f}MB"
            )
        except Exception as e:
            log_print(f"[MEM] error: {e}")

    # ===== debug / HUD =====

    def _ai_status(self) -> tuple:
        ai = self._ai
        if ai is None:
            return False, None
        return bool(getattr(ai, "loaded", False)), getattr(ai, "active_model", None)

    def get_info(self) -> dict:
        with self._cmd_lock:
            linked = self._master_linked
            sub_id = self._sub_id
            codes = list(self._codes)
            interval = self._interval_ms
        data = self._latest_task
        mode = DATA_ID_NAMES.get(sub_id, "IDLE") if sub_id else "IDLE"
        ai_loaded, active_model = self._ai_status()
        return {
            "state": mode,
            "state_id": sub_id,
            "link_active": linked,
            "codes": codes,
            "interval_ms": interval,
            "found": bool(data.get("found", False)),
            "pit": data.get("pit", 0),
            "pos": data.get("pos", 0),
            "err": data.get("err_fixed", 0),
            "latched_order": data.get("latched_order", []),
            "pending_count": data.get("pending_count", 0),
            "thresholds_ok": data.get("thresholds_ok", True),
            "fps": self._last_fps,
            "ai_loaded": ai_loaded,
            "ai_model": active_model,
            "last_nack": self._last_nack,
            "target_roi": self._target_roi_snapshot(),
        }

    def _target_roi_snapshot(self):
        processor = self._get_gks_processor()
        try:
            return processor.target_roi() if processor is not None else None
        except Exception:
            return None
