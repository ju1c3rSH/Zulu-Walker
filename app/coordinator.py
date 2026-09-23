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

        self._latch_cleared_by_idle = False

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

        self._enqueue(GateSignal(state_name=mode, codes=codes_snapshot))

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

    def apply_gate_task(self, mode: str, codes: tuple) -> None:
        """启用 gks_task 并切换模式（codes 为订阅帧快照）。

        处理器 set_task 内部判重: 同模式同码的重订阅不清锁存,
        换 data_id / 换任务码才清（main.py reset_task_latch 时机）。
        """
        task_enabled_before = self._task_enabled
        prev_task = None
        processor = self._get_gks_processor()
        if processor is not None:
            getter = getattr(processor, "current_task", None)
            if callable(getter):
                try:
                    prev_task = getter()
                except Exception:
                    prev_task = None
        if not task_enabled_before:
            if self._vision_manager:
                self._vision_manager.enable_task(_PIPELINE_ID, _TASK_GKS)
            self._task_enabled = True
        if processor is not None:
            processor.set_task(mode, tuple(codes))
            # 任务/模式真的变了 → 顺手清结果缓存。原因: RX 线程在收到订阅帧时立即改 _sub_id,
            # 而 processor 的 mode 要等主循环这一拍才改; 中间存在 1~2 帧窗口, 旧任务的
            # 载荷会被按新 data_id 打包推出 —— 切到 TARGET 会推 found=1/err=0(正好被底盘判成
            # "已收敛")、切到 RESCUE 会推 found=1/pos=0。清掉后这段窗口推零值心跳(found=0),
            # 是安全方向。同模式同码的幂等重订阅不清, 保证 steady-state 不丢帧。
            if prev_task != (mode, tuple(codes)):
                self._latest_task = {}
                self._latest_task_ts = 0.0
            # 判重描述需覆盖 STOP 清锁存保留模式的情况: STOP reset() 清锁存
            # 但 _mode/_codes 不变, 同码重订 prev 会命中现值, 实际锁存非"保留"
            if prev_task == (mode, tuple(codes)):
                latch_desc = ("cleared(锁存已被STOP清)" if self._latch_cleared_by_idle
                              else "kept(幂等重订阅)")
            else:
                latch_desc = "cleared(换任务/换模式)"
            self._latch_cleared_by_idle = False
            setup = "任务首次启用" if not task_enabled_before else "任务已启用"
            log_print(f"[UART ACTION] gate={mode} codes={list(codes)} "
                      f"latch={latch_desc} ({setup})")

    def apply_gate_idle(self) -> None:
        """停推: 关任务门, 清处理器锁存与结果缓存。"""
        was_active = self._task_enabled
        if self._task_enabled:
            if self._vision_manager:
                self._vision_manager.disable_task(_PIPELINE_ID, _TASK_GKS)
            self._task_enabled = False
        processor = self._get_gks_processor()
        if processor is not None:
            processor.reset()
        self._latest_task = {}
        self._latch_cleared_by_idle = True   # reset() 清锁存但保留 _mode/_codes
        setup = "任务已停用" if was_active else "已是空闲"
        log_print(f"[UART ACTION] gate=IDLE latch=cleared ({setup})")

    # ===== streaming payload builders =====

    def _build_stream_payload(self, data_id: int, now: Optional[float] = None) -> bytes:
        """恒返回 8B 子载荷: 无结果/缺阈值/结果陈旧都推零值心跳（Brain 1s stale 判定）。"""
        data = self._latest_task if self._task_enabled else {}
        if now is not None and (now - self._latest_task_ts) > _TASK_STREAM_STALE_S:
            data = {}   # 相机停帧/处理器异常: 不透传陈旧 found=1
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
