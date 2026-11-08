"""
工科赛近端视觉协调器（panzer_links_protocol.md v2.1 订阅制从站）。

Brain (panzer_Brain, 主站) 通过 USART1 驱动本端:
  CMD_REQUEST [data_id, min_interval_ms, 爆色, 靶色, 形状] -> ACK [did,60,10] -> 流
  CMD_STOP -> 停推（门控 idle, 处理器清锁存）

不设主机静默自动 idle: Brain 的看门狗是"收不到 CAM 数据 >1s 才重订阅",
健康推流期间主站本就静默, 若从站按静默判掉线会周期性断流并清锁存。
Brain 侧异常退出由 STOP 缺失 + 人质/视觉超时兜底。

v2.1 关键语义:
- 订阅期**零值心跳不断流**: 阈值缺失/无结果也按 interval 推全零帧,
  否则 Brain 1s 帧龄 stale 判定 + 1s 看门狗重订阅; 结果陈旧 >1s 同样归零。
- **结果驱动推流**: 视觉每产出一帧结果立即推一帧, 不受 min_interval_ms 限制;
  interval 语义降级为"无新结果时的心跳周期", 见 _push_allowed。策略可用
  project_config.yaml `gks_stream.push_on_new_result` 回退到纯 interval 限速。
- 同 data_id 同任务码的重订阅幂等, 不清视觉锁存（set_task 内部判重）;
  换 data_id / 换任务码才清。
- ACK [did,60,10] / NACK [did,reason] 照发保协议一致性
  （Brain on_cam1 只消费 0x24 DATA_STREAM, ACK/NACK 不被消费）。
- DATA_STREAM payload 恒 10B = seq + data_id + 8B 子载荷。

本类不解释像素（只消费 VisionResult dict）; 锁存/识别在 GksTaskProcessor。
线程模型: UART RX 回调线程只做事件入队; 门控变更在主循环应用,
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

        # 推流策略（结果驱动 + 心跳兜底, 见 _push_allowed）:
        #   视觉每产出一帧结果 -> _result_seq 前进 -> 立即推一帧（不受 interval 限制）
        #   无新结果 -> 按 interval 推心跳（保 Brain 1s stale 判定 / 1s 看门狗）
        # main.py 从 project_config.yaml `gks_stream:` 段注入（缺省即本处默认值）。
        self._result_seq = 0             # 已收到的 gks_task 结果数（仅主循环线程读写）
        self._sent_result_seq = 0        # 最近一次推流时已消费到第几个结果
        self._push_on_new_result = True
        self._min_push_interval_s = 0.0  # 两条放行路径的硬保护, 0 = 不限

        # 推流统计窗口（按秒汇总打一条; 结果驱动后可达相机帧率, 逐帧打印会刷屏）
        self._tx_window_start = 0.0
        self._tx_window_count = 0
        self._tx_hz = 0.0                # 最近一个统计窗口的实测推流率（HUD/台架对拍）

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

    def set_stream_policy(self, push_on_new_result: Optional[bool] = None,
                          min_push_interval_ms: Optional[float] = None) -> None:
        """注入推流策略（main.py 从 project_config.yaml `gks_stream:` 读取）。

        None = 保持当前值（配置缺段/缺键时即用类内默认: 结果驱动、无硬保护间隔），
        push_on_new_result=False 可一键回到旧的"纯 interval 限速"语义。
        """
        if push_on_new_result is not None:
            self._push_on_new_result = bool(push_on_new_result)
        if min_push_interval_ms is not None:
            self._min_push_interval_s = max(0.0, float(min_push_interval_ms) / 1000.0)
        log_print(f"[UART CFG] stream: push_on_new_result={int(self._push_on_new_result)} "
                  f"min_push_interval={self._min_push_interval_s * 1000:.0f}ms")

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
            if self._push_allowed(now):
                current_seq = self._stream_seq
                frame = build_data_stream_frame(current_seq, sub_id, payload)
                result_seq = self._result_seq
                if self._send(frame):
                    self._stream_seq = (current_seq + 1) & 0xFF
                    self._last_send_time = now
                    self._sent_result_seq = result_seq
                    self._note_streamed(sub_id, now)

        self._mem_log_counter += 1
        if self._mem_log_counter >= 900:
            self._mem_log_counter = 0
            self._log_memory()

    def _push_allowed(self, now: float) -> bool:
        """推流闸门 —— 结果驱动 + 心跳兜底（"一帧误差一帧输出"的实现点）。

        两条放行路径, 缺一不可:

        1) **新结果**: 视觉又产出了一帧结果（_result_seq 前进）-> 立即推, 不受
           min_interval_ms 限制。延迟只有一个主循环 tick（MAIN_LOOP_DELAY = 2ms,
           远小于相机帧间隔 16.7ms）, 不会被时间闸门吞掉中间帧。

        2) **心跳**: 没有新结果时按 interval 推一帧。interval 的语义因此从
           "每帧最小间隔"变为"心跳周期"—— Brain 的 1s 帧龄 stale 判定 + 1s
           看门狗要求订阅期必须持续有帧, 停推即被重订阅。下限仍为
           MIN_INTERVAL_MS(30ms), 即心跳最快 ~33Hz（协议"建议 30~50ms"）。

        两条路径都受 min_push_interval（项目配置, 默认 0 = 不限）保护, 用于压制
        相机恢复瞬间的突发连推; 上限天然由相机帧率封顶。push_on_new_result=False
        则退化为纯 interval 限速。
        """
        if (now - self._last_send_time) < self._min_push_interval_s:
            return False
        if self._push_on_new_result and self._result_seq != self._sent_result_seq:
            return True
        interval = max(self._interval_ms, MIN_INTERVAL_MS) / 1000.0
        return (now - self._last_send_time) >= interval

    def _note_streamed(self, sub_id: int, now: float) -> None:
        """推流日志按秒汇总一条（含实测 Hz）。

        结果驱动后推流率可达相机帧率, 逐帧打印会刷屏; 但"推流率是否等于帧率"
        是核心验收项, 所以按秒打 —— 台架上一眼就能看出是 60Hz 还是掉到 20Hz
        （`[FPS] pipeline` 可对照）。
        """
        if self._tx_window_start == 0.0:
            self._tx_window_start = now
        self._tx_window_count += 1
        elapsed = now - self._tx_window_start
        if elapsed < 1.0:
            return
        hz = self._tx_window_count / elapsed
        self._tx_hz = hz
        type_name = DATA_ID_NAMES.get(sub_id, f"0x{sub_id:02X}")
        last_seq = (self._stream_seq - 1) & 0xFF
        log_print(f"[UART TX] DATA_STREAM {hz:.1f}Hz seq={last_seq} "
                  f"type={type_name} {self._stream_payload_desc()}")
        self._tx_window_start = now
        self._tx_window_count = 0

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
        """消费视觉结果: 覆盖最新缓存 + 结果计数前进（结果驱动推流的"有新帧"判据）。

        计数在**每收到一个 gks_task 结果**时 +1, 归属/新鲜度两道判据统一在
        _active_result 里做。归属不符的结果也会消耗一个计数, 代价只是切任务
        瞬间多推一两帧零值, 不会漏推。
        """
        for _pipeline_id, results in all_results.items():
            for task_name, vision_result in results.items():
                if task_name == _TASK_GKS and isinstance(vision_result, VisionResult):
                    self._latest_task = vision_result.result_data or {}
                    self._latest_task_ts = time.monotonic()
                    self._result_seq += 1

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

        # ⚠ 先发帧再打日志: 协议 I/O 不能挂在"日志是否成功"上。日志层一旦抛异常,
        # ACK 就会被静默吞掉, Brain 收不到应答只会更猛地重订阅。
        self._send(build_cmd_ack_frame(data_id))
        log_print(f"[UART TX] CMD_ACK type={dt_name} freq={ACK_MAX_FREQ_HZ}Hz "
                  f"size={ACK_PAYLOAD_SIZE}B interval={self._interval_ms}ms "
                  f"codes={list(self._codes)} ({codes_action})")

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
        """本次订阅是否清视觉锁存:

            清锁存 ⟺ 完整订阅帧(带任务码) AND (data_id 变了 OR 任务码变了)

        两个细节, 缺一就不等价:
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

        注: set_task 在"mode 真的变了"时仍会自清一次, 覆盖"短帧重订阅却换了
        data_id"这个 coordinator 判不清的边缘; 多清一次幂等无害。
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
        """停推: 关任务门 + 清结果缓存, **不清处理器锁存**。

        锁存是否作废由**下次订阅**的判据决定, 这里只把 _prev_gate_id 归零:
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
        # 推流统计窗口归零: 停推期间没有发送, 不归零的话下次订阅的首帧会把整段
        # 空闲时间算进分母, 打出一条毫无意义的低 Hz 日志。
        self._tx_window_start = 0.0
        self._tx_window_count = 0
        setup = "任务已停用" if was_active else "已是空闲"
        log_print(f"[UART ACTION] gate=IDLE latch=kept(是否清理由下次订阅判定) ({setup})")

    # ===== streaming payload builders =====

    def _active_result(self, data_id: int, now: Optional[float] = None) -> dict:
        """本拍可上报的视觉结果; 返回 {} 表示推零值心跳。

        两道判据集中在这一个方法里, 推送与日志共用同一份, 免得两处判据分叉:

        1) 陈旧: 超过 _TASK_STREAM_STALE_S 未刷新 -> 不透传陈旧的 found=1。
        2) 归属: result_data["mode"] 必须等于"产生它的模式"。

        第 2 条是切任务脏帧的根治手段: 视觉结果异步产出, 切模式后仍可能落进
        _latest_task, 按新 data_id 打包会推出错模式的 found=1。
        在**数据消费点**验归属, 而不是在切换瞬间清空缓存 —— 后者对"清空之后
        才到达的异步残留帧"无效; 校验 mode 对任意到达时刻都成立, 且天然放行
        同模式的幂等重订阅。
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
            "tx_hz": self._tx_hz,
            "ai_loaded": ai_loaded,
            "ai_model": active_model,
            "last_nack": self._last_nack,
            "target_roi": self._target_roi_snapshot(),
            "explosive_roi": self._explosive_roi_snapshot(),
            "target_track_win": self._target_track_win_snapshot(),
            "target_zero": self._target_zero_snapshot(),
        }

    def _target_roi_snapshot(self):
        processor = self._get_gks_processor()
        try:
            return processor.target_roi() if processor is not None else None
        except Exception:
            return None

    def _explosive_roi_snapshot(self):
        processor = self._get_gks_processor()
        try:
            return processor.explosive_roi() if processor is not None else None
        except Exception:
            return None

    def _target_track_win_snapshot(self):
        processor = self._get_gks_processor()
        try:
            return processor.target_track_win() if processor is not None else None
        except Exception:
            return None

    def _target_zero_snapshot(self) -> int:
        processor = self._get_gks_processor()
        try:
            return processor.target_zero_px() if processor is not None else 0
        except Exception:
            return 0
