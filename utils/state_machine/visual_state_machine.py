# -*- coding: utf-8 -*-

from typing import Optional, Any, Dict, Callable
from dataclasses import dataclass, field
from time import time

from .base import BaseStateMachine, State


@dataclass
class VisualContext:
    """
    视觉状态机上下文

    在状态间共享的数据，外部代码可读写。
    """
    # === 检测结果（外部写入）===
    confidence: float = 0.0
    target_center: Optional[tuple] = None  # (x, y) 像素坐标
    target_found: bool = False

    # === 输出（状态机写入，外部读取）===
    output_position: Optional[tuple] = None
    error_code: int = 0

    # === 统计（外部更新）===
    consecutive_detected_frames: int = 0
    consecutive_lost_frames: int = 0

    # === 状态计时 ===
    state_entry_time: float = 0.0

    # === 自定义数据 ===
    custom: Dict[str, Any] = field(default_factory=dict)

    percent_error_x: int = 0
    percent_error_y: int = 0
    target_distance_mm: float = 0.0
    is_quad_detected: bool = False
    is_uv_spot_detected: bool = False

    def reset_stats(self):
        """重置统计"""
        self.consecutive_detected_frames = 0
        self.consecutive_lost_frames = 0
        self.percent_error_x = 0
        self.percent_error_y = 0
        self.target_distance_mm = 0.0
        self.is_quad_detected = False
        self.is_uv_spot_detected = False


# === 状态实现 ===

class IdleState(State):
    """IDLE 状态：待机/初始化"""

    def on_enter(self, context: VisualContext, from_state: str) -> None:
        print("[VisualStateMachine] Enter IDLE")
        context.reset_stats()
        context.error_code = 0
        context.output_position = None
        context.target_found = False

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None  # 等待外部事件

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class SearchState(State):
    """SEARCH 状态：全局搜索"""

    def on_enter(self, context: VisualContext, from_state: str) -> None:
        print("[VisualStateMachine] Enter SEARCH")
        context.reset_stats()
        context.state_entry_time = time()

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None  # 等待外部事件

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class TrackingState(State):
    """TRACKING 状态：精细跟踪"""

    def on_enter(self, context: VisualContext, from_state: str) -> None:
        print("[VisualStateMachine] Enter TRACKING")
        context.reset_stats()

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None  # 等待外部事件

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class RecoveryState(State):
    """RECOVERY 状态：丢失恢复"""

    def on_enter(self, context: VisualContext, from_state: str) -> None:
        print("[VisualStateMachine] Enter RECOVERY")
        context.reset_stats()

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None  # 等待外部事件

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        pass


class FailState(State):
    """FAIL 状态：异常终止"""

    def on_enter(self, context: VisualContext, from_state: str) -> None:
        print(f"[VisualStateMachine] Enter FAIL (error_code={context.error_code})")
        context.output_position = None

    def on_execute(self, context: VisualContext) -> Optional[str]:
        return None  # 等待外部事件

    def on_exit(self, context: VisualContext, to_state: str) -> None:
        context.error_code = 0


class VisualStateMachine(BaseStateMachine):
    """
    视觉跟踪状态机

    ## 状态定义

    | 状态 | 职责 | 关键行为 |
    |:---|:---|:---|
    | IDLE | 待机/初始化 | 等待 START 信号 |
    | SEARCH | 全局搜索 | 全图检测，等待目标捕获 |
    | TRACKING | 精细跟踪 | 动态 ROI 检测，输出坐标 |
    | RECOVERY | 丢失恢复 | 扩大搜索，维持跟踪 |
    | FAIL | 异常终止 | 输出错误码 |

    ## 状态转换

    - IDLE → SEARCH: START 事件
    - SEARCH → TRACKING: TARGET_FOUND 事件
    - SEARCH → FAIL: SEARCH_TIMEOUT 事件
    - TRACKING → SEARCH: TARGET_LOST 事件
    - RECOVERY → TRACKING: TARGET_RECOVERED 事件
    - RECOVERY → SEARCH: RECOVERY_FAILED 事件
    - 任意活动状态 → IDLE: STOP 事件
    - FAIL → IDLE: RESET 事件

    ## 使用方法

    ```python
    from utils.state_machine import VisualStateMachine

    # 1. 创建状态机
    sm = VisualStateMachine()

    # 2. 注册回调（可选）- 用于控制任务开关等
    def on_enter_search(context, from_state):
        # 启用全图检测任务
        camera.enable_task("global_detect")
        camera.disable_task("roi_detect")

    def on_enter_tracking(context, from_state):
        # 切换到 ROI 检测
        camera.disable_task("global_detect")
        camera.enable_task("roi_detect")

    sm.on_state_enter(VisualStateMachine.States.SEARCH, on_enter_search)
    sm.on_state_enter(VisualStateMachine.States.TRACKING, on_enter_tracking)

    # 3. 启动状态机
    sm.start()  # IDLE -> SEARCH

    # 4. 在异步循环中更新
    async def detection_loop():
        while True:
            frame = await get_frame()
            result = detector.detect(frame)

            # 更新上下文
            sm.context.target_found = result.success
            sm.context.confidence = result.confidence
            sm.context.target_center = result.center

            # 更新统计
            if result.success:
                sm.context.consecutive_detected_frames += 1
                sm.context.consecutive_lost_frames = 0
            else:
                sm.context.consecutive_lost_frames += 1
                sm.context.consecutive_detected_frames = 0

            # 根据检测结果触发事件
            if sm.is_searching():
                if result.success and result.confidence >= 0.6:
                    sm.trigger(VisualStateMachine.Events.TARGET_FOUND)
                elif sm.state_duration > 5.0:
                    sm.trigger(VisualStateMachine.Events.SEARCH_TIMEOUT)

            elif sm.is_tracking():
                if sm.context.consecutive_lost_frames >= 3:
                    sm.trigger(VisualStateMachine.Events.TARGET_LOST)

            elif sm.is_recovering():
                if result.success and result.confidence >= 0.5:
                    sm.trigger(VisualStateMachine.Events.TARGET_RECOVERED)
                elif sm.context.consecutive_lost_frames >= 10:
                    sm.trigger(VisualStateMachine.Events.RECOVERY_FAILED)

            # 获取输出
            if sm.is_tracking():
                output = sm.context.output_position
                send_to_gimbal(output)
    ```
    """

    class States:
        """状态名常量"""
        IDLE = "IDLE"
        SEARCH = "SEARCH"
        TRACKING = "TRACKING"
        RECOVERY = "RECOVERY"
        FAIL = "FAIL"

    class Events:
        """事件名常量"""
        START = "START"
        STOP = "STOP"
        RESET = "RESET"
        TARGET_FOUND = "TARGET_FOUND"
        TARGET_LOST = "TARGET_LOST"
        TARGET_RECOVERED = "TARGET_RECOVERED"
        RECOVERY_FAILED = "RECOVERY_FAILED"
        SEARCH_TIMEOUT = "SEARCH_TIMEOUT"

    def __init__(self):
        super().__init__()
        self.context = VisualContext()

        self._setup_states()
        self._setup_transitions()
        self.set_initial_state(self.States.IDLE)

    def _setup_states(self) -> None:
        """注册所有状态"""
        self.register_state(self.States.IDLE, IdleState())
        self.register_state(self.States.SEARCH, SearchState())
        self.register_state(self.States.TRACKING, TrackingState())
        self.register_state(self.States.RECOVERY, RecoveryState())
        self.register_state(self.States.FAIL, FailState())

    def _setup_transitions(self) -> None:
        """设置状态转换"""
        # IDLE -> SEARCH
        self.register_transition(
            self.States.IDLE, self.States.SEARCH,
            event=self.Events.START
        )

        # SEARCH -> TRACKING
        self.register_transition(
            self.States.SEARCH, self.States.TRACKING,
            event=self.Events.TARGET_FOUND
        )

        # SEARCH -> FAIL
        self.register_transition(
            self.States.SEARCH, self.States.FAIL,
            event=self.Events.SEARCH_TIMEOUT
        )

        # TRACKING -> SEARCH
        self.register_transition(
            self.States.TRACKING, self.States.SEARCH,
            event=self.Events.TARGET_LOST
        )

        # RECOVERY -> TRACKING
        self.register_transition(
            self.States.RECOVERY, self.States.TRACKING,
            event=self.Events.TARGET_RECOVERED
        )

        # RECOVERY -> SEARCH
        self.register_transition(
            self.States.RECOVERY, self.States.SEARCH,
            event=self.Events.RECOVERY_FAILED
        )

        # 任意活动状态 -> IDLE (STOP)
        for state in [self.States.SEARCH, self.States.TRACKING, self.States.RECOVERY]:
            self.register_transition(
                state, self.States.IDLE,
                event=self.Events.STOP
            )

        # FAIL -> IDLE
        self.register_transition(
            self.States.FAIL, self.States.IDLE,
            event=self.Events.RESET
        )

    # === 外部接口 ===

    def on_state_enter(
        self,
        state: str,
        callback: Callable[[VisualContext, str], None]
    ) -> None:
        """
        注册状态进入回调

        Args:
            state: 状态名（使用 VisualStateMachine.States.XXX）
            callback: 回调函数，参数为 (context, from_state)
        """
        self.add_enter_callback(state, callback)

    def on_state_exit(
        self,
        state: str,
        callback: Callable[[VisualContext, str], None]
    ) -> None:
        """
        注册状态退出回调

        Args:
            state: 状态名
            callback: 回调函数，参数为 (context, to_state)
        """
        self.add_exit_callback(state, callback)

    def on_state_change(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """
        注册状态变更回调

        Args:
            callback: 回调函数，参数为 (old_state, new_state, event)
        """
        self.add_state_change_callback(callback)

    # === 便捷方法 ===

    def start(self) -> bool:
        """启动状态机 (IDLE -> SEARCH)"""
        return self.trigger(self.Events.START)

    def stop(self) -> bool:
        """停止状态机 (任意活动状态 -> IDLE)"""
        return self.trigger(self.Events.STOP)

    def reset_machine(self) -> bool:
        """重置状态机 (FAIL -> IDLE)"""
        return self.trigger(self.Events.RESET)

    def is_idle(self) -> bool:
        """是否处于 IDLE 状态"""
        return self.is_in_state(self.States.IDLE)

    def is_searching(self) -> bool:
        """是否处于 SEARCH 状态"""
        return self.is_in_state(self.States.SEARCH)

    def is_tracking(self) -> bool:
        """是否处于 TRACKING 状态"""
        return self.is_in_state(self.States.TRACKING)

    def is_recovering(self) -> bool:
        """是否处于 RECOVERY 状态"""
        return self.is_in_state(self.States.RECOVERY)

    def is_failed(self) -> bool:
        """是否处于 FAIL 状态"""
        return self.is_in_state(self.States.FAIL)

    def is_active(self) -> bool:
        """是否处于活动状态（非 IDLE 和 FAIL）"""
        return self.current_state in (
            self.States.SEARCH, self.States.TRACKING, self.States.RECOVERY
        )

    def get_output(self) -> dict:
        """
        获取当前输出

        Returns:
            {
                "state": 当前状态名,
                "position": 输出坐标或 None,
                "confidence": 置信度,
                "error_code": 错误码
            }
        """
        return {
            "state": self.current_state,
            "position": self.context.output_position,
            "confidence": self.context.confidence,
            "error_code": self.context.error_code,
        }

    def get_info(self) -> dict:
        """获取状态机完整信息"""
        return {
            "state": self.current_state,
            "previous_state": self.previous_state,
            "state_duration": self.state_duration,
            "state_frame_count": self.state_frame_count,
            "context": {
                "target_found": self.context.target_found,
                "confidence": self.context.confidence,
                "target_center": self.context.target_center,
                "consecutive_detected": self.context.consecutive_detected_frames,
                "consecutive_lost": self.context.consecutive_lost_frames,
                "error_code": self.context.error_code,
            }
        }
