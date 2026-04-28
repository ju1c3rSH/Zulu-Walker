# -*- coding: utf-8 -*-
"""
全局状态机模块

提供全局单例状态机，用于跨模块共享和修改机器人状态。
基于 BaseStateMachine 架构重构，保持原有 API 兼容。
"""
import enum
import threading
from typing import Dict, Any, Optional, Callable, List
from time import time
from dataclasses import dataclass, field

from .state_machine.base import BaseStateMachine, State


class RobotState(enum.Enum):
    """机器人状态枚举"""
    IDLE = "IDLE"
    READ_QR = "READ_QR"
    NAV_TO_RAW = "NAV_TO_RAW"
    WAIT_TURNTABLE = "WAIT_TURNTABLE"
    PICK_AND_LOAD = "PICK_AND_LOAD"
    NAV_TO_PROC = "NAV_TO_PROC"
    UNLOAD_TO_PROC = "UNLOAD_TO_PROC"
    NAV_TO_TEMP = "NAV_TO_TEMP"
    STACKING = "STACKING"
    RETURN_HOME = "RETURN_HOME"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"


class EventType(enum.Enum):
    """事件类型枚举"""
    START = "START"
    QR_DECODED = "QR_DECODED"
    ARRIVED_AT_ZONE = "ARRIVED_AT_ZONE"
    TURNTABLE_READY = "TURNTABLE_READY"
    GRIP_SUCCESS = "GRIP_SUCCESS"
    LOAD_SUCCESS = "LOAD_SUCCESS"
    UNLOAD_SUCCESS = "UNLOAD_SUCCESS"
    STACK_SUCCESS = "STACK_SUCCESS"
    ALL_BATCHES_DONE = "ALL_BATCHES_DONE"
    HOME_REACHED = "HOME_REACHED"
    SENSOR_FAULT = "SENSOR_FAULT"
    RECOVERY_OK = "RECOVERY_OK"


@dataclass
class RobotContext:
    """
    机器人状态机上下文

    在状态间共享的数据，外部代码可读写。
    """
    # 上下文数据
    data: Dict[str, Any] = field(default_factory=dict)

    # 状态历史记录
    history: List[Dict[str, Any]] = field(default_factory=list)

    def set(self, key: str, value: Any):
        """设置上下文数据"""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文数据"""
        return self.data.get(key, default)

    def clear(self):
        """清空上下文数据"""
        self.data.clear()

    def record_transition(self, from_state: str, to_state: str, event: Optional[str], data: Any):
        """记录状态变更历史"""
        record = {
            "timestamp": time(),
            "from_state": from_state,
            "to_state": to_state,
            "event": event,
            "data": str(data)[:100] if data else None,
        }
        self.history.append(record)
        # 保留最近100条记录
        if len(self.history) > 100:
            self.history = self.history[-100:]


# === 状态实现 ===

class IdleState(State):
    """IDLE 状态：空闲"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class ReadQRState(State):
    """READ_QR 状态：读取二维码"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class NavToRawState(State):
    """NAV_TO_RAW 状态：导航到原料区"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class WaitTurntableState(State):
    """WAIT_TURNTABLE 状态：等待转盘"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class PickAndLoadState(State):
    """PICK_AND_LOAD 状态：抓取装载"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class NavToProcState(State):
    """NAV_TO_PROC 状态：导航到加工区"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class UnloadToProcState(State):
    """UNLOAD_TO_PROC 状态：卸载到加工区"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class NavToTempState(State):
    """NAV_TO_TEMP 状态：导航到暂存区"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class StackingState(State):
    """STACKING 状态：码垛"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class ReturnHomeState(State):
    """RETURN_HOME 状态：返回原点"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class FinishedState(State):
    """FINISHED 状态：完成"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class ErrorState(State):
    """ERROR 状态：错误"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        print(f"[RobotStateMachine] Enter ERROR state from {from_state}")

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class RecoveringState(State):
    """RECOVERING 状态：恢复中"""

    def on_enter(self, context: RobotContext, from_state: str) -> None:
        pass

    def on_execute(self, context: RobotContext) -> Optional[str]:
        return None

    def on_exit(self, context: RobotContext, to_state: str) -> None:
        pass


class RobotStateMachine(BaseStateMachine):
    """
    物流机器人状态机

    基于 BaseStateMachine 架构实现，保持原有 API 兼容。

    ## 状态定义

    | 状态 | 职责 |
    |:---|:---|
    | IDLE | 空闲 |
    | READ_QR | 读取二维码 |
    | NAV_TO_RAW | 导航到原料区 |
    | WAIT_TURNTABLE | 等待转盘 |
    | PICK_AND_LOAD | 抓取装载 |
    | NAV_TO_PROC | 导航到加工区 |
    | UNLOAD_TO_PROC | 卸载到加工区 |
    | NAV_TO_TEMP | 导航到暂存区 |
    | STACKING | 码垛 |
    | RETURN_HOME | 返回原点 |
    | FINISHED | 完成 |
    | ERROR | 错误 |
    | RECOVERING | 恢复中 |
    """

    # 状态名常量（与 RobotState 枚举值对应）
    IDLE = "IDLE"
    READ_QR = "READ_QR"
    NAV_TO_RAW = "NAV_TO_RAW"
    WAIT_TURNTABLE = "WAIT_TURNTABLE"
    PICK_AND_LOAD = "PICK_AND_LOAD"
    NAV_TO_PROC = "NAV_TO_PROC"
    UNLOAD_TO_PROC = "UNLOAD_TO_PROC"
    NAV_TO_TEMP = "NAV_TO_TEMP"
    STACKING = "STACKING"
    RETURN_HOME = "RETURN_HOME"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"

    def __init__(self):
        super().__init__()
        self.context = RobotContext()
        self._last_update_time: float = time()

        self._setup_states()
        self._setup_transitions()
        self.set_initial_state(self.IDLE)

        # 注册状态变更回调以更新时间和记录历史
        self.add_state_change_callback(self._on_state_change)

    def _setup_states(self) -> None:
        """注册所有状态"""
        self.register_state(self.IDLE, IdleState())
        self.register_state(self.READ_QR, ReadQRState())
        self.register_state(self.NAV_TO_RAW, NavToRawState())
        self.register_state(self.WAIT_TURNTABLE, WaitTurntableState())
        self.register_state(self.PICK_AND_LOAD, PickAndLoadState())
        self.register_state(self.NAV_TO_PROC, NavToProcState())
        self.register_state(self.UNLOAD_TO_PROC, UnloadToProcState())
        self.register_state(self.NAV_TO_TEMP, NavToTempState())
        self.register_state(self.STACKING, StackingState())
        self.register_state(self.RETURN_HOME, ReturnHomeState())
        self.register_state(self.FINISHED, FinishedState())
        self.register_state(self.ERROR, ErrorState())
        self.register_state(self.RECOVERING, RecoveringState())

    def _setup_transitions(self) -> None:
        """设置状态转换"""
        # IDLE -> READ_QR
        self.register_transition(self.IDLE, self.READ_QR, event=EventType.START.value)

        # READ_QR -> NAV_TO_RAW
        self.register_transition(self.READ_QR, self.NAV_TO_RAW, event=EventType.QR_DECODED.value)

        # NAV_TO_RAW -> WAIT_TURNTABLE
        self.register_transition(self.NAV_TO_RAW, self.WAIT_TURNTABLE, event=EventType.ARRIVED_AT_ZONE.value)

        # WAIT_TURNTABLE -> PICK_AND_LOAD
        self.register_transition(self.WAIT_TURNTABLE, self.PICK_AND_LOAD, event=EventType.TURNTABLE_READY.value)

        # PICK_AND_LOAD -> NAV_TO_PROC
        self.register_transition(self.PICK_AND_LOAD, self.NAV_TO_PROC, event=EventType.LOAD_SUCCESS.value)

        # NAV_TO_PROC -> UNLOAD_TO_PROC
        self.register_transition(self.NAV_TO_PROC, self.UNLOAD_TO_PROC, event=EventType.ARRIVED_AT_ZONE.value)

        # UNLOAD_TO_PROC -> NAV_TO_TEMP
        self.register_transition(self.UNLOAD_TO_PROC, self.NAV_TO_TEMP, event=EventType.UNLOAD_SUCCESS.value)

        # NAV_TO_TEMP -> STACKING
        self.register_transition(self.NAV_TO_TEMP, self.STACKING, event=EventType.ARRIVED_AT_ZONE.value)

        # STACKING -> RETURN_HOME 或 FINISHED
        self.register_transition(self.STACKING, self.RETURN_HOME, event=EventType.STACK_SUCCESS.value)
        self.register_transition(self.STACKING, self.FINISHED, event=EventType.ALL_BATCHES_DONE.value)

        # RETURN_HOME -> FINISHED
        self.register_transition(self.RETURN_HOME, self.FINISHED, event=EventType.HOME_REACHED.value)

        # ERROR -> IDLE
        self.register_transition(self.ERROR, self.IDLE, event=EventType.RECOVERY_OK.value)

        # RECOVERING -> IDLE
        self.register_transition(self.RECOVERING, self.IDLE, event=EventType.RECOVERY_OK.value)

        # SENSOR_FAULT: 任何状态都可以进入 ERROR（需要在 trigger 中特殊处理）

    def _on_state_change(self, old_state: str, new_state: str, event: Optional[str]):
        """状态变更回调：更新时间和记录历史"""
        self._last_update_time = time()
        self.context.record_transition(old_state, new_state, event, None)

    # === 兼容原有 API ===

    @property
    def state(self) -> RobotState:
        """获取当前状态（返回枚举值）"""
        return RobotState(self.current_state)

    def set_state(self, new_state: RobotState, reason: str = "") -> bool:
        """
        直接设置状态（不通过事件触发）

        Args:
            new_state: 目标状态
            reason: 状态变更原因

        Returns:
            bool: 是否成功变更
        """
        old_state = self.current_state
        if old_state == new_state.value:
            return False

        # 直接转换状态
        return self._do_transition(new_state.value, None)

    def trigger(self, event: EventType, data: Any = None) -> bool:
        """
        触发事件，尝试进行状态转换

        Args:
            event: 触发的事件
            data: 事件相关数据

        Returns:
            bool: 是否成功转换状态
        """
        # SENSOR_FAULT 特殊处理：任何状态都可以进入 ERROR
        if event == EventType.SENSOR_FAULT:
            if self.current_state != self.ERROR:
                return self._do_transition(self.ERROR, event.value)
            return False

        return super().trigger(event.value, data)

    def can_trigger(self, event: EventType) -> bool:
        """检查当前状态下是否可以触发指定事件"""
        if event == EventType.SENSOR_FAULT:
            return True
        return self.can_handle_event(event.value)

    def reset(self):
        """重置状态机到初始状态"""
        self.context.clear()
        self._do_transition(self.IDLE, None)

    # === 上下文数据管理 ===

    def set_context(self, key: str, value: Any):
        """设置上下文数据"""
        self.context.set(key, value)
        self._last_update_time = time()

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文数据"""
        return self.context.get(key, default)

    def clear_context(self):
        """清空上下文数据"""
        self.context.clear()

    # === 回调管理（兼容原有 API）===

    def add_callback(self, callback: Callable[[RobotState, RobotState, Optional[EventType]], None]):
        """
        添加状态变更回调

        回调函数签名: callback(old_state, new_state, event)
        """
        def wrapper(old_state: str, new_state: str, event: Optional[str]):
            try:
                callback(
                    RobotState(old_state),
                    RobotState(new_state),
                    EventType(event) if event else None
                )
            except Exception as e:
                print(f"[StateMachine] Callback error: {e}")

        self._callbacks_wrapped[callback] = wrapper
        self.add_state_change_callback(wrapper)

    def remove_callback(self, callback: Callable):
        """移除状态变更回调"""
        if callback in self._callbacks_wrapped:
            wrapper = self._callbacks_wrapped[callback]
            self.remove_state_change_callback(wrapper)
            del self._callbacks_wrapped[callback]

    _callbacks_wrapped: Dict[Callable, Callable] = {}

    # === 历史记录 ===

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取状态变更历史"""
        return self.context.history[-limit:]

    # === 工具方法 ===

    @property
    def last_update_time(self) -> float:
        """获取最后更新时间"""
        return self._last_update_time

    def is_idle(self) -> bool:
        """是否处于空闲状态"""
        return self.is_in_state(self.IDLE)

    def is_error(self) -> bool:
        """是否处于错误状态"""
        return self.is_in_state(self.ERROR)

    def is_running(self) -> bool:
        """是否正在执行任务（非IDLE、FINISHED、ERROR）"""
        return self.current_state not in (self.IDLE, self.FINISHED, self.ERROR)

    def get_info(self) -> Dict[str, Any]:
        """获取状态机完整信息"""
        return {
            "state": self.current_state,
            "last_update": self._last_update_time,
            "context": dict(self.context.data),
            "history_count": len(self.context.history),
        }


# === 向后兼容：保留原有 StateMachine 类名 ===
StateMachine = RobotStateMachine


# === 全局单例 ===

_state_machine: Optional[StateMachine] = None


def get_state_machine() -> StateMachine:
    """获取全局状态机实例"""
    global _state_machine
    if _state_machine is None:
        _state_machine = StateMachine()
    return _state_machine


# === 便捷函数 ===

def get_state() -> RobotState:
    """获取当前状态"""
    return get_state_machine().state


def set_state(new_state: RobotState, reason: str = "") -> bool:
    """设置状态"""
    return get_state_machine().set_state(new_state, reason)


def trigger_event(event: EventType, data: Any = None) -> bool:
    """触发事件"""
    return get_state_machine().trigger(event, data)


def on_state_change(callback: Callable[[RobotState, RobotState, Optional[EventType]], None]):
    """注册状态变更回调"""
    get_state_machine().add_callback(callback)
