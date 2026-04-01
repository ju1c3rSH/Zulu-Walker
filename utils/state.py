# -*- coding: utf-8 -*-
"""
全局状态机模块

提供全局单例状态机，用于跨模块共享和修改机器人状态。
"""
import enum
import threading
from typing import Dict, Any, Optional, Callable, List
from time import time


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


# 状态转换表：定义每个状态下接收到事件后应转换到哪个状态
# 格式: {当前状态: {事件: 目标状态}}
STATE_TRANSITIONS: Dict[RobotState, Dict[EventType, RobotState]] = {
    RobotState.IDLE: {
        EventType.START: RobotState.READ_QR,
    },
    RobotState.READ_QR: {
        EventType.QR_DECODED: RobotState.NAV_TO_RAW,
    },
    RobotState.NAV_TO_RAW: {
        EventType.ARRIVED_AT_ZONE: RobotState.WAIT_TURNTABLE,
    },
    RobotState.WAIT_TURNTABLE: {
        EventType.TURNTABLE_READY: RobotState.PICK_AND_LOAD,
    },
    RobotState.PICK_AND_LOAD: {
        EventType.LOAD_SUCCESS: RobotState.NAV_TO_PROC,
    },
    RobotState.NAV_TO_PROC: {
        EventType.ARRIVED_AT_ZONE: RobotState.UNLOAD_TO_PROC,
    },
    RobotState.UNLOAD_TO_PROC: {
        EventType.UNLOAD_SUCCESS: RobotState.NAV_TO_TEMP,
    },
    RobotState.NAV_TO_TEMP: {
        EventType.ARRIVED_AT_ZONE: RobotState.STACKING,
    },
    RobotState.STACKING: {
        EventType.STACK_SUCCESS: RobotState.RETURN_HOME,
        EventType.ALL_BATCHES_DONE: RobotState.FINISHED,
    },
    RobotState.RETURN_HOME: {
        EventType.HOME_REACHED: RobotState.FINISHED,
    },
    RobotState.ERROR: {
        EventType.RECOVERY_OK: RobotState.IDLE,
    },
    RobotState.RECOVERING: {
        EventType.RECOVERY_OK: RobotState.IDLE,
    },
    # 任何状态都可以接收 SENSOR_FAULT 进入 ERROR
}


class StateMachine:
    """
    全局状态机（单例）

    管理机器人的全局状态，支持：
    - 状态查询和修改
    - 事件驱动的状态转换
    - 状态变更回调通知
    - 线程安全访问
    """

    _instance: Optional["StateMachine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._state = RobotState.IDLE
        self._state_lock = threading.RLock()
        self._callbacks: List[Callable[[RobotState, RobotState, Optional[EventType]], None]] = []
        self._context: Dict[str, Any] = {}  # 状态相关的上下文数据
        self._last_update_time: float = time()
        self._state_history: List[Dict[str, Any]] = []  # 状态历史记录

    @property
    def state(self) -> RobotState:
        """获取当前状态"""
        with self._state_lock:
            return self._state

    def set_state(self, new_state: RobotState, reason: str = "") -> bool:
        """
        直接设置状态（不通过事件触发）

        Args:
            new_state: 目标状态
            reason: 状态变更原因

        Returns:
            bool: 是否成功变更
        """
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return False

            self._state = new_state
            self._last_update_time = time()
            self._record_history(old_state, new_state, None, reason)

        self._notify_callbacks(old_state, new_state, None)
        return True

    def trigger(self, event: EventType, data: Any = None) -> bool:
        """
        触发事件，尝试进行状态转换

        Args:
            event: 触发的事件
            data: 事件相关数据

        Returns:
            bool: 是否成功转换状态
        """
        with self._state_lock:
            old_state = self._state

            # 检查 SENSOR_FAULT 特殊情况（任何状态都可以进入 ERROR）
            if event == EventType.SENSOR_FAULT:
                new_state = RobotState.ERROR
            elif old_state in STATE_TRANSITIONS:
                transitions = STATE_TRANSITIONS[old_state]
                if event in transitions:
                    new_state = transitions[event]
                else:
                    print(f"[StateMachine] Event {event.value} not valid in state {old_state.value}")
                    return False
            else:
                print(f"[StateMachine] No transitions defined for state {old_state.value}")
                return False

            self._state = new_state
            self._last_update_time = time()
            self._record_history(old_state, new_state, event, data)

        self._notify_callbacks(old_state, new_state, event)
        return True

    def can_trigger(self, event: EventType) -> bool:
        """检查当前状态下是否可以触发指定事件"""
        with self._state_lock:
            if event == EventType.SENSOR_FAULT:
                return True
            return event in STATE_TRANSITIONS.get(self._state, {})

    def reset(self):
        """重置状态机到初始状态"""
        with self._state_lock:
            old_state = self._state
            self._state = RobotState.IDLE
            self._context.clear()
            self._last_update_time = time()
            self._record_history(old_state, RobotState.IDLE, None, "reset")

        self._notify_callbacks(old_state, RobotState.IDLE, None)

    # === 上下文数据管理 ===

    def set_context(self, key: str, value: Any):
        """设置上下文数据"""
        with self._state_lock:
            self._context[key] = value
            self._last_update_time = time()

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文数据"""
        with self._state_lock:
            return self._context.get(key, default)

    def clear_context(self):
        """清空上下文数据"""
        with self._state_lock:
            self._context.clear()

    # === 回调管理 ===

    def add_callback(self, callback: Callable[[RobotState, RobotState, Optional[EventType]], None]):
        """
        添加状态变更回调

        回调函数签名: callback(old_state, new_state, event)
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """移除状态变更回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, old_state: RobotState, new_state: RobotState, event: Optional[EventType]):
        """通知所有回调"""
        for callback in self._callbacks:
            try:
                callback(old_state, new_state, event)
            except Exception as e:
                print(f"[StateMachine] Callback error: {e}")

    # === 历史记录 ===

    def _record_history(self, old_state: RobotState, new_state: RobotState,
                        event: Optional[EventType], data: Any):
        """记录状态变更历史"""
        record = {
            "timestamp": time(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "event": event.value if event else None,
            "data": str(data)[:100] if data else None,
        }
        self._state_history.append(record)
        # 保留最近100条记录
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取状态变更历史"""
        with self._state_lock:
            return self._state_history[-limit:]

    # === 工具方法 ===

    @property
    def last_update_time(self) -> float:
        """获取最后更新时间"""
        return self._last_update_time

    def is_idle(self) -> bool:
        """是否处于空闲状态"""
        return self.state == RobotState.IDLE

    def is_error(self) -> bool:
        """是否处于错误状态"""
        return self.state == RobotState.ERROR

    def is_running(self) -> bool:
        """是否正在执行任务（非IDLE、FINISHED、ERROR）"""
        return self.state not in (RobotState.IDLE, RobotState.FINISHED, RobotState.ERROR)

    def get_info(self) -> Dict[str, Any]:
        """获取状态机完整信息"""
        with self._state_lock:
            return {
                "state": self._state.value,
                "last_update": self._last_update_time,
                "context": dict(self._context),
                "history_count": len(self._state_history),
            }


# 全局单例实例
_state_machine: Optional[StateMachine] = None


def get_state_machine() -> StateMachine:
    """获取全局状态机实例"""
    global _state_machine
    if _state_machine is None:
        _state_machine = StateMachine()
    return _state_machine


# 便捷函数
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
