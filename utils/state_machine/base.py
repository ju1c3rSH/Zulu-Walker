# -*- coding: utf-8 -*-
"""
通用状态机框架

提供可复用的状态机基类，支持状态转换、事件触发和回调机制。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Set
from enum import Enum
from dataclasses import dataclass
from time import time
import threading


class State(ABC):
    """
    状态基类

    子类需要实现 on_enter, on_execute, on_exit 方法。
    on_execute 返回目标状态名时，会自动触发状态转换。
    """

    @abstractmethod
    def on_enter(self, context: Any, from_state: str) -> None:
        """
        进入状态时调用

        Args:
            context: 状态机上下文
            from_state: 来源状态名
        """
        pass

    @abstractmethod
    def on_execute(self, context: Any) -> Optional[str]:
        """
        状态执行逻辑（每帧调用）

        Args:
            context: 状态机上下文

        Returns:
            目标状态名（触发转换）或 None（保持当前状态）
        """
        pass

    @abstractmethod
    def on_exit(self, context: Any, to_state: str) -> None:
        """
        退出状态时调用

        Args:
            context: 状态机上下文
            to_state: 目标状态名
        """
        pass


@dataclass
class Transition:
    """状态转换定义"""
    from_state: str
    to_state: str
    condition: Optional[Callable[[Any], bool]] = None
    event: Optional[str] = None


class BaseStateMachine(ABC):
    """
    通用状态机基类

    提供状态注册、转换管理和事件触发功能。
    子类需要实现具体的状态和转换逻辑。

    Example:
        >>> class MyStateMachine(BaseStateMachine):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.register_state("IDLE", IdleState())
        ...         self.register_state("RUNNING", RunningState())
        ...         self.set_initial_state("IDLE")
        ...
        >>> sm = MyStateMachine()
        >>> sm.trigger("START")  # 触发事件
        >>> sm.update()  # 执行当前状态
    """

    def __init__(self):
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, Dict[str, Transition]] = {}
        self._event_transitions: Dict[str, Dict[str, str]] = {}
        self._current_state: Optional[str] = None
        self._previous_state: Optional[str] = None
        self._context: Any = None

        self._lock = threading.RLock()
        self._state_callbacks: List[Callable[[str, str, Optional[str]], None]] = []
        self._enter_callbacks: Dict[str, List[Callable[[Any, str], None]]] = {}
        self._exit_callbacks: Dict[str, List[Callable[[Any, str], None]]] = {}

        self._state_entry_time: float = 0.0
        self._state_frame_count: int = 0

    # === 状态注册 ===

    def register_state(self, name: str, state: State) -> None:
        """注册状态"""
        with self._lock:
            self._states[name] = state
            if name not in self._transitions:
                self._transitions[name] = {}

    def register_transition(
        self,
        from_state: str,
        to_state: str,
        condition: Optional[Callable[[Any], bool]] = None,
        event: Optional[str] = None
    ) -> None:
        """
        注册状态转换

        Args:
            from_state: 源状态名
            to_state: 目标状态名
            condition: 自动转换条件函数（可选）
            event: 触发事件名（可选）
        """
        with self._lock:
            if from_state not in self._transitions:
                self._transitions[from_state] = {}

            self._transitions[from_state][to_state] = Transition(
                from_state=from_state,
                to_state=to_state,
                condition=condition,
                event=event
            )

            if event:
                if from_state not in self._event_transitions:
                    self._event_transitions[from_state] = {}
                self._event_transitions[from_state][event] = to_state

    def set_initial_state(self, state_name: str) -> None:
        """设置初始状态"""
        with self._lock:
            if state_name not in self._states:
                raise ValueError(f"State '{state_name}' not registered")
            self._current_state = state_name
            self._state_entry_time = time()
            self._state_frame_count = 0

    # === 核心操作 ===

    def trigger(self, event: str, data: Any = None) -> bool:
        """
        触发事件，尝试状态转换

        Args:
            event: 事件名
            data: 事件数据（存入 context.event_data）

        Returns:
            是否成功转换状态
        """
        with self._lock:
            if self._current_state is None:
                return False

            # 检查当前状态是否响应该事件
            event_map = self._event_transitions.get(self._current_state, {})
            if event not in event_map:
                return False

            target_state = event_map[event]

            # 设置事件数据
            if self._context is not None:
                setattr(self._context, 'event_data', data)
                setattr(self._context, 'triggered_event', event)

            return self._do_transition(target_state, event)

    def update(self) -> None:
        """
        更新状态机（应在主循环中每帧调用）

        执行当前状态的 on_execute，并检查自动转换条件。
        """
        with self._lock:
            if self._current_state is None:
                return

            current_state_obj = self._states.get(self._current_state)
            if current_state_obj is None:
                return

            self._state_frame_count += 1

            # 执行当前状态
            target_state = current_state_obj.on_execute(self._context)

            # 检查状态执行是否请求转换
            if target_state is not None and target_state != self._current_state:
                self._do_transition(target_state, None)
                return

            # 检查自动转换条件
            transitions = self._transitions.get(self._current_state, {})
            for transition in transitions.values():
                if transition.condition and transition.condition(self._context):
                    self._do_transition(transition.to_state, None)
                    return

    def run_to_completion(self, max_steps: int = 10) -> int:
        """持续调用 update() 直到状态不再变化。

        当 on_execute 在一次 enter 后连级联触发多次自动转换时
        (如 RING_DISCOVERY → ALIGN_ROUGH → PLACE_ROUGH),
        外部不需要显式多次调用 update()。

        有界循环 (max_steps) 防止无限级联, 死循环保护。
        兼容 PLACE_*/CHECK_LOAD 等 on_execute 返回 None 的稳定状态。

        Returns: 执行的转换次数。
        """
        steps = 0
        for _ in range(max_steps):
            prev = self._current_state
            self.update()
            if self._current_state == prev:
                break
            steps += 1
        return steps

    def _do_transition(self, target_state: str, event: Optional[str]) -> bool:
        """执行状态转换"""
        if target_state not in self._states:
            return False

        if self._current_state == target_state:
            return False

        old_state = self._current_state
        new_state = target_state

        # 退出当前状态
        if old_state and old_state in self._states:
            self._states[old_state].on_exit(self._context, new_state)
            self._trigger_exit_callbacks(old_state, new_state)

        # 更新状态
        self._previous_state = old_state
        self._current_state = new_state
        self._state_entry_time = time()
        self._state_frame_count = 0

        # 进入新状态
        self._states[new_state].on_enter(self._context, old_state or "")
        self._trigger_enter_callbacks(new_state, old_state or "")

        # 通知全局回调
        self._notify_callbacks(old_state or "", new_state, event)

        return True

    # === 回调管理 ===

    def add_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """
        添加全局状态变更回调

        Args:
            callback: 回调函数，参数为 (old_state, new_state, event)
        """
        self._state_callbacks.append(callback)

    def remove_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """移除全局状态变更回调"""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def on_state_enter(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """进入指定状态时触发回调 (callback(context, from_state))"""
        self.add_enter_callback(state, callback)

    def on_state_exit(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """退出指定状态时触发回调 (callback(context, to_state))"""
        self.add_exit_callback(state, callback)

    def on_state_change(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """任意状态发生变化时触发回调 (callback(old_state, new_state, event))"""
        self.add_state_change_callback(callback)

    def add_enter_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """添加状态进入回调"""
        if state_name not in self._enter_callbacks:
            self._enter_callbacks[state_name] = []
        self._enter_callbacks[state_name].append(callback)

    def add_exit_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """添加状态退出回调"""
        if state_name not in self._exit_callbacks:
            self._exit_callbacks[state_name] = []
        self._exit_callbacks[state_name].append(callback)

    def _notify_callbacks(self, old_state: str, new_state: str, event: Optional[str]) -> None:
        """通知全局回调"""
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state, event)
            except Exception as e:
                print(f"[StateMachine] Callback error: {e}")

    def _trigger_enter_callbacks(self, state_name: str, from_state: str) -> None:
        """触发状态进入回调"""
        for callback in self._enter_callbacks.get(state_name, []):
            try:
                callback(self._context, from_state)
            except Exception as e:
                print(f"[StateMachine] Enter callback error: {e}")

    def _trigger_exit_callbacks(self, state_name: str, to_state: str) -> None:
        """触发状态退出回调"""
        for callback in self._exit_callbacks.get(state_name, []):
            try:
                callback(self._context, to_state)
            except Exception as e:
                print(f"[StateMachine] Exit callback error: {e}")

    # === 属性访问 ===

    @property
    def current_state(self) -> Optional[str]:
        """当前状态名"""
        with self._lock:
            return self._current_state

    @property
    def previous_state(self) -> Optional[str]:
        """上一个状态名"""
        with self._lock:
            return self._previous_state

    @property
    def context(self) -> Any:
        """状态机上下文"""
        return self._context

    @context.setter
    def context(self, value: Any) -> None:
        """设置状态机上下文"""
        self._context = value

    @property
    def state_duration(self) -> float:
        """当前状态持续时间（秒）"""
        return time() - self._state_entry_time

    @property
    def state_frame_count(self) -> int:
        """当前状态已执行帧数"""
        return self._state_frame_count

    def is_in_state(self, state_name: str) -> bool:
        """检查是否处于指定状态"""
        with self._lock:
            return self._current_state == state_name

    def can_handle_event(self, event: str) -> bool:
        """检查当前状态是否可以处理指定事件"""
        with self._lock:
            if self._current_state is None:
                return False
            event_map = self._event_transitions.get(self._current_state, {})
            return event in event_map

    def reset(self) -> None:
        """重置状态机"""
        with self._lock:
            self._current_state = None
            self._previous_state = None
            self._state_entry_time = 0.0
            self._state_frame_count = 0
