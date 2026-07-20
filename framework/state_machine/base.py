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
from framework.log import fw_log as log_print


class State(ABC):
    """
    状态基类

    子类需要实现 on_enter, on_execute, on_exit 方法。
    on_execute 返回目标状态名时，会自动触发状态转换。
    """

    @abstractmethod
    def on_enter(self, context: Any, from_state: str) -> None:
        pass

    @abstractmethod
    def on_execute(self, context: Any) -> Optional[str]:
        pass

    @abstractmethod
    def on_exit(self, context: Any, to_state: str) -> None:
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
        with self._lock:
            if state_name not in self._states:
                raise ValueError(f"State '{state_name}' not registered")
            self._current_state = state_name
            self._state_entry_time = time()
            self._state_frame_count = 0

    # === 核心操作 ===

    def trigger(self, event: str, data: Any = None) -> bool:
        with self._lock:
            if self._current_state is None:
                return False

            event_map = self._event_transitions.get(self._current_state, {})
            if event not in event_map:
                return False

            target_state = event_map[event]

            if self._context is not None:
                setattr(self._context, 'event_data', data)
                setattr(self._context, 'triggered_event', event)

            return self._do_transition(target_state, event)

    def update(self) -> None:
        with self._lock:
            if self._current_state is None:
                return

            current_state_obj = self._states.get(self._current_state)
            if current_state_obj is None:
                return

            self._state_frame_count += 1

            target_state = current_state_obj.on_execute(self._context)

            if target_state is not None and target_state != self._current_state:
                self._do_transition(target_state, None)
                return

            transitions = self._transitions.get(self._current_state, {})
            for transition in transitions.values():
                if transition.condition and transition.condition(self._context):
                    self._do_transition(transition.to_state, None)
                    return

    def run_to_completion(self, max_steps: int = 10) -> int:
        steps = 0
        for _ in range(max_steps):
            prev = self._current_state
            self.update()
            if self._current_state == prev:
                break
            steps += 1
        return steps

    def _do_transition(self, target_state: str, event: Optional[str]) -> bool:
        if target_state not in self._states:
            return False

        if self._current_state == target_state:
            return False

        old_state = self._current_state
        new_state = target_state

        if old_state and old_state in self._states:
            self._states[old_state].on_exit(self._context, new_state)
            self._trigger_exit_callbacks(old_state, new_state)

        self._previous_state = old_state
        self._current_state = new_state
        self._state_entry_time = time()
        self._state_frame_count = 0

        self._states[new_state].on_enter(self._context, old_state or "")
        self._trigger_enter_callbacks(new_state, old_state or "")

        self._notify_callbacks(old_state or "", new_state, event)

        return True

    # === 回调管理 ===

    def add_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        self._state_callbacks.append(callback)

    def remove_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def on_state_enter(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        self.add_enter_callback(state, callback)

    def on_state_exit(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        self.add_exit_callback(state, callback)

    def on_state_change(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        self.add_state_change_callback(callback)

    def add_enter_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        if state_name not in self._enter_callbacks:
            self._enter_callbacks[state_name] = []
        self._enter_callbacks[state_name].append(callback)

    def add_exit_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        if state_name not in self._exit_callbacks:
            self._exit_callbacks[state_name] = []
        self._exit_callbacks[state_name].append(callback)

    def _notify_callbacks(self, old_state: str, new_state: str, event: Optional[str]) -> None:
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state, event)
            except Exception as e:
                log_print(f"[StateMachine] Callback error: {e}")

    def _trigger_enter_callbacks(self, state_name: str, from_state: str) -> None:
        for callback in self._enter_callbacks.get(state_name, []):
            try:
                callback(self._context, from_state)
            except Exception as e:
                log_print(f"[StateMachine] Enter callback error: {e}")

    def _trigger_exit_callbacks(self, state_name: str, to_state: str) -> None:
        for callback in self._exit_callbacks.get(state_name, []):
            try:
                callback(self._context, to_state)
            except Exception as e:
                log_print(f"[StateMachine] Exit callback error: {e}")

    # === 属性访问 ===

    @property
    def current_state(self) -> Optional[str]:
        with self._lock:
            return self._current_state

    @property
    def previous_state(self) -> Optional[str]:
        with self._lock:
            return self._previous_state

    @property
    def context(self) -> Any:
        return self._context

    @context.setter
    def context(self, value: Any) -> None:
        self._context = value

    @property
    def state_duration(self) -> float:
        return time() - self._state_entry_time

    @property
    def state_frame_count(self) -> int:
        return self._state_frame_count

    def is_in_state(self, state_name: str) -> bool:
        with self._lock:
            return self._current_state == state_name

    def can_handle_event(self, event: str) -> bool:
        with self._lock:
            if self._current_state is None:
                return False
            event_map = self._event_transitions.get(self._current_state, {})
            return event in event_map

    def reset(self) -> None:
        with self._lock:
            self._current_state = None
            self._previous_state = None
            self._state_entry_time = 0.0
            self._state_frame_count = 0
