# -*- coding: utf-8 -*-
"""
通用 State → Action 桥接层

声明式绑定状态转换与动作执行，不绑定任何特定状态机实现。
"""

from typing import Callable, Optional, Union, Set
from .base import BaseStateMachine


class StateActionBridge:
    """
    监听 BaseStateMachine 的状态变化，自动触发预注册的动作。

    Usage:
        bridge = StateActionBridge(mission_sm)
        bridge.when_enter("READ_QR", lambda: activate("qr_detect"))
        bridge.when_enter({"ALIGN_RAW", "ALIGN_TEMP"}, lambda: activate("track"))
        bridge.when_enter({"NAV_TO_RAW", "ERROR"}, deactivate_all)
    """

    def __init__(self, sm: BaseStateMachine):
        self._enter_actions: dict = {}
        self._exit_actions: dict = {}
        sm.add_state_change_callback(self._on_change)

    def when_enter(
        self,
        states: Union[str, Set[str]],
        action: Callable[[], None]
    ) -> None:
        if isinstance(states, str):
            self._enter_actions[states] = action
        else:
            for s in states:
                self._enter_actions[s] = action

    def when_exit(
        self,
        states: Union[str, Set[str]],
        action: Callable[[], None]
    ) -> None:
        if isinstance(states, str):
            self._exit_actions[states] = action
        else:
            for s in states:
                self._exit_actions[s] = action

    def _on_change(self, old: str, new: str, event: Optional[str]) -> None:
        if new in self._enter_actions:
            self._enter_actions[new]()
        if old in self._exit_actions:
            self._exit_actions[old]()
