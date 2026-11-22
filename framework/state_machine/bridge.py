# -*- coding: utf-8 -*-
"""Declaratively bind state changes to actions.

The bridge listens to a BaseStateMachine through its state-change callbacks
and invokes registered actions; it contains no mission-specific logic.
"""

from typing import Callable, Optional, Union, Set
from framework.state_machine.base import BaseStateMachine


class StateActionBridge:
    """Run registered actions when observed states are entered or exited.

    Example:
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
