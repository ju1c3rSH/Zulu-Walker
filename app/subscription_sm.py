# -*- coding: utf-8 -*-
"""订阅门控状态机。

以 framework/state_machine/base.py 引擎承载 GKS 从站门控四态:
IDLE / EXPLOSIVE / TARGET / RESCUE 单激活。门控变更（enable_task/disable_task/
set_task/reset）在各状态 on_enter 与同状态重订阅/已 IDLE 的 STOP 直接落地执行;
推送流控（seq/interval/零值心跳）不属于本状态机。
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from framework.state_machine.base import BaseStateMachine, State


STATE_IDLE = "IDLE"
STOP_EVENT = "CMD_STOP"

_TASK_STATES = ("EXPLOSIVE", "TARGET", "RESCUE")


def request_event_name(state_name: str) -> str:
    return f"REQ_{state_name}"


@dataclass
class GateSignal:
    """门控指令: 订阅(REQ)或停推(STOP), 附任务码快照。"""
    state_name: Optional[str] = None
    codes: Tuple[int, int, int] = (0, 0, 0)
    is_stop: bool = False


class SubscriptionContext:
    """状态机上下文: 持 coordinator 与最近一次门控信号。"""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.signal: Optional[GateSignal] = None


class IdleGateState(State):
    """未订阅: 关任务门并清处理器锁存。"""

    def on_enter(self, context: SubscriptionContext, from_state: str) -> None:
        context.coordinator.apply_gate_idle()

    def on_execute(self, context: SubscriptionContext):
        return None

    def on_exit(self, context: SubscriptionContext, to_state: str) -> None:
        pass


class TaskGateState(State):
    """单激活任务态: 进入即按任务码快照 enable gks_task 并切换处理器模式。"""

    def __init__(self, state_name: str):
        self._state_name = state_name

    def on_enter(self, context: SubscriptionContext, from_state: str) -> None:
        signal = context.signal
        codes = signal.codes if signal else (0, 0, 0)
        context.coordinator.apply_gate_task(self._state_name, codes)

    def on_execute(self, context: SubscriptionContext):
        return None

    def on_exit(self, context: SubscriptionContext, to_state: str) -> None:
        pass


class SubscriptionStateMachine(BaseStateMachine):
    """GKS 订阅门控四态机, 事件驱动, 跨任务态互切与 STOP 均 直达转换。"""

    def __init__(self, coordinator):
        super().__init__()
        self.context = SubscriptionContext(coordinator)
        self.register_state(STATE_IDLE, IdleGateState())
        for state_name in _TASK_STATES:
            self.register_state(state_name, TaskGateState(state_name))
            self.register_transition(state_name, STATE_IDLE, event=STOP_EVENT)
            for from_state in (STATE_IDLE,) + _TASK_STATES:
                if from_state != state_name:
                    self.register_transition(
                        from_state, state_name, event=request_event_name(state_name))
        self.set_initial_state(STATE_IDLE)

    def dispatch(self, signal: GateSignal) -> None:
        """应用一条门控信号; 同状态重订阅/已 IDLE 的 STOP 直接落地。"""
        self.context.signal = signal
        if signal.is_stop:
            if self.current_state == STATE_IDLE:
                self.context.coordinator.apply_gate_idle()
                return
            event = STOP_EVENT
        else:
            if self.current_state == signal.state_name:
                self.context.coordinator.apply_gate_task(
                    signal.state_name, signal.codes)
                return
            event = request_event_name(signal.state_name)
        if not self.trigger(event, signal):
            if signal.is_stop:
                self.context.coordinator.apply_gate_idle()
            else:
                self.context.coordinator.apply_gate_task(
                    signal.state_name, signal.codes)
