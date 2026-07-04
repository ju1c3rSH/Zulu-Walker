# -*- coding: utf-8 -*-
"""
状态机框架

提供可复用的状态机基类和通用桥接工具。

- base.py: 通用状态机基类 (BaseStateMachine, State)
- bridge.py: State → Action 桥接层 (StateActionBridge)

VisualStateMachine 已移至 context/ 包，本模块通过 re-export 保持向后兼容。
"""

from .base import BaseStateMachine, State, Transition
from .bridge import StateActionBridge

# re-export VisualStateMachine from context/ for backward compatibility
from context.visual_state_machine import (
    VisualStateMachine,
    VisualContext,
    IdleState,
    SearchState,
    TrackingState,
    RecoveryState,
    FailState,
)

__all__ = [
    "BaseStateMachine",
    "State",
    "Transition",
    "StateActionBridge",
    "VisualStateMachine",
    "VisualContext",
    "IdleState",
    "SearchState",
    "TrackingState",
    "RecoveryState",
    "FailState",
]
