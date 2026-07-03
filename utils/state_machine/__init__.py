# -*- coding: utf-8 -*-
"""
状态机框架

提供可复用的状态机基类和视觉跟踪专用状态机实现。

## 模块结构

- base.py: 通用状态机基类 (BaseStateMachine, State)
- visual_state_machine.py: 视觉跟踪状态机 (VisualStateMachine, VisualContext)

## 快速开始

```python
from utils.state_machine import VisualStateMachine, VisualContext

# 创建状态机
sm = VisualStateMachine()

# 注册回调
def on_search(context, from_state):
    print("进入搜索状态")
    # 启用全图检测任务...

sm.on_state_enter(VisualStateMachine.States.SEARCH, on_search)

# 启动
sm.start()  # IDLE -> SEARCH

# 在异步循环中更新
async def loop():
    while True:
        # 更新上下文
        sm.context.target_found = detect_result.success
        sm.context.confidence = detect_result.confidence

        # 触发事件
        if sm.is_searching() and sm.context.target_found:
            sm.trigger(VisualStateMachine.Events.TARGET_FOUND)
```
"""

from .base import BaseStateMachine, State, Transition
from .visual_state_machine import (
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
    "VisualStateMachine",
    "VisualContext",
    "IdleState",
    "SearchState",
    "TrackingState",
    "RecoveryState",
    "FailState",
]
