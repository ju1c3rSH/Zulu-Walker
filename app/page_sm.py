# -*- coding: utf-8 -*-
"""UI 页面状态机（framework/state_machine 引擎承载）。

四态: MAIN 正常运行 / EXPLOSIVE / TARGET / RESCUE 三调试页。
页面语义: 调试页独占相机（enter 暂停视觉采集）, 回 MAIN 恢复;
页面切换由 app.main DebugPageFactory 驱动, UI 实例由其懒建。
"""
from framework.state_machine.base import BaseStateMachine, State


STATE_MAIN = "MAIN"
_DEBUG_STATES = ("EXPLOSIVE", "TARGET", "RESCUE")
PAGE_CYCLE = (STATE_MAIN,) + _DEBUG_STATES


class PageContext:
    """页面上下文: 持 VisionManager（暂停/恢复采集的执行者）。"""

    def __init__(self, vision_manager):
        self.vision_manager = vision_manager


class MainState(State):
    """正常运行页: 视觉采集恢复、HUD+推流正常工作。"""

    def on_enter(self, context: PageContext, from_state: str) -> None:
        if context.vision_manager:
            context.vision_manager.set_paused(False)

    def on_execute(self, context: PageContext):
        return None

    def on_exit(self, context: PageContext, to_state: str) -> None:
        pass


class DebugPageState(State):
    """调试页: 暂停视觉采集, 相机让位为调试页唯一读取者。"""

    def on_enter(self, context: PageContext, from_state: str) -> None:
        if context.vision_manager:
            context.vision_manager.set_paused(True)

    def on_execute(self, context: PageContext):
        return None

    def on_exit(self, context: PageContext, to_state: str) -> None:
        pass


class PageStateMachine(BaseStateMachine):
    """四页状态机; 页面切换事件为 GOTO_<目标页>。"""

    def __init__(self, vision_manager):
        super().__init__()
        self.context = PageContext(vision_manager)
        self.register_state(STATE_MAIN, MainState())
        for state_name in _DEBUG_STATES:
            self.register_state(state_name, DebugPageState())
            for from_state in (STATE_MAIN,) + _DEBUG_STATES:
                if from_state != state_name:
                    self.register_transition(
                        from_state, state_name, event=f"GOTO_{state_name}")
        for from_state in _DEBUG_STATES:
            self.register_transition(
                from_state, STATE_MAIN, event=f"GOTO_{STATE_MAIN}")
        self.set_initial_state(STATE_MAIN)

    def select(self, target_name: str) -> bool:
        """切页（已在该页时为无操作）。返回是否发生切换。"""
        if self.current_state == target_name:
            return False
        return self.trigger(f"GOTO_{target_name}")
