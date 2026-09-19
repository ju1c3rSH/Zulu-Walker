# -*- coding: utf-8 -*-
"""Phase A2/B2 新增组件测试: 订阅门控引擎 / 页面状态机 / 运行时参数。"""
from types import SimpleNamespace

from app.page_router import AppPageState
from app.page_sm import PAGE_CYCLE, PageStateMachine, STATE_MAIN
from app.subscription_sm import GateSignal, SubscriptionStateMachine
from modules.zw_opencv_module.processors.gks_task_processor import GksTaskProcessor


class StubCoordinator:
    def __init__(self):
        self.applied = []

    def apply_gate_task(self, mode, codes):
        self.applied.append(("task", mode, codes))

    def apply_gate_idle(self):
        self.applied.append(("idle",))


def test_subscription_sm_matches_sub_state():
    machine = SubscriptionStateMachine(StubCoordinator())
    assert machine.current_state == "IDLE"
    machine.dispatch(GateSignal(state_name="EXPLOSIVE", codes=(2, 1, 3)))
    assert machine.current_state == "EXPLOSIVE"
    machine.dispatch(GateSignal(state_name="RESCUE", codes=(0, 0, 3)))
    assert machine.current_state == "RESCUE"
    machine.dispatch(GateSignal(is_stop=True))
    assert machine.current_state == "IDLE"
    machine.dispatch(GateSignal(is_stop=True))
    assert machine.current_state == "IDLE"


def test_subscription_sm_same_state_direct_apply():
    stub = StubCoordinator()
    machine = SubscriptionStateMachine(stub)
    machine.dispatch(GateSignal(state_name="TARGET", codes=(5, 5, 5)))
    first_calls = len(stub.applied)
    assert machine.current_state == "TARGET"
    machine.dispatch(GateSignal(state_name="TARGET", codes=(1, 2, 3)))
    assert machine.current_state == "TARGET"
    assert (len(stub.applied), stub.applied[-1]) == (first_calls + 1, ("task", "TARGET", (1, 2, 3)))


def test_page_state_machine_pause_on_debug_pages():
    vm = SimpleNamespace(paused=False, set_paused=lambda v: setattr(vm, "paused", v))
    pages = PageStateMachine(vm)
    assert pages.current_state == STATE_MAIN
    for target in PAGE_CYCLE[1:]:
        assert pages.select(target) is True
        assert vm.paused is True
    assert pages.select(STATE_MAIN) is True
    assert pages.current_state == STATE_MAIN
    assert vm.paused is False


def test_page_cycle_order_with_camera_gate():
    class CycleVM:
        def set_paused(self, value):
            pass

    pages = AppPageState(CycleVM())
    pages.set_camera_gate(lambda: True)
    order = []
    for _ in range(4):
        order.append(pages.cycle())
    assert order == ["EXPLOSIVE", "TARGET", "RESCUE", STATE_MAIN]
    assert pages.name == STATE_MAIN


def test_page_cycle_refused_gate_should_stay():
    class CycleVM:
        def set_paused(self, value):
            pass

    pages = AppPageState(CycleVM())
    pages.set_camera_gate(lambda: False)
    for _ in range(4):
        assert pages.cycle() == STATE_MAIN
    assert pages.name == STATE_MAIN


def test_runtime_param_enabled():
    processor = GksTaskProcessor()
    processor.set_runtime_param("rescue", "conf_th", 0.55)
    assert abs(processor.config["rescue"]["conf_th"] - 0.55) < 1e-9
    processor.set_runtime_param("pit_left_is_1", "pit_left_is_1", False)
    assert processor.config["pit_left_is_1"] is False
    try:
        processor.set_runtime_param("rescue", "nope", 1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pit_latch_state_machine_fields():
    processor = GksTaskProcessor()
    status = processor.pit_status
    assert status["pit"] == 0 and status["pit_flag"] == 0
