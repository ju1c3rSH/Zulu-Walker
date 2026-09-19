# -*- coding: utf-8 -*-
"""页面路由 — KEY_OK 循环切页 + AppPageState 门面。

KEY_OK 在释放沿/未知沿循环切页（0.5s 连发兜底防抖；press 沿忽略防双翻转）。
进调试页前置条件: 流水线相机可用（调试页独占相机时相机让位为
调试页唯一读取者, 结果 >1s 陈旧由 coordinator 归零心跳自动兜底、UART 不断流）。
key 回调线程只 select 页面（页面 enter/exit 在 trigger 内同步完成, 仅置暂停位）。
"""
from app.page_sm import STATE_MAIN, PAGE_CYCLE, PageStateMachine
import time

from utils.log_util import log_print


PAGE_ENTER_DEBOUNCE_S = 0.5


class AppPageState:
    """页面状态门面（旧名兼容）: .name 读当前页, .select 切页。"""

    def __init__(self, vision_manager):
        self._sm = PageStateMachine(vision_manager)
        self._camera_gate = lambda: True

    @property
    def name(self) -> str:
        return self._sm.current_state or STATE_MAIN

    def is_main(self) -> bool:
        return self.name == STATE_MAIN

    def select(self, page_name: str) -> bool:
        return self._sm.select(page_name)

    def set_camera_gate(self, gate) -> None:
        """进调试页前置条件回调（相机不存在的页面切换被拒绝）。"""
        self._camera_gate = gate

    def cycle(self) -> str:
        """KEY_OK 循环切页（相机 gate 拒绝后停留在当前页）。"""
        target = PAGE_CYCLE[(PAGE_CYCLE.index(self.name) + 1) % len(PAGE_CYCLE)]
        if target != STATE_MAIN and not self._camera_gate():
            log_print(f"[KEY] enter page {target} refused (camera unavailable)")
            return self.name
        self._sm.select(target)
        return self.name


class KeyCycleRouter:
    """KEY_OK 循环切页路由: 边沿/防抖判定 + Key 实例持有防 GC。"""

    def __init__(self, pages: AppPageState):
        self._pages = pages
        self._last_state = None
        self._last_toggle = 0.0
        self._key = None
        self._install()

    def _install(self) -> None:
        try:
            from maix import key as maix_key
            self._key = maix_key.Key(self._on_key)
            log_print("[KEY] KEY_OK cycles pages (default exit-app disabled)")
        except Exception as e:
            log_print(f"[KEY] init fail: {e}")

    def _on_key(self, key_id, state) -> None:
        prev = self._last_state
        self._last_state = state
        try:
            log_print(f"[KEY] event id={key_id} state={state}")
        except Exception:
            pass
        if state == prev:
            return
        if _key_edge(state) == "press":
            return
        now = time.monotonic()
        if now - self._last_toggle < PAGE_ENTER_DEBOUNCE_S:
            return
        self._last_toggle = now
        try:
            page = self._pages.cycle()
            log_print(f"[KEY] page -> {page}")
        except Exception as e:
            log_print(f"[KEY] page toggle fail: {e}")


def _key_edge(state) -> str:
    """key 状态串化判沿（不依赖枚举对象）。"""
    s = str(state).upper()
    if "PRESS" in s:
        return "press"
    if "RELEASE" in s:
        return "release"
    return "unknown"
