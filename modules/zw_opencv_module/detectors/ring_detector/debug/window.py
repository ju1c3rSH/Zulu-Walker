from ..._shared.base_debug_window import BaseDebugWindow


class RingDebugWindow(BaseDebugWindow):
    def __init__(self, **kwargs):
        super().__init__(title="Ring Debug", method_count=3, **kwargs)
