from ..._shared.base_debug_window import BaseDebugWindow


class CargoDebugWindow(BaseDebugWindow):
    def __init__(self, **kwargs):
        super().__init__(title="Cargo Debug", method_count=2, **kwargs)
