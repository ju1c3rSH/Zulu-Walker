from typing import Dict, Optional

import cv2
import numpy as np

from ..._shared.base_debug_window import BaseDebugWindow


class CircleTargetDebugWindow(BaseDebugWindow):
    def __init__(self, **kwargs):
        super().__init__(title="Circle Target Debug", method_count=4, **kwargs)
