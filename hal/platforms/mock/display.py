from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MockDisplay:
    def show(self, frame: np.ndarray) -> bool:
        logger.debug("MockDisplay.show() frame=%s", frame.shape if frame is not None else "None")
        return True

    def close(self) -> None:
        logger.info("MockDisplay closed")
