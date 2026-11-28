"""Read a frame without knowing which platform produced it.

Three shapes reach the pipeline: a maix Image (zero-copy on the device), a
``LazyFrame`` proxy, and a BGR ndarray (the desktop, and anywhere cv2 runs).
They agree on nothing except that a frame has a size and can be copied, which
is all display composition and inference input preparation need.

Keeping that knowledge here means no consumer has to guess: a wrong guess used
to surface as ``AttributeError: 'numpy.ndarray' object has no attribute
'height'`` on one platform and work fine on the other.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = ["frame_size", "frame_for_display"]


def frame_size(frame: Any) -> Tuple[int, int]:
    """Return ``(width, height)`` of a platform frame.

    A maix Image exposes ``width()`` and ``height()`` as methods; ndarray and
    ``LazyFrame`` expose ``shape``.  Anything else yields ``(0, 0)`` so the
    caller can skip a frame instead of raising inside the vision thread.
    """
    width_getter = getattr(frame, "width", None)
    height_getter = getattr(frame, "height", None)
    if callable(width_getter) and callable(height_getter):
        return int(width_getter()), int(height_getter())

    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])

    return 0, 0


def frame_for_display(camera: Any) -> Optional[Any]:
    """Return the newest frame display composition may copy.

    ``last_raw`` is the platform's native buffer, and it is None wherever maix
    does not exist - that is what the Camera protocol promises, and the
    inference path relies on the distinction to pick its zero-copy route.

    Display needs only "a frame that can be copied", so this falls back to
    ``last_frame``, the BGR ndarray the capture thread already kept.  Reading
    it does not consume a frame from the queue.

    ``last_frame`` is not part of the Camera protocol: it exists on the
    platforms that keep a BGR view (linux, mock) and not on the ones whose
    ``last_raw`` is always a native image (maixcam1, maixcam2).  That is why
    the fallback is a ``getattr`` rather than an attribute access.
    """
    if camera is None:
        return None
    frame = getattr(camera, "last_raw", None)
    if frame is not None:
        return frame
    return getattr(camera, "last_frame", None)
