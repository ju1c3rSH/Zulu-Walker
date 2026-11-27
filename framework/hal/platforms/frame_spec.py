"""The frame contract a platform hands to the inference kernel.

The configuration states what the camera delivers; the kernel converts that
into what the model wants.  Only the first half is platform knowledge, so the
parsing lives here instead of being copied into every platform package: the
three platforms disagree about almost everything, but they agree on this.
"""

from __future__ import annotations

from typing import Any

from framework.inference.runtime import FrameSpec

__all__ = ["DEFAULT_FRAME_SPEC", "frame_spec_from_config"]

# What an OpenCV pipeline produces, and therefore what a config that says
# nothing about its frame is assumed to deliver.
DEFAULT_FRAME_SPEC = FrameSpec(color="bgr", dtype="uint8", layout="hwc")


def frame_spec_from_config(raw: Any) -> FrameSpec:
    """Build the platform frame contract from an ``ai.frame`` block.

    An already built FrameSpec passes through, a mapping is read field by
    field, and anything else falls back to the default.  The result is a
    declaration, not a check: the kernel converts a frame that disagrees
    with it rather than refusing it.
    """
    if isinstance(raw, FrameSpec):
        return raw
    if isinstance(raw, dict):
        return FrameSpec(
            color=str(raw.get("color", DEFAULT_FRAME_SPEC.color)),
            dtype=str(raw.get("dtype", DEFAULT_FRAME_SPEC.dtype)),
            layout=str(raw.get("layout", DEFAULT_FRAME_SPEC.layout)),
        )
    return DEFAULT_FRAME_SPEC
