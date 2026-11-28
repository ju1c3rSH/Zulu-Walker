"""Data models emitted by the detector methods.

Import a model from its own module (``models.cargo``, ``models.circle``,
``models.ring``).  The package deliberately re-exports nothing: a device
package that only runs the YOLO path would otherwise carry three detector
dataclasses it never constructs.
"""
from .color import Color

__all__ = ["Color"]
