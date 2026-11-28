"""Layout registry: a HUD is a named function from state to a Scene.

Built-in layouts live in ``framework.ui.layouts``.  A second-party layout
registers itself the same way:

    # myapp/hud.py
    class MyHud:
        name = "mine"
        def scene(self, state, constraints, measure):
            ...

    # at import time
    from framework.ui import layout
    layout.register(MyHud())

and the config names it:

    display:
      layout: "myapp.hud:MyHud"

An unknown name raises instead of falling back: a typo that silently shows
the built-in HUD is harder to notice than a startup failure.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Protocol, Tuple

from framework.ui.measurer import Measurer
from framework.ui.scene import Scene
from framework.ui.space import Constraints

__all__ = ["Layout", "register", "get", "load", "available", "LayoutNotFound"]


class LayoutNotFound(Exception):
    """The configured layout name is neither registered nor importable."""


class Layout(Protocol):
    """One way of drawing a screen.

    ``scene`` must be pure: same state and constraints produce the same Scene.
    Extra keyword arguments (buttons, pressed) are allowed; a caller that only
    knows the protocol passes the three positional ones.
    """

    name: str

    def scene(self, state: Any, constraints: Constraints,
              measure: Measurer) -> Scene:
        ...


_REGISTRY: Dict[str, Layout] = {}


def register(layout: Layout) -> Layout:
    """Add *layout* under its ``name``; replacing an existing one is allowed."""
    name = getattr(layout, "name", "")
    if not name:
        raise ValueError("a layout must define a non-empty name")
    _REGISTRY[name] = layout
    return layout


def available() -> Tuple[str, ...]:
    """The registered layout names, sorted."""
    _load_builtins()
    return tuple(sorted(_REGISTRY))


def get(name: str) -> Layout:
    """The layout registered as *name*."""
    _load_builtins()
    layout = _REGISTRY.get(name)
    if layout is None:
        raise LayoutNotFound(
            f"unknown layout {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return layout


def load(spec: str) -> Layout:
    """Resolve ``"name"`` or ``"module.path:ClassName"`` to a layout."""
    if ":" not in spec:
        return get(spec)

    module_path, separator, class_name = spec.partition(":")
    if not separator or not class_name:
        raise LayoutNotFound(f"malformed layout spec {spec!r}")

    module = importlib.import_module(module_path)
    candidate = getattr(module, class_name, None)
    if candidate is None:
        raise LayoutNotFound(f"{module_path} has no attribute {class_name!r}")

    instance = candidate() if isinstance(candidate, type) else candidate
    return register(instance)


def _load_builtins() -> None:
    """Import the built-in layouts once, so their registration runs."""
    if _REGISTRY:
        return
    from framework.ui.layouts import desktop  # noqa: F401  (registers on import)
