"""Built-in wrapper registry: dialect key -> wrapper factory and metadata.

A manifest names its task adaptation layer with a dialect key.  Two forms
are accepted:

* a builtin name such as "vision.det.raw", resolved through BUILTIN
* an extension reference of the form "module:attr", imported directly

Every builtin entry carries the metadata the framework needs at load time:
target (where the factory lives), families (model families the dialect
accepts), provides (task capabilities the wrapper exposes) and output_form
(how the engine output must be read).

This module is intentionally free of runtime, engine, manifest and wrapper
imports: resolving a dialect happens lazily, so importing the registry costs
nothing but a table.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from framework.hal.interface.ai import E_DIALECT_UNSUPPORTED, ModelConfigError

__all__ = [
    "BUILTIN",
    "get_wrapper_metadata",
    "resolve_wrapper_factory",
    "wrapper_families",
]

BUILTIN: dict[str, dict[str, Any]] = {
    "vision.det.raw": {
        "target": "framework.inference.wrappers.vision_det:build_raw",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "raw",
    },
    "vision.det.end2end": {
        "target": "framework.inference.wrappers.vision_det:build_end2end",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "end2end",
    },
}


def get_wrapper_metadata(key: str) -> Mapping[str, Any]:
    """Return the metadata of a builtin dialect.

    Raises ModelConfigError with code E_DIALECT_UNSUPPORTED when the key is
    not a builtin dialect.
    """
    entry = BUILTIN.get(key)
    if entry is None:
        raise ModelConfigError(
            f"unknown dialect '{key}' (builtin dialects: {sorted(BUILTIN)})",
            code=E_DIALECT_UNSUPPORTED,
        )
    return {
        "target": entry["target"],
        "families": frozenset(entry["families"]),
        "provides": frozenset(entry["provides"]),
        "output_form": entry["output_form"],
    }


def wrapper_families(key: str) -> frozenset[str]:
    """Return the model families a builtin dialect accepts."""
    return frozenset(get_wrapper_metadata(key)["families"])


def resolve_wrapper_factory(name: str) -> Callable[..., Any]:
    """Resolve a wrapper reference to the factory that builds the task API.

    A "module:attr" reference is imported directly; any other name is looked
    up in BUILTIN and its target resolved the same way.  An unresolvable
    reference raises ModelConfigError with code E_DIALECT_UNSUPPORTED.
    """
    if ":" in name:
        return _import_factory(name)

    entry = BUILTIN.get(name)
    if entry is None:
        raise ModelConfigError(
            f"unknown dialect '{name}' (builtin dialects: {sorted(BUILTIN)})",
            code=E_DIALECT_UNSUPPORTED,
        )
    return _import_factory(entry["target"])


def _import_factory(reference: str) -> Callable[..., Any]:
    module_name, _, attr = reference.partition(":")
    if not module_name or not attr:
        raise ModelConfigError(
            f"wrapper reference '{reference}' is not a 'module:attr' pair",
            code=E_DIALECT_UNSUPPORTED,
        )
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ModelConfigError(
            f"wrapper reference '{reference}' could not be resolved: {exc}",
            code=E_DIALECT_UNSUPPORTED,
        ) from exc
    if not callable(factory):
        raise ModelConfigError(
            f"wrapper reference '{reference}' does not point at a callable",
            code=E_DIALECT_UNSUPPORTED,
        )
    return factory
