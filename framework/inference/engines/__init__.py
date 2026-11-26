"""Registry of the inference engines known to the framework.

This module is the single place where an engine id is bound to a factory.
It lists every engine the framework supports, installed or not, because the
complete list answers two questions at once: what exists on this device and
which import failed.

A factory performs its third-party import lazily and returns None when the
runtime library is missing; the registry entry stays in place either way, so
"not installed" and "installed but the import failed" remain distinguishable.
No entry_points are used: the table is auditable, works offline and stays
checkable by the platform isolation script.

The package must not import any third-party inference library at import
time, and it must not depend on platform code, the manifest, the runtime or
the wrappers.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

from framework.inference.contract import InferenceEngine

logger = logging.getLogger(__name__)

EngineFactory = Callable[[], Optional[InferenceEngine]]

__all__ = [
    "EngineFactory",
    "REGISTRY",
    "available_engines",
    "engine_factory",
    "register_engine",
]


def _maix_nn_factory() -> Optional[InferenceEngine]:
    """Build a maix.nn engine, or None when the firmware runtime is absent."""
    try:
        adapter = importlib.import_module("framework.inference.engines.maix_nn")
    except ImportError as exc:
        logger.info("Engine adapter 'maix_nn' is not importable: %s", exc)
        return None
    create_engine = getattr(adapter, "create_engine", None)
    if create_engine is None:
        logger.info("Engine adapter 'maix_nn' exposes no create_engine()")
        return None
    return create_engine()


def _library_backed_factory(engine_id: str, library: str) -> EngineFactory:
    """Build a factory that needs both the adapter module and its runtime library.

    The adapter module and the library are probed separately so that a
    missing adapter and a missing library are reported as different causes
    of the same "unavailable" verdict.
    """

    def factory() -> Optional[InferenceEngine]:
        try:
            adapter = importlib.import_module(f"framework.inference.engines.{engine_id}")
        except ImportError as exc:
            logger.info("Engine adapter '%s' is not importable: %s", engine_id, exc)
            return None
        try:
            importlib.import_module(library)
        except Exception as exc:
            logger.info("Engine library '%s' is not importable: %s", library, exc)
            return None
        create_engine = getattr(adapter, "create_engine", None)
        if create_engine is None:
            logger.info("Engine adapter '%s' exposes no create_engine()", engine_id)
            return None
        return create_engine()

    return factory


REGISTRY: dict[str, EngineFactory] = {
    "maix_nn": _maix_nn_factory,
    "ncnn": _library_backed_factory("ncnn", "ncnn"),
    "onnxruntime": _library_backed_factory("onnxruntime", "onnxruntime"),
    "rknn_lite": _library_backed_factory("rknn_lite", "rknn_lite"),
}


def register_engine(engine_id: str, factory: EngineFactory) -> None:
    """Bind engine_id to factory, replacing a previous entry for that id."""
    REGISTRY[engine_id] = factory


def engine_factory(engine_id: str) -> Optional[EngineFactory]:
    """Return the factory bound to engine_id, or None for an unknown id."""
    return REGISTRY.get(engine_id)


def available_engines() -> list[str]:
    """Return the ids whose factory yields a usable engine on this device."""
    available: list[str] = []
    for engine_id, factory in REGISTRY.items():
        try:
            usable = factory() is not None
        except Exception as exc:
            logger.warning("Engine '%s' availability probe failed: %s", engine_id, exc)
            continue
        if usable:
            available.append(engine_id)
    return available
