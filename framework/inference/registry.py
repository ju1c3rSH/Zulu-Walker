"""Engine availability probing and engine selection.

The selection policy is a single explicit engine.  There is no auto mode,
no preferred ordering, no candidate chain, no fallback to a stub and no
per-model engine override: changing engine means changing accelerator (NPU
versus CPU), and a silent downgrade yields a system that runs an order of
magnitude slower while nobody notices why the speed changed.  An engine
that is missing or unusable is therefore a startup failure.

This module knows no engine internals.  It asks the registry for a factory
and reports what the factory returns.
"""

from __future__ import annotations

import logging

from framework.hal.interface.ai import E_ENGINE_UNAVAILABLE, ModelConfigError
from framework.inference.contract import InferenceEngine
from framework.inference.engines import REGISTRY, engine_factory

logger = logging.getLogger(__name__)

__all__ = ["engine_availability", "resolve_engine"]


def resolve_engine(engine_id: str) -> InferenceEngine:
    """Return the engine instance registered under engine_id.

    Raises ModelConfigError with E_ENGINE_UNAVAILABLE when the id is not
    registered and when the engine's runtime library is missing.  No other
    engine is ever tried.
    """
    factory = engine_factory(engine_id)
    if factory is None:
        raise ModelConfigError(
            f"Engine '{engine_id}' is not registered; "
            f"known engines: {sorted(REGISTRY)}",
            E_ENGINE_UNAVAILABLE,
        )
    try:
        engine = factory()
    except Exception as exc:
        raise ModelConfigError(
            f"Engine '{engine_id}' could not be created: {exc}",
            E_ENGINE_UNAVAILABLE,
        ) from exc
    if engine is None:
        raise ModelConfigError(
            f"Engine '{engine_id}' is unavailable on this device",
            E_ENGINE_UNAVAILABLE,
        )
    return engine


def engine_availability() -> dict[str, bool]:
    """Report whether every registered engine can be created on this device."""
    availability: dict[str, bool] = {}
    for engine_id in REGISTRY:
        factory = engine_factory(engine_id)
        if factory is None:
            availability[engine_id] = False
            continue
        try:
            availability[engine_id] = factory() is not None
        except Exception as exc:
            logger.warning(
                "Availability probe for engine '%s' failed: %s", engine_id, exc
            )
            availability[engine_id] = False
    return availability
