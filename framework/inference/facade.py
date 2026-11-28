"""The facade that exposes the AIInference contract to the business layer.

The facade owns the registry of models, the single active runtime and the
statistics.  It is the only object the business layer talks to.

The engine is given to the facade by its caller: there is no auto mode, no
preferred ordering, no candidate chain and no fallback to a stub.  A model
is registered without being loaded, switching to a registered model loads
it when it is not loaded yet, and a switch that has to wait for the running
inferences gives up with E_SWITCH_TIMEOUT instead of hanging.

The facade imports no platform code.  The camera is duck-typed: only its
last_raw is read, and only to decide whether the native frame may be used.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from framework.hal.interface.ai import (
    AIError,
    Detection,
    E_CONTRACT_MISSING,
    E_DIALECT_UNSUPPORTED,
    E_ENGINE_UNAVAILABLE,
    E_INFER_FAILED,
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    E_OUTPUT_LAYOUT_MISMATCH,
    E_SWITCH_TIMEOUT,
    ManifestError,
    ModelConfigError,
    SegmentResult,
)
from framework.inference.contract import ArtifactRef, InferenceEngine
from framework.inference.manifest import (
    ModelManifest,
    check_artifact_against_config,
    load_manifest,
)
from framework.inference.registry import resolve_engine
from framework.inference.runtime import (
    CanonicalSpec,
    EngineStats,
    FrameSpec,
    ModelRuntime,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AIFacade",
    "CanonicalSpec",
    "EngineStats",
    "FrameSpec",
    "ModelEntry",
    "ModelRuntime",
]

DEFAULT_SWITCH_TIMEOUT_MS = 2000
DEFAULT_FRAME_SPEC = FrameSpec(color="bgr", dtype="uint8", layout="hwc")
HARD_FAILURE_CODES = frozenset(
    {
        E_CONTRACT_MISSING,
        E_DIALECT_UNSUPPORTED,
        E_MODEL_TYPE_UNSUPPORTED,
        E_OUTPUT_LAYOUT_MISMATCH,
    }
)


@dataclass
class ModelEntry:
    """One registered model: its manifest, its artifact and its runtime."""

    nick_name: str
    model_path: str
    model_type: str
    manifest: ModelManifest
    artifact: ArtifactRef
    runtime: Optional[ModelRuntime] = None
    runtime_options: dict[str, Any] = field(default_factory=dict)


class AIFacade:
    """The stable inference contract of the framework.

    The facade keeps one active runtime, hands the task interfaces to the
    business layer and reports every failure with a stable error code.
    """

    def __init__(
        self,
        engine_id: str,
        *,
        switch_timeout_ms: int = DEFAULT_SWITCH_TIMEOUT_MS,
        frame_spec: Optional[FrameSpec] = None,
        engine_factory: Optional[Callable[[], InferenceEngine]] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._engine_id = engine_id
        self._switch_timeout_ms = int(switch_timeout_ms)
        self._frame_spec = frame_spec or DEFAULT_FRAME_SPEC
        self._runtime_options = dict(runtime_options or {})
        self._engine_factory = engine_factory or (
            lambda: resolve_engine(engine_id)
        )
        self._entries: dict[str, ModelEntry] = {}
        self._active: Optional[ModelRuntime] = None
        self._active_nick = ""
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._inflight = 0
        self._load_lock = threading.Lock()
        self._stats_by_engine: dict[str, EngineStats] = {}
        self._last_error_code = ""
        self._error_count = 0
        self._degraded = False
        self._probe_engine()

    # ------------------------------------------------------------------ #
    # Registry API
    # ------------------------------------------------------------------ #

    @property
    def models(self) -> list[str]:
        """The nick names of the registered models."""
        return list(self._entries.keys())

    @property
    def active_model(self) -> str:
        """The nick name of the active model, empty when none is active."""
        return self._active_nick

    @property
    def model_type(self) -> str:
        """The model type the active model was registered with."""
        entry = self._entries.get(self._active_nick)
        return entry.model_type if entry is not None else ""

    @property
    def model_family(self) -> str:
        """The model family of the active model, as declared by its manifest."""
        runtime = self._active
        return runtime.family if runtime is not None else ""

    @property
    def loaded(self) -> bool:
        """True while a runtime is active."""
        return self._active is not None

    @property
    def input_width(self) -> int:
        """The input width of the active model."""
        canonical = self._canonical()
        return canonical.target_size[0] if canonical is not None else 0

    @property
    def input_height(self) -> int:
        """The input height of the active model."""
        canonical = self._canonical()
        return canonical.target_size[1] if canonical is not None else 0

    @property
    def labels(self) -> list[str]:
        """The label list of the active model."""
        runtime = self._active
        return list(runtime.labels) if runtime is not None else []

    @property
    def model_path(self) -> str:
        """The artifact file of the active model."""
        entry = self._entries.get(self._active_nick)
        return entry.model_path if entry is not None else ""

    @property
    def last_error_code(self) -> str:
        """The code of the most recent failure, empty while the backend is healthy."""
        return self._last_error_code

    @property
    def error_count(self) -> int:
        """The number of failures recorded since the last successful inference."""
        return self._error_count

    @property
    def degraded(self) -> bool:
        """True while the facade runs but does not produce valid output."""
        return self._degraded

    @property
    def infer_fps(self) -> float:
        """The inference rate of the active runtime, 0.0 while none ran.

        The board backends report this from their own counters, so the
        kernel exposes the same pair; a consumer that reads the attribute
        must not have to know which backend it holds.
        """
        runtime = self._active
        return runtime.stats.infer_fps if runtime is not None else 0.0

    @property
    def infer_avg_ms(self) -> float:
        """The average inference time of the active runtime, 0.0 while none ran."""
        runtime = self._active
        return runtime.stats.infer_avg_ms if runtime is not None else 0.0

    def add(
        self,
        nick_name: str,
        model_path: str,
        model_type: str = "auto",
        *,
        manifest_path: Optional[str] = None,
        runtime_options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """Register a model without loading it.

        The manifest carries the frame contract of the model and declares
        the artifact of the configured engine.  A model file that disagrees
        with the manifest is a contract error, not a warning.

        runtime_options is the accelerator choice of the deployment.  It is
        merged over the block the artifact declares, and a model entry
        without one inherits the facade wide choice.
        """
        if not manifest_path:
            raise ManifestError(
                f"model '{nick_name}' needs a manifest path: the manifest "
                f"carries the frame contract of the model"
            )
        manifest = load_manifest(manifest_path)
        artifact = check_artifact_against_config(manifest, self._engine_id, model_path)
        entry = ModelEntry(
            nick_name=nick_name,
            model_path=model_path,
            model_type=model_type,
            manifest=manifest,
            artifact=artifact,
            runtime_options=dict(runtime_options or {}),
        )
        with self._lock:
            self._entries[nick_name] = entry
        logger.info(
            "Registered model '%s' -> %s (engine=%s)", nick_name, model_path, self._engine_id
        )
        return True

    def remove(self, nick_name: str) -> None:
        """Unregister a model and release the engine resources it holds."""
        with self._lock:
            entry = self._entries.pop(nick_name, None)
            is_active = entry is not None and self._active is entry.runtime
        if entry is None:
            logger.warning("Attempt to remove unknown model '%s'", nick_name)
            return
        if entry.runtime is not None:
            if is_active:
                self._await_quiescence()
            self._release(entry.runtime)
        if is_active:
            with self._lock:
                self._active = None
                self._active_nick = ""

    def switch(self, nick_name: str) -> bool:
        """Activate a registered model.

        A model that is already loaded is activated by swapping the pointer.
        A model that is not loaded yet is loaded first; that load waits for
        the running inferences and gives up with E_SWITCH_TIMEOUT.
        """
        entry = self._entries.get(nick_name)
        if entry is None:
            raise ModelConfigError(
                f"Cannot switch to the unknown model '{nick_name}'",
                E_MODEL_NOT_LOADED,
            )
        with self._load_lock:
            runtime = entry.runtime
            if runtime is None:
                self._await_quiescence()
                runtime = self._build_runtime(entry)
                entry.runtime = runtime
                self._stats_by_engine[runtime.stats.engine_id] = runtime.stats
            with self._lock:
                self._active = runtime
                self._active_nick = nick_name
        self._clear_error()
        logger.info("Switched to model '%s' (engine=%s)", nick_name, self._engine_id)
        return True

    def load(
        self,
        model_path: str,
        model_type: str = "auto",
        *,
        manifest_path: Optional[str] = None,
        nick_name: Optional[str] = None,
        runtime_options: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> bool:
        """Register a model under a nick name and make it the active one."""
        nick = nick_name or "default"
        self.add(
            nick,
            model_path,
            model_type,
            manifest_path=manifest_path,
            runtime_options=runtime_options,
        )
        return self.switch(nick)

    def unload(self) -> None:
        """Release the engine resources of the active model."""
        with self._lock:
            runtime = self._active
            nick = self._active_nick
            self._active = None
            self._active_nick = ""
        if runtime is None:
            return
        self._await_quiescence()
        self._release(runtime)
        entry = self._entries.get(nick)
        if entry is not None:
            entry.runtime = None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def detect(
        self,
        frame: np.ndarray,
        camera: Any = None,
        **kwargs: Any,
    ) -> list[Detection]:
        """Run object detection on one frame.

        The native frame of the camera is used when the engine declares that
        it accepts one and the camera has one; otherwise the numpy frame is
        used.  The decision is read from the engine capabilities.
        """
        runtime = self._active
        if runtime is None:
            self._report_error(E_MODEL_NOT_LOADED, "detect() called but no model is loaded")
            return []
        capability = self._require_capability(runtime, "detect")
        started = time.perf_counter()
        self._enter_inference()
        try:
            payload = self._select_frame(runtime, frame, camera)
            detections = capability.detect(payload, **kwargs)
        except Exception as exc:
            self._exit_inference()
            self._handle_inference_failure(runtime, exc)
            return []
        self._exit_inference()
        runtime.record_inference((time.perf_counter() - started) * 1000.0)
        self._clear_error()
        return detections

    def segment(
        self,
        frame: np.ndarray,
        camera: Any = None,
        **kwargs: Any,
    ) -> list[SegmentResult]:
        """Run segmentation on one frame through the active capability."""
        runtime = self._active
        if runtime is None:
            raise ModelConfigError(
                "segment() called but no model is loaded", E_MODEL_NOT_LOADED
            )
        capability = self._require_capability(runtime, "segment")
        started = time.perf_counter()
        self._enter_inference()
        try:
            payload = self._select_frame(runtime, frame, camera)
            results = capability.segment(payload, **kwargs)
        except Exception as exc:
            self._exit_inference()
            self._handle_inference_failure(runtime, exc)
            return []
        self._exit_inference()
        runtime.record_inference((time.perf_counter() - started) * 1000.0)
        self._clear_error()
        return results

    def classify(
        self,
        frame: np.ndarray,
        camera: Any = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Run classification on one frame through the active capability."""
        runtime = self._active
        if runtime is None:
            raise ModelConfigError(
                "classify() called but no model is loaded", E_MODEL_NOT_LOADED
            )
        capability = self._require_capability(runtime, "classify")
        started = time.perf_counter()
        self._enter_inference()
        try:
            payload = self._select_frame(runtime, frame, camera)
            results = capability.classify(payload, **kwargs)
        except Exception as exc:
            self._exit_inference()
            self._handle_inference_failure(runtime, exc)
            return []
        self._exit_inference()
        runtime.record_inference((time.perf_counter() - started) * 1000.0)
        self._clear_error()
        return results

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        """The mask of the last segmentation, through the active capability."""
        runtime = self._active
        if runtime is None:
            raise ModelConfigError(
                "get_mask() called but no model is loaded", E_MODEL_NOT_LOADED
            )
        capability = self._require_capability(runtime, "get_mask")
        return capability.get_mask(index)

    def stats_by_engine(self) -> dict[str, EngineStats]:
        """The statistics of every engine that served a model, keyed by engine."""
        return dict(self._stats_by_engine)

    def __enter__(self) -> "AIFacade":
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _canonical(self) -> Optional[CanonicalSpec]:
        """The configuration protocol of the active runtime."""
        runtime = self._active
        return runtime.canonical if runtime is not None else None

    def _probe_engine(self) -> None:
        """Fail at construction when the configured engine is not usable."""
        try:
            engine = self._engine_factory()
        except ModelConfigError:
            raise
        except Exception as exc:
            raise ModelConfigError(
                f"Engine '{self._engine_id}' could not be created: {exc}",
                E_ENGINE_UNAVAILABLE,
            ) from exc
        if engine is None:
            raise ModelConfigError(
                f"Engine '{self._engine_id}' is unavailable on this device",
                E_ENGINE_UNAVAILABLE,
            )

    def _new_engine(self) -> InferenceEngine:
        """Create the engine instance of a new runtime."""
        try:
            engine = self._engine_factory()
        except ModelConfigError:
            raise
        except Exception as exc:
            raise ModelConfigError(
                f"Engine '{self._engine_id}' could not be created: {exc}",
                E_ENGINE_UNAVAILABLE,
            ) from exc
        if engine is None:
            raise ModelConfigError(
                f"Engine '{self._engine_id}' is unavailable on this device",
                E_ENGINE_UNAVAILABLE,
            )
        return engine

    def _build_runtime(self, entry: ModelEntry) -> ModelRuntime:
        """Load the engine of a model and let its dialect build the capability."""
        engine = self._new_engine()
        options = {**self._runtime_options, **entry.runtime_options}
        return ModelRuntime(
            engine,
            entry.artifact,
            manifest=entry.manifest,
            frame_spec=self._frame_spec,
            runtime_options=options,
        )

    def _release(self, runtime: ModelRuntime) -> None:
        """Release the engine resources of a runtime."""
        try:
            runtime.engine.unload()
        except Exception as exc:
            logger.warning("Engine unload failed: %s", exc)

    def _select_frame(
        self,
        runtime: ModelRuntime,
        frame: np.ndarray,
        camera: Any,
    ) -> Any:
        """The frame the engine gets: the native one when it may take it."""
        try:
            accepts = runtime.engine.capabilities().input_accepts
        except Exception as exc:
            raise AIError(
                f"the engine capabilities could not be read: {exc}", E_INFER_FAILED
            ) from exc
        if "native_image" in accepts and camera is not None:
            native = getattr(camera, "last_raw", None)
            if native is not None:
                return native
        return frame

    def _require_capability(self, runtime: ModelRuntime, name: str) -> Any:
        """The capability object, when the dialect declares the task."""
        capability = runtime.capability
        provides = frozenset(getattr(capability, "provides", frozenset()))
        if name not in provides:
            error = ModelConfigError(
                f"the dialect of the active model provides no {name}()",
                E_DIALECT_UNSUPPORTED,
            )
            self._report_error(error.code, error.message)
            raise error
        return capability

    def _enter_inference(self) -> None:
        """Account one inference that is about to run."""
        with self._condition:
            self._inflight += 1

    def _exit_inference(self) -> None:
        """Account one inference that finished."""
        with self._condition:
            self._inflight -= 1
            self._condition.notify_all()

    def _await_quiescence(self) -> None:
        """Wait for the running inferences, up to the switch timeout."""
        deadline = time.monotonic() + self._switch_timeout_ms / 1000.0
        with self._condition:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise AIError(
                        f"switch timed out after {self._switch_timeout_ms} ms with "
                        f"{self._inflight} inferences in flight",
                        E_SWITCH_TIMEOUT,
                    )
                self._condition.wait(remaining)

    def _handle_inference_failure(self, runtime: ModelRuntime, exc: Exception) -> None:
        """Record a failed inference and decide whether it is a hard failure."""
        code = getattr(exc, "code", E_INFER_FAILED)
        runtime.record_error(code)
        self._report_error(code, str(exc))
        if code in HARD_FAILURE_CODES:
            raise exc

    def _report_error(self, code: str, message: str) -> None:
        """Record a recoverable failure so it stays visible to the caller."""
        self._last_error_code = code
        self._error_count += 1
        self._degraded = True
        logger.error("[%s] %s", code, message)

    def _clear_error(self) -> None:
        """Mark the facade healthy again after a successful inference."""
        self._last_error_code = ""
        self._error_count = 0
        self._degraded = False
