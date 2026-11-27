"""Engine adapter for ONNX Runtime, the reference tensor level engine.

The adapter is tensor level: the runtime hands it the canonical tensor and
it hands back the raw output tensor.  Decoding that tensor and suppressing
the boxes belong to the wrapper, so nothing here knows what a detection is.

Three facts shape the implementation:

* onnxruntime is the only engine of the four that wants NCHW, so the
  adapter pays one transpose per frame; the canonical form stays HWC.
* the session releases the GIL while it runs but holds it while it is
  created, so a model load must be started from the main loop.
* a requested execution provider that is not installed is dropped by the
  runtime without an error, so the adapter compares the request with the
  session and reports a silent downgrade instead of hiding it.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import numpy as np

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    E_INFER_FAILED,
    E_INVALID_INPUT,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    AIError,
    ContractError,
    ModelConfigError,
    ModelLoadError,
)
from framework.inference.contract import ArtifactRef, Capabilities, TensorSpec
from framework.inference.engines.version import check_runtime_version

logger = logging.getLogger(__name__)

__all__ = ["ENGINE_ID", "OnnxRuntimeEngine", "create_engine"]

ENGINE_ID = "onnxruntime"
LOGICAL_INPUT_NAME = "image"

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_GRAPH_OPTIMIZATION_LEVEL = "ORT_ENABLE_ALL"
DEFAULT_EXECUTION_MODE = "ORT_SEQUENTIAL"

# The providers that mean "not the CPU", and the device name each of them
# reports.  The first one that is actually active wins.
_GPU_PROVIDERS = {
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "dml",
    "ROCMExecutionProvider": "rocm",
    "TensorrtExecutionProvider": "tensorrt",
}

_DTYPE_NAMES = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(uint8)": "uint8",
    "tensor(int8)": "int8",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
}


class OnnxRuntimeEngine:
    """InferenceEngine implementation backed by onnxruntime."""

    engine_id = ENGINE_ID

    def __init__(self) -> None:
        self._session: Any = None
        self._artifact: Optional[ArtifactRef] = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._logical_output: str = ""
        self._providers: tuple[str, ...] = ()
        self._requested_providers: tuple[str, ...] = ()
        self._input_dtype: Optional[np.dtype] = None
        self._quant: str = "fp32"

    # ------------------------------------------------------------------ #
    # InferenceEngine
    # ------------------------------------------------------------------ #

    def load(self, artifact: ArtifactRef, **opts: Any) -> None:
        """Open the session and bind the io names of the artifact.

        The physical names come from the manifest, because only the manifest
        knows them; the session is asked whether they exist, so a wrong name
        fails at load time instead of at the first frame.
        """
        runtime = _runtime_options(artifact, opts)
        ort = _import_onnxruntime()
        check_runtime_version(
            runtime.get("version"),
            getattr(ort, "__version__", ""),
            ENGINE_ID,
            package="onnxruntime",
        )
        providers = self._resolve_providers(runtime)
        options = self._build_session_options(ort, runtime)
        try:
            session = ort.InferenceSession(
                artifact.path, sess_options=options, providers=list(providers)
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{artifact.path}': {exc}", E_MODEL_LOAD_FAILED
            ) from exc
        self._check_io(artifact, session)
        self._session = session
        self._artifact = artifact
        self._requested_providers = providers
        self._providers = tuple(session.get_providers())
        self._input_dtype = self._resolve_input_dtype(session)
        self._quant = self._resolve_quant()
        self._report_providers()

    def input_specs(self) -> list[TensorSpec]:
        """The tensor specs the session declares for its inputs."""
        if self._session is None:
            return []
        return self._specs(self._session.get_inputs())

    def output_specs(self) -> list[TensorSpec]:
        """The tensor specs the session declares for its outputs."""
        if self._session is None:
            return []
        return self._specs(self._session.get_outputs())

    def capabilities(self) -> Capabilities:
        """Describe the runtime that is actually in use."""
        device = "cpu"
        for name in self._providers:
            if name in _GPU_PROVIDERS:
                device = _GPU_PROVIDERS[name]
                break
        return Capabilities(
            device=device,
            quant=self._quant,
            layouts=("nchw",),
            dynamic_shape=self._dynamic_shape(),
            input_accepts=("numpy_bgr",),
            io_binding="named",
            releases_gil=True,
            load_holds_gil=True,
            memory_hint="session graph and activations",
            load_cost_hint="200-800ms",
        )

    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]:
        """Run one inference and return the raw output tensor.

        The plain run() call is deliberate: the io binding call has a
        history of holding the GIL, which would stall the UART thread.
        """
        if self._session is None:
            raise AIError(
                f"Engine '{ENGINE_ID}' has no model loaded", E_MODEL_NOT_LOADED
            )
        if LOGICAL_INPUT_NAME not in inputs:
            raise ModelConfigError(
                f"engine.run() requires the logical input '{LOGICAL_INPUT_NAME}'",
                E_INVALID_INPUT,
            )
        tensor = self._to_nchw(inputs[LOGICAL_INPUT_NAME])
        try:
            outputs = self._session.run(
                [self._output_name], {self._input_name: tensor}
            )
        except Exception as exc:
            raise AIError(f"onnxruntime run() failed: {exc}", E_INFER_FAILED) from exc
        if not outputs:
            raise AIError("onnxruntime returned no output", E_INFER_FAILED)
        return {self._logical_output: outputs[0]}

    def unload(self) -> None:
        """Release the session."""
        self._session = None
        self._artifact = None
        self._providers = ()
        self._requested_providers = ()
        self._input_dtype = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_providers(self, runtime: dict[str, Any]) -> tuple[str, ...]:
        """The execution providers the artifact asks for."""
        raw = runtime.get("providers") or DEFAULT_PROVIDERS
        if isinstance(raw, str):
            raw = [raw]
        providers = tuple(str(item) for item in raw)
        if not providers:
            raise ModelConfigError(
                "runtime.providers must not be empty", E_CONTRACT_MISSING
            )
        return providers

    def _build_session_options(self, ort: Any, runtime: dict[str, Any]) -> Any:
        """Build the session options the artifact asks for."""
        options = ort.SessionOptions()
        num_threads = runtime.get("num_threads")
        if num_threads:
            options.intra_op_num_threads = int(num_threads)
        mode = str(runtime.get("execution_mode", DEFAULT_EXECUTION_MODE))
        options.execution_mode = getattr(
            ort.ExecutionMode, mode, ort.ExecutionMode.ORT_SEQUENTIAL
        )
        level = str(
            runtime.get("graph_optimization_level", DEFAULT_GRAPH_OPTIMIZATION_LEVEL)
        )
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel,
            level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        return options

    def _check_io(self, artifact: ArtifactRef, session: Any) -> None:
        """Bind the physical names and verify that the session has them."""
        inputs = {meta.name for meta in session.get_inputs()}
        outputs = {meta.name for meta in session.get_outputs()}
        logical_input = str(artifact.io.inputs.get(LOGICAL_INPUT_NAME, ""))
        if not logical_input:
            raise ModelConfigError(
                f"The artifact declares no logical input '{LOGICAL_INPUT_NAME}'",
                E_CONTRACT_MISSING,
            )
        if logical_input not in inputs:
            raise ModelConfigError(
                f"The model has no input named '{logical_input}' "
                f"(it declares {sorted(inputs)})",
                E_CONTRACT_MISSING,
            )
        declared = artifact.io.outputs
        if not declared:
            raise ModelConfigError(
                "The artifact declares no logical output", E_CONTRACT_MISSING
            )
        self._logical_output = next(iter(declared))
        self._output_name = str(declared[self._logical_output])
        if self._output_name not in outputs:
            raise ContractError(
                f"The model has no output named '{self._output_name}' "
                f"(it declares {sorted(outputs)})"
            )
        self._input_name = logical_input

    def _to_nchw(self, frame: Any) -> np.ndarray:
        """Turn the canonical HWC tensor into the NCHW tensor the model wants."""
        array = np.asarray(frame)
        if array.ndim == 3:
            array = np.expand_dims(array, 0)
        if array.ndim != 4:
            raise AIError(
                f"the canonical frame has shape {array.shape}, expected a batch",
                E_INVALID_INPUT,
            )
        # (N, H, W, C) -> (N, C, H, W)
        tensor = np.ascontiguousarray(np.transpose(array, (0, 3, 1, 2)))
        if self._input_dtype is not None and tensor.dtype != self._input_dtype:
            tensor = tensor.astype(self._input_dtype)
        return tensor

    def _resolve_input_dtype(self, session: Any) -> Optional[np.dtype]:
        """The dtype the session expects, as a numpy dtype."""
        metas = session.get_inputs()
        if not metas:
            return None
        name = _DTYPE_NAMES.get(str(metas[0].type), "")
        if not name:
            return None
        return np.dtype(name)

    def _resolve_quant(self) -> str:
        """The quantization of the model, read from its input dtype."""
        if self._input_dtype == np.dtype("float16"):
            return "fp16"
        if self._input_dtype == np.dtype("uint8"):
            return "int8"
        return "fp32"

    def _dynamic_shape(self) -> bool:
        """True when the session declares a dynamic dimension."""
        for spec in self.input_specs():
            if any(dim < 0 for dim in spec.shape):
                return True
        return False

    def _report_providers(self) -> None:
        """Report a provider that was asked for but is not in use."""
        missing = [name for name in self._requested_providers if name not in self._providers]
        if missing:
            logger.warning(
                "Requested execution providers %s are not active; "
                "the session runs on %s",
                missing,
                list(self._providers),
            )
        logger.info("onnxruntime session ready, providers=%s", list(self._providers))

    @staticmethod
    def _specs(metas: Any) -> list[TensorSpec]:
        """Turn the session metadata into tensor specs."""
        specs: list[TensorSpec] = []
        for meta in metas:
            shape = tuple(
                int(dim) if isinstance(dim, int) else -1 for dim in (meta.shape or ())
            )
            specs.append(
                TensorSpec(name=str(meta.name), shape=shape, dtype=str(meta.type))
            )
        return specs


def _runtime_options(artifact: ArtifactRef, opts: dict[str, Any]) -> dict[str, Any]:
    """The run-time options of an artifact, with the caller's overrides.

    The artifact declares the options in its manifest; a caller may replace
    them for one load, which is what the tests and the benchmarks do.
    """
    runtime = dict(artifact.runtime or {})
    override = opts.get("runtime")
    if isinstance(override, dict):
        runtime.update(override)
    return runtime


def _import_onnxruntime() -> Any:
    """Import the runtime, or report it as an unavailable engine."""
    try:
        return importlib.import_module("onnxruntime")
    except Exception as exc:
        raise ModelConfigError(
            f"The onnxruntime library is not importable: {exc}",
            E_ENGINE_UNAVAILABLE,
        ) from exc


def create_engine() -> Optional[OnnxRuntimeEngine]:
    """Build an unloaded onnxruntime engine, or None when the runtime is absent."""
    try:
        importlib.import_module("onnxruntime")
    except Exception as exc:
        logger.info("onnxruntime is not importable: %s", exc)
        return None
    return OnnxRuntimeEngine()
