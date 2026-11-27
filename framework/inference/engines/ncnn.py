"""Engine adapter for ncnn, the CPU and Vulkan engine of the framework.

The adapter is tensor level: it hands ncnn the canonical tensor and hands
back the raw output blob.  Decoding the blob and suppressing the boxes
belong to the wrapper, so nothing here knows what a detection is.

Three properties of the ncnn Python binding shape the implementation, and
all three are easy to get wrong:

* A Mat is planar.  The memory of a (C, H, W) array is the C planes of an
  H x W image, which is the layout an NCHW tensor has.  The canonical form
  of the framework is HWC, so the adapter transposes once - the same memory
  the onnxruntime adapter produces, only with other axis names.
* substract_mean_normalize multiplies by its second argument and wants one
  value per channel.  The framework formula is (x - mean) / std, so the
  adapter passes 1 / std as the second argument.
* extract(name, 1) returns a reference into the extractor.  The copy into a
  numpy array must happen while the extractor is still alive, otherwise the
  data is read after it was released.

The ncnn artifact is a pair of files, a .param with the graph and a .bin
with the weights.  The .bin carries neither the tensor names nor the
shapes, so the manifest is the only declaration of the io mapping and the
graph is only cross-checked against it.
"""

from __future__ import annotations

import importlib
import logging
import os
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
from framework.inference import ops
from framework.inference.contract import ArtifactRef, Capabilities, TensorSpec
from framework.inference.engines.version import check_runtime_version

logger = logging.getLogger(__name__)

__all__ = ["ENGINE_ID", "NcnnEngine", "create_engine"]

ENGINE_ID = "ncnn"
LOGICAL_INPUT_NAME = "image"

DEFAULT_DEVICE = "cpu"
ALIGNMENT_BYTES = 16


class NcnnEngine:
    """InferenceEngine implementation backed by ncnn."""

    engine_id = ENGINE_ID

    def __init__(self) -> None:
        self._net: Any = None
        self._artifact: Optional[ArtifactRef] = None
        self._input_blob: str = ""
        self._output_blob: str = ""
        self._logical_output: str = ""
        self._device: str = DEFAULT_DEVICE
        self._quant: str = "fp32"
        self._canonical: Any = None
        self._norm_owner: str = "framework"
        self._norm_params: Optional[tuple[list[float], list[float]]] = None
        self._buffer: Optional[np.ndarray] = None
        self._buffer_key: Optional[tuple[Any, ...]] = None

    # ------------------------------------------------------------------ #
    # InferenceEngine
    # ------------------------------------------------------------------ #

    def load(self, artifact: ArtifactRef, **opts: Any) -> None:
        """Load the graph and bind the io names of the artifact."""
        runtime = _runtime_options(artifact, opts)
        self._canonical = opts.get("canonical")
        self._norm_owner = artifact.preprocess.norm_owner
        if self._norm_owner == "engine" and self._canonical is None:
            # The parameters of the in-device normalisation live in the
            # canonical contract, which only the runtime holds.
            raise ModelConfigError(
                "the engine needs the canonical input contract to normalise "
                "on the device",
                E_CONTRACT_MISSING,
            )
        ncnn = _import_ncnn()
        check_runtime_version(
            runtime.get("version"),
            getattr(ncnn, "__version__", ""),
            ENGINE_ID,
            package="ncnn",
        )
        device = str(runtime.get("device", DEFAULT_DEVICE)).strip().lower()
        if device not in ("cpu", "vulkan"):
            raise ModelConfigError(
                f"unknown ncnn device '{device}', expected 'cpu' or 'vulkan'",
                E_CONTRACT_MISSING,
            )
        net = ncnn.Net()
        if device == "vulkan":
            self._prepare_vulkan(ncnn, net, runtime)
        num_threads = int(runtime.get("num_threads", 0) or 0)
        if num_threads > 0:
            net.opt.num_threads = num_threads
        if runtime.get("use_fp16_storage"):
            option = getattr(net.opt, "use_fp16_storage", None)
            if option is not None:
                net.opt.use_fp16_storage = True
        try:
            self._load_graph(net, artifact)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model '{artifact.path}': {exc}", E_MODEL_LOAD_FAILED
            ) from exc
        self._net = net
        self._artifact = artifact
        self._device = device
        self._quant = self._resolve_quant(artifact)
        self._bind_io(artifact)

    def input_specs(self) -> list[TensorSpec]:
        """Empty: ncnn declares no tensor level specs.

        The .bin carries neither the names nor the shapes of the tensors, so
        the manifest is the only declaration and the graph is cross-checked
        against it at load time.
        """
        return []

    def output_specs(self) -> list[TensorSpec]:
        """Empty, for the same reason as the input specs."""
        return []

    def capabilities(self) -> Capabilities:
        """Describe the runtime that is actually in use."""
        return Capabilities(
            device=self._device,
            quant=self._quant,
            layouts=("nhwc",),
            dynamic_shape=False,
            input_accepts=("numpy_pixels", "numpy_bgr"),
            io_binding="named",
            releases_gil=False,
            load_holds_gil=True,
            memory_hint="graph weights plus the input buffer",
            load_cost_hint="fast",
        )

    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]:
        """Run one inference and return the raw output blob."""
        if self._net is None:
            raise AIError(
                f"Engine '{ENGINE_ID}' has no model loaded", E_MODEL_NOT_LOADED
            )
        if LOGICAL_INPUT_NAME not in inputs:
            raise ModelConfigError(
                f"engine.run() requires the logical input '{LOGICAL_INPUT_NAME}'",
                E_INVALID_INPUT,
            )
        ncnn = _import_ncnn()
        tensor = self._prepare_tensor(inputs[LOGICAL_INPUT_NAME])
        mat = ncnn.Mat(tensor)
        if self._norm_owner == "engine":
            mean, inv_std = self._engine_norm_params()
            mat.substract_mean_normalize(mean, inv_std)
        with self._net.create_extractor() as ex:
            ex.input(self._input_blob, mat)
            ret, out_mat = ex.extract(self._output_blob, 1)
            if ret != 0:
                raise AIError(
                    f"ncnn extract('{self._output_blob}') failed with {ret}",
                    E_INFER_FAILED,
                )
            # The copy must happen here: outside the extractor the referenced
            # blob is already released.
            result = np.array(out_mat)
        return {self._logical_output: result}

    def unload(self) -> None:
        """Release the graph."""
        self._net = None
        self._artifact = None
        self._buffer = None
        self._buffer_key = None
        self._norm_params = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prepare_vulkan(self, ncnn: Any, net: Any, runtime: dict[str, Any]) -> None:
        """Bring up the Vulkan device, or fail loudly.

        A Vulkan request that cannot be honoured is a startup failure: a
        silent fallback to the CPU would look like a working configuration
        while running an order of magnitude slower.
        """
        get_count = getattr(ncnn, "get_gpu_count", None)
        if get_count is None:
            raise ModelConfigError(
                "this ncnn build has no Vulkan support", E_ENGINE_UNAVAILABLE
            )
        try:
            count = int(get_count())
        except Exception as exc:
            raise ModelConfigError(
                f"the ncnn Vulkan device could not be queried: {exc}",
                E_ENGINE_UNAVAILABLE,
            ) from exc
        if count <= 0:
            raise ModelConfigError(
                "no Vulkan device is available for ncnn", E_ENGINE_UNAVAILABLE
            )
        create_instance = getattr(ncnn, "create_gpu_instance", None)
        if create_instance is not None:
            try:
                create_instance()
            except Exception as exc:
                raise ModelConfigError(
                    f"the ncnn Vulkan instance could not be created: {exc}",
                    E_ENGINE_UNAVAILABLE,
                ) from exc
        device_index = int(runtime.get("vulkan_device", 0) or 0)
        setter = getattr(net, "set_vulkan_device", None)
        if setter is not None:
            setter(device_index)
        net.opt.use_vulkan_compute = True

    def _load_graph(self, net: Any, artifact: ArtifactRef) -> None:
        """Load the .param and the .bin of the artifact.

        The ncnn binding reports a refused file through a return code and
        not through an exception, so the codes are checked here; otherwise a
        missing weight file would look like a loaded model.
        """
        bin_path = (artifact.aux_paths or {}).get("bin")
        if bin_path:
            self._check_graph_result(
                net.load_param(artifact.path),
                f"the graph '{artifact.path}' was refused",
            )
            self._check_graph_result(
                net.load_model(bin_path),
                f"the weights '{bin_path}' were refused",
            )
            return
        if artifact.path.endswith(".param.bin"):
            self._check_graph_result(
                net.load_param_bin(artifact.path),
                f"the graph '{artifact.path}' was refused",
            )
            return
        raise ModelLoadError(
            f"The ncnn artifact '{artifact.path}' has no companion .bin file",
            E_MODEL_LOAD_FAILED,
        )

    @staticmethod
    def _check_graph_result(code: Any, message: str) -> None:
        """Turn a non-zero ncnn return code into a load failure."""
        try:
            failed = int(code) != 0
        except (TypeError, ValueError):
            failed = code is not None
        if failed:
            raise ModelLoadError(message, E_MODEL_LOAD_FAILED)

    def _bind_io(self, artifact: ArtifactRef) -> None:
        """Bind the blob names and cross-check them against the graph."""
        logical_input = str(artifact.io.inputs.get(LOGICAL_INPUT_NAME, ""))
        if not logical_input:
            raise ModelConfigError(
                f"The artifact declares no logical input '{LOGICAL_INPUT_NAME}'",
                E_CONTRACT_MISSING,
            )
        declared = artifact.io.outputs
        if not declared:
            raise ModelConfigError(
                "The artifact declares no logical output", E_CONTRACT_MISSING
            )
        self._input_blob = logical_input
        self._logical_output = next(iter(declared))
        self._output_blob = str(declared[self._logical_output])
        names = self._graph_names("input_names")
        if names and self._input_blob not in names:
            raise ContractError(
                f"The graph has no input blob named '{self._input_blob}' "
                f"(it declares {sorted(names)})"
            )
        names = self._graph_names("output_names")
        if names and self._output_blob not in names:
            raise ContractError(
                f"The graph has no output blob named '{self._output_blob}' "
                f"(it declares {sorted(names)})"
            )

    def _graph_names(self, accessor: str) -> list[str]:
        """The blob names the graph declares, when the build reports them."""
        getter = getattr(self._net, accessor, None)
        if getter is None:
            return []
        try:
            return [str(name) for name in getter()]
        except Exception as exc:
            logger.debug("The ncnn graph did not report %s: %s", accessor, exc)
            return []

    def _prepare_tensor(self, frame: Any) -> np.ndarray:
        """Turn the canonical HWC tensor into the planar buffer ncnn wants."""
        array = np.asarray(frame)
        if array.ndim != 3:
            raise AIError(
                f"the canonical frame has shape {array.shape}, expected HWC",
                E_INVALID_INPUT,
            )
        planar = np.transpose(array, (2, 0, 1))
        dtype = np.float32 if self._norm_owner == "engine" else planar.dtype
        shape = (
            int(planar.shape[0]),
            int(planar.shape[1]),
            int(planar.shape[2]),
        )
        buffer = self._buffer_for(shape, dtype)
        np.copyto(buffer, planar, casting="unsafe")
        return buffer

    def _buffer_for(self, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        """The aligned input buffer, reused across the frames of one size."""
        key = (shape, str(np.dtype(dtype)))
        if self._buffer is None or self._buffer_key != key:
            self._buffer = ops.alloc_aligned_buffer(
                shape, dtype, alignment=ALIGNMENT_BYTES
            )
            self._buffer_key = key
        return self._buffer

    def _engine_norm_params(self) -> tuple[list[float], list[float]]:
        """The mean and the reciprocal std for the in-engine normalisation.

        The conversion itself lives in ops.py, which is the only place the
        framework computes it; the reciprocal is there because ncnn
        multiplies by its second argument while the framework divides.
        """
        if self._norm_params is None:
            canonical = self._canonical
            if canonical is None:
                raise AIError(
                    "the engine carries no canonical input contract",
                    E_CONTRACT_MISSING,
                )
            mean_e, std_e = ops.convert_norm_params(
                canonical.norm_scale, canonical.norm_mean, canonical.norm_std
            )
            channels = int(canonical.channels)
            mean_list = _per_channel(mean_e, channels)
            std_list = _per_channel(std_e, channels)
            self._norm_params = (
                mean_list,
                [(1.0 / value) if value != 0.0 else 0.0 for value in std_list],
            )
        return self._norm_params

    def _resolve_quant(self, artifact: ArtifactRef) -> str:
        """The quantization the artifact declares."""
        build = artifact.build or {}
        if build.get("int8") not in (None, "none", ""):
            return "int8"
        if build.get("fp16") not in (None, "none", ""):
            return "fp16"
        return "fp32"


def _per_channel(values: Any, channels: int) -> list[float]:
    """Broadcast the parameters to one value per channel."""
    array = np.atleast_1d(np.asarray(values, dtype=np.float32)).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, channels)
    if array.size != channels:
        raise ModelConfigError(
            f"the normalisation declares {array.size} values for {channels} channels",
            E_CONTRACT_MISSING,
        )
    return [float(value) for value in array]


def _runtime_options(artifact: ArtifactRef, opts: dict[str, Any]) -> dict[str, Any]:
    """The run-time options of an artifact, with the caller's overrides."""
    runtime = dict(artifact.runtime or {})
    override = opts.get("runtime")
    if isinstance(override, dict):
        runtime.update(override)
    return runtime


def _import_ncnn() -> Any:
    """Import the runtime, or report it as an unavailable engine."""
    try:
        return importlib.import_module("ncnn")
    except Exception as exc:
        raise ModelConfigError(
            f"The ncnn library is not importable: {exc}", E_ENGINE_UNAVAILABLE
        ) from exc


def create_engine() -> Optional[NcnnEngine]:
    """Build an unloaded ncnn engine, or None when the runtime is absent."""
    try:
        importlib.import_module("ncnn")
    except Exception as exc:
        logger.info("ncnn is not importable: %s", exc)
        return None
    return NcnnEngine()
