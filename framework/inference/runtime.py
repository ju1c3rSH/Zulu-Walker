"""Model runtime: one engine instance bound to one artifact and one wrapper.

A runtime is the unit the facade owns.  It holds the engine instance, the
artifact contract, the wrapper's capability object and the statistics, and
it owns the canonical preprocess stage: the frame shape conversion, the
resize and the normalisation, in that order, driven by the artifact.

The runtime knows no task semantics and imports no platform code.  The
frame it starts from is declared by the platform through FrameSpec; the
frame it has to produce is declared by the manifest through CanonicalSpec.

The manifest parser stays pure and never touches the disk.  The file checks
belong to the load path: the runtime verifies that the artifact file and
its companions exist and are readable before the engine loads anything, so
a missing file is reported with its path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from framework.hal.interface.ai import (
    AIError,
    E_CONTRACT_MISSING,
    E_INVALID_INPUT,
    E_MODEL_LOAD_FAILED,
    E_MODEL_TYPE_UNSUPPORTED,
    ModelConfigError,
    ModelLoadError,
)
from framework.hal.interface.frame import frame_size
from framework.inference import ops
from framework.inference.contract import (
    ArtifactRef,
    GeometryTransform,
    InferenceEngine,
    PreprocessResult,
)
from framework.inference.wrappers import (
    get_wrapper_metadata,
    resolve_wrapper_factory,
)

if TYPE_CHECKING:
    from framework.inference.manifest import ModelManifest

__all__ = [
    "CanonicalSpec",
    "EngineStats",
    "FrameSpec",
    "ModelRuntime",
]

CHANNELS_BY_COLOR = {"rgb": 3, "bgr": 3, "rgba": 4}
RESIZE_MODES_WITH_SCRATCH = ("letterbox", "crop")


@dataclass(frozen=True)
class FrameSpec:
    """The frame shape the platform delivers: channel order, dtype and layout."""

    color: str
    dtype: str
    layout: str = "hwc"

    @property
    def colour(self) -> str:
        """The channel order, for callers that spell the field the British way."""
        return self.color


@dataclass(frozen=True)
class CanonicalSpec:
    """The framework configuration protocol of one logical model.

    It is the part of the manifest that describes what the model wants to
    eat: the frame shape, the target resolution, the resize policy and the
    normalisation parameters.
    """

    color: str
    dtype: str
    layout: str
    target_size: tuple[int, int]
    resize_mode: str
    resize_anchor: str
    resize_value: int | str
    norm_scale: float
    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]

    @classmethod
    def from_manifest(cls, manifest: "ModelManifest") -> "CanonicalSpec":
        """Take the configuration protocol of a parsed manifest."""
        return cls(
            color=manifest.frame_color,
            dtype=manifest.frame_dtype,
            layout=manifest.frame_layout,
            target_size=(int(manifest.target_size[0]), int(manifest.target_size[1])),
            resize_mode=manifest.resize_mode,
            resize_anchor=manifest.resize_anchor,
            resize_value=manifest.resize_value,
            norm_scale=manifest.norm_scale,
            norm_mean=tuple(manifest.norm_mean),
            norm_std=tuple(manifest.norm_std),
        )

    @property
    def channels(self) -> int:
        """The number of channels the canonical frame carries."""
        return CHANNELS_BY_COLOR[self.color]


@dataclass
class EngineStats:
    """The run-time statistics of one engine and model pair."""

    engine_id: str
    model_name: str
    infer_count: int = 0
    infer_avg_ms: float = 0.0
    infer_fps: float = 0.0
    error_count: int = 0
    last_error_code: str = ""
    load_ms: float = 0.0


class ModelRuntime:
    """One loaded engine, its artifact and the capability the wrapper built.

    The runtime owns the canonical preprocess stage.  The wrapper never
    implements resizing or normalisation itself; it calls preprocess() and
    uses the returned geometry to map results back to the source frame.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        artifact: ArtifactRef,
        *,
        manifest: Optional["ModelManifest"] = None,
        canonical: Optional[CanonicalSpec] = None,
        labels: Optional[Sequence[str]] = None,
        family: str = "",
        frame_spec: Optional[FrameSpec] = None,
        wrapper_factory: Optional[Callable[["ModelRuntime"], Any]] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._engine = engine
        self._artifact = artifact
        self._canonical = canonical
        if self._canonical is None and manifest is not None:
            self._canonical = CanonicalSpec.from_manifest(manifest)
        self._labels: tuple[str, ...] = tuple(
            labels if labels is not None else (manifest.labels if manifest is not None else ())
        )
        self._family = family or (manifest.family if manifest is not None else "")
        self._source_spec = frame_spec
        self._stats = EngineStats(
            engine_id=artifact.engine_id or getattr(engine, "engine_id", ""),
            model_name=manifest.name if manifest is not None else os.path.basename(artifact.path),
        )
        self._started_at = time.monotonic()
        self._resize_buffer: Optional[np.ndarray] = None
        self._resize_shape: Optional[tuple[Any, ...]] = None
        self._norm_buffer: Optional[np.ndarray] = None
        self._scratch_buffer: Optional[np.ndarray] = None
        self._scratch_shape: Optional[tuple[int, ...]] = None
        self._convert_buffer: Optional[np.ndarray] = None
        self._convert_shape: Optional[tuple[Any, ...]] = None
        self._engine_norm_params: Optional[tuple[np.ndarray, np.ndarray]] = None

        self._verify_artifact_files()
        started = time.perf_counter()
        # The canonical contract travels with the artifact: an engine that
        # normalises on the device needs the parameters it must apply.
        # The runtime options are the deployment's choice of accelerator;
        # the engine merges them over the block the artifact declares.
        engine.load(
            artifact,
            canonical=self._canonical,
            runtime=dict(runtime_options or {}),
        )
        self._stats.load_ms = (time.perf_counter() - started) * 1000.0

        self._capability = self._build_capability(wrapper_factory)

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #

    @property
    def engine(self) -> InferenceEngine:
        """The engine instance this runtime drives."""
        return self._engine

    @property
    def artifact(self) -> ArtifactRef:
        """The artifact contract of the loaded model file."""
        return self._artifact

    @property
    def params(self) -> dict[str, Any]:
        """The wrapper parameters declared by the artifact."""
        return self._artifact.wrapper_params

    @property
    def stats(self) -> EngineStats:
        """The run-time statistics of this runtime."""
        return self._stats

    @property
    def capability(self) -> Any:
        """The capability object the wrapper built."""
        return self._capability

    @property
    def labels(self) -> tuple[str, ...]:
        """The label list of the logical model."""
        return self._labels

    @property
    def family(self) -> str:
        """The model family of the logical model."""
        return self._family

    @property
    def canonical(self) -> Optional[CanonicalSpec]:
        """The configuration protocol the canonical stage produces."""
        return self._canonical

    def preprocess(self, frame: Any) -> PreprocessResult:
        """Prepare one frame for the engine.

        The stage runs in the documented order: the frame shape conversion,
        then the resize, then the normalisation.  When the artifact declares
        both preprocess segments as baked the whole stage is skipped and the
        frame is handed to the engine untouched; the decision belongs to the
        artifact and never to the frame source.
        """
        spec = self._artifact.preprocess
        if spec.resize_owner == "baked" and spec.norm_owner == "baked":
            return self._bypass(frame)
        if self._canonical is None:
            raise AIError(
                "the runtime carries no canonical input contract", E_CONTRACT_MISSING
            )
        if not isinstance(frame, np.ndarray):
            raise AIError(
                "the canonical stage needs a numpy frame", E_INVALID_INPUT
            )
        image = self._convert_frame(frame)
        image, transform = self._resize(image)
        tensor = self._normalize(image)
        return PreprocessResult(inputs=self._inputs(tensor), transform=transform)

    def record_inference(self, elapsed_ms: float) -> None:
        """Account one inference in the statistics."""
        stats = self._stats
        stats.infer_count += 1
        stats.infer_avg_ms = stats.infer_avg_ms + (
            elapsed_ms - stats.infer_avg_ms
        ) / stats.infer_count
        elapsed = time.monotonic() - self._started_at
        stats.infer_fps = stats.infer_count / elapsed if elapsed > 0.0 else 0.0

    def record_error(self, code: str) -> None:
        """Account one failure in the statistics."""
        self._stats.error_count += 1
        self._stats.last_error_code = code

    # ------------------------------------------------------------------ #
    # Canonical stage
    # ------------------------------------------------------------------ #

    def _bypass(self, frame: Any) -> PreprocessResult:
        """Pass the frame through untouched for a fully baked artifact."""
        width, height = frame_size(frame)
        transform = GeometryTransform(
            mode="stretch",
            scale_x=1.0,
            scale_y=1.0,
            offset_x=0,
            offset_y=0,
            src_w=width,
            src_h=height,
        )
        return PreprocessResult(inputs=self._inputs(frame), transform=transform)

    def _inputs(self, tensor: Any) -> dict[str, Any]:
        """Name the canonical tensor by the logical input names of the artifact."""
        return {name: tensor for name in self._artifact.io.inputs}

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Bring the frame into the shape the manifest declares."""
        canonical = self._canonical
        source = self._effective_source_spec(frame)
        if (
            source.color == canonical.color
            and source.dtype == canonical.dtype
            and source.layout == canonical.layout
        ):
            return frame
        if source.layout != "hwc" or canonical.layout != "hwc":
            raise AIError(
                f"cannot convert a '{source.layout}' frame into a "
                f"'{canonical.layout}' frame",
                E_INVALID_INPUT,
            )
        return self._convert_array(frame, source, canonical)

    def _effective_source_spec(self, frame: np.ndarray) -> FrameSpec:
        """The frame shape of the incoming frame, from the platform or the data."""
        if self._source_spec is not None:
            return self._source_spec
        dtype = "uint8" if frame.dtype == np.uint8 else "float32"
        color = self._canonical.color if self._canonical is not None else "bgr"
        return FrameSpec(color=color, dtype=dtype, layout="hwc")

    def _convert_array(
        self,
        frame: np.ndarray,
        source: FrameSpec,
        canonical: CanonicalSpec,
    ) -> np.ndarray:
        """Convert colour order, channel count and dtype in one copy."""
        source_channels = frame.shape[2] if frame.ndim == 3 else 1
        target_channels = canonical.channels
        if target_channels == 4 and source_channels < 4:
            raise AIError(
                f"the model wants a '{canonical.color}' frame but the platform "
                f"frame has no alpha channel",
                E_INVALID_INPUT,
            )
        buffer = self._convert_buffer_for(frame, target_channels)
        if source_channels == 1 and target_channels == 3:
            buffer[...] = frame if frame.ndim == 3 else frame[:, :, None]
            return buffer
        if source_channels == 4 and target_channels == 3:
            if source.color == canonical.color:
                buffer[...] = frame[..., :3]
            else:
                buffer[...] = frame[..., 2::-1]
            return buffer
        if source_channels == 3 and target_channels == 3:
            if source.color == canonical.color:
                buffer[...] = frame
            else:
                buffer[...] = frame[..., ::-1]
            return buffer
        raise AIError(
            f"cannot convert a {source_channels}-channel frame into the "
            f"{target_channels}-channel '{canonical.color}' frame",
            E_INVALID_INPUT,
        )

    def _resize(self, image: np.ndarray) -> tuple[np.ndarray, GeometryTransform]:
        """Bring the frame to the target resolution of the model."""
        canonical = self._canonical
        width, height = canonical.target_size
        if image.shape[1] == width and image.shape[0] == height:
            transform = GeometryTransform(
                mode="stretch",
                scale_x=1.0,
                scale_y=1.0,
                offset_x=0,
                offset_y=0,
                src_w=int(image.shape[1]),
                src_h=int(image.shape[0]),
            )
            return image, transform
        out = self._resize_buffer_for(image)
        scratch = self._scratch_for(image, width, height, canonical.resize_mode)
        resized, transform = ops.resize_frame(
            image,
            (width, height),
            mode=canonical.resize_mode,
            value=canonical.resize_value,
            anchor=canonical.resize_anchor,
            out=out,
            scratch=scratch,
        )
        return resized, transform

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply the normalisation the artifact declares, if the framework owns it."""
        owner = self._artifact.preprocess.norm_owner
        if owner != "framework":
            return image
        formula = self._artifact.preprocess.formula
        canonical = self._canonical
        buffer = self._norm_buffer_for(image)
        if formula == "yolo":
            return ops.normalize_yolo(
                image,
                canonical.norm_scale,
                canonical.norm_mean,
                canonical.norm_std,
                out=buffer,
            )
        if formula == "substract_mean_normalize":
            mean, std = self._engine_norm_params()
            return ops.normalize_engine(image, mean, std, out=buffer)
        raise AIError(
            f"unknown normalisation formula '{formula}'", E_CONTRACT_MISSING
        )

    def _engine_norm_params(self) -> tuple[np.ndarray, np.ndarray]:
        """The normalisation parameters in the engine form, converted once."""
        if self._engine_norm_params is None:
            canonical = self._canonical
            self._engine_norm_params = ops.convert_norm_params(
                canonical.norm_scale, canonical.norm_mean, canonical.norm_std
            )
        return self._engine_norm_params

    # ------------------------------------------------------------------ #
    # Working buffers
    # ------------------------------------------------------------------ #

    def _convert_buffer_for(self, frame: np.ndarray, channels: int) -> np.ndarray:
        """The buffer the frame shape conversion writes into."""
        shape = (int(frame.shape[0]), int(frame.shape[1]), channels)
        dtype = self._canonical.dtype if self._canonical is not None else frame.dtype
        if self._convert_buffer is None or self._convert_shape != (shape, str(dtype)):
            self._convert_buffer = np.empty(shape, dtype=dtype)
            self._convert_shape = (shape, str(dtype))
        return self._convert_buffer

    def _resize_buffer_for(self, image: np.ndarray) -> np.ndarray:
        """The buffer the resized frame is written into."""
        canonical = self._canonical
        shape = (canonical.target_size[1], canonical.target_size[0], canonical.channels)
        dtype = canonical.dtype
        if self._resize_buffer is None or self._resize_shape != (shape, str(dtype)):
            self._resize_buffer = np.empty(shape, dtype=dtype)
            self._resize_shape = (shape, str(dtype))
        return self._resize_buffer

    def _norm_buffer_for(self, image: np.ndarray) -> np.ndarray:
        """The float32 buffer the normalised frame is written into."""
        shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
        if self._norm_buffer is None or self._norm_buffer.shape != shape:
            self._norm_buffer = np.empty(shape, dtype=np.float32)
        return self._norm_buffer

    def _scratch_for(
        self,
        image: np.ndarray,
        width: int,
        height: int,
        mode: str,
    ) -> Optional[np.ndarray]:
        """The buffer for the intermediate scaled image of letterbox and crop.

        The scaled size follows the same rule ops.py applies, so the buffer
        is only sized here; the resize itself stays in ops.py.
        """
        if mode not in RESIZE_MODES_WITH_SCRATCH:
            return None
        src_h, src_w = image.shape[:2]
        if mode == "letterbox":
            scale = min(width / src_w, height / src_h)
            new_w = min(width, max(1, int(round(src_w * scale))))
            new_h = min(height, max(1, int(round(src_h * scale))))
        else:
            scale = max(width / src_w, height / src_h)
            new_w = max(width, int(round(src_w * scale)))
            new_h = max(height, int(round(src_h * scale)))
        shape = (new_h, new_w) + tuple(image.shape[2:])
        if self._scratch_buffer is None or self._scratch_shape != shape:
            self._scratch_buffer = np.empty(shape, dtype=image.dtype)
            self._scratch_shape = shape
        return self._scratch_buffer

    # ------------------------------------------------------------------ #
    # Load time validation
    # ------------------------------------------------------------------ #

    def _verify_artifact_files(self) -> None:
        """Check that the artifact file and its companions exist and are readable.

        The manifest parser stays free of disk access, so the file check
        belongs to the load path.  It runs before the engine loads anything,
        so a missing file is reported with its path instead of with whatever
        the engine makes of it.
        """
        self._require_readable_file(self._artifact.path, "the artifact file")
        for role, path in sorted(self._artifact.aux_paths.items()):
            self._require_readable_file(path, f"the companion '{role}' file")

    @staticmethod
    def _require_readable_file(path: str, description: str) -> None:
        """Read one byte of a file, or report it as an unloadable artifact."""
        try:
            with open(path, "rb") as handle:
                handle.read(1)
        except OSError as exc:
            raise ModelLoadError(
                f"{description} '{path}' cannot be read ({exc.strerror or exc})",
                E_MODEL_LOAD_FAILED,
            ) from exc

    # ------------------------------------------------------------------ #
    # Capability construction
    # ------------------------------------------------------------------ #

    def _build_capability(
        self,
        wrapper_factory: Optional[Callable[["ModelRuntime"], Any]],
    ) -> Any:
        """Resolve the dialect of the artifact and let it build the capability."""
        if wrapper_factory is None:
            factory = resolve_wrapper_factory(self._artifact.wrapper)
        else:
            factory = wrapper_factory
        self._check_family()
        return factory(self)

    def _check_family(self) -> None:
        """Refuse a dialect that does not accept the family of the model."""
        key = self._artifact.wrapper
        if ":" in key or not self._family:
            return
        try:
            families = frozenset(get_wrapper_metadata(key)["families"])
        except ModelConfigError:
            return
        if self._family not in families:
            raise ModelConfigError(
                f"dialect '{key}' does not accept the model family "
                f"'{self._family}'",
                E_MODEL_TYPE_UNSUPPORTED,
            )
