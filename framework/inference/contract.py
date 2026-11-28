"""Core contracts of the inference kernel.

The structures and protocols defined here are the single definition point
shared by engines, manifests and wrappers.  The module stays free of
platform and engine dependencies: it may only use the standard library, and
it must not import framework.inference.manifest at runtime because the
manifest module imports ArtifactRef from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from framework.hal.interface.ai import Detection
    from framework.inference.manifest import IOMapping, PreprocessSpec

__all__ = [
    "ArtifactRef",
    "Capabilities",
    "DetectionAPI",
    "GeometryTransform",
    "InferenceEngine",
    "PreprocessResult",
    "TensorSpec",
]


@dataclass(frozen=True)
class TensorSpec:
    """Shape and dtype of one engine tensor; -1 marks a dynamic dimension."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class Capabilities:
    """What an engine can do, cross-checked against platform and model at load time.

    device names the accelerator class ("cpu", "cuda", "vulkan", ...);
    device_detail carries the concrete device the runtime actually bound,
    such as the Vulkan device name and index, or the list of active
    execution providers.  It is empty when the runtime reports nothing
    beyond the class, and it exists because "vulkan" alone does not tell an
    operator whether the integrated or the discrete GPU is in use.
    """

    device: str
    quant: str
    layouts: tuple[str, ...]
    dynamic_shape: bool
    input_accepts: tuple[str, ...]
    io_binding: str
    releases_gil: bool
    load_holds_gil: bool
    memory_hint: str | None = None
    load_cost_hint: str | None = None
    device_detail: str = ""


@dataclass(frozen=True)
class ArtifactRef:
    """Load-time description of the concrete model file an engine has to load.

    path is the primary artifact file; aux_paths carries companion files
    keyed by role, for example "bin" for the ncnn weight file next to the
    .param graph.  runtime and build carry optional package/version and build
    recipe information and default to empty mappings when the manifest omits
    them.
    """

    engine_id: str
    path: str
    wrapper: str
    wrapper_params: dict[str, Any]
    preprocess: PreprocessSpec
    io: IOMapping
    runtime: dict[str, Any] = field(default_factory=dict)
    build: dict[str, Any] = field(default_factory=dict)
    aux_paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GeometryTransform:
    """Mapping from source frame pixels to input image pixels.

    The inverse mapping a wrapper applies to box coordinates is
    x_src = (x_input - offset_x) / scale_x and
    y_src = (y_input - offset_y) / scale_y.
    mode is "letterbox", "crop" or "stretch"; offset_x is the left padding
    for letterbox, the signed left crop amount for crop and zero for
    stretch, and scale_x / scale_y differ per axis for stretch.
    """

    mode: str
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int
    src_w: int
    src_h: int


@dataclass(slots=True)
class PreprocessResult:
    """Canonical preprocess output: engine-ready tensors plus the geometry to map results back."""

    inputs: dict[str, Any]
    transform: GeometryTransform


class InferenceEngine(Protocol):
    """One inference step: named tensors in, named tensors out, no postprocessing.

    An engine knows nothing about task semantics, never interprets its own
    outputs and never receives a raw frame.

    The run options carry the numeric parameters a dialect wants the engine
    to apply, such as the score threshold of an end-to-end engine that
    filters internally.  The engine forwards them to its runtime and ignores
    the ones its runtime does not know; it never interprets them.
    """

    engine_id: str

    def load(self, artifact: ArtifactRef, **opts: Any) -> None:
        """Load the artifact the runtime bound to this engine.

        ``opts`` is an open set rather than a schema: the runtime always
        passes ``canonical`` and ``runtime``, and an engine consumes the
        keys it understands and ignores the rest.  Rejecting an unknown key
        would make every engine break the moment the framework adds one.
        """
        ...

    def input_specs(self) -> list[TensorSpec]:
        ...

    def output_specs(self) -> list[TensorSpec]:
        ...

    def capabilities(self) -> Capabilities:
        ...

    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]:
        ...

    def unload(self) -> None:
        ...


class DetectionAPI(Protocol):
    """Task-level detection interface a wrapper exposes to the business layer."""

    def detect(self, frame: Any, **kw: Any) -> list[Detection]:
        ...
