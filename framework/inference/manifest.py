"""Parser and validator for the inference manifest schema v3.

A manifest is the single source of the run-time contract: labels, the
frame contract, the per-artifact preprocessing ownership and the io
mapping.  This module turns a manifest file into typed, validated
objects and refuses anything it cannot guarantee.

The module stays free of platform and engine knowledge: it knows the
schema, not the engines.  It therefore never checks that an artifact
file exists on disk (files are provisioned separately) and never
validates an engine id against a registry (a manifest may declare
artifacts for engines other than the one in use).

Every contract failure raises ManifestError, whose code is
E_CONTRACT_MISSING.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from framework.hal.interface.ai import ManifestError
from framework.inference.contract import ArtifactRef

__all__ = [
    "IOMapping",
    "ModelManifest",
    "PreprocessSpec",
    "SUPPORTED_SCHEMA_VERSION",
    "check_artifact_against_config",
    "load_manifest",
]

SUPPORTED_SCHEMA_VERSION = 3

RESIZE_OWNERS = frozenset({"framework", "baked"})
NORM_OWNERS = frozenset({"framework", "engine", "baked"})
NORM_FORMULAS = frozenset({"yolo", "substract_mean_normalize"})
RESIZE_MODES = frozenset({"letterbox", "crop", "stretch"})
CROP_ANCHORS = frozenset({"center", "top", "bottom", "left", "right"})
FRAME_COLORS = frozenset({"rgb", "bgr", "rgba"})
FRAME_DTYPES = frozenset({"uint8", "float32"})
FRAME_LAYOUTS = frozenset({"hwc"})
COLOR_CHANNELS = {"rgb": 3, "bgr": 3, "rgba": 4}

LOGICAL_INPUT_NAME = "image"
DEFAULT_RESIZE_VALUE = 114
DEFAULT_CROP_ANCHOR = "center"


@dataclass(frozen=True)
class PreprocessSpec:
    """Who performs each preprocessing stage for one artifact.

    resize_owner is "framework" or "baked", norm_owner is "framework",
    "engine" or "baked".  formula names the normalisation formula and is
    absent exactly when norm_owner is "baked".
    """

    resize_owner: str
    norm_owner: str
    formula: str | None = None

    def __post_init__(self) -> None:
        _validate_preprocess(self.resize_owner, self.norm_owner, self.formula, "preprocess")


@dataclass(frozen=True)
class IOMapping:
    """Logical-to-physical name maps of one artifact.

    The output map keeps the YAML writing order, so its first key is the
    primary output the wrapper reads.
    """

    inputs: dict[str, str | int]
    outputs: dict[str, str | int]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ManifestError(
                f"io.inputs: must declare the logical input {LOGICAL_INPUT_NAME!r} "
                f"(got {self.inputs!r})"
            )
        if not self.outputs:
            raise ManifestError(
                f"io.outputs: must declare at least one output (got {self.outputs!r})"
            )

    @property
    def main_output(self) -> str:
        """The logical name of the primary output."""
        return next(iter(self.outputs))

    @property
    def main_output_physical(self) -> str | int:
        """The physical name or index of the primary output."""
        return self.outputs[self.main_output]


@dataclass(frozen=True)
class ModelManifest:
    """The validated run-time contract of one logical model.

    The frame contract describes the tensor the model expects, the resize
    policy and the normalisation parameters are the model-level ones, and
    artifacts holds the per-engine artifact descriptions keyed by engine
    id.
    """

    schema_version: int
    name: str
    family: str
    frame_color: str
    frame_dtype: str
    frame_layout: str
    target_size: tuple[int, int]
    resize_mode: str
    resize_anchor: str
    resize_value: int | str
    norm_scale: float
    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]
    labels: tuple[str, ...]
    artifacts: dict[str, ArtifactRef]


def load_manifest(path: str | Path) -> ModelManifest:
    """Read a manifest file and return the validated contract it declares."""
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(
            f"manifest: cannot read {manifest_path} ({exc.strerror or exc})"
        ) from exc
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest: cannot parse {manifest_path} ({exc})") from exc
    if not isinstance(document, dict):
        raise ManifestError(
            f"manifest: the top level of {manifest_path} must be a mapping "
            f"(got {type(document).__name__})"
        )
    return _parse_document(document, manifest_path.parent.resolve())


def check_artifact_against_config(
    manifest: ModelManifest, engine_id: str, model_path: str | Path
) -> ArtifactRef:
    """Cross-check the run-time configuration against the manifest.

    The runtime names one engine and one exact artifact file.  The
    manifest must declare that engine and that same file; there is no
    directory form and no fallback.  Returns the matching artifact.
    """
    artifact = manifest.artifacts.get(engine_id)
    if artifact is None:
        raise ManifestError(
            f"artifacts: the manifest declares no artifact for engine "
            f"{engine_id!r} (declared engines: {sorted(manifest.artifacts)})"
        )
    declared = _normalise_path(artifact.path)
    configured = _normalise_path(str(model_path))
    if declared != configured and Path(declared).name != Path(configured).name:
        raise ManifestError(
            f"artifacts.{engine_id}.file: the configured model "
            f"{str(model_path)!r} is not the declared artifact file "
            f"{artifact.path!r}"
        )
    return artifact


def _parse_document(document: dict[str, Any], base: Path) -> ModelManifest:
    version = document.get("schema_version")
    if version is None:
        raise _fail("schema_version", version, "is required")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise _fail(
            "schema_version",
            version,
            f"is not a known schema version (this parser knows "
            f"{SUPPORTED_SCHEMA_VERSION})",
        )

    name = _require_str(document.get("name"), "name")
    family = _require_str(document.get("family"), "family")
    input_block = _require_mapping(document.get("input"), "input")

    color, dtype, layout = _parse_frame(input_block.get("frame"))
    channels = COLOR_CHANNELS[color]
    target_size = _parse_size(input_block.get("size"))
    resize_mode, resize_anchor, resize_value = _parse_resize(
        input_block.get("resize"), channels
    )
    norm_scale, norm_mean, norm_std = _parse_norm(input_block.get("norm"), channels)
    labels = _parse_labels(document, base)
    artifacts = _parse_artifacts(document.get("artifacts"), base)

    return ModelManifest(
        schema_version=version,
        name=name,
        family=family,
        frame_color=color,
        frame_dtype=dtype,
        frame_layout=layout,
        target_size=target_size,
        resize_mode=resize_mode,
        resize_anchor=resize_anchor,
        resize_value=resize_value,
        norm_scale=norm_scale,
        norm_mean=norm_mean,
        norm_std=norm_std,
        labels=labels,
        artifacts=artifacts,
    )


def _parse_frame(raw: Any) -> tuple[str, str, str]:
    frame = _require_mapping(raw, "input.frame")
    color = _require_str(frame.get("color"), "input.frame.color")
    if color not in FRAME_COLORS:
        raise _fail("input.frame.color", color, f"must be one of {sorted(FRAME_COLORS)}")
    dtype = _require_str(frame.get("dtype"), "input.frame.dtype")
    if dtype not in FRAME_DTYPES:
        raise _fail("input.frame.dtype", dtype, f"must be one of {sorted(FRAME_DTYPES)}")
    layout = _require_str(frame.get("layout"), "input.frame.layout")
    if layout not in FRAME_LAYOUTS:
        raise _fail("input.frame.layout", layout, f"must be one of {sorted(FRAME_LAYOUTS)}")
    return color, dtype, layout


def _parse_size(raw: Any) -> tuple[int, int]:
    size = _require_sequence(raw, "input.size")
    if len(size) != 2:
        raise _fail("input.size", size, "must hold exactly two values (width, height)")
    width = _require_int(size[0], "input.size[0]")
    height = _require_int(size[1], "input.size[1]")
    if width <= 0 or height <= 0:
        raise _fail("input.size", size, "must hold two positive integers")
    return width, height


def _parse_resize(raw: Any, channels: int) -> tuple[str, str, int | str]:
    resize = _require_mapping(raw, "input.resize")
    mode = _require_str(resize.get("mode"), "input.resize.mode")
    if mode not in RESIZE_MODES:
        raise _fail("input.resize.mode", mode, f"must be one of {sorted(RESIZE_MODES)}")

    anchor = resize.get("anchor")
    if anchor is None:
        anchor = DEFAULT_CROP_ANCHOR
    else:
        anchor = _require_str(anchor, "input.resize.anchor")
        if anchor not in CROP_ANCHORS:
            raise _fail(
                "input.resize.anchor", anchor, f"must be one of {sorted(CROP_ANCHORS)}"
            )
        if mode != "crop":
            raise _fail(
                "input.resize.anchor",
                anchor,
                f"is only meaningful for mode 'crop' (the mode is {mode!r})",
            )

    value = resize.get("value", DEFAULT_RESIZE_VALUE)
    if isinstance(value, str):
        if value != "transparent":
            raise _fail(
                "input.resize.value",
                value,
                "must be a non-negative int or the literal 'transparent'",
            )
        if channels != 4:
            raise _fail(
                "input.resize.value",
                value,
                f"'transparent' padding needs a 4-channel frame (rgba), the "
                f"frame has {channels} channels",
            )
    else:
        value = _require_int(value, "input.resize.value")
        if value < 0:
            raise _fail("input.resize.value", value, "must not be negative")
    return mode, anchor, value


def _parse_norm(raw: Any, channels: int) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    norm = _require_mapping(raw, "input.norm")
    if "scale" not in norm:
        raise _fail("input.norm.scale", norm.get("scale"), "is required")
    scale = _require_number(norm["scale"], "input.norm.scale")
    mean = _require_sequence(norm.get("mean"), "input.norm.mean")
    std = _require_sequence(norm.get("std"), "input.norm.std")
    if len(mean) != channels:
        raise _fail(
            "input.norm.mean", mean, f"must hold {channels} entries for the frame colour"
        )
    if len(std) != channels:
        raise _fail(
            "input.norm.std", std, f"must hold {channels} entries for the frame colour"
        )
    mean_values = tuple(_require_number(item, "input.norm.mean") for item in mean)
    std_values = tuple(_require_number(item, "input.norm.std") for item in std)
    return scale, mean_values, std_values


def _parse_labels(document: dict[str, Any], base: Path) -> tuple[str, ...]:
    labels_file = document.get("labels_file")
    inline_labels = document.get("labels")
    if labels_file is not None:
        labels_path = (base / str(labels_file)).resolve()
        if not labels_path.is_file():
            raise _fail("labels_file", labels_file, f"does not exist at {labels_path}")
        try:
            text = labels_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _fail(
                "labels_file", labels_file, f"cannot be read ({exc.strerror or exc})"
            ) from exc
        lines = tuple(line.strip() for line in text.splitlines() if line.strip())
        if not lines:
            raise _fail("labels_file", labels_file, f"holds no label at {labels_path}")
        return lines
    if inline_labels is None:
        raise _fail(
            "labels",
            inline_labels,
            "one of 'labels' or 'labels_file' is required",
        )
    items = _require_sequence(inline_labels, "labels")
    if not items:
        raise _fail("labels", inline_labels, "must not be empty")
    return tuple(
        _require_str(item, f"labels[{index}]").strip()
        for index, item in enumerate(items)
    )


def _parse_artifacts(raw: Any, base: Path) -> dict[str, ArtifactRef]:
    block = _require_mapping(raw, "artifacts")
    if not block:
        raise _fail("artifacts", block, "must declare at least one engine artifact")
    artifacts: dict[str, ArtifactRef] = {}
    for engine_id, entry in block.items():
        artifacts[str(engine_id)] = _parse_artifact(str(engine_id), entry, base)
    return artifacts


def _parse_artifact(engine_id: str, raw: Any, base: Path) -> ArtifactRef:
    path = f"artifacts.{engine_id}"
    block = _require_mapping(raw, path)

    file_name = _require_str(block.get("file"), f"{path}.file")
    artifact_path = (base / file_name).resolve()
    wrapper = _require_str(block.get("wrapper"), f"{path}.wrapper")
    wrapper_params = _optional_mapping(block.get("wrapper_params"), f"{path}.wrapper_params")
    runtime = _optional_mapping(block.get("runtime"), f"{path}.runtime")
    build = _optional_mapping(block.get("build"), f"{path}.build")
    preprocess = _parse_preprocess(block.get("preprocess"), f"{path}.preprocess")
    io = _parse_io(block.get("io"), f"{path}.io")

    aux_paths: dict[str, str] = {}
    bin_name = block.get("bin")
    if bin_name is not None:
        aux_paths["bin"] = str((base / _require_str(bin_name, f"{path}.bin")).resolve())

    return ArtifactRef(
        engine_id=engine_id,
        path=str(artifact_path),
        wrapper=wrapper,
        wrapper_params=wrapper_params,
        preprocess=preprocess,
        io=io,
        runtime=runtime,
        build=build,
        aux_paths=aux_paths,
    )


def _parse_preprocess(raw: Any, path: str) -> PreprocessSpec:
    block = _require_mapping(raw, path)
    resize_owner = _require_str(block.get("resize"), f"{path}.resize")
    norm_owner = _require_str(block.get("norm"), f"{path}.norm")
    formula = block.get("formula")
    _validate_preprocess(resize_owner, norm_owner, formula, path)
    return PreprocessSpec(
        resize_owner=resize_owner,
        norm_owner=norm_owner,
        formula=formula if isinstance(formula, str) else None,
    )


def _validate_preprocess(
    resize_owner: str, norm_owner: str, formula: Any, path: str
) -> None:
    if resize_owner not in RESIZE_OWNERS:
        raise _fail(f"{path}.resize", resize_owner, f"must be one of {sorted(RESIZE_OWNERS)}")
    if norm_owner not in NORM_OWNERS:
        raise _fail(f"{path}.norm", norm_owner, f"must be one of {sorted(NORM_OWNERS)}")
    if norm_owner == "baked":
        if formula is not None:
            raise _fail(
                f"{path}.formula",
                formula,
                "must be absent when the normalisation is baked into the artifact",
            )
        return
    if formula is None:
        raise _fail(
            f"{path}.formula",
            formula,
            f"is required when the norm owner is {norm_owner!r}",
        )
    if formula not in NORM_FORMULAS:
        raise _fail(f"{path}.formula", formula, f"must be one of {sorted(NORM_FORMULAS)}")


def _parse_io(raw: Any, path: str) -> IOMapping:
    block = _require_mapping(raw, path)
    inputs = _require_mapping(block.get("inputs"), f"{path}.inputs")
    outputs = _require_mapping(block.get("outputs"), f"{path}.outputs")
    if not inputs:
        raise _fail(f"{path}.inputs", inputs, "must not be empty")
    if not outputs:
        raise _fail(f"{path}.outputs", outputs, "must not be empty")
    for logical_name in inputs:
        if logical_name != LOGICAL_INPUT_NAME:
            raise _fail(
                f"{path}.inputs",
                logical_name,
                f"the logical input name must be {LOGICAL_INPUT_NAME!r}",
            )
    return IOMapping(inputs=dict(inputs), outputs=dict(outputs))


def _optional_mapping(raw: Any, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    return dict(_require_mapping(raw, path))


def _normalise_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _fail(field_path: str, value: Any, reason: str) -> ManifestError:
    return ManifestError(f"{field_path}: {reason} (got {value!r})")


def _require_mapping(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(field_path, value, "must be a mapping")
    return value


def _require_sequence(value: Any, field_path: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise _fail(field_path, value, "must be a list")
    return list(value)


def _require_str(value: Any, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(field_path, value, "must be a non-empty string")
    return value


def _require_int(value: Any, field_path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(field_path, value, "must be an integer")
    return value


def _require_number(value: Any, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(field_path, value, "must be a number")
    return float(value)
