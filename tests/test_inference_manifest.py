"""Tests for the inference manifest parser and validator.

Run: python -m pytest tests/test_inference_manifest.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from framework.hal.interface.ai import E_CONTRACT_MISSING, ManifestError
from framework.inference.manifest import (
    IOMapping,
    ModelManifest,
    PreprocessSpec,
    check_artifact_against_config,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_MANIFEST = REPO_ROOT / "deploy" / "maixcam" / "models" / "steelball_1280x352" / "manifest.yaml"
MODULE_PATH = REPO_ROOT / "framework" / "inference" / "manifest.py"


def _valid_document() -> dict:
    return {
        "schema_version": 3,
        "name": "demo",
        "family": "yolo11",
        "input": {
            "frame": {"color": "rgb", "dtype": "uint8", "layout": "hwc"},
            "size": [640, 352],
            "resize": {"mode": "letterbox", "value": 114},
            "norm": {
                "scale": 0.00392156862745098,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
        },
        "labels": ["ball"],
        "artifacts": {
            "ncnn": {
                "file": "demo.param",
                "bin": "demo.bin",
                "wrapper": "vision.det.raw",
                "preprocess": {
                    "resize": "framework",
                    "norm": "engine",
                    "formula": "substract_mean_normalize",
                },
                "io": {
                    "inputs": {"image": "data"},
                    "outputs": {"pred": "output"},
                },
            }
        },
    }


def _write_manifest(directory: Path, document: dict, name: str = "manifest.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _expect_error(directory: Path, document: dict) -> ManifestError:
    path = _write_manifest(directory, document)
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)
    assert excinfo.value.code == E_CONTRACT_MISSING
    return excinfo.value


def _mutate(**changes) -> dict:
    document = _valid_document()
    for path, value in changes.items():
        keys = path.split(".")
        target = document
        for key in keys[:-1]:
            target = target[key]
        if value is None:
            target.pop(keys[-1], None)
        else:
            target[keys[-1]] = value
    return document


# ---------- the happy path ---------- #


def test_valid_manifest_parses_into_the_expected_structures(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    assert isinstance(manifest, ModelManifest)
    assert manifest.schema_version == 3
    assert manifest.name == "demo"
    assert manifest.family == "yolo11"
    assert manifest.frame_color == "rgb"
    assert manifest.frame_dtype == "uint8"
    assert manifest.frame_layout == "hwc"
    assert manifest.target_size == (640, 352)
    assert manifest.resize_mode == "letterbox"
    assert manifest.resize_anchor == "center"
    assert manifest.resize_value == 114
    assert manifest.norm_scale == pytest.approx(1.0 / 255.0)
    assert manifest.norm_mean == (0.0, 0.0, 0.0)
    assert manifest.norm_std == (1.0, 1.0, 1.0)
    assert manifest.labels == ("ball",)


def test_artifact_is_parsed_into_an_artifact_ref(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    artifact = manifest.artifacts["ncnn"]
    assert artifact.engine_id == "ncnn"
    assert artifact.wrapper == "vision.det.raw"
    assert artifact.wrapper_params == {}
    assert artifact.runtime == {}
    assert artifact.build == {}
    assert isinstance(artifact.preprocess, PreprocessSpec)
    assert artifact.preprocess.resize_owner == "framework"
    assert artifact.preprocess.norm_owner == "engine"
    assert artifact.preprocess.formula == "substract_mean_normalize"
    assert isinstance(artifact.io, IOMapping)
    assert artifact.io.inputs == {"image": "data"}
    assert artifact.io.outputs == {"pred": "output"}


def test_artifact_paths_are_resolved_relative_to_the_manifest_directory(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    artifact = manifest.artifacts["ncnn"]
    assert artifact.path == str((tmp_path / "demo.param").resolve())
    assert artifact.aux_paths == {"bin": str((tmp_path / "demo.bin").resolve())}


def test_labels_file_is_resolved_relative_to_the_manifest_directory(tmp_path):
    (tmp_path / "labels.txt").write_text("ball\nsteel\n", encoding="utf-8")
    document = _valid_document()
    del document["labels"]
    document["labels_file"] = "labels.txt"

    manifest = load_manifest(_write_manifest(tmp_path, document))

    assert manifest.labels == ("ball", "steel")


def test_optional_artifact_blocks_default_to_empty_mappings(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["wrapper_params"] = {"output_layout": "auto"}
    document["artifacts"]["ncnn"]["runtime"] = {"package": "ncnn", "version": "1.0"}
    document["artifacts"]["ncnn"]["build"] = {"tool": "ncnnoptimize"}

    manifest = load_manifest(_write_manifest(tmp_path, document))

    artifact = manifest.artifacts["ncnn"]
    assert artifact.wrapper_params == {"output_layout": "auto"}
    assert artifact.runtime == {"package": "ncnn", "version": "1.0"}
    assert artifact.build == {"tool": "ncnnoptimize"}


def test_missing_artifact_files_are_not_a_parse_error(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    assert not (tmp_path / "demo.param").exists()
    assert manifest.artifacts["ncnn"].path.endswith("demo.param")


def test_repo_manifest_parses_cleanly():
    manifest = load_manifest(REPO_MANIFEST)

    assert manifest.name == "steelball_1280x352"
    assert manifest.family == "yolo11"
    assert sorted(manifest.artifacts) == ["maix_nn", "ncnn", "onnxruntime"]
    assert manifest.artifacts["ncnn"].aux_paths["bin"].endswith("steelball_1280x352.bin")


# ---------- io mapping ---------- #


def test_main_output_is_the_first_key_of_the_output_map(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["io"]["outputs"] = {
        "pred": "output",
        "extra": "output1",
    }

    manifest = load_manifest(_write_manifest(tmp_path, document))

    io = manifest.artifacts["ncnn"].io
    assert io.main_output == "pred"
    assert io.main_output_physical == "output"


def test_empty_output_map_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["io"]["outputs"] = {}

    error = _expect_error(tmp_path, document)

    assert "io.outputs" in error.message


def test_empty_output_map_is_rejected_when_the_mapping_is_built_directly():
    with pytest.raises(ManifestError) as excinfo:
        IOMapping(inputs={"image": "data"}, outputs={})
    assert excinfo.value.code == E_CONTRACT_MISSING


# ---------- schema version ---------- #


def test_missing_schema_version_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"schema_version": None}))

    assert "schema_version" in error.message


def test_unknown_schema_version_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(schema_version=99))

    assert "schema_version" in error.message


# ---------- top level fields ---------- #


def test_missing_name_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(name=None))

    assert "name" in error.message


def test_missing_family_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(family=None))

    assert "family" in error.message


def test_missing_input_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(input=None))

    assert "input" in error.message


def test_missing_labels_and_labels_file_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(labels=None))

    assert "labels" in error.message


def test_empty_artifacts_mapping_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(artifacts={}))

    assert "artifacts" in error.message


# ---------- frame contract ---------- #


def test_unknown_frame_colour_is_rejected(tmp_path):
    document = _valid_document()
    document["input"]["frame"]["color"] = "yuv"

    error = _expect_error(tmp_path, document)

    assert "input.frame.color" in error.message


def test_unknown_frame_dtype_is_rejected(tmp_path):
    document = _valid_document()
    document["input"]["frame"]["dtype"] = "int8"

    error = _expect_error(tmp_path, document)

    assert "input.frame.dtype" in error.message


def test_unknown_frame_layout_is_rejected(tmp_path):
    document = _valid_document()
    document["input"]["frame"]["layout"] = "chw"

    error = _expect_error(tmp_path, document)

    assert "input.frame.layout" in error.message


def test_input_size_must_hold_two_integers(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.size": [640]}))

    assert "input.size" in error.message


def test_input_size_must_hold_positive_integers(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.size": [640, 0]}))

    assert "input.size" in error.message


# ---------- resize policy ---------- #


def test_unknown_resize_mode_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.resize.mode": "pad"}))

    assert "input.resize.mode" in error.message


def test_anchor_outside_crop_mode_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.resize.anchor": "top"}))

    assert "input.resize.anchor" in error.message


def test_crop_anchor_defaults_to_center(tmp_path):
    document = _valid_document()
    document["input"]["resize"] = {"mode": "crop", "value": 0}

    manifest = load_manifest(_write_manifest(tmp_path, document))

    assert manifest.resize_anchor == "center"


def test_negative_resize_value_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.resize.value": -1}))

    assert "input.resize.value" in error.message


def test_transparent_padding_needs_a_four_channel_frame(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.resize.value": "transparent"}))

    assert "input.resize.value" in error.message


def test_transparent_padding_is_accepted_for_rgba(tmp_path):
    document = _valid_document()
    document["input"]["frame"]["color"] = "rgba"
    document["input"]["norm"]["mean"] = [0, 0, 0, 0]
    document["input"]["norm"]["std"] = [1, 1, 1, 1]
    document["input"]["resize"]["value"] = "transparent"

    manifest = load_manifest(_write_manifest(tmp_path, document))

    assert manifest.resize_value == "transparent"


# ---------- normalisation ---------- #


def test_missing_norm_scale_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.norm.scale": None}))

    assert "input.norm.scale" in error.message


def test_missing_norm_mean_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.norm.mean": None}))

    assert "input.norm.mean" in error.message


def test_missing_norm_std_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.norm.std": None}))

    assert "input.norm.std" in error.message


def test_norm_length_must_match_the_channel_count(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.norm.mean": [0, 0]}))

    assert "input.norm.mean" in error.message


def test_norm_std_length_must_match_the_channel_count(tmp_path):
    error = _expect_error(tmp_path, _mutate(**{"input.norm.std": [1, 1]}))

    assert "input.norm.std" in error.message


# ---------- labels ---------- #


def test_missing_labels_file_is_rejected(tmp_path):
    error = _expect_error(tmp_path, _mutate(labels=None, labels_file="absent.txt"))

    assert "labels_file" in error.message


def test_empty_labels_file_is_rejected(tmp_path):
    (tmp_path / "labels.txt").write_text("\n   \n", encoding="utf-8")

    error = _expect_error(tmp_path, _mutate(labels=None, labels_file="labels.txt"))

    assert "labels_file" in error.message


# ---------- artifacts ---------- #


def test_artifact_without_file_is_rejected(tmp_path):
    document = _valid_document()
    del document["artifacts"]["ncnn"]["file"]

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.file" in error.message


def test_artifact_without_wrapper_is_rejected(tmp_path):
    document = _valid_document()
    del document["artifacts"]["ncnn"]["wrapper"]

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.wrapper" in error.message


def test_artifact_without_preprocess_is_rejected(tmp_path):
    document = _valid_document()
    del document["artifacts"]["ncnn"]["preprocess"]

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess" in error.message


def test_artifact_without_io_is_rejected(tmp_path):
    document = _valid_document()
    del document["artifacts"]["ncnn"]["io"]

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.io" in error.message


def test_artifact_with_empty_io_inputs_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["io"]["inputs"] = {}

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.io.inputs" in error.message


def test_artifact_with_unknown_logical_input_name_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["io"]["inputs"] = {"picture": "data"}

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.io.inputs" in error.message


def test_artifact_with_unknown_resize_owner_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["preprocess"]["resize"] = "engine"

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess.resize" in error.message


def test_artifact_with_unknown_norm_owner_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["preprocess"]["norm"] = "wrapper"

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess.norm" in error.message


def test_artifact_without_formula_is_rejected(tmp_path):
    document = _valid_document()
    del document["artifacts"]["ncnn"]["preprocess"]["formula"]

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess.formula" in error.message


def test_artifact_with_unknown_formula_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["preprocess"]["formula"] = "imagenet"

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess.formula" in error.message


def test_artifact_with_formula_and_baked_norm_is_rejected(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["preprocess"]["norm"] = "baked"

    error = _expect_error(tmp_path, document)

    assert "artifacts.ncnn.preprocess.formula" in error.message


def test_baked_norm_without_formula_is_accepted(tmp_path):
    document = _valid_document()
    document["artifacts"]["ncnn"]["preprocess"] = {"resize": "baked", "norm": "baked"}

    manifest = load_manifest(_write_manifest(tmp_path, document))

    preprocess = manifest.artifacts["ncnn"].preprocess
    assert preprocess.resize_owner == "baked"
    assert preprocess.norm_owner == "baked"
    assert preprocess.formula is None


# ---------- configuration cross-check ---------- #


def test_cross_check_accepts_the_matching_engine_and_file(tmp_path):
    document = _valid_document()
    document["artifacts"]["maix_nn"] = {
        "file": "demo.mud",
        "wrapper": "vision.det.end2end",
        "preprocess": {"resize": "baked", "norm": "baked"},
        "io": {"inputs": {"image": "image"}, "outputs": {"objects": "objects"}},
    }
    manifest = load_manifest(_write_manifest(tmp_path, document))

    artifact = check_artifact_against_config(
        manifest, "maix_nn", tmp_path / "demo.mud"
    )

    assert artifact.engine_id == "maix_nn"


def test_cross_check_rejects_an_unknown_engine(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    with pytest.raises(ManifestError) as excinfo:
        check_artifact_against_config(manifest, "onnxruntime", tmp_path / "demo.param")
    assert excinfo.value.code == E_CONTRACT_MISSING


def test_cross_check_rejects_a_mismatched_file_name(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    with pytest.raises(ManifestError) as excinfo:
        check_artifact_against_config(manifest, "ncnn", tmp_path / "other.param")
    assert excinfo.value.code == E_CONTRACT_MISSING


def test_cross_check_rejects_a_directory_instead_of_a_file(tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path, _valid_document()))

    with pytest.raises(ManifestError) as excinfo:
        check_artifact_against_config(manifest, "ncnn", tmp_path)
    assert excinfo.value.code == E_CONTRACT_MISSING


# ---------- module hygiene ---------- #


def test_manifest_module_imports_no_platform_or_engine_library():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    for forbidden in ("numpy", "cv2", "maix", "ncnn", "onnxruntime", "rknn_lite", "torch"):
        assert forbidden not in roots
