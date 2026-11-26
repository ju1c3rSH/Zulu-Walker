"""Tests for scripts/check_manifest.py.

Run: python -m pytest tests/test_manifest_checker.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_manifest.py"
INSTANCE_MANIFEST = (REPO_ROOT / "deploy" / "maixcam" / "models" /
                     "steelball_1280x352" / "manifest.yaml")


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_manifest", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_manifest"] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _base_manifest() -> dict:
    return {
        "schema_version": 3,
        "name": "demo",
        "family": "yolo11",
        "input": {
            "frame": {"color": "rgb", "dtype": "uint8", "layout": "hwc"},
            "size": [640, 352],
            "resize": {"mode": "letterbox", "value": 114},
            "norm": {"scale": 0.00392156862745098,
                     "mean": [0, 0, 0],
                     "std": [1, 1, 1]},
        },
        "labels": ["a", "b"],
        "artifacts": {
            "maix_nn": {
                "file": "demo.mud",
                "wrapper": "vision.det.end2end",
                "wrapper_params": {},
                "preprocess": {"resize": "baked", "norm": "baked"},
                "io": {
                    "inputs": {"image": "image"},
                    "outputs": {"objects": "objects"},
                },
            },
        },
    }


def _write_manifest(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, doc: dict) -> int:
    path = _write_manifest(tmp_path, doc)
    return checker.main([str(SCRIPT_PATH), str(path)])


def _errors(output: str) -> list[str]:
    return [line[len("ERROR "):] for line in output.splitlines()
            if line.startswith("ERROR ")]


def _warnings(output: str) -> list[str]:
    return [line[len("WARN  "):] for line in output.splitlines()
            if line.startswith("WARN  ")]


def test_valid_manifest_is_clean(tmp_path, capsys):
    code = _run(tmp_path, _base_manifest())
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []
    assert "0 error(s)" in out


def test_empty_io_inputs_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["io"]["inputs"] = {}
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("io.inputs is empty" in e for e in _errors(out))


def test_empty_io_outputs_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["io"]["outputs"] = {}
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("io.outputs is empty" in e for e in _errors(out))


def test_logical_input_name_must_be_image(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["io"]["inputs"] = {"blob": "data"}
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("logical input" in e and "blob" in e for e in _errors(out))


def test_norm_mean_must_match_the_channel_count(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["norm"]["mean"] = [0, 0]
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("'input.norm.mean' has 2 entries but the frame has 3 channels"
               in e for e in _errors(out))


def test_norm_std_must_match_the_channel_count(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["norm"]["std"] = [1, 1, 1, 1]
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("'input.norm.std' has 4 entries but the frame has 3 channels"
               in e for e in _errors(out))


def test_rgba_frame_accepts_four_channel_norm(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["frame"]["color"] = "rgba"
    doc["input"]["norm"]["mean"] = [0, 0, 0, 0]
    doc["input"]["norm"]["std"] = [1, 1, 1, 1]
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []


def test_transparent_resize_on_a_three_channel_frame_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["resize"]["value"] = "transparent"
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("'transparent' needs a 4-channel frame" in e
               for e in _errors(out))


def test_transparent_resize_on_a_four_channel_frame_is_allowed(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["frame"]["color"] = "rgba"
    doc["input"]["norm"]["mean"] = [0, 0, 0, 0]
    doc["input"]["norm"]["std"] = [1, 1, 1, 1]
    doc["input"]["resize"]["value"] = "transparent"
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []


def test_unknown_build_layout_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["build"] = {
        "input_size": [640, 352],
        "layout": "channels_first",
        "color": "rgb",
    }
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("build.layout must be one of" in e for e in _errors(out))


def test_build_layout_may_differ_from_the_frame_layout(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["build"] = {
        "input_size": [640, 352],
        "layout": "nchw",
        "color": "rgb",
    }
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []


def test_runtime_version_must_match_build_version(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["runtime"] = {"package": "rknn-toolkit-lite2",
                                              "version": "2.3.0"}
    doc["artifacts"]["maix_nn"]["build"] = {
        "input_size": [640, 352],
        "layout": "nhwc",
        "color": "rgb",
        "version": "2.2.0",
    }
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("runtime.version '2.3.0' != build.version '2.2.0'" in e
               for e in _errors(out))


def test_matching_runtime_and_build_version_is_clean(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["runtime"] = {"package": "rknn-toolkit-lite2",
                                              "version": "2.3.0"}
    doc["artifacts"]["maix_nn"]["build"] = {
        "input_size": [640, 352],
        "layout": "nhwc",
        "color": "rgb",
        "version": "2.3.0",
    }
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []


def test_unknown_engine_id_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    artifact = doc["artifacts"].pop("maix_nn")
    doc["artifacts"]["my_engine"] = artifact
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("unknown engine id" in e for e in _errors(out))


def test_unknown_wrapper_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["artifacts"]["maix_nn"]["wrapper"] = "vision.det.nope"
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("unknown wrapper" in e for e in _errors(out))


def test_end2end_wrapper_accepts_yolov5(tmp_path, capsys):
    doc = _base_manifest()
    doc["family"] = "yolov5"
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []


def test_missing_manifest_is_a_usage_error():
    assert checker.main([str(SCRIPT_PATH), "no_such_manifest.yaml"]) == 2


def test_no_arguments_is_a_usage_error():
    assert checker.main([str(SCRIPT_PATH)]) == 2


def test_repository_instance_manifest_has_no_errors(capsys):
    code = checker.main([str(SCRIPT_PATH), str(INSTANCE_MANIFEST)])
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []
    assert "0 error(s)" in out
    assert len(_warnings(out)) == 4


# --- the checker must never be laxer than the run-time parser ---------------

def _parser_rejects(tmp_path: Path, doc: dict) -> str:
    """Assert the run-time parser rejects the manifest and return its reason."""
    path = _write_manifest(tmp_path, doc)
    runtime = checker.load_runtime()
    with pytest.raises(runtime.manifest_error) as excinfo:
        runtime.load_manifest(path)
    return str(excinfo.value)


def test_unknown_schema_version_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["schema_version"] = 2
    reason = _parser_rejects(tmp_path, doc)
    assert "schema_version" in reason
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("schema_version" in e for e in _errors(out))


def test_resize_anchor_on_a_non_crop_mode_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["resize"]["anchor"] = "center"
    reason = _parser_rejects(tmp_path, doc)
    assert "anchor" in reason
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("anchor" in e and "crop" in e for e in _errors(out))


def test_missing_labels_file_is_an_error(tmp_path, capsys):
    doc = _base_manifest()
    doc.pop("labels")
    doc["labels_file"] = "missing_labels.txt"
    reason = _parser_rejects(tmp_path, doc)
    assert "labels_file" in reason
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("labels_file" in e for e in _errors(out))


def test_norm_scale_must_be_a_number(tmp_path, capsys):
    doc = _base_manifest()
    doc["input"]["norm"]["scale"] = "abc"
    reason = _parser_rejects(tmp_path, doc)
    assert "scale" in reason
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("scale" in e for e in _errors(out))


def test_label_entries_must_be_non_empty_strings(tmp_path, capsys):
    doc = _base_manifest()
    doc["labels"] = [1, 2]
    reason = _parser_rejects(tmp_path, doc)
    assert "labels[0]" in reason
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("labels[0]" in e for e in _errors(out))


def test_a_manifest_the_runtime_parser_accepts_passes_the_checker(tmp_path,
                                                                 capsys):
    doc = _base_manifest()
    path = _write_manifest(tmp_path, doc)
    runtime = checker.load_runtime()
    assert runtime.load_manifest(path).name == "demo"
    code = checker.main([str(SCRIPT_PATH), str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []
    assert "0 error(s)" in out


def test_the_real_registries_are_used():
    from framework.inference.engines import REGISTRY
    from framework.inference.wrappers import BUILTIN

    runtime = checker.load_runtime()
    assert runtime.engines == set(REGISTRY)
    assert set(runtime.builtin) == set(BUILTIN)


# --- the fallback path (no framework package) --------------------------------

_FRAMEWORK_MODULES = (
    "framework",
    "framework.hal",
    "framework.hal.interface",
    "framework.hal.interface.ai",
    "framework.inference",
    "framework.inference.manifest",
    "framework.inference.engines",
    "framework.inference.wrappers",
)


def _hide_framework(monkeypatch) -> None:
    """Make the framework package unimportable, as on a bare checkout."""
    for name in _FRAMEWORK_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def test_fallback_runs_when_the_framework_cannot_be_imported(tmp_path, capsys,
                                                             monkeypatch):
    _hide_framework(monkeypatch)
    code = _run(tmp_path, _base_manifest())
    out = capsys.readouterr().out
    assert code == 0
    assert _errors(out) == []
    assert any("run-time registry unavailable" in w for w in _warnings(out))


def test_fallback_still_rejects_an_invalid_manifest(tmp_path, capsys,
                                                     monkeypatch):
    _hide_framework(monkeypatch)
    doc = _base_manifest()
    doc["schema_version"] = 2
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("schema_version" in e for e in _errors(out))


def test_fallback_still_rejects_a_missing_labels_file(tmp_path, capsys,
                                                       monkeypatch):
    _hide_framework(monkeypatch)
    doc = _base_manifest()
    doc.pop("labels")
    doc["labels_file"] = "missing_labels.txt"
    code = _run(tmp_path, doc)
    out = capsys.readouterr().out
    assert code == 1
    assert any("labels_file" in e for e in _errors(out))
