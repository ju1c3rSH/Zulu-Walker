"""Contract tests for the built-in wrapper registry.

The registry is the single place where a manifest dialect key turns into a
callable plus the metadata the framework needs at load time.  It must stay
free of runtime, engine, manifest and wrapper imports so that resolving a
dialect stays lazy.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from framework.hal.interface.ai import E_DIALECT_UNSUPPORTED, ModelConfigError
from framework.inference.wrappers import (
    BUILTIN,
    get_wrapper_metadata,
    resolve_wrapper_factory,
    wrapper_families,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_DET_MODULE = "framework.inference.wrappers.vision_det"

EXPECTED_METADATA = {
    "vision.det.raw": {
        "target": "framework.inference.wrappers.vision_det:build_raw",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "raw",
    },
    "vision.det.end2end": {
        "target": "framework.inference.wrappers.vision_det:build_end2end",
        "families": {"yolo11", "yolov5"},
        "provides": {"detect"},
        "output_form": "end2end",
    },
}


def test_builtin_registry_holds_exactly_the_vision_det_dialects():
    assert sorted(BUILTIN) == ["vision.det.end2end", "vision.det.raw"]


@pytest.mark.parametrize("key", sorted(EXPECTED_METADATA))
def test_each_builtin_entry_carries_the_four_metadata_fields(key):
    entry = BUILTIN[key]
    assert set(entry) == {"target", "families", "provides", "output_form"}
    assert entry["target"] == EXPECTED_METADATA[key]["target"]
    assert entry["families"] == EXPECTED_METADATA[key]["families"]
    assert entry["provides"] == EXPECTED_METADATA[key]["provides"]
    assert entry["output_form"] == EXPECTED_METADATA[key]["output_form"]


@pytest.mark.parametrize("key", sorted(EXPECTED_METADATA))
def test_metadata_lookup_returns_the_registry_entry(key):
    metadata = get_wrapper_metadata(key)
    assert metadata["target"] == EXPECTED_METADATA[key]["target"]
    assert metadata["families"] == EXPECTED_METADATA[key]["families"]
    assert metadata["provides"] == EXPECTED_METADATA[key]["provides"]
    assert metadata["output_form"] == EXPECTED_METADATA[key]["output_form"]


def test_module_attr_reference_resolves_to_the_named_callable():
    assert resolve_wrapper_factory("json:dumps") is json.dumps


def test_unknown_dialect_name_raises_dialect_unsupported():
    with pytest.raises(ModelConfigError) as excinfo:
        resolve_wrapper_factory("vision.det.seg")
    assert excinfo.value.code == E_DIALECT_UNSUPPORTED


def test_unknown_dialect_metadata_lookup_raises_dialect_unsupported():
    with pytest.raises(ModelConfigError) as excinfo:
        get_wrapper_metadata("vision.det.seg")
    assert excinfo.value.code == E_DIALECT_UNSUPPORTED


def test_unimportable_module_attr_reference_raises_dialect_unsupported():
    with pytest.raises(ModelConfigError) as excinfo:
        resolve_wrapper_factory("framework.inference.no_such_module:build")
    assert excinfo.value.code == E_DIALECT_UNSUPPORTED


@pytest.mark.parametrize("key", sorted(EXPECTED_METADATA))
def test_families_helper_returns_the_accepted_families(key):
    assert wrapper_families(key) == EXPECTED_METADATA[key]["families"]


@pytest.mark.parametrize("key", sorted(EXPECTED_METADATA))
def test_builtin_targets_are_well_formed_module_attr_references(key):
    target = BUILTIN[key]["target"]
    module_name, separator, attr = target.partition(":")
    assert separator == ":"
    assert module_name
    assert attr

    if importlib.util.find_spec(VISION_DET_MODULE) is None:
        pytest.skip(f"{VISION_DET_MODULE} is not implemented yet")
    assert callable(resolve_wrapper_factory(target))


def test_importing_the_registry_does_not_pull_in_runtime_or_engines():
    probe = (
        "import sys\n"
        "import framework.inference.wrappers\n"
        "forbidden = [\n"
        "    'framework.inference.runtime',\n"
        "    'framework.inference.engines',\n"
        "    'framework.inference.manifest',\n"
        "    'framework.inference.wrappers.vision_det',\n"
        "]\n"
        "imported = [name for name in forbidden if name in sys.modules]\n"
        "assert not imported, imported\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
