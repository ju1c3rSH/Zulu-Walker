"""Contract tests for the extended AI error-code vocabulary.

Phase 0 widens the HAL AI error contract with four failure codes and one
exception class while keeping the six existing codes unchanged, so callers
branch on stable identifiers instead of message text or exception types.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from framework.hal.interface.ai import (
    AIError,
    ContractError,
    E_CONTRACT_MISSING,
    E_DIALECT_UNSUPPORTED,
    E_ENGINE_UNAVAILABLE,
    E_INVALID_INPUT,
    E_INFER_FAILED,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    E_OUTPUT_LAYOUT_MISMATCH,
    E_SWITCH_TIMEOUT,
    ManifestError,
    ModelConfigError,
)

EXISTING_ERROR_CODES = [
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    E_MODEL_LOAD_FAILED,
    E_OUTPUT_LAYOUT_MISMATCH,
    E_INFER_FAILED,
    E_INVALID_INPUT,
]

NEW_ERROR_CODES = [
    E_ENGINE_UNAVAILABLE,
    E_CONTRACT_MISSING,
    E_DIALECT_UNSUPPORTED,
    E_SWITCH_TIMEOUT,
]


@pytest.mark.parametrize(
    "code,expected",
    [
        (E_ENGINE_UNAVAILABLE, "E_ENGINE_UNAVAILABLE"),
        (E_CONTRACT_MISSING, "E_CONTRACT_MISSING"),
        (E_DIALECT_UNSUPPORTED, "E_DIALECT_UNSUPPORTED"),
        (E_SWITCH_TIMEOUT, "E_SWITCH_TIMEOUT"),
    ],
)
def test_new_error_codes_have_stable_string_values(code, expected):
    assert code == expected
    assert isinstance(code, str)


@pytest.mark.parametrize(
    "code,expected",
    [
        (E_MODEL_NOT_LOADED, "E_MODEL_NOT_LOADED"),
        (E_MODEL_TYPE_UNSUPPORTED, "E_MODEL_TYPE_UNSUPPORTED"),
        (E_MODEL_LOAD_FAILED, "E_MODEL_LOAD_FAILED"),
        (E_OUTPUT_LAYOUT_MISMATCH, "E_OUTPUT_LAYOUT_MISMATCH"),
        (E_INFER_FAILED, "E_INFER_FAILED"),
        (E_INVALID_INPUT, "E_INVALID_INPUT"),
    ],
)
def test_existing_error_codes_are_unchanged(code, expected):
    assert code == expected


def test_manifest_error_is_an_ai_error_but_not_a_contract_error():
    assert issubclass(ManifestError, AIError)
    assert not issubclass(ManifestError, ContractError)


def test_manifest_error_reports_the_missing_contract_code():
    assert ManifestError().code == E_CONTRACT_MISSING


def test_contract_error_keeps_its_own_code():
    assert ContractError("output layout mismatch").code == E_OUTPUT_LAYOUT_MISMATCH


def test_manifest_error_accepts_an_instance_level_code_override():
    error = ManifestError("manifest is incomplete", code=E_SWITCH_TIMEOUT)
    assert error.code == E_SWITCH_TIMEOUT
    assert error.message == "manifest is incomplete"


def test_model_config_error_carries_the_engine_unavailable_code():
    error = ModelConfigError("no such engine", code=E_ENGINE_UNAVAILABLE)
    assert error.code == E_ENGINE_UNAVAILABLE
    assert isinstance(error, AIError)


def test_model_config_error_carries_the_dialect_unsupported_code():
    error = ModelConfigError("cannot parse dialect", code=E_DIALECT_UNSUPPORTED)
    assert error.code == E_DIALECT_UNSUPPORTED


def test_model_config_error_keeps_its_default_code():
    assert ModelConfigError("bad model type").code == E_MODEL_TYPE_UNSUPPORTED


def test_base_ai_error_carries_the_switch_timeout_code():
    error = AIError("switch timed out", code=E_SWITCH_TIMEOUT)
    assert error.code == E_SWITCH_TIMEOUT
    assert type(error) is AIError


def test_base_ai_error_keeps_its_default_code():
    assert AIError("inference failed").code == E_INFER_FAILED


@pytest.mark.parametrize(
    "code",
    [E_ENGINE_UNAVAILABLE, E_DIALECT_UNSUPPORTED, E_SWITCH_TIMEOUT],
)
def test_only_the_manifest_code_has_a_dedicated_exception_class(code):
    assert not any(cls.code == code for cls in AIError.__subclasses__())


def test_the_interface_package_reexports_the_ai_contract():
    import framework.hal.interface as interface

    assert interface.ManifestError is ManifestError
    assert interface.AIError is AIError
    assert interface.ContractError is ContractError
    for name in (
        "AIInference",
        "ClassifyKwargs",
        "DetectKwargs",
        "Detection",
        "Keypoint",
        "ManifestError",
        "MaskStats",
        "ModelConfigError",
        "ModelLoadError",
        "SegmentResult",
        "filter_detections",
    ):
        assert hasattr(interface, name), name
    for code in EXISTING_ERROR_CODES + NEW_ERROR_CODES:
        assert getattr(interface, code) == code


def test_the_interface_package_does_not_import_platform_implementations():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "framework"
        / "hal"
        / "interface"
        / "__init__.py"
    )
    source = source_path.read_text(encoding="utf-8-sig")
    assert "platform" not in source

