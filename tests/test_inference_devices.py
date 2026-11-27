"""Unit tests for the accelerator vocabulary shared by the engine adapters.

Run: python -m pytest tests/test_inference_devices.py -q

Everything here is a pure function: no GPU, no driver and no inference
library is touched, so the policy is testable on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    ModelConfigError,
)
from framework.inference.engines.devices import (
    FALLBACK_ALLOW,
    FALLBACK_FORBID,
    ProviderRequest,
    describe_vulkan_devices,
    install_hint,
    missing_accelerators,
    normalise_ncnn_device,
    parse_provider_request,
    select_vulkan_device,
)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
TRT = "TensorrtExecutionProvider"

DEVICES = ((0, "Intel(R) RaptorLake-S Mobile Graphics Controller"), (1, "NVIDIA GeForce RTX 5060 Laptop GPU"))


# --------------------------------------------------------------------- #
# Provider requests
# --------------------------------------------------------------------- #


def test_an_absent_block_means_the_cpu_and_no_fallback_promise():
    request = parse_provider_request(None)

    assert request.names == (CPU,)
    assert request.options == {}
    assert request.fallback == FALLBACK_ALLOW
    assert request.accelerators == ()


def test_a_single_name_is_accepted_as_a_shorthand():
    request = parse_provider_request(CUDA)

    assert request.names == (CUDA,)
    assert request.accelerators == (CUDA,)
    assert request.fallback == FALLBACK_FORBID


def test_the_order_of_the_list_is_the_ort_priority_order():
    request = parse_provider_request([TRT, CUDA, CPU])

    assert request.names == (TRT, CUDA, CPU)
    assert request.accelerators == (TRT, CUDA)


def test_an_ort_style_dict_entry_carries_its_provider_options():
    request = parse_provider_request([{CUDA: {"device_id": 1}}, CPU])

    assert request.names == (CUDA, CPU)
    assert request.options == {CUDA: {"device_id": 1}}


def test_a_dict_entry_without_options_is_accepted():
    request = parse_provider_request([{CUDA: None}, CPU])

    assert request.names == (CUDA, CPU)
    assert request.options == {}


def test_the_explicit_options_mapping_wins_over_the_inline_one():
    request = parse_provider_request(
        [{CUDA: {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}}],
        options={CUDA: {"device_id": 2}},
    )

    assert request.options[CUDA]["device_id"] == 2
    assert request.options[CUDA]["cudnn_conv_algo_search"] == "DEFAULT"


def test_the_session_argument_keeps_plain_names_plain_and_options_as_dicts():
    request = parse_provider_request([{CUDA: {"device_id": 1}}, CPU])

    assert request.session_argument() == [{CUDA: {"device_id": 1}}, CPU]


def test_an_unknown_provider_is_a_contract_error_naming_the_known_ones():
    with pytest.raises(ModelConfigError) as caught:
        parse_provider_request(["CUDAExecutionProvder"])

    assert caught.value.code == E_CONTRACT_MISSING
    assert "CUDAExecutionProvider" in str(caught.value)


def test_an_empty_list_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        parse_provider_request([])

    assert caught.value.code == E_CONTRACT_MISSING


def test_an_empty_name_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        parse_provider_request(["  "])

    assert caught.value.code == E_CONTRACT_MISSING


def test_an_unknown_fallback_mode_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        parse_provider_request([CUDA], fallback="warn")

    assert caught.value.code == E_CONTRACT_MISSING


def test_an_explicit_allow_keeps_an_accelerator_request_soft():
    request = parse_provider_request([CUDA], fallback=FALLBACK_ALLOW)

    assert request.fallback == FALLBACK_ALLOW


def test_a_request_without_any_accelerator_never_forbids_a_fallback():
    request = parse_provider_request([CPU], fallback=FALLBACK_FORBID)

    assert request.fallback == FALLBACK_ALLOW


def test_an_unknown_request_type_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        parse_provider_request(3)

    assert caught.value.code == E_CONTRACT_MISSING


def test_missing_accelerators_keeps_the_requested_order():
    request = parse_provider_request([TRT, CUDA, CPU])

    assert missing_accelerators(request, [CPU]) == (TRT, CUDA)
    assert missing_accelerators(request, [CUDA, CPU]) == (TRT,)
    assert missing_accelerators(request, [TRT, CUDA, CPU]) == ()


def test_a_cpu_only_request_has_no_missing_accelerator():
    request = parse_provider_request([CPU])

    assert missing_accelerators(request, [CPU]) == ()


def test_the_install_hint_names_the_wheel_to_install():
    assert "onnxruntime-gpu" in install_hint(CUDA)
    assert "onnxruntime-directml" in install_hint(DML)
    assert install_hint(CPU) == ""


def test_the_request_is_immutable():
    request = ProviderRequest(names=(CPU,), options={}, fallback=FALLBACK_ALLOW)

    with pytest.raises(Exception):
        request.names = (CUDA,)


# --------------------------------------------------------------------- #
# ncnn devices
# --------------------------------------------------------------------- #


def test_a_missing_device_spec_is_an_alias_for_auto():
    assert select_vulkan_device(None, DEVICES, 1) == 1
    assert select_vulkan_device("auto", DEVICES, 1) == 1


def test_auto_follows_the_library_default_instead_of_index_zero():
    """The library ranks devices; overriding it with 0 picks the iGPU."""

    assert select_vulkan_device("auto", DEVICES, 1) == 1


def test_an_integer_index_is_taken_as_is():
    assert select_vulkan_device(0, DEVICES, 1) == 0
    assert select_vulkan_device(1, DEVICES, 1) == 1


def test_a_numeric_string_is_read_as_an_index():
    assert select_vulkan_device("1", DEVICES, 0) == 1


def test_a_name_selects_the_device_that_contains_it():
    assert select_vulkan_device("nvidia", DEVICES, 0) == 1
    assert select_vulkan_device("RTX 5060", DEVICES, 0) == 1
    assert select_vulkan_device("intel", DEVICES, 1) == 0


def test_a_name_that_matches_several_devices_fails_and_lists_them():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device("intel", DEVICES + ((2, "Intel(R) UHD Graphics"),), 0)

    assert caught.value.code == E_ENGINE_UNAVAILABLE
    assert "UHD" in str(caught.value)


def test_a_name_that_matches_nothing_fails_and_lists_the_devices():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device("radeon", DEVICES, 0)

    assert caught.value.code == E_ENGINE_UNAVAILABLE
    assert "RTX 5060" in str(caught.value)


def test_an_index_outside_the_device_list_fails():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device(7, DEVICES, 0)

    assert caught.value.code == E_ENGINE_UNAVAILABLE


def test_a_negative_index_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device(-1, DEVICES, 0)

    assert caught.value.code == E_CONTRACT_MISSING


def test_a_machine_without_devices_fails_rather_than_guessing():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device("auto", (), 0)

    assert caught.value.code == E_ENGINE_UNAVAILABLE


def test_a_device_spec_of_an_unknown_type_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        select_vulkan_device(1.5, DEVICES, 0)

    assert caught.value.code == E_CONTRACT_MISSING


def test_the_device_list_is_rendered_with_the_default_marked():
    text = describe_vulkan_devices(DEVICES, 1)

    assert "[1]" in text
    assert "RTX 5060" in text
    assert "default" in text


def test_the_ncnn_device_name_is_whitespace_and_case_insensitive():
    assert normalise_ncnn_device(" Vulkan ") == "vulkan"
    assert normalise_ncnn_device(None) == "cpu"


def test_an_unknown_ncnn_device_is_a_contract_error():
    with pytest.raises(ModelConfigError) as caught:
        normalise_ncnn_device("metal")

    assert caught.value.code == E_CONTRACT_MISSING
