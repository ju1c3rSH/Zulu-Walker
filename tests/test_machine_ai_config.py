"""Unit tests for the AI configuration of the machine.

Run: python -m pytest tests/test_machine_ai_config.py -q

The configuration has two shapes: one that names an engine, which selects
the kernel driven backend, and one that does not, which keeps the backend
of the platform.  The tests cover both, and the rule that an explicitly
configured engine that cannot be brought up is a startup failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _write_config(directory: Path, ai: dict[str, Any]) -> Path:
    """Write a configuration with one AI section and no camera."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "platform": "linux",
        "display_enabled": False,
        "cameras": [],
        "uart_defaults": {"port": "auto", "baudrate": 921600},
        "ai": ai,
    }
    path = directory / "project_config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_a_configuration_without_an_engine_keeps_the_platform_backend(tmp_path):
    """The device configuration has no engine and must keep working."""
    from framework.hal import Machine

    config_path = _write_config(tmp_path, {"models": [], "active": ""})
    machine = Machine.create(str(config_path))
    assert machine.ai is None or machine.ai.models == []


def test_a_configured_engine_is_handed_to_the_platform(tmp_path):
    from framework.hal import Machine

    config_path = _write_config(
        tmp_path,
        {
            "engine": "onnxruntime",
            "switch_timeout_ms": 1000,
            "frame": {"color": "bgr", "dtype": "uint8", "layout": "hwc"},
            "models": [],
            "active": "",
        },
    )
    try:
        machine = Machine.create(str(config_path))
    except Exception as exc:
        pytest.skip(f"the onnxruntime runtime is not usable here: {exc}")
    assert machine.ai is not None
    assert machine.ai.model_family == ""


def test_an_engine_that_cannot_be_brought_up_is_a_startup_failure(tmp_path):
    """A silent downgrade would run the system slower without a visible cause."""
    from framework.hal import Machine

    config_path = _write_config(
        tmp_path,
        {
            "engine": "no_such_engine",
            "models": [],
            "active": "",
        },
    )
    with pytest.raises(Exception) as exc:
        Machine.create(str(config_path))
    assert "E_ENGINE_UNAVAILABLE" in str(exc.value) or exc.value.code == "E_ENGINE_UNAVAILABLE"
