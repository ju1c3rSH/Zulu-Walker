"""Tests for the framework isolation checker.

The checker encodes the dependency boundary of
docs/architecture/inference/01-scope-and-boundaries.md S1.2: platform
libraries stay inside the platform layer and the engine adapters, engine
libraries stay inside the platform layer and the engine adapters, and cv2
stays in framework/inference/ops.py, in framework/inference/engines/ and
under framework/hal/platforms/.

The check is AST based, so the dynamic import forms are covered too.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_platform_isolation.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_platform_isolation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _violations(root: Path) -> list[str]:
    return checker.scan(root / "framework")


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "framework").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------- cv2: directory exemptions plus the single ops.py file ---------- #


def test_cv2_outside_ops_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/hal/interface/ai.py", "import cv2\n")
    assert _violations(root) == ["framework/hal/interface/ai.py:1: import cv2"]


def test_cv2_in_ops_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/ops.py", "import cv2\nimport numpy as np\n")
    assert _violations(root) == []


def test_cv2_in_engines_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/engines/ncnn.py", "import cv2\n")
    assert _violations(root) == []


def test_cv2_in_platforms_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/hal/platforms/linux/camera.py", "import cv2\n")
    assert _violations(root) == []


def test_cv2_in_runtime_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "import cv2\n")
    assert _violations(root) == ["framework/inference/runtime.py:1: import cv2"]


# ---------- engine libraries: engines/ directory exemption ---------- #


def test_engine_library_in_engines_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/engines/ncnn.py", "import ncnn\n")
    assert _violations(root) == []


def test_engine_library_in_runtime_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "import ncnn\n")
    assert _violations(root) == ["framework/inference/runtime.py:1: import ncnn"]


def test_rknn_lite_is_flagged_although_rknn_is_a_prefix(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "import rknn_lite\n")
    assert _violations(root) == [
        "framework/inference/runtime.py:1: import rknn_lite"
    ]


def test_engine_library_in_platforms_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/hal/platforms/linux/ai.py", "import onnxruntime\n")
    assert _violations(root) == []


# ---------- platform libraries: no exemption outside the platform layer ---------- #


def test_platform_library_in_engines_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/engines/maix_nn.py", "import maix.nn\n")
    assert _violations(root) == []


def test_platform_library_in_the_inference_kernel_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "import maix.image\n")
    assert _violations(root) == [
        "framework/inference/runtime.py:1: import maix.image"
    ]


def test_from_maix_import_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "from maix import nn\n")
    assert _violations(root) == [
        "framework/inference/runtime.py:1: from maix import nn"
    ]


def test_serial_is_flagged_outside_the_platform_layer(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/hal/interface/uart.py", "import serial\n")
    assert _violations(root) == ["framework/hal/interface/uart.py:1: import serial"]


def test_serial_is_not_flagged_inside_the_platform_layer(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/hal/platforms/linux/uart.py", "import serial\n")
    assert _violations(root) == []


# ---------- import forms ---------- #


def test_from_import_form_is_matched(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "from ncnn import Model\n")
    assert _violations(root) == [
        "framework/inference/runtime.py:1: from ncnn import Model"
    ]


def test_indented_import_is_matched(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "def f():\n    import torch\n")
    assert _violations(root) == [
        "framework/inference/runtime.py:2: import torch"
    ]


def test_longer_module_name_is_not_matched(tmp_path):
    root = _tree(tmp_path)
    _write(
        root,
        "framework/inference/manifest.py",
        "from maiximage import loader\nimport maixcamera_stub\n",
    )
    assert _violations(root) == []


# ---------- dynamic imports ---------- #


def test_importlib_import_module_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(
        root,
        "framework/inference/wrappers/loader.py",
        "import importlib\n\n\ndef load():\n    return importlib.import_module('maix.nn')\n",
    )
    assert _violations(root) == [
        "framework/inference/wrappers/loader.py:5: "
        "return importlib.import_module('maix.nn')"
    ]


def test_importlib_import_module_of_an_engine_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(
        root,
        "framework/inference/runtime.py",
        "import importlib\n\nncnn = importlib.import_module(\"ncnn\")\n",
    )
    assert _violations(root) == [
        'framework/inference/runtime.py:3: ncnn = importlib.import_module("ncnn")'
    ]


def test_dunder_import_is_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "mod = __import__(\"maix.nn\")\n")
    assert _violations(root) == [
        'framework/inference/runtime.py:1: mod = __import__("maix.nn")'
    ]


def test_importlib_import_module_in_an_exempt_directory_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(
        root,
        "framework/inference/engines/maix_nn.py",
        "import importlib\n\n\ndef load():\n    return importlib.import_module('maix.nn')\n",
    )
    assert _violations(root) == []


def test_dynamic_import_of_a_longer_name_is_not_flagged(tmp_path):
    root = _tree(tmp_path)
    _write(
        root,
        "framework/inference/runtime.py",
        "import importlib\n\nmod = importlib.import_module('maiximage.loader')\n",
    )
    assert _violations(root) == []


# ---------- unreadable and unparsable files ---------- #


def test_unparsable_file_is_reported_as_an_error(tmp_path):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "def broken(:\n")
    try:
        _violations(root)
    except checker.UnparsableFile as exc:
        assert exc.path.name == "runtime.py"
    else:
        raise AssertionError("an unparsable file must not be skipped silently")


def test_unparsable_file_makes_the_checker_fail(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path)
    _write(root, "framework/inference/runtime.py", "def broken(:\n")
    monkeypatch.setattr(checker, "SCAN_DIR", root / "framework")
    assert checker.main() == 1
    assert "cannot parse" in capsys.readouterr().err


# ---------- the real repository ---------- #


def test_repository_tree_passes_the_isolation_check():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check_platform_isolation: OK" in result.stdout


def test_repository_tree_is_clean_under_the_checker_api():
    assert checker.scan(REPO_ROOT / "framework") == []
