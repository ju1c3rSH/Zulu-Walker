"""The device package must be a closed dependency set.

Run: python -m pytest tests/test_package_manifest.py -q

A missing file in ``deploy/maixcam/app.yaml`` only fails on the board, after
flashing.  These tests move that discovery to the laptop: one asserts the real
manifest is closed, the other asserts the checker actually notices a gap (a
checker that always passes is worse than none).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_package import check  # noqa: E402

MANIFEST = "deploy/maixcam/app.yaml"


def test_the_device_manifest_is_closed():
    problems = check(ROOT / MANIFEST, ROOT)
    assert problems == [], "device package dependencies are not closed:\n" + "\n".join(problems)


def test_a_listed_path_that_does_not_exist_is_reported(tmp_path):
    manifest = tmp_path / "app.yaml"
    manifest.write_text(
        "files:\n"
        "  - app/main.py\n"
        "  - modules/this_file_was_deleted.py\n",
        encoding="utf-8",
    )

    problems = check(manifest, ROOT)

    assert any("listed but missing" in p for p in problems)


def test_a_dependency_outside_the_list_is_reported(tmp_path):
    """app/main.py imports framework.*; a list with only main.py must fail."""
    manifest = tmp_path / "app.yaml"
    manifest.write_text("files:\n  - app/main.py\n", encoding="utf-8")

    problems = check(manifest, ROOT)

    assert any("not listed" in p for p in problems)


def test_relative_imports_are_resolved(tmp_path):
    """``from .battery import X`` in a platform package must pull battery in."""
    manifest = tmp_path / "app.yaml"
    manifest.write_text(
        "files:\n"
        "  - framework/hal/platforms/maixcam2/__init__.py\n",
        encoding="utf-8",
    )

    problems = check(manifest, ROOT)

    assert any("battery" in p for p in problems)


def test_a_case_typo_is_caught_despite_a_case_insensitive_host(tmp_path):
    """The device is Linux; ``Machine.py`` is not ``machine.py`` there.

    ``Path.exists()`` would resolve it on Windows and hide the mistake, so the
    checker compares path components instead.
    """
    manifest = tmp_path / "app.yaml"
    manifest.write_text(
        "files:\n"
        "  - framework/hal/__init__.py\n"
        "  - framework/hal/Machine.py\n",
        encoding="utf-8",
    )

    problems = check(manifest, ROOT)

    assert any("Machine.py" in p and "missing" in p for p in problems)
