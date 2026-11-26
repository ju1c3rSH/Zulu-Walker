#!/usr/bin/env python
"""Architecture fitness check: platform and engine isolation of the framework layer.

Rule set - docs/architecture/inference/01-scope-and-boundaries.md S1.2:

* platform libraries (maix, serial) may only be imported under
  framework/hal/platforms/,
* engine libraries (ncnn, onnxruntime, rknn_lite, rknn, torch, espdl,
  llama_cpp) may only be imported under framework/inference/engines/,
* cv2 may only be imported by framework/inference/ops.py.

Platform knowledge enters only through HAL protocols and probed capability
hooks; engine knowledge enters only through the inference kernel, so a
platform swap or an engine swap never leaks into the other layer.

Exit code 0 = clean, 1 = violations found. No third-party dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIR = ROOT / "framework"

PLATFORM_LIBS = re.compile(r"^\s*(?:import|from)\s+(maix|serial)\b")
ENGINE_LIBS = re.compile(
    r"^\s*(?:import|from)\s+(ncnn|onnxruntime|rknn_lite|rknn|torch|espdl|llama_cpp)\b"
)
CV2_LIBS = re.compile(r"^\s*(?:import|from)\s+(cv2)\b")

PLATFORM_EXEMPT_DIRS = ("hal/platforms",)
ENGINE_EXEMPT_DIRS = ("hal/platforms", "inference/engines")
CV2_EXEMPT_DIRS = ("hal/platforms",)
CV2_EXEMPT_FILES = ("inference/ops.py",)


class Rule(NamedTuple):
    """A forbidden import family together with its exemptions."""

    label: str
    pattern: re.Pattern
    exempt_dirs: tuple[str, ...]
    exempt_files: tuple[str, ...]
    hint: str


RULES = (
    Rule(
        "platform library",
        PLATFORM_LIBS,
        PLATFORM_EXEMPT_DIRS,
        (),
        "Platform modules belong under framework/hal/platforms/<name>/, "
        "behind HAL protocols or probed capability hooks.",
    ),
    Rule(
        "engine library",
        ENGINE_LIBS,
        ENGINE_EXEMPT_DIRS,
        (),
        "Engine libraries belong under framework/inference/engines/, "
        "behind the inference kernel contract.",
    ),
    Rule(
        "cv2",
        CV2_LIBS,
        CV2_EXEMPT_DIRS,
        CV2_EXEMPT_FILES,
        "cv2 carries no platform knowledge and belongs in "
        "framework/inference/ops.py only.",
    ),
)


class UnreadableFile(Exception):
    """Raised when a scanned file cannot be read as UTF-8 text."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"cannot read {path}: {cause}")
        self.path = path
        self.cause = cause


def _is_exempt(path: Path, framework_dir: Path, rule: Rule) -> bool:
    """True when the file sits under an exempt directory or is an exempt file."""
    for rel_dir in rule.exempt_dirs:
        exempt_dir = (framework_dir / rel_dir).resolve()
        if path == exempt_dir or exempt_dir in path.parents:
            return True
    for rel_file in rule.exempt_files:
        if path == (framework_dir / rel_file).resolve():
            return True
    return False


def scan(framework_dir: Path) -> list[str]:
    """Return one violation line per forbidden import under framework_dir."""
    framework_dir = framework_dir.resolve()
    violations: list[str] = []
    for py in sorted(framework_dir.rglob("*.py")):
        path = py.resolve()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise UnreadableFile(path, exc) from exc
        rel = path.relative_to(framework_dir.parent).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule in RULES:
                if _is_exempt(path, framework_dir, rule):
                    continue
                if rule.pattern.match(line):
                    violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def main() -> int:
    if not SCAN_DIR.is_dir():
        print(f"check_platform_isolation: {SCAN_DIR} missing", file=sys.stderr)
        return 1

    try:
        violations = scan(SCAN_DIR)
    except UnreadableFile as exc:
        print(
            f"check_platform_isolation: cannot read {exc.path}: {exc.cause}",
            file=sys.stderr,
        )
        return 1

    if violations:
        print("isolation rule violations found under framework/:")
        for v in violations:
            print(f"  {v}")
        print("")
        for rule in RULES:
            print(f"- {rule.label}: {rule.hint}")
        return 1

    print("check_platform_isolation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
