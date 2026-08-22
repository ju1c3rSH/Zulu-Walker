#!/usr/bin/env python
"""Architecture fitness check: platform isolation of the framework layer.

The framework (and everything under it except ``framework/hal/platforms/``)
must not import platform-specific modules (maix, cv2, serial) directly.
Platform knowledge enters only through HAL protocols and probed capability
hooks - see docs/architecture/thread_tick_topology.md and the retrospective
ARCH-* entries.

Exit code 0 = clean, 1 = violations found. No third-party dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(r"^\s*(?:import|from)\s+(maix|cv2|serial)\b")
ROOT = Path(__file__).resolve().parent.parent
SCAN_DIR = ROOT / "framework"
EXEMPT = ROOT / "framework" / "hal" / "platforms"


def main() -> int:
    if not SCAN_DIR.is_dir():
        print(f"check_platform_isolation: {SCAN_DIR} missing", file=sys.stderr)
        return 1

    violations = []
    for py in sorted(SCAN_DIR.rglob("*.py")):
        if EXEMPT in py.parents or py == EXEMPT:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"check_platform_isolation: cannot read {py}: {e}", file=sys.stderr)
            return 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.match(line):
                rel = py.relative_to(ROOT)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    if violations:
        print("platform imports leaked into the framework layer:")
        for v in violations:
            print(f"  {v}")
        print(
            "\nPlatform modules belong under framework/hal/platforms/<name>/, "
            "behind HAL protocols or probed capability hooks."
        )
        return 1

    print("check_platform_isolation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
