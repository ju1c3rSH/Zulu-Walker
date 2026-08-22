#!/usr/bin/env python
"""Scope-order check for app/main.py: flag reads of locals before assignment.

Catches the ``UnboundLocalError`` class that py_compile cannot see - e.g.
referencing ``machine`` inside ``main()`` before ``Machine.create`` assigns
it. Introduced after exactly such a bug shipped in an ARCH-07 refactor.

Deliberately conservative: only names that ARE assigned later in the same
function are flagged (true use-before-def), so module-level names and
function parameters pass through.

Exit code 0 = clean, 1 = violations. No third-party dependencies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "app" / "main.py"


def scan(fn: ast.FunctionDef):
    stores = {
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    problems = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in stores
        ):
            # Find whether any Store of this name textually precedes the Load;
            # if not, the first touch of the local is this very read.
            first_store = min(
                (n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Name) and n.id == node.id
                 and isinstance(n.ctx, ast.Store)),
                default=None,
            )
            if first_store is not None and node.lineno < first_store:
                problems.append((node.id, node.lineno, first_store))
    return problems


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.exists():
        print(f"check_boot_scope: target not found: {target}", file=sys.stderr)
        return 1
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for name, used_at, stored_at in scan(node):
                violations.append(
                    f"main(): '{name}' read at line {used_at} but first "
                    f"assigned at line {stored_at}"
                )
    if violations:
        print("use-before-def detected:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("check_boot_scope: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
