#!/usr/bin/env python
"""Architecture fitness check: platform and engine isolation of the framework layer.

Rule set - docs/architecture/inference/01-scope-and-boundaries.md S1.2:

* platform libraries (maix, serial) may only be imported under
  framework/hal/platforms/ or framework/inference/engines/,
* engine libraries (ncnn, onnxruntime, rknn_lite, rknn, torch, espdl,
  llama_cpp) may only be imported under framework/hal/platforms/ or
  framework/inference/engines/,
* cv2 may only be imported under framework/hal/platforms/ or
  framework/inference/engines/, or by the single file
  framework/inference/ops.py.

A file is exempt from a rule when it sits under one of the rule's exempt
directories or when it is exactly the rule's exempt file.

The check is AST based, so it sees the import as the interpreter sees it
and not as a line of text: an import inside a function, an import after a
semicolon and a dynamic import are all imports.  The forms that are
inspected are

* "import maix.nn" - every alias of an import statement,
* "from maix import nn" - the module of a from-import,
* importlib.import_module("maix.nn") and __import__("maix.nn") - the
  string literal of the call.

The root package of the imported name decides which rule applies, so
"maiximage" is not the "maix" platform library and "rknn_lite" is its own
engine library.

Platform knowledge enters only through HAL protocols and probed capability
hooks; engine knowledge enters only through the inference kernel, so a
platform swap or an engine swap never leaks into the other layer.

Exit code 0 = clean, 1 = violations found or a file could not be read or
parsed. No third-party dependencies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIR = ROOT / "framework"

PLATFORM_ROOTS = frozenset({"maix", "serial"})
ENGINE_ROOTS = frozenset(
    {"ncnn", "onnxruntime", "rknn_lite", "rknn", "torch", "espdl", "llama_cpp"}
)
CV2_ROOTS = frozenset({"cv2"})

PLATFORM_EXEMPT_DIRS = ("hal/platforms", "inference/engines")
ENGINE_EXEMPT_DIRS = ("hal/platforms", "inference/engines")
CV2_EXEMPT_DIRS = ("hal/platforms", "inference/engines")
CV2_EXEMPT_FILES = ("inference/ops.py",)


class Rule(NamedTuple):
    """A forbidden import family together with its exemptions."""

    label: str
    roots: frozenset
    exempt_dirs: tuple[str, ...]
    exempt_files: tuple[str, ...]
    hint: str


RULES = (
    Rule(
        "platform library",
        PLATFORM_ROOTS,
        PLATFORM_EXEMPT_DIRS,
        (),
        "Platform modules belong under framework/hal/platforms/<name>/, "
        "behind HAL protocols or probed capability hooks. The engine "
        "adapters are the one other place a platform binding may appear.",
    ),
    Rule(
        "engine library",
        ENGINE_ROOTS,
        ENGINE_EXEMPT_DIRS,
        (),
        "Engine libraries belong under framework/inference/engines/, "
        "behind the inference kernel contract.",
    ),
    Rule(
        "cv2",
        CV2_ROOTS,
        CV2_EXEMPT_DIRS,
        CV2_EXEMPT_FILES,
        "cv2 carries no platform knowledge. It belongs in "
        "framework/inference/ops.py, in framework/inference/engines/ and "
        "under framework/hal/platforms/.",
    ),
)


class ScanError(Exception):
    """Base class of the errors that stop a scan."""


class UnreadableFile(ScanError):
    """Raised when a scanned file cannot be read as UTF-8 text."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"cannot read {path}: {cause}")
        self.path = path
        self.cause = cause


class UnparsableFile(ScanError):
    """Raised when a scanned file is not valid Python."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"cannot parse {path}: {cause}")
        self.path = path
        self.cause = cause


def _root_package(name: str) -> str:
    """The root package of a dotted module name."""
    return name.split(".", 1)[0]


def _string_argument(node: ast.Call) -> Optional[str]:
    """The first argument of a call when it is a plain string literal."""
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _dynamic_import_root(node: ast.Call) -> Optional[str]:
    """The root package a dynamic import call reaches, when it is a literal."""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id != "__import__":
            return None
    elif isinstance(func, ast.Attribute):
        if func.attr != "import_module":
            return None
        if not isinstance(func.value, ast.Name) or func.value.id != "importlib":
            return None
    else:
        return None
    literal = _string_argument(node)
    if literal is None:
        return None
    return _root_package(literal)


def _import_roots(tree: ast.AST) -> Iterator[tuple[int, int, str]]:
    """Yield (lineno, col_offset, root package) of every import of the tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, node.col_offset, _root_package(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                # A relative import stays inside the framework.
                continue
            yield node.lineno, node.col_offset, _root_package(node.module)
        elif isinstance(node, ast.Call):
            root = _dynamic_import_root(node)
            if root is not None:
                yield node.lineno, node.col_offset, root


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
            # utf-8-sig: a BOM is a valid Python source marker, not an error.
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise UnreadableFile(path, exc) from exc
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise UnparsableFile(path, exc) from exc
        rel = path.relative_to(framework_dir.parent).as_posix()
        lines = text.splitlines()
        hits = sorted(_import_roots(tree))
        reported: set[tuple[int, str]] = set()
        for lineno, _col, root in hits:
            for rule in RULES:
                if root not in rule.roots:
                    continue
                if _is_exempt(path, framework_dir, rule):
                    continue
                if (lineno, rule.label) in reported:
                    continue
                reported.add((lineno, rule.label))
                source = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
                violations.append(f"{rel}:{lineno}: {source}")
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
    except UnparsableFile as exc:
        print(
            f"check_platform_isolation: cannot parse {exc.path}: {exc.cause}",
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
