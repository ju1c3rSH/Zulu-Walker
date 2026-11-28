"""Check that a device package lists every file its imports need.

``deploy/maixcam/app.yaml`` names the files maixpack ships, and that list is a
precise allow-list.  A file whose dependency is missing still fails on the
device, at import time - the worst moment, because the board is already
flashed.

The walk starts from the listed files and follows both absolute and relative
imports.  It reports:

* a listed path that no longer exists (a file was renamed or deleted);
* a local module imported by a listed file that is not listed;
* a local module imported by a listed file that does not exist on disk;
* the configured platform package, when the config names one that is not listed.

``if TYPE_CHECKING:`` blocks are skipped: those imports never run, and pulling
them in would ship files the device never loads.

Usage:
    python scripts/check_package.py [manifest] [--project-root DIR]

Exit code is 1 when anything is wrong, so it can run as a test or a hook.
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterator

LOCAL_PREFIXES = ("framework", "utils", "modules", "app")


@lru_cache(maxsize=1024)
def _dir_names(path: str) -> frozenset[str]:
    try:
        return frozenset(p.name for p in Path(path).iterdir())
    except OSError:
        return frozenset()


def _exists(root: Path, relative: str) -> bool:
    """Case-sensitive existence check.

    ``Path.exists()`` follows the host filesystem, so on Windows it resolves
    ``framework/hal/Machine.py`` to ``machine.py``.  The device runs Linux,
    where that import is a ModuleNotFoundError - the check has to agree with
    the device, not with the machine doing the checking.
    """
    current = root
    for part in relative.split("/"):
        if part not in _dir_names(str(current)):
            return False
        current = current / part
    return True


class _ImportCollector(ast.NodeVisitor):
    """Collect (module, level, names) for every import that actually runs."""

    def __init__(self) -> None:
        self.found: list[tuple[str, int, tuple[str, ...]]] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking(node.test):
            # Type-only imports never execute; including them would force the
            # device to ship modules it never loads.
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = tuple(alias.name for alias in node.names)
        self.found.append((node.module or "", node.level, names))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.found.append((alias.name, 0, ()))


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _candidates(base: Path, module: str, root: Path) -> list[str]:
    """Paths a module may resolve to, expressed relative to *root*."""
    if not module:
        return []
    target = base / module.replace(".", "/")
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError:
        return []
    return [f"{relative}.py", f"{relative}/__init__.py"]


def _imports(path: Path) -> Iterator[tuple[str, int, tuple[str, ...]]]:
    source = path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    collector = _ImportCollector()
    collector.visit(tree)
    yield from collector.found


def _resolve(importer: Path, module: str, level: int, names: tuple[str, ...],
             root: Path) -> list[tuple[list[str], bool]]:
    """Candidate groups for one import statement, with whether each must exist.

    A statement can name several files: ``from pkg import a, b`` may load
    ``pkg/a.py`` and ``pkg/b.py`` as well as ``pkg`` itself.  The module part
    must exist (a stale module is a bug), while a name may just be a class or
    a function - it is only a file when one is there.
    """
    groups: list[tuple[list[str], bool]] = []

    if level:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        if module:
            groups.append((_candidates(base, module, root), True))
        else:
            # ``from . import sibling``: the names are modules of this package.
            for name in names:
                groups.append((_candidates(base, name, root), True))
        return groups

    groups.append((_candidates(root, module, root), True))
    for name in names:
        sub = f"{module}.{name}" if module else name
        groups.append((_candidates(root, sub, root), False))
    return groups


def check(manifest: str | Path, root: str | Path) -> list[str]:
    """Return a list of problems; empty means the manifest is closed."""
    import yaml

    root = Path(root)
    data = yaml.safe_load(Path(manifest).read_text(encoding="utf-8")) or {}
    listed = {f.replace("\\", "/") for f in data.get("files", [])}
    problems: list[str] = []

    for entry in sorted(listed):
        if not _exists(root, entry):
            problems.append(f"listed but missing: {entry}")

    for entry in sorted(listed):
        if not entry.endswith(".py") or not _exists(root, entry):
            continue
        for module, level, names in _imports(root / entry):
            is_local = bool(level) or module.split(".")[0] in LOCAL_PREFIXES
            if not is_local:
                continue
            for group, required in _resolve(root / entry, module, level, names, root):
                if not group:
                    continue
                if any(c in listed for c in group):
                    continue
                if any(_exists(root, c) for c in group):
                    label = module or ".".join(names)
                    problems.append(
                        f"{entry} imports {label!r}, which is not listed"
                    )
                elif required:
                    # Nothing on disk under any candidate: a stale import that
                    # would raise on the device no matter what the list says.
                    problems.append(
                        f"{entry} imports {module!r}, which does not exist"
                    )

    problems.extend(_check_platform(manifest, root, listed))
    return sorted(set(problems))


def _check_platform(manifest: Path, root: Path, listed: set[str]) -> list[str]:
    """The platform package named by the config next to the manifest must ship.

    ``machine.py`` imports it without a fallback, so a config that names a
    platform the package does not carry fails at startup.
    """
    config = Path(manifest).parent.parent.parent / "project_config.yaml"
    if not config.exists():
        return []
    import yaml

    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    platform = cfg.get("platform")
    if not platform:
        return []
    entry = f"framework/hal/platforms/{platform}/__init__.py"
    if entry in listed:
        return []
    if not _exists(root, entry):
        return [f"config names platform {platform!r}, which does not exist"]
    return [f"config names platform {platform!r}, which is not listed"]


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    manifest: Path | None = None
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--project-root" and i + 1 < len(args):
            root = Path(args[i + 1])
            i += 2
        elif args[i].startswith("--"):
            i += 1
        else:
            manifest = Path(args[i])
            i += 1
    manifest = manifest or root / "deploy/maixcam/app.yaml"

    problems = check(manifest, root)
    if problems:
        print(f"{manifest}: {len(problems)} problem(s)")
        for p in problems:
            print("  ", p)
        return 1

    import yaml

    entries = len((yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("files", []))
    print(f"{manifest}: closed ({entries} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
