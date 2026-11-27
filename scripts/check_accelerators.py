#!/usr/bin/env python
"""Report what acceleration this host really offers, and whether it is enough.

The runtime never guesses: a configuration names an accelerator explicitly and
the engines refuse to start when it cannot be honoured.  This script answers
the question that comes before that: what does this machine have?

It reports, without loading any model:

* every onnxruntime build reachable from this interpreter, its version, its
  device and every execution provider it can provide;
* every Vulkan device ncnn can see, the one the library ranks first, and the
  heap budget the adapter would bind to;
* the accelerators the configuration asks for, and whether they are there.

The last part is a check, so the exit code is meaningful in a build script:
0 when the configuration is satisfiable, 1 when it is not, 2 when the check
itself could not run.

Usage:
    python scripts/check_accelerators.py
    python scripts/check_accelerators.py --config configs/project_config.desktop.yaml
    python scripts/check_accelerators.py --config configs/project_config.desktop.yaml --strict

--strict also fails when an engine named by the configuration is not
installed at all, which is what a deployment gate wants; the default only
fails on an accelerator that was asked for and is missing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.inference.engines.devices import (  # noqa: E402
    ORT_PROVIDER_DEVICES,
    describe_vulkan_devices,
    install_hint,
    missing_accelerators,
    parse_provider_request,
    select_vulkan_device,
)

EXIT_OK = 0
EXIT_UNSATISFIED = 1
EXIT_CANNOT_CHECK = 2

DEFAULT_CONFIG = "configs/project_config.desktop.yaml"


# --------------------------------------------------------------------- #
# onnxruntime
# --------------------------------------------------------------------- #


def probe_onnxruntime() -> dict[str, Any]:
    """Describe the onnxruntime build this interpreter reaches."""
    report: dict[str, Any] = {"installed": False, "providers": [], "problems": []}
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:
        report["problems"].append(f"onnxruntime is not importable: {exc}")
        return report
    report["installed"] = True
    report["version"] = str(getattr(ort, "__version__", "?"))
    try:
        report["device"] = str(ort.get_device())
    except Exception as exc:
        report["device"] = f"unknown ({exc})"
    try:
        report["providers"] = [str(name) for name in ort.get_available_providers()]
    except Exception as exc:
        report["problems"].append(f"the provider list could not be read: {exc}")
    report["kind"] = _build_kind(report["providers"])
    return report


def _build_kind(providers: list[str]) -> str:
    """Name the wheel from the providers it carries."""
    if "DmlExecutionProvider" in providers:
        return "directml"
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        return "gpu"
    if "ROCMExecutionProvider" in providers or "MIGraphXExecutionProvider" in providers:
        return "rocm"
    return "cpu"


# --------------------------------------------------------------------- #
# ncnn
# --------------------------------------------------------------------- #


def probe_ncnn() -> dict[str, Any]:
    """Describe the Vulkan side of ncnn, if this wheel has one.

    get_gpu_count() brings the Vulkan instance up as a side effect and leaves
    it alive.  A process that queries it and then exits without
    destroy_gpu_instance() dies with a segmentation fault in the driver
    teardown, so the probe pairs the two calls itself.
    """
    report: dict[str, Any] = {
        "installed": False,
        "vulkan": False,
        "devices": (),
        "default_index": None,
        "problems": [],
    }
    try:
        ncnn = importlib.import_module("ncnn")
    except Exception as exc:
        report["problems"].append(f"ncnn is not importable: {exc}")
        return report
    report["installed"] = True
    report["version"] = str(getattr(ncnn, "__version__", "?"))
    get_count = getattr(ncnn, "get_gpu_count", None)
    destroy = getattr(ncnn, "destroy_gpu_instance", None)
    if get_count is None:
        report["problems"].append("this ncnn build has no Vulkan support (no get_gpu_count)")
        return report
    create = getattr(ncnn, "create_gpu_instance", None)
    try:
        if create is not None:
            create()
        _fill_ncnn_devices(ncnn, report, get_count)
    finally:
        if destroy is not None:
            try:
                destroy()
            except Exception:
                pass
    return report


def _fill_ncnn_devices(ncnn: Any, report: dict[str, Any], get_count: Any) -> None:
    """Read the device table into the report."""
    try:
        count = int(get_count())
    except Exception as exc:
        report["problems"].append(f"the Vulkan device count could not be read: {exc}")
        return
    if count <= 0:
        report["problems"].append("ncnn reports no Vulkan device")
        return
    get_info = getattr(ncnn, "get_gpu_info", None)
    devices = []
    for index in range(count):
        name = f"device {index}"
        budget = ""
        if get_info is not None:
            try:
                info = get_info(index)
                name = str(info.device_name())
                budget = f", heap {int(info.get_heap_budget()) // (1 << 20)} MiB"
            except Exception:
                pass
        devices.append((index, f"{name}{budget}"))
    report["vulkan"] = True
    report["devices"] = tuple(devices)
    get_default = getattr(ncnn, "get_default_gpu_index", None)
    if get_default is not None:
        try:
            report["default_index"] = int(get_default())
        except Exception:
            report["default_index"] = 0


# --------------------------------------------------------------------- #
# Configuration check
# --------------------------------------------------------------------- #


def load_config(path: Path) -> dict[str, Any]:
    """Read the ai block of a project configuration."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ai = document.get("ai") or {}
    return ai if isinstance(ai, dict) else {}


def check_onnxruntime(ai: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """List the accelerators the configuration asks for and this build lacks."""
    problems: list[str] = []
    for label, block in runtime_blocks(ai):
        request = parse_provider_request(
            block.get("providers"),
            options=block.get("provider_options"),
            fallback=block.get("fallback"),
        )
        missing = missing_accelerators(request, report.get("providers", []))
        for name in missing:
            hint = install_hint(name)
            problems.append(
                f"{label}: asks for {name}, which this build does not provide"
                + (f" ({hint})" if hint else "")
            )
    return problems


def check_ncnn(ai: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """List the Vulkan devices the configuration asks for and cannot get."""
    problems: list[str] = []
    devices = report.get("devices", ())
    for label, block in runtime_blocks(ai):
        device = str(block.get("device", "cpu")).strip().lower()
        if device != "vulkan":
            continue
        if not report.get("vulkan"):
            problems.append(f"{label}: asks for vulkan, this ncnn build has none")
            continue
        spec = block.get("vulkan_device")
        if spec is None:
            continue
        try:
            select_vulkan_device(spec, devices, report.get("default_index") or 0)
        except Exception as exc:
            problems.append(f"{label}: vulkan_device {spec!r} cannot be bound ({exc})")
    return problems


def runtime_blocks(ai: dict[str, Any]):
    """Yield the runtime blocks of a configuration, with a label each."""
    yield "ai.runtime", ai.get("runtime") or {}
    for model in ai.get("models") or []:
        if isinstance(model, dict) and isinstance(model.get("runtime"), dict):
            yield f"models[{model.get('nick_name', '?')}].runtime", model["runtime"]


# --------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------- #


def print_report(ort: dict[str, Any], ncnn: dict[str, Any]) -> None:
    print("== onnxruntime ==")
    if not ort["installed"]:
        print("  not installed")
    else:
        accelerators = [
            f"{name} -> {ORT_PROVIDER_DEVICES[name]}"
            for name in ort["providers"]
            if name in ORT_PROVIDER_DEVICES
        ]
        print(f"  version   {ort['version']}")
        print(f"  build     {ort.get('kind', '?')} (device {ort.get('device', '?')})")
        print(f"  provides  {', '.join(ort['providers']) or 'nothing'}")
        print(f"  gpu eps   {', '.join(accelerators) or 'none'}")
    for problem in ort["problems"]:
        print(f"  ! {problem}")

    print("== ncnn ==")
    if not ncnn["installed"]:
        print("  not installed")
    else:
        print(f"  version   {ncnn['version']}")
        print(f"  vulkan    {'yes' if ncnn['vulkan'] else 'no'}")
        if ncnn["vulkan"]:
            print(f"  devices   {describe_vulkan_devices(ncnn['devices'], ncnn['default_index'] or 0)}")
    for problem in ncnn["problems"]:
        print(f"  ! {problem}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="project configuration to check")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when the configured engine is not installed at all",
    )
    parser.add_argument("--json", action="store_true", help="print the raw probe as JSON")
    args = parser.parse_args(argv)

    ort = probe_onnxruntime()
    ncnn = probe_ncnn()
    if args.json:
        print(json.dumps({"onnxruntime": ort, "ncnn": ncnn}, default=str, indent=2))
        return EXIT_OK

    print_report(ort, ncnn)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"\n! configuration {config_path} not found; the probe above is all there is")
        return EXIT_CANNOT_CHECK

    ai = load_config(config_path)
    engine = str(ai.get("engine") or "").strip()
    print(f"\n== configuration: {config_path} ==")
    print(f"  engine    {engine or '(platform default)'}")

    problems: list[str] = []
    if engine == "onnxruntime":
        problems += check_onnxruntime(ai, ort)
        if args.strict and not ort["installed"]:
            problems.append("the configuration names onnxruntime, which is not installed")
    elif engine == "ncnn":
        problems += check_ncnn(ai, ncnn)
        if args.strict and not ncnn["installed"]:
            problems.append("the configuration names ncnn, which is not installed")
    elif engine and args.strict:
        print("  (no accelerator of this engine is described by this script)")

    if problems:
        print("\n== unsatisfied ==")
        for problem in problems:
            print(f"  ! {problem}")
        return EXIT_UNSATISFIED

    print("  ok        every accelerator the configuration asks for is present")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
