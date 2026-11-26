"""Validate an inference manifest against schema_version 3.

The manifest is the contract between a logical model and its per-engine
artifacts.  A wrong field here shows up as silently wrong boxes on the
device, so the checks are fail-loud: errors exit 1, warnings exit 0.

Usage:
    python scripts/check_manifest.py deploy/maixcam/models/steelball_1280x352/manifest.yaml
    python scripts/check_manifest.py <manifest> --engine maix_nn --model /root/models/x/x.mud

With --engine / --model the manifest is cross-checked against the runtime
configuration: the manifest must declare an artifact for that engine, and
that artifact's file must be the one the runtime loads.

Checks:
  - schema_version is known
  - the logical-model block (name / family / input / labels) is complete
  - the frame contract (input.frame) and the resize policy (input.resize) are sane
  - every artifact declares a file, a wrapper, a preprocess triple and an io map
  - every artifact uses the SAME logical input name
  - the model family is one the chosen wrapper accepts
  - the build recipe agrees with the logical model (size / colour) and the
    runtime block pins a version
  - artifact files exist (missing binaries are a warning, not an error)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_SCHEMA = 3
RESIZE_OWNERS = {"framework", "baked"}
NORM_OWNERS = {"framework", "engine", "baked"}
RESIZE_MODES = {"letterbox", "crop", "stretch"}
CROP_ANCHORS = {"center", "top", "bottom", "left", "right"}
FORMULAS = {"yolo", "substract_mean_normalize"}
COLORS = {"rgb", "bgr"}
DTYPES = {"uint8", "float32"}
LAYOUTS = {"hwc"}
CANONICAL_INPUT_NAMES = {"image"}

# Fallback copy of framework.inference.wrappers.BUILTIN.  That module does not
# exist yet (Phase 0), so the checker carries its own copy of the names and the
# families they accept.  Once the module lands, the real registry wins and the
# two cannot drift.
_FALLBACK_BUILTIN: dict[str, dict[str, Any]] = {
    "vision.det.raw": {"families": {"yolo11", "yolov5"}, "provides": {"detect"}},
    "vision.det.end2end": {"families": {"yolo11"}, "provides": {"detect"}},
}


def builtin_wrappers() -> dict[str, dict[str, Any]]:
    try:
        from framework.inference.wrappers import BUILTIN  # type: ignore
    except Exception:
        return _FALLBACK_BUILTIN
    return dict(BUILTIN)


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_logical_model(doc: dict[str, Any], base: Path, rep: Report) -> None:
    name = doc.get("name")
    if not name:
        rep.error("missing top-level 'name'")
    elif not isinstance(name, str):
        rep.error(f"'name' must be a string, got {type(name).__name__}")

    family = doc.get("family")
    if not family:
        rep.error("missing top-level 'family' (model family, e.g. 'yolo11')")
    elif not isinstance(family, str):
        rep.error(f"'family' must be a string, got {type(family).__name__}")

    inp = doc.get("input")
    if not isinstance(inp, dict):
        rep.error("missing 'input' block")
        return

    frame = inp.get("frame")
    if not isinstance(frame, dict):
        rep.error("missing 'input.frame' (the frame shape the model expects)")
    else:
        if frame.get("color") not in COLORS:
            rep.error(f"'input.frame.color' must be one of {sorted(COLORS)}, "
                      f"got {frame.get('color')!r}")
        if frame.get("dtype") not in DTYPES:
            rep.error(f"'input.frame.dtype' must be one of {sorted(DTYPES)}, "
                      f"got {frame.get('dtype')!r}")
        if frame.get("layout") not in LAYOUTS:
            rep.error(f"'input.frame.layout' must be one of {sorted(LAYOUTS)}, "
                      f"got {frame.get('layout')!r}")

    size = inp.get("size")
    if not (isinstance(size, list) and len(size) == 2
            and all(isinstance(v, int) and v > 0 for v in size)):
        rep.error(f"'input.size' must be two positive ints, got {size!r}")

    resize = inp.get("resize")
    if not isinstance(resize, dict) or "mode" not in resize:
        rep.error("'input.resize' must be a mapping with a 'mode'")
    else:
        mode = resize.get("mode")
        if mode not in RESIZE_MODES:
            rep.error(f"'input.resize.mode' must be one of {sorted(RESIZE_MODES)}, "
                      f"got {mode!r}")
        anchor = resize.get("anchor")
        if anchor is not None:
            if anchor not in CROP_ANCHORS:
                rep.error(f"'input.resize.anchor' must be one of "
                          f"{sorted(CROP_ANCHORS)}, got {anchor!r}")
            if mode != "crop":
                rep.warn(f"'input.resize.anchor' only applies to mode='crop', "
                         f"mode is {mode!r}")
        value = resize.get("value")
        if value is not None and value != "transparent":
            if not (isinstance(value, int) and value >= 0):
                rep.error(f"'input.resize.value' must be a non-negative int or "
                          f"'transparent', got {value!r}")

    norm = inp.get("norm")
    if not isinstance(norm, dict) or "scale" not in norm:
        rep.error("'input.norm' must be a mapping with a 'scale'")
    else:
        mean = norm.get("mean")
        std = norm.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            rep.error("'input.norm' needs both 'mean' and 'std' lists")
        elif len(mean) != len(std):
            rep.error(f"'input.norm.mean' has {len(mean)} entries but 'std' "
                      f"has {len(std)}")

    labels = doc.get("labels")
    labels_file = doc.get("labels_file")
    if labels is None and labels_file is None:
        rep.error("one of 'labels' or 'labels_file' is required")
    elif labels_file is not None:
        p = base / str(labels_file)
        if not p.is_file():
            rep.warn(f"labels_file not found: {p}")
        else:
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            if not lines:
                rep.error(f"labels_file is empty: {p}")
            elif isinstance(labels, list) and len(labels) != len(lines):
                rep.error(f"'labels' has {len(labels)} entries but "
                          f"{labels_file} has {len(lines)}")


def check_wrapper(wrapper: Any, engine: str, family: str, rep: Report) -> None:
    if not wrapper or not isinstance(wrapper, str):
        rep.error(f"{engine}: missing 'wrapper'")
        return
    if ":" in wrapper:
        mod, _, attr = wrapper.partition(":")
        if not mod or not attr:
            rep.error(f"{engine}: bad wrapper path '{wrapper}'")
        return
    builtin = builtin_wrappers()
    entry = builtin.get(wrapper)
    if entry is None:
        rep.error(f"{engine}: unknown wrapper '{wrapper}' "
                  f"(builtin: {sorted(builtin)})")
        return
    families = entry.get("families")
    if families and family and family not in families:
        rep.error(f"{engine}: model family '{family}' is not accepted by "
                  f"wrapper '{wrapper}' (accepts: {sorted(families)})")


def check_preprocess(pre: Any, engine: str, rep: Report) -> None:
    if not isinstance(pre, dict):
        rep.error(f"{engine}: missing 'preprocess'")
        return
    if pre.get("resize") not in RESIZE_OWNERS:
        rep.error(f"{engine}: preprocess.resize must be one of "
                  f"{sorted(RESIZE_OWNERS)}, got {pre.get('resize')!r}")
    norm = pre.get("norm")
    if norm not in NORM_OWNERS:
        rep.error(f"{engine}: preprocess.norm must be one of "
                  f"{sorted(NORM_OWNERS)}, got {norm!r}")
        return
    formula = pre.get("formula")
    if norm == "baked":
        if formula is not None:
            rep.error(f"{engine}: preprocess.formula must be absent when "
                      f"norm='baked' (the normalisation is baked in)")
    elif formula is None:
        rep.error(f"{engine}: preprocess.formula is required when norm='{norm}'")
    elif formula not in FORMULAS:
        rep.error(f"{engine}: preprocess.formula must be one of "
                  f"{sorted(FORMULAS)}, got {formula!r}")


def check_build(build: Any, engine: str, doc: dict[str, Any], rep: Report) -> None:
    if not isinstance(build, dict):
        return
    inp = doc.get("input") or {}
    size = inp.get("size")
    if "input_size" in build and build["input_size"] != size:
        rep.error(f"{engine}: build.input_size {build['input_size']} != "
                  f"input.size {size}")
    frame = inp.get("frame") or {}
    if "color" in build and "color" in frame and build["color"] != frame["color"]:
        rep.error(f"{engine}: build.color {build['color']!r} != "
                  f"input.frame.color {frame['color']!r}")


def check_artifacts(doc: dict[str, Any], base: Path, rep: Report) -> list[str]:
    arts = doc.get("artifacts")
    if not isinstance(arts, dict) or not arts:
        rep.error("'artifacts' must be a non-empty mapping")
        return []

    family = doc.get("family") or ""
    logical_inputs: set[str] = set()
    engines: list[str] = []

    for engine, art in arts.items():
        engines.append(engine)
        if not isinstance(art, dict):
            rep.error(f"artifact '{engine}' must be a mapping")
            continue

        file = art.get("file")
        if not file:
            rep.error(f"{engine}: missing 'file'")
        else:
            p = base / str(file)
            if not p.is_file():
                rep.warn(f"{engine}: artifact file not present yet: {p.name}")
        if art.get("bin"):
            p = base / str(art["bin"])
            if not p.is_file():
                rep.warn(f"{engine}: weight file not present yet: {p.name}")

        check_wrapper(art.get("wrapper"), engine, family, rep)
        check_preprocess(art.get("preprocess"), engine, rep)

        io = art.get("io")
        if not isinstance(io, dict):
            rep.error(f"{engine}: missing 'io'")
            continue
        ins = io.get("inputs")
        outs = io.get("outputs")
        if not isinstance(ins, dict) or not isinstance(outs, dict):
            rep.error(f"{engine}: io.inputs / io.outputs must be mappings")
            continue
        if not ins:
            logical_inputs |= CANONICAL_INPUT_NAMES  # implicit io (maix.nn)
            continue
        if not outs:
            rep.error(f"{engine}: io.outputs is empty but io.inputs is not")
        for key in ins:
            if key not in CANONICAL_INPUT_NAMES:
                rep.error(f"{engine}: logical input '{key}' is not one of "
                          f"{sorted(CANONICAL_INPUT_NAMES)}")
            logical_inputs.add(key)

        runtime = art.get("runtime")
        if isinstance(runtime, dict) and not runtime.get("version"):
            rep.error(f"{engine}: runtime.version is required (the runtime "
                      f"version must be pinned and compared at load time)")

        check_build(art.get("build"), engine, doc, rep)

    if len(logical_inputs) > 1:
        rep.error(f"artifacts disagree on the logical input name: "
                  f"{sorted(logical_inputs)}")
    return engines


def check_against_config(doc: dict[str, Any], engine: str | None,
                          model: str | None, rep: Report) -> None:
    """Cross-check the manifest against the runtime configuration.

    The runtime names exactly one engine and one exact artifact file
    (no directory, no fallback).  The manifest must describe that same
    engine and that same file, otherwise the runtime would load a weight
    the manifest does not describe.
    """
    arts = doc.get("artifacts") or {}
    if engine is not None:
        if engine not in arts:
            rep.error(f"config engine '{engine}' has no artifact in this "
                      f"manifest (declared: {sorted(arts)})")
            return
    if model is None:
        return
    want = Path(model).name
    candidates = [engine] if engine in arts else list(arts)
    for name in candidates:
        declared = str((arts.get(name) or {}).get("file") or "")
        if declared and Path(declared).name == want:
            return
    rep.error(f"config model '{model}' does not match any declared artifact "
              f"file ({sorted(arts)})")


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    # Named cfg_* so the summary loop below (which iterates artifact
    # engines) cannot shadow them.
    cfg_engine: str | None = None
    cfg_model: str | None = None
    for flag in ("--engine", "--model"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                print(f"check_manifest: {flag} needs a value", file=sys.stderr)
                return 2
            if flag == "--engine":
                cfg_engine = args[i + 1]
            else:
                cfg_model = args[i + 1]
            del args[i:i + 2]
    if not args:
        print(__doc__)
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"check_manifest: {path} not found", file=sys.stderr)
        return 2

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    rep = Report()

    version = doc.get("schema_version")
    if version != SUPPORTED_SCHEMA:
        if isinstance(version, int) and version < SUPPORTED_SCHEMA:
            rep.warn(f"schema_version {version} is older than {SUPPORTED_SCHEMA}; "
                     f"the parser will branch on it")
        else:
            rep.error(f"unknown schema_version {version!r} "
                      f"(this checker knows {SUPPORTED_SCHEMA})")

    check_logical_model(doc, base, rep)
    engines = check_artifacts(doc, base, rep)
    check_against_config(doc, cfg_engine, cfg_model, rep)

    print(f"manifest: {path}")
    print(f"  schema_version: {version}")
    print(f"  model: {doc.get('name')} (family={doc.get('family')})")
    inp = doc.get("input") or {}
    frame = inp.get("frame") or {}
    print(f"  frame: color={frame.get('color')} dtype={frame.get('dtype')} "
          f"layout={frame.get('layout')}")
    print(f"  input: size={inp.get('size')} resize={(inp.get('resize') or {}).get('mode')}")
    for engine, art in (doc.get("artifacts") or {}).items():
        if not isinstance(art, dict):
            continue
        pre = art.get("preprocess") or {}
        io = art.get("io") or {}
        ins = ",".join((io.get("inputs") or {}).keys()) or "(implicit)"
        outs = ",".join((io.get("outputs") or {}).keys()) or "(implicit)"
        print(f"  - {engine}: file={art.get('file')} wrapper={art.get('wrapper')}")
        print(f"      preprocess resize={pre.get('resize')} norm={pre.get('norm')} "
              f"formula={pre.get('formula')}")
        print(f"      io in={ins} out={outs}")

    for w in rep.warnings:
        print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")
    if cfg_engine is not None:
        print(f"  config engine: {cfg_engine}"
              + (f" model: {cfg_model}" if cfg_model else ""))
    print(f"check_manifest: {len(rep.errors)} error(s), "
          f"{len(rep.warnings)} warning(s), {len(engines)} artifact(s)")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
