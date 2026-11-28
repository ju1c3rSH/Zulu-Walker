"""Validate an inference manifest against schema_version 3.

The manifest is the contract between a logical model and its per-engine
artifacts.  A wrong field here shows up as silently wrong boxes on the
device, so the checks are fail-loud: errors exit 1, warnings exit 0.

The run-time parser (framework.inference.manifest) is the authority.  This
script runs it over the manifest first, so a manifest the checker accepts
is a manifest the device accepts at startup.  The checks below therefore
add what the parser cannot know (artifact files on disk, engine ids,
wrapper dialects, the build recipe, the runtime configuration) and must
never be laxer than the parser.

Usage:
    python scripts/check_manifest.py deploy/maixcam/models/steelball_1280x352/manifest.yaml
    python scripts/check_manifest.py <manifest> --engine maix_nn --model /root/models/x/x.mud

With --engine / --model the manifest is cross-checked against the runtime
configuration: the manifest must declare an artifact for that engine, and
that artifact's file must be the one the runtime loads.

Checks:
  - the run-time parser accepts the manifest (schema, labels, artifacts)
  - schema_version is known
  - the logical-model block (name / family / input / labels) is complete
  - the frame contract (input.frame) and the resize policy (input.resize) are sane
  - the normalisation lists match the channel count of the frame
  - every artifact declares a file, a wrapper, a preprocess triple and an io map
  - every io map is non-empty and uses the SAME logical input name
  - every artifact key is a known engine id
  - the model family is one the chosen wrapper accepts
  - the build recipe agrees with the logical model (size / colour / layout) and
    the runtime block pins a version that matches the build version
  - artifact files exist (missing binaries are a warning, not an error)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# The repository root has to be importable: under the documented invocation
# `python scripts/check_manifest.py ...` sys.path[0] is the scripts
# directory, so `framework` would not be found without this.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_SCHEMA = 3
RESIZE_OWNERS = {"framework", "baked"}
NORM_OWNERS = {"framework", "engine", "baked"}
RESIZE_MODES = {"letterbox", "crop", "stretch"}
CROP_ANCHORS = {"center", "top", "bottom", "left", "right"}
FORMULAS = {"yolo", "substract_mean_normalize"}
COLORS = {"rgb", "bgr", "rgba"}
COLOR_CHANNELS = {"rgb": 3, "bgr": 3, "rgba": 4}
DTYPES = {"uint8", "float32"}
LAYOUTS = {"hwc"}
BUILD_LAYOUTS = {"hwc", "nchw", "nhwc"}
CANONICAL_INPUT_NAMES = {"image"}

# The keys a runtime block may carry.  Anything else is a warning, because a
# misspelled accelerator key is otherwise silently ignored and the run lands
# on the CPU looking perfectly healthy.
RUNTIME_KEYS = {
    "version",
    "providers",
    "provider_options",
    "fallback",
    "device",
    "vulkan_device",
    "fp16",
    "use_fp16_storage",
    "num_threads",
    "execution_mode",
    "graph_optimization_level",
}
NCNN_DEVICES = {"cpu", "vulkan"}
FALLBACK_MODES = {"forbid", "allow"}
FP16_MODES = {"auto", "on", "off"}
KNOWN_PROVIDERS = {
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "TensorrtExecutionProvider",
    "VitisAIExecutionProvider",
    "QNNExecutionProvider",
    "OpenVINOExecutionProvider",
    "AzureExecutionProvider",
    "CoreMLExecutionProvider",
    "XnnpackExecutionProvider",
    "WebGpuExecutionProvider",
}

# Fallback copy of the engine registry, used only when the framework
# package cannot be imported at all.
KNOWN_ENGINES = {"maix_nn", "ncnn", "onnxruntime", "rknn_lite"}

# Fallback copy of framework.inference.wrappers.BUILTIN.  It carries the
# names and the families they accept, so the checker keeps working when
# the framework package is unavailable.
_FALLBACK_BUILTIN: dict[str, dict[str, Any]] = {
    "vision.det.raw": {"families": {"yolo11", "yolov5"}, "provides": {"detect"}},
    "vision.det.end2end": {"families": {"yolo11", "yolov5"}, "provides": {"detect"}},
}


class Runtime:
    """The real run-time parser and registries.

    Held together so the checker always uses one consistent view of the
    framework: the parser that validates a manifest, the wrapper dialects
    it accepts and the engine ids it knows.
    """

    def __init__(self, load_manifest: Any, check_artifact_against_config: Any,
                 manifest_error: type[BaseException],
                 builtin: dict[str, dict[str, Any]],
                 engines: set[str]) -> None:
        self.load_manifest = load_manifest
        self.check_artifact_against_config = check_artifact_against_config
        self.manifest_error = manifest_error
        self.builtin = builtin
        self.engines = engines


def load_runtime() -> Runtime:
    """Import the run-time parser and the real registries.

    Raises ImportError when the framework package is not available.  No
    other exception is caught: a syntax error or a broken initialiser
    inside the framework is a bug that must surface, not a reason to fall
    back to a stale copy of the registries.
    """
    from framework.hal.interface.ai import ManifestError
    from framework.inference.engines import REGISTRY
    from framework.inference.manifest import (
        check_artifact_against_config,
        load_manifest,
    )
    from framework.inference.wrappers import BUILTIN

    return Runtime(
        load_manifest=load_manifest,
        check_artifact_against_config=check_artifact_against_config,
        manifest_error=ManifestError,
        builtin=dict(BUILTIN),
        engines=set(REGISTRY),
    )


def builtin_wrappers(runtime: Runtime | None = None) -> dict[str, dict[str, Any]]:
    if runtime is not None:
        return runtime.builtin
    return _FALLBACK_BUILTIN


def known_engines(runtime: Runtime | None = None) -> set[str]:
    if runtime is not None:
        return runtime.engines
    return set(KNOWN_ENGINES)


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
    channels: int | None = None
    if not isinstance(frame, dict):
        rep.error("missing 'input.frame' (the frame shape the model expects)")
    else:
        if frame.get("color") not in COLORS:
            rep.error(f"'input.frame.color' must be one of {sorted(COLORS)}, "
                      f"got {frame.get('color')!r}")
        else:
            channels = COLOR_CHANNELS[frame["color"]]
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
        # The parser validates the mode first and stops there, so the
        # anchor is only judged on a valid mode.
        if anchor is not None and mode in RESIZE_MODES:
            if anchor not in CROP_ANCHORS:
                rep.error(f"'input.resize.anchor' must be one of "
                          f"{sorted(CROP_ANCHORS)}, got {anchor!r}")
            if mode != "crop":
                rep.error(f"'input.resize.anchor' is only meaningful for "
                          f"mode='crop', the mode is {mode!r}")
        value = resize.get("value")
        if value == "transparent":
            if channels is not None and channels != 4:
                rep.error(f"'input.resize.value' = 'transparent' needs a "
                          f"4-channel frame (rgba), the frame has {channels} "
                          f"channels")
        elif value is not None:
            if not (isinstance(value, int) and value >= 0):
                rep.error(f"'input.resize.value' must be a non-negative int or "
                          f"'transparent', got {value!r}")

    norm = inp.get("norm")
    if not isinstance(norm, dict) or "scale" not in norm:
        rep.error("'input.norm' must be a mapping with a 'scale'")
    else:
        scale = norm.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            rep.error(f"'input.norm.scale' must be a number, got {scale!r}")
        mean = norm.get("mean")
        std = norm.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            rep.error("'input.norm' needs both 'mean' and 'std' lists")
        else:
            if len(mean) != len(std):
                rep.error(f"'input.norm.mean' has {len(mean)} entries but 'std' "
                          f"has {len(std)}")
            if channels is not None:
                if len(mean) != channels:
                    rep.error(f"'input.norm.mean' has {len(mean)} entries but the "
                              f"frame has {channels} channels")
                if len(std) != channels:
                    rep.error(f"'input.norm.std' has {len(std)} entries but the "
                              f"frame has {channels} channels")
            for field, values in (("mean", mean), ("std", std)):
                for index, item in enumerate(values):
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        rep.error(f"'input.norm.{field}[{index}]' must be a "
                                  f"number, got {item!r}")

    labels = doc.get("labels")
    labels_file = doc.get("labels_file")
    if labels is None and labels_file is None:
        rep.error("one of 'labels' or 'labels_file' is required")
    elif labels_file is not None:
        p = base / str(labels_file)
        if not p.is_file():
            rep.error(f"labels_file does not exist: {p}")
        else:
            try:
                text = p.read_text(encoding="utf-8")
            except OSError as exc:
                rep.error(f"labels_file cannot be read: {p} ({exc})")
                text = ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not lines and text:
                rep.error(f"labels_file is empty: {p}")
            elif isinstance(labels, list) and len(labels) != len(lines):
                rep.error(f"'labels' has {len(labels)} entries but "
                          f"{labels_file} has {len(lines)}")
    if labels is not None and labels_file is None:
        # The parser reads the labels from labels_file when both are
        # given, so the inline list is only judged on its own.
        if not isinstance(labels, list):
            rep.error(f"'labels' must be a list, got {type(labels).__name__}")
        elif not labels:
            rep.error("'labels' must not be empty")
        else:
            for index, item in enumerate(labels):
                if not isinstance(item, str) or not item.strip():
                    rep.error(f"labels[{index}] must be a non-empty string, "
                              f"got {item!r}")


def check_wrapper(wrapper: Any, engine: str, family: str, rep: Report,
                  runtime: Runtime | None = None) -> None:
    if not wrapper or not isinstance(wrapper, str):
        rep.error(f"{engine}: missing 'wrapper'")
        return
    if ":" in wrapper:
        mod, _, attr = wrapper.partition(":")
        if not mod or not attr:
            rep.error(f"{engine}: bad wrapper path '{wrapper}'")
        return
    builtin = builtin_wrappers(runtime)
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
    if "layout" in build and build["layout"] not in BUILD_LAYOUTS:
        rep.error(f"{engine}: build.layout must be one of "
                  f"{sorted(BUILD_LAYOUTS)}, got {build['layout']!r}")


def check_version_agreement(art: dict[str, Any], engine: str,
                            rep: Report) -> None:
    """The pinned runtime version and the recorded build version must agree."""
    runtime_block = art.get("runtime")
    build = art.get("build")
    if not isinstance(runtime_block, dict) or not isinstance(build, dict):
        return
    runtime_version = runtime_block.get("version")
    build_version = build.get("version")
    if runtime_version is None or build_version is None:
        return
    if str(runtime_version) != str(build_version):
        rep.error(f"{engine}: runtime.version {runtime_version!r} != "
                  f"build.version {build_version!r}")


def check_runtime_block(block: dict[str, Any], engine: str, rep: Report) -> None:
    """Validate the accelerator keys of a runtime block.

    The block is read by the engine adapter, which refuses an unknown
    provider and an unknown device, but a misspelled *key* would simply be
    ignored by everyone.  That case is caught here, together with the value
    domains the adapter enforces at load time, so the manifest fails at check
    time instead of at deployment.
    """
    for key in block:
        if key not in RUNTIME_KEYS:
            rep.warn(f"{engine}: runtime.{key} is not a known key "
                     f"(known: {sorted(RUNTIME_KEYS)})")

    providers = block.get("providers")
    if providers is not None:
        if not isinstance(providers, (str, list)):
            rep.error(f"{engine}: runtime.providers must be a name or a list")
            entries: list[Any] = []
        else:
            entries = providers if isinstance(providers, list) else [providers]
        if not entries:
            rep.error(f"{engine}: runtime.providers must name at least one provider")
        for entry in entries:
            name: Any = entry
            if isinstance(entry, dict):
                if len(entry) != 1:
                    rep.error(f"{engine}: a runtime.providers mapping entry must "
                              f"hold exactly one provider")
                    continue
                name = next(iter(entry))
                if not isinstance(entry[name], dict):
                    rep.error(f"{engine}: runtime.providers[{name}] must carry a "
                              f"mapping of options")
            if not isinstance(name, str):
                rep.error(f"{engine}: runtime.providers entry {entry!r} is not a name")
            elif name.strip() not in KNOWN_PROVIDERS:
                rep.error(f"{engine}: runtime.providers names the unknown provider "
                          f"{name!r}")

    options = block.get("provider_options")
    if options is not None:
        if not isinstance(options, dict):
            rep.error(f"{engine}: runtime.provider_options must be a mapping")
        else:
            for name, values in options.items():
                if not isinstance(values, dict):
                    rep.error(f"{engine}: runtime.provider_options.{name} must be "
                              f"a mapping")
                elif providers is not None and not _names_provider(providers, str(name)):
                    rep.error(f"{engine}: runtime.provider_options.{name} is set "
                              f"but {name} is not in runtime.providers")

    fallback = block.get("fallback")
    if fallback is not None and str(fallback).strip().lower() not in FALLBACK_MODES:
        rep.error(f"{engine}: runtime.fallback must be one of "
                  f"{sorted(FALLBACK_MODES)} (got {fallback!r})")

    device = block.get("device")
    if device is not None:
        if engine != "ncnn":
            rep.warn(f"{engine}: runtime.device is only read by the ncnn adapter")
        elif str(device).strip().lower() not in NCNN_DEVICES:
            rep.error(f"{engine}: runtime.device must be one of "
                      f"{sorted(NCNN_DEVICES)} (got {device!r})")

    vulkan_device = block.get("vulkan_device")
    if vulkan_device is not None and not isinstance(vulkan_device, (int, str)):
        rep.error(f"{engine}: runtime.vulkan_device must be 'auto', an index or "
                  f"a device name (got {vulkan_device!r})")

    fp16 = block.get("fp16")
    if fp16 is not None and str(fp16).strip().lower() not in FP16_MODES:
        rep.error(f"{engine}: runtime.fp16 must be one of "
                  f"{sorted(FP16_MODES)} (got {fp16!r})")

    num_threads = block.get("num_threads")
    if num_threads is not None and (isinstance(num_threads, bool)
                                    or not isinstance(num_threads, int)
                                    or num_threads < 0):
        rep.error(f"{engine}: runtime.num_threads must be a non-negative integer "
                  f"(got {num_threads!r})")


def _names_provider(providers: Any, name: str) -> bool:
    """Whether a providers block already mentions the given provider."""
    entries = providers if isinstance(providers, list) else [providers]
    for entry in entries:
        if entry == name:
            return True
        if isinstance(entry, dict) and name in entry:
            return True
    return False


def check_artifacts(doc: dict[str, Any], base: Path, rep: Report,
                    runtime: Runtime | None = None) -> list[str]:
    arts = doc.get("artifacts")
    if not isinstance(arts, dict) or not arts:
        rep.error("'artifacts' must be a non-empty mapping")
        return []

    family = doc.get("family") or ""
    logical_inputs: set[str] = set()
    engines: list[str] = []
    known = known_engines(runtime)

    for engine, art in arts.items():
        engines.append(engine)
        if not isinstance(art, dict):
            rep.error(f"artifact '{engine}' must be a mapping")
            continue
        if engine not in known:
            rep.error(f"artifact '{engine}': unknown engine id "
                      f"(known: {sorted(known)})")

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
        elif "bin" in art and not isinstance(art.get("bin"), str):
            rep.error(f"{engine}: 'bin' must be a string")

        for block in ("wrapper_params", "runtime", "build"):
            if block in art and not isinstance(art.get(block), dict):
                rep.error(f"{engine}: '{block}' must be a mapping")

        check_wrapper(art.get("wrapper"), engine, family, rep, runtime)
        check_preprocess(art.get("preprocess"), engine, rep)

        io = art.get("io")
        if not isinstance(io, dict):
            rep.error(f"{engine}: missing 'io'")
        else:
            ins = io.get("inputs")
            outs = io.get("outputs")
            if not isinstance(ins, dict) or not isinstance(outs, dict):
                rep.error(f"{engine}: io.inputs / io.outputs must be mappings")
            else:
                if not ins:
                    rep.error(f"{engine}: io.inputs is empty (the logical input "
                              f"name must be declared)")
                if not outs:
                    rep.error(f"{engine}: io.outputs is empty")
                for key in ins:
                    if key not in CANONICAL_INPUT_NAMES:
                        rep.error(f"{engine}: logical input '{key}' is not one of "
                                  f"{sorted(CANONICAL_INPUT_NAMES)}")
                    logical_inputs.add(key)

        runtime_block = art.get("runtime")
        if isinstance(runtime_block, dict) and not runtime_block.get("version"):
            rep.error(f"{engine}: runtime.version is required (the runtime "
                      f"version must be pinned and compared at load time)")
        if isinstance(runtime_block, dict):
            check_runtime_block(runtime_block, engine, rep)

        check_build(art.get("build"), engine, doc, rep)
        check_version_agreement(art, engine, rep)

    if len(logical_inputs) > 1:
        rep.error(f"artifacts disagree on the logical input name: "
                  f"{sorted(logical_inputs)}")
    return engines


def check_against_config(doc: dict[str, Any], engine: str | None,
                         model: str | None, rep: Report,
                         runtime: Runtime | None = None,
                         runtime_manifest: Any = None) -> None:
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
    if runtime is not None and runtime_manifest is not None and engine is not None:
        # The run-time parser owns this comparison.
        try:
            runtime.check_artifact_against_config(runtime_manifest, engine, model)
        except runtime.manifest_error as exc:
            rep.error(f"config model '{model}' does not match the artifact "
                      f"declared for engine '{engine}': {exc}")
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

    rep = Report()

    # The run-time parser and the real registries are the authority.  Only
    # an ImportError (the framework package is absent) falls back to the
    # built-in copies; any other failure is a bug in the framework and is
    # allowed to surface.
    runtime: Runtime | None = None
    try:
        runtime = load_runtime()
    except ImportError as exc:
        rep.warn(f"run-time registry unavailable ({exc}); the checker falls back "
                 f"to its built-in copies of the engine and wrapper registries")

    doc: Any = None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"check_manifest: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        rep.error(f"cannot parse {path}: {exc}")
        doc = None
    if doc is not None and not isinstance(doc, dict):
        rep.error(f"the top level of {path} must be a mapping "
                  f"(got {type(doc).__name__})")
        doc = None

    # The run-time parser decides the schema verdict.
    runtime_manifest: Any = None
    if runtime is not None:
        try:
            runtime_manifest = runtime.load_manifest(path)
        except runtime.manifest_error as exc:
            rep.error(f"run-time parser rejected the manifest: {exc}")

    engines: list[str] = []
    if doc is not None:
        version = doc.get("schema_version")
        if version != SUPPORTED_SCHEMA:
            rep.error(f"unknown schema_version {version!r} "
                      f"(this checker knows {SUPPORTED_SCHEMA})")

        check_logical_model(doc, path.parent, rep)
        engines = check_artifacts(doc, path.parent, rep, runtime)
        check_against_config(doc, cfg_engine, cfg_model, rep, runtime,
                             runtime_manifest)

    printable = doc if isinstance(doc, dict) else {}
    print(f"manifest: {path}")
    print(f"  schema_version: {printable.get('schema_version')}")
    print(f"  model: {printable.get('name')} (family={printable.get('family')})")
    inp = printable.get("input") or {}
    frame = inp.get("frame") or {}
    print(f"  frame: color={frame.get('color')} dtype={frame.get('dtype')} "
          f"layout={frame.get('layout')}")
    print(f"  input: size={inp.get('size')} resize={(inp.get('resize') or {}).get('mode')}")
    for engine, art in (printable.get("artifacts") or {}).items():
        if not isinstance(art, dict):
            continue
        pre = art.get("preprocess") or {}
        io = art.get("io") or {}
        ins = ",".join((io.get("inputs") or {}).keys()) or "(none)"
        outs = ",".join((io.get("outputs") or {}).keys()) or "(none)"
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
