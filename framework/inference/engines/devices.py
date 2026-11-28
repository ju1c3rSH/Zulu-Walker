"""Accelerator vocabulary shared by the engine adapters.

Two questions are answered here, and both answers are pure data
transformations so they can be tested without a GPU, a driver or the
inference library itself:

* which execution providers an artifact asks for, in which shape the
  onnxruntime session wants them, and what to do when one of them is not
  active;
* which Vulkan device an index, a name or ``auto`` resolves to for ncnn.

The module owns the policy, not the mechanism.  It never imports a runtime
library, so a typo in a manifest is reported the same way on a machine with
a GPU and on one without.

Three rules are encoded here, and each of them exists because the
alternative is a silent slowdown:

1. A provider name that is not in the known list is a contract error.  The
   runtime drops an unknown name without complaining, so a misspelled
   accelerator would look exactly like a working CPU configuration.
2. A request that names an accelerator forbids a fallback unless it says
   otherwise.  Running an order of magnitude slower because the driver was
   missing is precisely the failure this project refuses to hide.
3. ``auto`` for the Vulkan device means the library default, not index 0.
   ncnn ranks its devices, and index 0 is frequently the integrated GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    ModelConfigError,
)

__all__ = [
    "DEFAULT_PROVIDERS",
    "FALLBACK_ALLOW",
    "FALLBACK_FORBID",
    "KNOWN_ORT_PROVIDERS",
    "NCNN_DEVICES",
    "ORT_PROVIDER_DEVICES",
    "VULKAN_AUTO",
    "ProviderRequest",
    "describe_vulkan_devices",
    "install_hint",
    "missing_accelerators",
    "normalise_ncnn_device",
    "parse_provider_request",
    "select_vulkan_device",
]

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)

FALLBACK_FORBID = "forbid"
FALLBACK_ALLOW = "allow"
FALLBACK_MODES = frozenset({FALLBACK_FORBID, FALLBACK_ALLOW})

# The providers that mean "not the CPU", and the device name each of them
# stands for in Capabilities.device.
ORT_PROVIDER_DEVICES: dict[str, str] = {
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "dml",
    "ROCMExecutionProvider": "rocm",
    "MIGraphXExecutionProvider": "migraphx",
    "TensorrtExecutionProvider": "tensorrt",
    "VitisAIExecutionProvider": "vitisai",
    "QNNExecutionProvider": "qnn",
}

# Every provider the adapters accept by name.  The list is a whitelist on
# purpose: an unknown name is refused instead of being dropped by the
# runtime, where it would show up as an unexplained slow run.
KNOWN_ORT_PROVIDERS = frozenset(
    set(ORT_PROVIDER_DEVICES)
    | {
        "CPUExecutionProvider",
        "AzureExecutionProvider",
        "OpenVINOExecutionProvider",
        "CoreMLExecutionProvider",
        "XnnpackExecutionProvider",
        "WebGpuExecutionProvider",
    }
)

INSTALL_HINTS: dict[str, str] = {
    "CUDAExecutionProvider": "pip install onnxruntime-gpu (needs CUDA 12.x and cuDNN 9)",
    "TensorrtExecutionProvider": "pip install onnxruntime-gpu plus TensorRT (needs CUDA)",
    "ROCMExecutionProvider": "pip install onnxruntime-rocm (Linux, ROCm 6)",
    "MIGraphXExecutionProvider": "pip install onnxruntime-rocm (Linux, ROCm 6)",
    "DmlExecutionProvider": "pip install onnxruntime-directml (Windows, any DX12 GPU)",
    "VitisAIExecutionProvider": "pip install onnxruntime-vitisai (needs the Vitis AI runtime)",
    "QNNExecutionProvider": "pip install onnxruntime-qnn (needs the Qualcomm AI runtime)",
}

NCNN_DEVICES = frozenset({"cpu", "vulkan"})
VULKAN_AUTO = "auto"


@dataclass(frozen=True)
class ProviderRequest:
    """The execution providers one session is asked for, with their policy.

    names is the priority order onnxruntime was given, options carries the
    per-provider option mappings the artifact or the deployment declared,
    and fallback says whether an accelerator that is not active is a
    startup failure (forbid) or a warning (allow).
    """

    names: tuple[str, ...]
    options: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallback: str = FALLBACK_ALLOW

    @property
    def accelerators(self) -> tuple[str, ...]:
        """The requested providers that are not the CPU, in request order."""
        return tuple(name for name in self.names if name in ORT_PROVIDER_DEVICES)

    def session_argument(self) -> list[Any]:
        """The provider list in the shape InferenceSession expects.

        A provider with options is passed as a one-key mapping, a provider
        without options as a plain name; onnxruntime accepts both in the
        same list.
        """
        argument: list[Any] = []
        for name in self.names:
            options = self.options.get(name)
            argument.append({name: dict(options)} if options else name)
        return argument


def parse_provider_request(
    raw: Any,
    *,
    options: Optional[Mapping[str, Any]] = None,
    fallback: Optional[str] = None,
    default: Sequence[str] = DEFAULT_PROVIDERS,
) -> ProviderRequest:
    """Turn the runtime block of an artifact or a deployment into a request.

    raw accepts the three shapes the manifests and the configs use: absent,
    a single name, or a list whose items are either names or one-key
    mappings of name to options.  options is a mapping applied on top of the
    inline ones, which is how a deployment overrides what an artifact
    declared without rewriting the manifest.
    """
    names: list[str] = []
    inline: dict[str, dict[str, Any]] = {}

    if raw is None:
        names = [str(name) for name in default]
    elif isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping):
                names.extend(_read_mapping_entry(item, inline))
            else:
                raise ModelConfigError(
                    f"runtime.providers: every entry must be a name or a "
                    f"one-key mapping (got {item!r})",
                    E_CONTRACT_MISSING,
                )
    else:
        raise ModelConfigError(
            f"runtime.providers: must be a name or a list (got {raw!r})",
            E_CONTRACT_MISSING,
        )

    if not names:
        raise ModelConfigError(
            "runtime.providers: must name at least one execution provider",
            E_CONTRACT_MISSING,
        )

    cleaned: list[str] = []
    for name in names:
        stripped = str(name).strip()
        if not stripped:
            raise ModelConfigError(
                "runtime.providers: a provider name must not be empty",
                E_CONTRACT_MISSING,
            )
        if stripped not in KNOWN_ORT_PROVIDERS:
            raise ModelConfigError(
                f"runtime.providers: unknown provider {stripped!r} "
                f"(known providers: {', '.join(sorted(KNOWN_ORT_PROVIDERS))})",
                E_CONTRACT_MISSING,
            )
        if stripped not in cleaned:
            cleaned.append(stripped)

    merged = dict(inline)
    for name, values in (options or {}).items():
        key = str(name).strip()
        if key not in cleaned:
            raise ModelConfigError(
                f"runtime.provider_options: options were given for "
                f"{key!r}, which is not in runtime.providers",
                E_CONTRACT_MISSING,
            )
        merged[key] = {**_as_option_mapping(key, merged.get(key)), **_as_option_mapping(key, values)}

    mode = _resolve_fallback(fallback, cleaned)

    return ProviderRequest(names=tuple(cleaned), options=merged, fallback=mode)


def missing_accelerators(
    request: ProviderRequest, active: Sequence[str]
) -> tuple[str, ...]:
    """The requested accelerators the session did not activate.

    onnxruntime drops an unavailable provider without raising, so the only
    way to notice a downgrade is to compare what was asked for with what the
    session reports.
    """
    present = {str(name) for name in active}
    return tuple(name for name in request.accelerators if name not in present)


def install_hint(provider: str) -> str:
    """The wheel an operator has to install to get that provider."""
    return INSTALL_HINTS.get(str(provider), "")


def select_vulkan_device(
    spec: Any,
    devices: Sequence[tuple[int, str]],
    default_index: int,
) -> int:
    """Resolve the runtime.vulkan_device setting to a device index.

    spec is ``auto`` (or absent) for the library's own ranking, an integer
    index, or a case-insensitive substring of the device name.  An index
    that is not there and a name that matches zero or several devices are
    failures: picking a device when the request is ambiguous is how a run
    ends up on the integrated GPU.
    """
    if not devices:
        raise ModelConfigError(
            "ncnn reports no Vulkan device on this machine",
            E_ENGINE_UNAVAILABLE,
        )

    indices = [int(index) for index, _ in devices]

    if spec is None:
        return _default_device(indices, default_index)

    if isinstance(spec, bool):
        raise ModelConfigError(
            f"runtime.vulkan_device: must be 'auto', an index or a device "
            f"name (got {spec!r})",
            E_CONTRACT_MISSING,
        )

    if isinstance(spec, int):
        return _index_device(spec, indices, devices)

    if not isinstance(spec, str):
        raise ModelConfigError(
            f"runtime.vulkan_device: must be 'auto', an index or a device "
            f"name (got {spec!r})",
            E_CONTRACT_MISSING,
        )

    text = spec.strip()
    if not text or text.lower() == VULKAN_AUTO:
        return _default_device(indices, default_index)

    if text.isdigit():
        return _index_device(int(text), indices, devices)

    wanted = text.lower()
    matches = [(index, name) for index, name in devices if wanted in str(name).lower()]
    if not matches:
        raise ModelConfigError(
            f"runtime.vulkan_device: no Vulkan device matches {text!r} "
            f"(available devices:{_render(devices)})",
            E_ENGINE_UNAVAILABLE,
        )
    if len(matches) > 1:
        raise ModelConfigError(
            f"runtime.vulkan_device: {text!r} matches several Vulkan devices "
            f"({_render(matches)}); name a single one or use an index",
            E_ENGINE_UNAVAILABLE,
        )
    return int(matches[0][0])


def describe_vulkan_devices(
    devices: Sequence[tuple[int, str]], default_index: int
) -> str:
    """Render the device table, marking the one the library prefers."""
    lines: list[str] = []
    for index, name in devices:
        marker = "  <- default" if int(index) == int(default_index) else ""
        lines.append(f"[{int(index)}] {name}{marker}")
    return "; ".join(lines)


def normalise_ncnn_device(raw: Any) -> str:
    """The device setting of the ncnn adapter, in its canonical form."""
    if raw is None:
        return "cpu"
    text = str(raw).strip().lower()
    if not text:
        return "cpu"
    if text not in NCNN_DEVICES:
        raise ModelConfigError(
            f"runtime.device: unknown ncnn device {str(raw)!r}, expected one "
            f"of {sorted(NCNN_DEVICES)}",
            E_CONTRACT_MISSING,
        )
    return text


def _read_mapping_entry(
    item: Mapping[Any, Any], inline: dict[str, dict[str, Any]]
) -> list[str]:
    """Read one onnxruntime style ``{provider: options}`` entry."""
    if len(item) != 1:
        raise ModelConfigError(
            f"runtime.providers: a mapping entry must hold exactly one "
            f"provider (got {item!r})",
            E_CONTRACT_MISSING,
        )
    name, values = next(iter(item.items()))
    key = str(name).strip()
    if values is not None:
        inline[key] = _as_option_mapping(key, values)
    return [key]


def _as_option_mapping(name: str, values: Any) -> dict[str, Any]:
    """Read one provider option block, refusing anything but a mapping."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ModelConfigError(
            f"runtime provider options for {name!r} must be a mapping "
            f"(got {values!r})",
            E_CONTRACT_MISSING,
        )
    return {str(key): value for key, value in values.items()}


def _resolve_fallback(fallback: Optional[str], names: Sequence[str]) -> str:
    """Decide whether an inactive accelerator is fatal.

    An explicit mode is honoured, except that ``forbid`` is meaningless for
    a request that names no accelerator, so it is reported as ``allow``.  An
    absent mode forbids the fallback exactly when an accelerator was asked
    for.
    """
    has_accelerator = any(name in ORT_PROVIDER_DEVICES for name in names)
    if fallback is None:
        return FALLBACK_FORBID if has_accelerator else FALLBACK_ALLOW
    mode = str(fallback).strip().lower()
    if mode not in FALLBACK_MODES:
        raise ModelConfigError(
            f"runtime.fallback: must be one of {sorted(FALLBACK_MODES)} "
            f"(got {fallback!r})",
            E_CONTRACT_MISSING,
        )
    if mode == FALLBACK_FORBID and not has_accelerator:
        return FALLBACK_ALLOW
    return mode


def _default_device(indices: Sequence[int], default_index: int) -> int:
    """The library default when it is one of the devices, else the first."""
    if int(default_index) in indices:
        return int(default_index)
    return int(indices[0])


def _index_device(
    index: int, indices: Sequence[int], devices: Sequence[tuple[int, str]]
) -> int:
    """Validate an explicit index against the device list."""
    if index < 0:
        raise ModelConfigError(
            f"runtime.vulkan_device: a device index must not be negative "
            f"(got {index})",
            E_CONTRACT_MISSING,
        )
    if index not in indices:
        raise ModelConfigError(
            f"runtime.vulkan_device: there is no Vulkan device with index "
            f"{index} (available devices:{_render(devices)})",
            E_ENGINE_UNAVAILABLE,
        )
    return index


def _render(devices: Iterable[tuple[int, str]]) -> str:
    """Render a device list for an error message."""
    return "".join(f"\n  [{int(index)}] {name}" for index, name in devices)
