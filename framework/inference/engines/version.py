"""Comparison of the pinned runtime version with the installed one.

The manifest pins the runtime version an artifact was validated against.  The
pin is compared when the engine loads the artifact, because a runtime that
differs from the validated one may return different numbers for the same
graph; such a difference has to surface at load time instead of drifting into
the accuracy of the detector unnoticed.

The comparison is component wise.  Every component the pin names has to match
the installed version exactly, while a pin that names fewer components than
the installed version leaves the rest free.  The pin "1.26" therefore accepts
"1.26.0" and "1.26.3" but not "1.27.0", and the pin "1.0.20261020" accepts
only that exact ncnn build, because ncnn numbers its releases by date.

A missing pin means "no version is claimed": the check is skipped.  A pin that
cannot be read, or a library that reports no version at all, is a broken
contract and fails loudly - a pin that cannot be verified is not a pin.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_MODEL_LOAD_FAILED,
    ModelConfigError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

__all__ = ["check_runtime_version", "version_components"]

_LEADING_DIGITS = re.compile(r"^\d+")


def version_components(text: str) -> tuple[int, ...]:
    """The numeric components of a version string, in order.

    A leading "v" is dropped, a component that does not start with a digit
    ends the list, and everything behind it is ignored.  That makes "1.26",
    "1.26.0" and "1.26.x" behave as the prefixes they look like.
    """
    components: list[int] = []
    for chunk in str(text).strip().lstrip("vV").split("."):
        match = _LEADING_DIGITS.match(chunk)
        if not match:
            break
        components.append(int(match.group()))
    return tuple(components)


def check_runtime_version(pinned: Optional[str], installed: Optional[str],
                          engine_id: str, package: str = "") -> None:
    """Fail when the installed runtime does not match the pinned version.

    The package name is only used in the message, so a failure names the
    library the operator has to install.
    """
    if pinned is None or not str(pinned).strip():
        # Nothing is claimed, so nothing is compared.
        logger.debug("Engine '%s' pins no runtime version", engine_id)
        return

    wanted = version_components(str(pinned))
    if not wanted:
        raise ModelConfigError(
            f"The pinned runtime version {str(pinned)!r} of engine "
            f"'{engine_id}' cannot be read as a version number",
            E_CONTRACT_MISSING,
        )

    if installed is None or not str(installed).strip():
        raise ModelLoadError(
            f"The installed {package or engine_id} reports no version, so the "
            f"pinned version {str(pinned)!r} cannot be verified",
            E_MODEL_LOAD_FAILED,
        )

    present = version_components(str(installed))
    if present[: len(wanted)] != wanted:
        raise ModelLoadError(
            f"The manifest pins {engine_id} {str(pinned)!r} but the installed "
            f"{package or engine_id} is {str(installed)!r}; install the pinned "
            f"version or update the pin in the manifest",
            E_MODEL_LOAD_FAILED,
        )
    logger.info(
        "Engine '%s' runtime version %s matches the pin %s",
        engine_id, installed, pinned,
    )
