"""Single source of truth for the public SelfTR identity.

Change :data:`DEFAULT_METHOD_NAME` to rename the method in generated metadata,
logs, and the public Python API.  The implementation identifier ``selftr`` is
kept stable intentionally, so changing the display name never invalidates old
commands or experiment artifacts.
"""

from __future__ import annotations

import os


DEFAULT_METHOD_NAME = "SelfTR"
"""Official display name. Edit this one value for a future rename."""

METHOD_ID = "selftr"
"""Stable machine-readable identifier used by the API and CLI."""

LEGACY_METHOD_ALIASES = frozenset({"u-m", "u_m", "um"})
"""Deprecated identifiers accepted only to keep existing experiment commands working."""


def resolve_method_name(name: str | None = None) -> str:
    """Return a non-empty display name, honoring ``SELFTR_METHOD_NAME``.

    A command-line value takes precedence over the environment, which in turn
    takes precedence over :data:`DEFAULT_METHOD_NAME`.
    """

    resolved = name or os.environ.get("SELFTR_METHOD_NAME") or DEFAULT_METHOD_NAME
    resolved = str(resolved).strip()
    if not resolved:
        raise ValueError("The SelfTR display name must not be empty")
    return resolved


def canonical_frame_fusion_mode(mode: str) -> str:
    """Normalize the public SelfTR mode and its deprecated U-M aliases."""

    normalized = str(mode).strip().lower().replace("_", "-")
    return METHOD_ID if normalized in LEGACY_METHOD_ALIASES else normalized


def is_selftr_mode(mode: str) -> bool:
    """Whether *mode* resolves to the canonical SelfTR frame-fusion mode."""

    return canonical_frame_fusion_mode(mode) == METHOD_ID


__all__ = [
    "DEFAULT_METHOD_NAME",
    "METHOD_ID",
    "LEGACY_METHOD_ALIASES",
    "canonical_frame_fusion_mode",
    "is_selftr_mode",
    "resolve_method_name",
]
