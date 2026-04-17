# -*- coding: utf-8 -*-
"""
Tool versioning utilities for comparing Tool versions and detecting breaking changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime import Tool, ToolVersion


def version_breaks(v1: "ToolVersion", v2: "ToolVersion") -> bool:
    """True if v2 has a breaking change relative to v1 (major version bump)."""
    return v2.major != v1.major


def version_adds_required_params(
    v1: "Tool", v2: "Tool"
) -> list[str]:
    """Return sorted names of new required params in v2 not present in v1."""
    v1_names = {p.name for p in v1.parameters}
    v2_names = {p.name for p in v2.parameters}
    v2_required = {p.name for p in v2.parameters if p.required}
    new_required = sorted(v2_names - v1_names & v2_required)
    return new_required
