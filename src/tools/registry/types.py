from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict

from ..runtime import ToolParameter


class _OpenAIToolFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _OpenAIToolSchema(TypedDict):
    type: str
    function: _OpenAIToolFunction


class _BuiltinToolDefinition(TypedDict):
    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable[..., Any]
    doc: str


ExecutionGlobals: TypeAlias = dict[str, Any]
ToolExecutionStats: TypeAlias = dict[str, dict[str, Any]]
_SKILL_PREFIX_SEPARATORS = (":", "/")
_COMMON_PARAM_ALIASES: dict[str, set[str]] = {
    "pattern": {
        "type",
        "kind",
        "sample_type",
        "series_type",
        "mode",
        "style",
    },
    "n_points": {
        "points",
        "num_points",
        "n",
        "count",
        "length",
        "size",
        "num_samples",
    },
    "noise_level": {"noise", "noise_std", "noise_sigma", "sigma", "std"},
    "filepath": {"path", "file", "filename"},
}
