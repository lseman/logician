from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_SYMBOL_TO_MODULE = {
    "Agent": "core",
    "AgentResponse": "trace",
    "create_agent": "factory",
    "plot_tool_calls_by_iteration": "trace",
    "TurnResult": "types",
}

__all__ = list(_SYMBOL_TO_MODULE.keys())


if TYPE_CHECKING:
    from .core import Agent
    from .factory import create_agent
    from .trace import AgentResponse, plot_tool_calls_by_iteration
    from .types import TurnResult


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
