from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_SYMBOL_TO_MODULE = {
    "Agent": "agent",
    "AgentResponse": "agent",
    "create_agent": "agent",
    "plot_tool_calls_by_iteration": "agent",
    "Config": "config",
    "MessageDB": "db",
    "DocumentDB": "db",
    "Message": "messages",
    "MessageRole": "messages",
    "ToolRegistry": "tools",
    "ToolParameter": "tools",
    "ToolCall": "tools",
    "AppState": "tools",
    "Context": "tools",
    "SSRReasoner": "reasoners",
    "SocraticStep": "reasoners",
    "run_eoh": "eoh",
    "EoHConfig": "eoh",
}

__all__ = list(_SYMBOL_TO_MODULE.keys())


if TYPE_CHECKING:
    from .agent import Agent, AgentResponse, create_agent, plot_tool_calls_by_iteration
    from .config import Config
    from .db import DocumentDB, MessageDB
    from .eoh import EoHConfig, run_eoh
    from .messages import Message, MessageRole
    from .reasoners import SocraticStep, SSRReasoner
    from .tools import AppState, Context, ToolCall, ToolParameter, ToolRegistry


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
