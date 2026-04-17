# -*- coding: utf-8 -*-
"""
Hook system for logician - compatible with OpenClaude's hook architecture.

This module provides:
- types.py: Hook type definitions matching OpenClaude's schema
- loader.py: Plugin hook loading from registry
- engine.py: Hook execution engine for SessionStart and other hooks
"""

from __future__ import annotations

from .engine import HookEngine, SessionStartResult
from .loader import HookLoader, LoadedHook
from .types import (
    HookCommand,
    HookCommandType,
    HookDefinition,
    HookEventType,
    HookExecutionResult,
    HookMatcher,
    SessionStartHookOutput,
    SyncHookResponse,
    load_plugin_hooks,
    parse_hook_response,
)

__all__ = [
    "HookEngine",
    "HookLoader",
    "HookExecutionResult",
    "LoadedHook",
    "SessionStartResult",
    "SessionStartHookOutput",
    "SyncHookResponse",
    "HookCommand",
    "HookCommandType",
    "HookDefinition",
    "HookEventType",
    "HookMatcher",
    "load_plugin_hooks",
    "parse_hook_response",
]

# For backwards compatibility
HookResult = HookExecutionResult
