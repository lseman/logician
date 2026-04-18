# -*- coding: utf-8 -*-
"""
Hook type definitions compatible with OpenClaude's hook schema.

OpenClaude hook contract:
https://github.com/anthropics/claude-code/blob/main/src/schemas/hooks.ts
https://github.com/anthropics/claude-code/blob/main/src/types/hooks.ts
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HookEventType(Enum):
    """Hook event types matching OpenClaude's HOOK_EVENTS."""

    SESSION_START = "SessionStart"
    SETUP = "Setup"
    STOP = "Stop"
    NOTIFICATION = "Notification"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"


class HookCommandType(Enum):
    """Types of hook commands supported."""

    COMMAND = "command"  # Bash command
    PROMPT = "prompt"  # Inline prompt string
    AGENT = "agent"  # Agent sub-task
    HTTP = "http"  # HTTP request


@dataclass
class HookMatcher:
    """Matcher for hook conditions."""

    matcher: str | None = None  # Optional matcher pattern


@dataclass
class HookCommand:
    """A single hook command to execute."""

    type: HookCommandType
    command: str | None = None  # For command/http types
    prompt: str | None = None  # For prompt type
    agent: str | None = None  # For agent type
    http_url: str | None = None  # For http type
    http_headers: dict[str, str] | None = None
    timeout: float | None = None  # Optional per-hook timeout in seconds


@dataclass
class HookDefinition:
    """A defined hook with matchers and commands."""

    matcher: HookMatcher | None = None
    hooks: list[HookCommand] = field(default_factory=list)


# ── Hook Output Types ────────────────────────────────────────────────────────────


@dataclass
class SessionStartHookOutput:
    """Output from a SessionStart hook execution."""

    hook_event_name: str = "SessionStart"
    additional_context: str | None = None
    initial_user_message: str | None = None
    watch_paths: list[str] = field(default_factory=list)


@dataclass
class SyncHookResponse:
    """Full sync hook response matching OpenClaude's syncHookResponseSchema."""

    continue_: bool | None = None
    suppress_output: bool | None = None
    stop_reason: str | None = None
    decision: str | None = None  # "approve" or "block"
    reason: str | None = None
    system_message: str | None = None
    hook_specific_output: SessionStartHookOutput | None = None


@dataclass
class HookExecutionResult:
    """Aggregated result from executing hooks."""

    additional_contexts: list[str] = field(default_factory=list)
    initial_user_message: str | None = None
    watch_paths: list[str] = field(default_factory=list)
    raw_output: str = ""


# ── Parsing Helpers ────────────────────────────────────────────────────────────


def parse_hook_response(raw_output: str) -> HookExecutionResult:
    """Parse hook command output into HookExecutionResult.

    Accepts either:
    - JSON with hookSpecificOutput.additionalContext (OpenClaude format)
    - Plain string (treated as additional context directly)
    """
    result = HookExecutionResult(raw_output=raw_output)

    # Try JSON first
    try:
        data = json.loads(raw_output)
    except Exception:
        # Plain string - treat as additional context
        if raw_output.strip():
            result.additional_contexts.append(raw_output.strip())
        return result

    # Parse OpenClaude format
    hook_specific = data.get("hookSpecificOutput", {})
    if isinstance(hook_specific, dict):
        if hook_specific.get("hookEventName") == "SessionStart":
            additional = hook_specific.get("additionalContext")
            if additional and isinstance(additional, str):
                result.additional_contexts.append(additional)

            initial_msg = hook_specific.get("initialUserMessage")
            if initial_msg and isinstance(initial_msg, str):
                result.initial_user_message = initial_msg

            watch_paths = hook_specific.get("watchPaths", [])
            if isinstance(watch_paths, list):
                result.watch_paths = [p for p in watch_paths if isinstance(p, str)]

    return result


def load_plugin_hooks(plugin_dir: Path) -> dict[HookEventType, list[HookDefinition]]:
    """Load hooks from a plugin's directory.

    Checks for:
    1. hooks/hooks.json (standard external hooks file)
    2. .claude-plugin/plugin.json hooks section
    """
    hooks: dict[HookEventType, list[HookDefinition]] = {}

    # Check .claude-plugin/plugin.json first
    plugin_json = plugin_dir / ".claude-plugin" / "plugin.json"
    if plugin_json.exists():
        try:
            data = json.loads(plugin_json.read_text(encoding="utf-8"))
            hooks_data = data.get("hooks", {})
            if hooks_data:
                hooks = _parse_hooks_dict(hooks_data)
        except Exception:
            pass

    # Check hooks/hooks.json
    hooks_json = plugin_dir / "hooks" / "hooks.json"
    if hooks_json.exists():
        try:
            data = json.loads(hooks_json.read_text(encoding="utf-8"))
            if data:
                # hooks.json has "hooks" as top-level key containing the actual hooks
                # Also handle case where hooks are at top level (direct format)
                hooks_data = data.get("hooks", data)
                existing = hooks
                hooks = _parse_hooks_dict(hooks_data)
                # Merge with existing
                for event_type, defs in existing.items():
                    if event_type not in hooks:
                        hooks[event_type] = defs
                    else:
                        hooks[event_type].extend(defs)
        except Exception:
            pass

    return hooks


def _parse_hooks_dict(data: dict[str, Any]) -> dict[HookEventType, list[HookDefinition]]:
    """Parse hooks dict into structured HookDefinition objects."""
    result: dict[HookEventType, list[HookDefinition]] = {}

    for event_name, matchers in data.items():
        try:
            event_type = HookEventType(event_name)
        except ValueError:
            continue

        if not isinstance(matchers, list):
            continue

        definitions: list[HookDefinition] = []
        for matcher_data in matchers:
            if not isinstance(matcher_data, dict):
                continue

            matcher_str = matcher_data.get("matcher")
            matcher = HookMatcher(matcher=matcher_str) if matcher_str else None

            hooks_list: list[HookCommand] = []
            for hook_data in matcher_data.get("hooks", []):
                if not isinstance(hook_data, dict):
                    continue

                hook_type_str = hook_data.get("type", "command")
                try:
                    hook_type = HookCommandType(hook_type_str)
                except ValueError:
                    hook_type = HookCommandType.COMMAND

                hook_cmd = HookCommand(
                    type=hook_type,
                    command=hook_data.get("command"),
                    prompt=hook_data.get("prompt"),
                    agent=hook_data.get("agent"),
                    http_url=hook_data.get("url"),
                    http_headers=hook_data.get("headers"),
                    timeout=(
                        float(hook_data.get("timeout"))
                        if isinstance(hook_data.get("timeout"), (int, float))
                        else None
                    ),
                )
                hooks_list.append(hook_cmd)

            if hooks_list:
                definitions.append(HookDefinition(matcher=matcher, hooks=hooks_list))

        if definitions:
            result[event_type] = definitions

    return result
