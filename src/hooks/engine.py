# -*- coding: utf-8 -*-
"""
Hook execution engine - executes SessionStart hooks and aggregates their results.

This module provides the HookEngine class that:
1. Loads SessionStart hooks from all enabled plugins
2. Executes each hook and captures output
3. Parses hook output for additionalContext
4. Aggregates results for injection into agent context
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .loader import HookLoader, LoadedHook
from .types import (
    HookCommand,
    HookCommandType,
    HookEventType,
    HookExecutionResult,
    build_hook_input,
    parse_hook_response,
)


class _HookExecutionTimeout(RuntimeError):
    pass


@dataclass
class SessionStartResult:
    """Aggregated result from all SessionStart hook executions."""

    additional_contexts: list[str] = field(default_factory=list)
    initial_user_message: str | None = None
    watch_paths: list[str] = field(default_factory=list)
    hook_count: int = 0
    errors: list[str] = field(default_factory=list)


class HookEngine:
    """Executes hooks and collects their results."""

    def __init__(
        self,
        timeout_seconds: int = 30,
        *,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        loader: HookLoader | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback
        self.startup_command_timeout_seconds = self._env_float(
            "LOGICIAN_STARTUP_HOOK_TIMEOUT_MS",
            default=1200.0,
        ) / 1000.0
        self.startup_total_budget_seconds = self._env_float(
            "LOGICIAN_STARTUP_HOOK_BUDGET_MS",
            default=1800.0,
        ) / 1000.0
        self.startup_max_parallelism = max(
            1,
            int(
                self._env_float(
                    "LOGICIAN_STARTUP_HOOK_MAX_PARALLELISM",
                    default=8.0,
                )
            ),
        )
        self.loader = loader or HookLoader()

    def execute_session_start_hooks(
        self, source: str = "startup", **payload: Any
    ) -> SessionStartResult:
        """Execute all SessionStart hooks and aggregate their results."""
        result = SessionStartResult()
        hook_input = build_hook_input(
            HookEventType.SESSION_START,
            session_id=payload.get("session_id", "") or "",
            transcript_path=payload.get("transcript_path", "") or "",
            source=source,
        )
        try:
            hooks = self.loader.get_session_start_hooks(source)
        except TypeError:
            # Tests may monkeypatch with a zero-arg lambda; fall back gracefully.
            hooks = self.loader.get_session_start_hooks()
        result.hook_count = len(hooks)
        self._notify("discovered", source=source, hook_count=result.hook_count)
        deadline = None
        source_name = str(source or "").strip().lower()
        if source_name == "startup":
            deadline = time.perf_counter() + max(
                0.0,
                self.startup_total_budget_seconds,
                self._recommended_startup_budget_seconds(hooks),
            )
            self._execute_startup_hooks_parallel(
                hooks, result, source, deadline, hook_input
            )
            self._notify(
                "completed",
                source=source,
                hook_count=result.hook_count,
                context_count=len(result.additional_contexts),
                errors=list(result.errors),
            )
            return result

        for ordinal, loaded_hook in enumerate(hooks):
            self._notify(
                "hook_started",
                source=source,
                ordinal=ordinal,
                plugin_id=loaded_hook.plugin_id,
                plugin_name=loaded_hook.plugin_name,
            )
            if deadline is not None and time.perf_counter() >= deadline:
                result.errors.append("startup hook budget exhausted")
                self._notify(
                    "error",
                    source=source,
                    ordinal=ordinal,
                    plugin_id=loaded_hook.plugin_id,
                    plugin_name=loaded_hook.plugin_name,
                    error="startup hook budget exhausted",
                )
                break
            hook_result = self._execute_hook(
                loaded_hook, source, deadline=deadline, hook_input=hook_input
            )
            if hook_result:
                result.additional_contexts.extend(hook_result.additional_contexts)
                if hook_result.initial_user_message and not result.initial_user_message:
                    result.initial_user_message = hook_result.initial_user_message
                result.watch_paths.extend(hook_result.watch_paths)
                self._notify_result(source, ordinal, loaded_hook, hook_result)
                self._notify(
                    "hook_finished",
                    source=source,
                    ordinal=ordinal,
                    plugin_id=loaded_hook.plugin_id,
                    plugin_name=loaded_hook.plugin_name,
                )

        self._notify(
            "completed",
            source=source,
            hook_count=result.hook_count,
            context_count=len(result.additional_contexts),
            errors=list(result.errors),
        )
        return result

    # ── Generic hook dispatch ───────────────────────────────────────────────

    def execute_hooks(
        self,
        event_type: HookEventType,
        *,
        matcher_value: str | None = None,
        session_id: str = "",
        transcript_path: str = "",
        cwd: str | None = None,
        prompt: str | None = None,
        tool_name: str | None = None,
        tool_input: Any = None,
        tool_response: Any = None,
        stop_hook_active: bool | None = None,
        reason: str | None = None,
        timeout_seconds: float | None = None,
    ) -> HookExecutionResult:
        """Run all hooks for a non-SessionStart event and aggregate results.

        `matcher_value` is used to filter hook definitions: e.g. the tool name
        for PreToolUse/PostToolUse, or "*" when undefined. Hook stdin receives
        the event-specific JSON payload.
        """
        aggregated = HookExecutionResult()
        hook_input = build_hook_input(
            event_type,
            session_id=session_id,
            transcript_path=transcript_path,
            cwd=cwd,
            source=matcher_value,
            prompt=prompt,
            tool_name=tool_name or matcher_value,
            tool_input=tool_input,
            tool_response=tool_response,
            stop_hook_active=stop_hook_active,
            reason=reason,
        )
        hooks = self.loader.get_hooks_for(event_type, matcher_value=matcher_value)
        if not hooks:
            return aggregated

        effective_timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(self.timeout_seconds)
        )
        deadline = time.perf_counter() + max(0.0, effective_timeout)

        for ordinal, loaded_hook in enumerate(hooks):
            if time.perf_counter() >= deadline:
                break
            try:
                hook_result = self._execute_hook(
                    loaded_hook,
                    event_type.value,
                    deadline=deadline,
                    hook_input=hook_input,
                )
            except _HookExecutionTimeout:
                aggregated.raw_output = "timeout"
                continue
            except Exception:
                continue
            if hook_result:
                aggregated.additional_contexts.extend(hook_result.additional_contexts)
                if hook_result.initial_user_message and not aggregated.initial_user_message:
                    aggregated.initial_user_message = hook_result.initial_user_message
                aggregated.watch_paths.extend(hook_result.watch_paths)
                if hook_result.raw_output:
                    aggregated.raw_output = hook_result.raw_output
                self._notify_result(
                    event_type.value, ordinal, loaded_hook, hook_result
                )

        return aggregated

    def execute_hooks_async(
        self,
        event_type: HookEventType,
        **kwargs: Any,
    ) -> None:
        """Fire-and-forget variant of execute_hooks. Returns immediately.

        Used for events whose output the agent does not consume
        (PostToolUse observations, SessionEnd finalization, etc.).
        """
        import threading

        def _run() -> None:
            try:
                self.execute_hooks(event_type, **kwargs)
            except Exception:
                return

        threading.Thread(target=_run, daemon=True).start()

    def _execute_startup_hooks_parallel(
        self,
        hooks: list[LoadedHook],
        result: SessionStartResult,
        source: str,
        deadline: float | None,
        hook_input: str,
    ) -> None:
        tasks: list[tuple[int, LoadedHook]] = list(enumerate(hooks))

        if not tasks:
            return

        max_workers = min(len(tasks), self.startup_max_parallelism)
        collected: dict[int, tuple[LoadedHook, HookExecutionResult]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._execute_hook,
                    loaded_hook,
                    source,
                    deadline,
                    hook_input,
                ): (ordinal, loaded_hook)
                for ordinal, loaded_hook in tasks
            }
            for ordinal, loaded_hook in tasks:
                self._notify(
                    "hook_started",
                    source=source,
                    ordinal=ordinal,
                    plugin_id=loaded_hook.plugin_id,
                    plugin_name=loaded_hook.plugin_name,
                )

            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.perf_counter())

            try:
                for future in concurrent.futures.as_completed(future_map, timeout=wait_timeout):
                    ordinal, loaded_hook = future_map[future]
                    try:
                        cmd_result = future.result()
                    except _HookExecutionTimeout:
                        err = "startup hook command timed out"
                        result.errors.append(err)
                        self._notify(
                            "error",
                            source=source,
                            ordinal=ordinal,
                            plugin_id=loaded_hook.plugin_id,
                            plugin_name=loaded_hook.plugin_name,
                            error=err,
                        )
                        continue
                    except Exception:
                        continue
                    if cmd_result:
                        collected[ordinal] = (loaded_hook, cmd_result)
                        self._notify_result(source, ordinal, loaded_hook, cmd_result)
                    self._notify(
                        "hook_finished",
                        source=source,
                        ordinal=ordinal,
                        plugin_id=loaded_hook.plugin_id,
                        plugin_name=loaded_hook.plugin_name,
                    )
            except concurrent.futures.TimeoutError:
                err = "startup hook budget exhausted"
                result.errors.append(err)
                self._notify("error", source=source, error=err)

        for ordinal in sorted(collected):
            _loaded_hook, cmd_result = collected[ordinal]
            result.additional_contexts.extend(cmd_result.additional_contexts)
            if cmd_result.initial_user_message and not result.initial_user_message:
                result.initial_user_message = cmd_result.initial_user_message
            result.watch_paths.extend(cmd_result.watch_paths)

    def _execute_hook(
        self,
        loaded_hook: LoadedHook,
        source: str,
        deadline: float | None = None,
        hook_input: str | None = None,
    ) -> HookExecutionResult | None:
        """Execute a single loaded hook and return its result.

        Iterates through all commands in the hook definition and collects
        additional contexts from each, continuing even if earlier commands
        return empty results.
        """
        aggregated = HookExecutionResult()
        aggregated.raw_output = loaded_hook.definition.hooks[0].command if loaded_hook.definition.hooks else ""

        for command in loaded_hook.definition.hooks:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            try:
                cmd_result = self._execute_command(
                    command,
                    loaded_hook,
                    source=source,
                    deadline=deadline,
                    hook_input=hook_input,
                )
                if cmd_result:
                    aggregated.additional_contexts.extend(cmd_result.additional_contexts)
                    if cmd_result.initial_user_message and not aggregated.initial_user_message:
                        aggregated.initial_user_message = cmd_result.initial_user_message
                    aggregated.watch_paths.extend(cmd_result.watch_paths)
                    if cmd_result.raw_output:
                        aggregated.raw_output = cmd_result.raw_output
            except _HookExecutionTimeout:
                raise
            except Exception:
                # Log error but continue with other commands
                continue

        # Return aggregated result if we collected any contexts, else None
        return (
            aggregated
            if (
                aggregated.additional_contexts
                or aggregated.initial_user_message
                or aggregated.watch_paths
            )
            else None
        )

    def _execute_command(
        self,
        command: HookCommand,
        loaded_hook: LoadedHook,
        *,
        source: str,
        deadline: float | None = None,
        hook_input: str | None = None,
    ) -> HookExecutionResult | None:
        """Execute a single hook command based on its type."""
        timeout_seconds = self._command_timeout_seconds(
            source,
            deadline,
            command_timeout=command.timeout,
        )
        if timeout_seconds is not None and timeout_seconds <= 0:
            return None
        if command.type == HookCommandType.COMMAND:
            return self._execute_bash_hook(
                command.command,
                loaded_hook,
                source=source,
                timeout_seconds=timeout_seconds,
                hook_input=hook_input,
            )
        elif command.type == HookCommandType.PROMPT:
            return self._execute_prompt_hook(command.prompt)
        elif command.type == HookCommandType.AGENT:
            return self._execute_agent_hook(command.agent)
        elif command.type == HookCommandType.HTTP:
            return self._execute_http_hook(command, timeout_seconds=timeout_seconds)
        return None

    def _execute_bash_hook(
        self,
        cmd: str | None,
        loaded_hook: LoadedHook,
        *,
        source: str,
        timeout_seconds: float | None = None,
        hook_input: str | None = None,
    ) -> HookExecutionResult | None:
        """Execute a bash command hook."""
        if not cmd:
            return None

        try:
            import os

            # Build environment with CLAUDE_PLUGIN_ROOT set to plugin root
            env = os.environ.copy()
            env["CLAUDE_PLUGIN_ROOT"] = str(loaded_hook.plugin_dir)
            if hook_input is None:
                hook_input = self._hook_input_json(source)

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                input=hook_input,
                timeout=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return parse_hook_response(output.strip())
        except subprocess.TimeoutExpired:
            raise _HookExecutionTimeout("hook command timed out")
        except Exception:
            return None

    def _execute_prompt_hook(
        self, prompt: str | None
    ) -> HookExecutionResult | None:
        """Execute a prompt hook (returns the prompt as context)."""
        if not prompt:
            return None

        result = HookExecutionResult()
        result.additional_contexts.append(prompt)
        return result

    def _execute_agent_hook(
        self, agent: str | None
    ) -> HookExecutionResult | None:
        """Execute an agent hook (placeholder - would need agent infrastructure)."""
        # Agent hooks would require spawning a sub-agent
        # For now, we'll skip these as they require more complex infra
        return None

    def _execute_http_hook(
        self,
        command: HookCommand,
        *,
        timeout_seconds: float | None = None,
    ) -> HookExecutionResult | None:
        """Execute an HTTP hook."""
        import urllib.request

        if not command.http_url:
            return None

        try:
            req = urllib.request.Request(command.http_url)
            if command.http_headers:
                for key, value in command.http_headers.items():
                    req.add_header(key, value)

            with urllib.request.urlopen(
                req,
                timeout=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            ) as resp:
                output = resp.read().decode("utf-8")
                return parse_hook_response(output.strip())
        except Exception:
            return None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _command_timeout_seconds(
        self,
        source: str,
        deadline: float | None = None,
        *,
        command_timeout: float | None = None,
    ) -> float | None:
        timeout = float(command_timeout) if command_timeout is not None else float(self.timeout_seconds)
        if str(source or "").strip().lower() == "startup" and command_timeout is None:
            timeout = min(timeout, max(0.0, float(self.startup_command_timeout_seconds)))
        if deadline is None:
            return timeout
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return 0.0
        return min(timeout, remaining)

    def _recommended_startup_budget_seconds(self, hooks: list[LoadedHook]) -> float:
        budget = 0.0
        for loaded_hook in hooks:
            chain_budget = 0.0
            for command in loaded_hook.definition.hooks:
                timeout = command.timeout
                if timeout is None:
                    timeout = float(self.startup_command_timeout_seconds)
                chain_budget += max(0.0, float(timeout))
            budget = max(budget, chain_budget)
        return budget

    def _hook_input_json(self, source: str) -> str:
        """Legacy fallback: build a SessionStart payload with the given source."""
        return build_hook_input(HookEventType.SESSION_START, source=source)

    def execute_hook_by_path(
        self, hook_path: Path, source: str = "startup"
    ) -> HookExecutionResult | None:
        """Execute a single hook from a file path (for testing)."""
        if not hook_path.exists():
            return None

        try:
            cmd = hook_path.read_text(encoding="utf-8").strip()
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            output = result.stdout if result.returncode == 0 else result.stderr
            return parse_hook_response(output.strip())
        except Exception:
            return None

    def _notify(self, kind: str, **payload: Any) -> None:
        cb = self.progress_callback
        if cb is None:
            return
        try:
            cb(kind, payload)
        except Exception:
            return

    def _notify_result(
        self,
        source: str,
        ordinal: int,
        loaded_hook: LoadedHook,
        hook_result: HookExecutionResult,
    ) -> None:
        for context in hook_result.additional_contexts:
            self._notify(
                "context",
                source=source,
                ordinal=ordinal,
                plugin_id=loaded_hook.plugin_id,
                plugin_name=loaded_hook.plugin_name,
                context=context,
            )
        if hook_result.initial_user_message:
            self._notify(
                "initial_user_message",
                source=source,
                ordinal=ordinal,
                plugin_id=loaded_hook.plugin_id,
                plugin_name=loaded_hook.plugin_name,
                message=hook_result.initial_user_message,
            )
