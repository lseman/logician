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
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import HookLoader, LoadedHook
from .types import (
    HookCommand,
    HookCommandType,
    HookExecutionResult,
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

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds
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
        self.loader = HookLoader()

    def execute_session_start_hooks(
        self, source: str = "startup"
    ) -> SessionStartResult:
        """Execute all SessionStart hooks and aggregate their results."""
        result = SessionStartResult()
        hooks = self.loader.get_session_start_hooks()
        result.hook_count = len(hooks)
        deadline = None
        source_name = str(source or "").strip().lower()
        if source_name == "startup":
            deadline = time.perf_counter() + max(0.0, self.startup_total_budget_seconds)
            self._execute_startup_hooks_parallel(hooks, result, source, deadline)
            return result

        for loaded_hook in hooks:
            if deadline is not None and time.perf_counter() >= deadline:
                result.errors.append("startup hook budget exhausted")
                break
            hook_result = self._execute_hook(loaded_hook, source, deadline=deadline)
            if hook_result:
                result.additional_contexts.extend(hook_result.additional_contexts)
                if hook_result.initial_user_message and not result.initial_user_message:
                    result.initial_user_message = hook_result.initial_user_message
                result.watch_paths.extend(hook_result.watch_paths)

        return result

    def _execute_startup_hooks_parallel(
        self,
        hooks: list[LoadedHook],
        result: SessionStartResult,
        source: str,
        deadline: float | None,
    ) -> None:
        tasks: list[tuple[int, LoadedHook]] = list(enumerate(hooks))

        if not tasks:
            return

        max_workers = min(len(tasks), self.startup_max_parallelism)
        collected: dict[int, HookExecutionResult] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._execute_hook,
                    loaded_hook,
                    source,
                    deadline,
                ): ordinal
                for ordinal, loaded_hook in tasks
            }

            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.perf_counter())

            try:
                for future in concurrent.futures.as_completed(future_map, timeout=wait_timeout):
                    ordinal = future_map[future]
                    try:
                        cmd_result = future.result()
                    except _HookExecutionTimeout:
                        result.errors.append("startup hook command timed out")
                        continue
                    except Exception:
                        continue
                    if cmd_result:
                        collected[ordinal] = cmd_result
            except concurrent.futures.TimeoutError:
                result.errors.append("startup hook budget exhausted")

        for ordinal in sorted(collected):
            cmd_result = collected[ordinal]
            result.additional_contexts.extend(cmd_result.additional_contexts)
            if cmd_result.initial_user_message and not result.initial_user_message:
                result.initial_user_message = cmd_result.initial_user_message
            result.watch_paths.extend(cmd_result.watch_paths)

    def _execute_hook(
        self, loaded_hook: LoadedHook, source: str, deadline: float | None = None
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
        return aggregated if aggregated.additional_contexts else None

    def _execute_command(
        self,
        command: HookCommand,
        loaded_hook: LoadedHook,
        *,
        source: str,
        deadline: float | None = None,
    ) -> HookExecutionResult | None:
        """Execute a single hook command based on its type."""
        timeout_seconds = self._command_timeout_seconds(source, deadline)
        if timeout_seconds is not None and timeout_seconds <= 0:
            return None
        if command.type == HookCommandType.COMMAND:
            return self._execute_bash_hook(
                command.command,
                loaded_hook,
                timeout_seconds=timeout_seconds,
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
        timeout_seconds: float | None = None,
    ) -> HookExecutionResult | None:
        """Execute a bash command hook."""
        if not cmd:
            return None

        try:
            import os

            # Build environment with CLAUDE_PLUGIN_ROOT set to plugin root
            env = os.environ.copy()
            env["CLAUDE_PLUGIN_ROOT"] = str(loaded_hook.plugin_dir)

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
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
        self, source: str, deadline: float | None = None
    ) -> float | None:
        timeout = float(self.timeout_seconds)
        if str(source or "").strip().lower() == "startup":
            timeout = min(timeout, max(0.0, float(self.startup_command_timeout_seconds)))
        if deadline is None:
            return timeout
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return 0.0
        return min(timeout, remaining)

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
