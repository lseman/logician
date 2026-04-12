from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from .bootstrap import _find_local_venv  # noqa: F401 (re-exported for skills that import it here)


class _LegacyCodingRuntime:
    def __init__(self, config: Mapping[str, Any], run_cmd: Callable[..., Any]) -> None:
        self._config = dict(config)
        self._run_cmd = run_cmd

    def config(self) -> dict[str, Any]:
        return self._config

    def cwd(self) -> str | None:
        value = self._config.get("default_cwd")
        return str(value) if value else None

    def venv_path(self) -> str | None:
        value = self._config.get("venv_path")
        if value:
            return str(value)
        return _find_local_venv(self.cwd())

    def set_cwd(self, path: str | None) -> str | None:
        resolved = str(Path(path).expanduser().resolve()) if path else None
        self._config["default_cwd"] = resolved
        return resolved

    def set_venv_path(self, path: str | None) -> str | None:
        resolved = str(Path(path).expanduser().resolve()) if path else None
        self._config["venv_path"] = resolved
        return resolved

    def resolve_cwd(self, cwd: str | None) -> str | None:
        if cwd:
            return str(Path(cwd).expanduser().resolve())
        return self.cwd()

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and self.cwd():
            candidate = Path(self.cwd() or ".") / candidate
        return candidate.resolve()

    def build_shell_prefix(self, venv_path: str | None) -> str:
        venv = venv_path or self.venv_path()
        if not venv:
            return ""
        activate = Path(venv).expanduser() / "bin" / "activate"
        if not activate.exists():
            return ""
        return f". {activate} && "

    def run_cmd(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_cmd(*args, **kwargs)


def get_coding_runtime(globalns: Mapping[str, Any] | None = None) -> Any:
    namespace = dict(globalns or {})
    rt = (
        namespace.get("coding_runtime")
        or namespace.get("_coding_runtime")
        or getattr(builtins, "coding_runtime", None)
        or getattr(builtins, "_coding_runtime", None)
    )
    if rt is None:
        legacy_config = getattr(builtins, "_coding_config", None)
        legacy_run_cmd = getattr(builtins, "_run_cmd", None)
        if isinstance(legacy_config, Mapping) and callable(legacy_run_cmd):
            return _LegacyCodingRuntime(legacy_config, legacy_run_cmd)
    if rt is None:
        raise RuntimeError(
            "coding_runtime is not available — skill must be loaded via the registry"
        )
    return rt


def _format_skill_doc_context(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""

    def _normalize_items(key: str, *, limit: int = 4) -> list[str]:
        raw = meta.get(key, [])
        if isinstance(raw, (str, bytes)):
            raw_items = [raw]
        else:
            raw_items = list(raw or [])
        items = [str(item).strip() for item in raw_items if str(item).strip()]
        return items[:limit]

    lines: list[str] = []
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    aliases = _normalize_items("aliases", limit=8)
    triggers = _normalize_items("triggers", limit=8)
    preferred_tools = _normalize_items("preferred_tools", limit=8)
    example_queries = _normalize_items("example_queries", limit=4)
    when_not_to_use = _normalize_items("when_not_to_use", limit=4)
    next_skills = _normalize_items("next_skills", limit=6)
    workflow = _normalize_items("workflow", limit=5)
    entry_criteria = _normalize_items("entry_criteria", limit=4)
    decision_rules = _normalize_items("decision_rules", limit=4)
    failure_recovery = _normalize_items("failure_recovery", limit=4)
    exit_criteria = _normalize_items("exit_criteria", limit=4)
    anti_patterns = _normalize_items("anti_patterns", limit=4)
    preferred_sequence = _normalize_items("preferred_sequence", limit=6)

    if name:
        lines.append(f"Skill: {name}")
    if description:
        lines.append(f"Skill purpose: {description}")
    if aliases:
        lines.append(f"Skill aliases: {', '.join(aliases)}")
    if triggers:
        lines.append(f"Skill triggers: {', '.join(triggers)}")
    if preferred_tools:
        lines.append(f"Preferred tools in this skill: {', '.join(preferred_tools)}")
    if example_queries:
        lines.append(f"Example queries: {'; '.join(example_queries)}")
    if when_not_to_use:
        lines.append("Skill avoid when:")
        lines.extend(f"- {item}" for item in when_not_to_use)
    if next_skills:
        lines.append(f"Typical next skills: {', '.join(next_skills)}")
    if preferred_sequence:
        lines.append(f"Typical sequence: {' -> '.join(preferred_sequence)}")
    if entry_criteria:
        lines.append("Enter this skill when:")
        lines.extend(f"- {item}" for item in entry_criteria)
    if decision_rules:
        lines.append("Skill decision rules:")
        lines.extend(f"- {item}" for item in decision_rules)
    if workflow:
        lines.append("Skill workflow:")
        lines.extend(f"- {step}" for step in workflow)
    if failure_recovery:
        lines.append("Skill failure recovery:")
        lines.extend(f"- {item}" for item in failure_recovery)
    if exit_criteria:
        lines.append("Skill exit criteria:")
        lines.extend(f"- {item}" for item in exit_criteria)
    if anti_patterns:
        lines.append("Skill anti-patterns:")
        lines.extend(f"- {item}" for item in anti_patterns)

    if not lines:
        return ""
    return "\n".join(["Skill context:", *lines])


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Legacy compatibility decorator.

    This decorator no longer exports tools automatically. Use __tools__ in skill
    modules to explicitly declare exported tool callables.
    """

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        meta = dict(getattr(inner, "__llm_tool_meta__", {}) or {})
        meta_name = str(name or meta.get("name") or getattr(inner, "__name__", "")).strip()
        meta["name"] = meta_name or getattr(inner, "__name__", "")
        if description is not None:
            meta["description"] = description
        setattr(inner, "__llm_tool_meta__", meta)
        skill_context = _format_skill_doc_context(inner.__globals__.get("__skill__"))
        marker = "Skill context:"
        base_doc = str(getattr(inner, "__doc__", "") or "").rstrip()
        if skill_context and marker not in base_doc:
            inner.__doc__ = f"{base_doc}\n\n{skill_context}" if base_doc else skill_context
        return inner

    if func is None:
        return decorator
    return decorator(func)
