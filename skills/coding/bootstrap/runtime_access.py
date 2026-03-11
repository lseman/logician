from __future__ import annotations

import builtins
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping


class LegacyCodingRuntime:
    """Compatibility runtime for direct imports outside the registry loader."""

    def __init__(self, globalns: Mapping[str, Any] | None = None) -> None:
        namespace = dict(globalns or {})
        self._config = getattr(
            builtins,
            "_coding_config",
            namespace.get("_coding_config", {}),
        )
        self._run_cmd = getattr(
            builtins,
            "_run_cmd",
            namespace.get("_run_cmd"),
        )

    def cwd(self) -> str | None:
        value = self._config.get("default_cwd")
        return str(value) if value else None

    def venv_path(self) -> str | None:
        value = self._config.get("venv_path")
        return str(value) if value else None

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
        p = Path(path).expanduser()
        base = self.cwd()
        if not p.is_absolute() and base:
            p = Path(base) / p
        return p.resolve()

    def build_shell_prefix(self, venv_path: str | None) -> str:
        venv = venv_path or self.venv_path()
        if not venv:
            return ""
        activate = Path(venv).expanduser() / "bin" / "activate"
        if not activate.exists():
            return ""
        return f". {shlex.quote(str(activate))} && "

    def run_cmd(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 60,
        venv_path: str | None = None,
        shell: bool = True,
    ) -> dict[str, Any]:
        del shell
        if callable(self._run_cmd):
            return self._run_cmd(
                command,
                cwd=cwd,
                timeout=timeout,
                venv_path=venv_path,
            )
        raise RuntimeError("coding runtime is not available")


def get_coding_runtime(globalns: Mapping[str, Any] | None = None) -> Any:
    namespace = dict(globalns or {})
    return (
        namespace.get("coding_runtime")
        or namespace.get("_coding_runtime")
        or getattr(builtins, "coding_runtime", None)
        or getattr(builtins, "_coding_runtime", None)
        or LegacyCodingRuntime(namespace)
    )


def _format_skill_doc_context(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""

    lines: list[str] = []
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    aliases = [str(item).strip() for item in meta.get("aliases", []) if str(item).strip()]
    triggers = [str(item).strip() for item in meta.get("triggers", []) if str(item).strip()]
    preferred_tools = [
        str(item).strip() for item in meta.get("preferred_tools", []) if str(item).strip()
    ]
    example_queries = [
        str(item).strip() for item in meta.get("example_queries", []) if str(item).strip()
    ]
    when_not_to_use = [
        str(item).strip() for item in meta.get("when_not_to_use", []) if str(item).strip()
    ]
    next_skills = [
        str(item).strip() for item in meta.get("next_skills", []) if str(item).strip()
    ]
    workflow = [str(item).strip() for item in meta.get("workflow", []) if str(item).strip()]

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
        lines.append(f"Skill avoid when: {when_not_to_use[0]}")
    if next_skills:
        lines.append(f"Typical next skills: {', '.join(next_skills)}")
    if workflow:
        lines.append("Skill workflow:")
        lines.extend(f"- {step}" for step in workflow)

    if not lines:
        return ""
    return "\n".join(["Skill context:", *lines])


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Mark a coding skill function as an exported tool."""

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        meta = dict(getattr(inner, "__llm_tool_meta__", {}) or {})
        meta_name = str(name or meta.get("name") or getattr(inner, "__name__", "")).strip()
        meta["name"] = meta_name or getattr(inner, "__name__", "")
        if description is not None:
            meta["description"] = description
        setattr(inner, "__llm_tool_meta__", meta)
        setattr(inner, "__tool__", True)
        skill_context = _format_skill_doc_context(inner.__globals__.get("__skill__"))
        marker = "Skill context:"
        base_doc = str(getattr(inner, "__doc__", "") or "").rstrip()
        if skill_context and marker not in base_doc:
            inner.__doc__ = f"{base_doc}\n\n{skill_context}" if base_doc else skill_context
        return inner

    if func is None:
        return decorator
    return decorator(func)
