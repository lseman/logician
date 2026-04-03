# -*- coding: utf-8 -*-
"""
Unified tools package with skill-aware loading and prompt rendering.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:  # Python 3.11+
    from typing import Self
except ImportError:  # pragma: no cover - Python 3.10 fallback
    Self = Any  # type: ignore[assignment,misc]

from ..logging_utils import get_logger
from .parser import parse_tool_call_strict, parse_tool_calls
from .registry import (
    ExecutionGlobals,
    RegistryExecutionMixin,
    RegistryIntrospectionMixin,
    RegistryLoadingMixin,
    RegistryPromptingMixin,
    RegistryRoutingMixin,
    SkillCatalog,
    ToolExecutionStats,
)
from .runtime import (
    HAS_TOON,
    AppState,
    Context,
    SkillCard,
    SkillSelection,
    Tool,
    ToolCall,
    ToolParameter,
    ToolPermissionRule,
    _safe_json_fallback,
    check_optional_deps,
    materialize_tool,
)


class _LazyModuleProxy:
    """Import an optional dependency only when a skill actually touches it."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(type(self)) + dir(self._load())))

    def __repr__(self) -> str:
        return f"<lazy module proxy {self._module_name!r}>"


class ToolRegistry(
    RegistryExecutionMixin,
    RegistryPromptingMixin,
    RegistryLoadingMixin,
    RegistryIntrospectionMixin,
    RegistryRoutingMixin,
):
    """
    Registry for managing and executing tools loaded from SKILLS sources.
    """

    def __init__(self, auto_load_from_skills: bool = True) -> None:
        self._tools: dict[str, Tool] = {}
        self._log = get_logger("agent.tools")
        default_root = Path(__file__).resolve().parents[2]
        default_skills_dir = default_root / "skills"
        self.skills_md_path, self.skills_dir_path = self._resolve_skills_paths()
        if self.skills_dir_path != default_skills_dir and self.skills_dir_path.is_dir():
            self._log.info(
                "Resolved skills path override: %s (default was %s)",
                self.skills_dir_path,
                default_skills_dir,
            )
        self._catalog = SkillCatalog(
            skills_md_path=self.skills_md_path,
            skills_dir_path=self.skills_dir_path,
            log=self._log,
        )

        self._execution_globals: ExecutionGlobals = {
            "__builtins__": __builtins__,
            "json": json,
            "np": _LazyModuleProxy("numpy"),
            "pd": _LazyModuleProxy("pandas"),
            "ctx": None,
        }
        self._execution_globals["call_tool"] = self.call_tool
        self._execution_globals["_safe_json"] = _safe_json_fallback
        self._bootstrapped: bool = False
        self._active_lazy_skill_groups: set[str] = set()
        self._version: int = 0
        self._forced_skill_ids: list[str] = []
        self._forced_skill_reason: str = ""
        self._tool_exec_stats: ToolExecutionStats = {}
        self._grammars: dict[str, str] = {}
        self._strict_tool_arg_validation: bool = True
        self._coerce_tool_argument_types: bool = True
        self._skill_resolution_epoch: int = 0

        if auto_load_from_skills:
            _sources = self._catalog.iter_skills_sources()
            if _sources or self.skills_dir_path.is_dir():
                self._log.info("Auto-loading tools from %d source(s)", len(_sources))
                self.load_tools_from_skills()

    @staticmethod
    def _normalize_lazy_skill_group_name(value: str) -> str:
        text = str(value or "").strip().lower()
        text = text.strip("/").replace("-", "_").replace(" ", "_")
        if text.startswith("lazy_"):
            text = text[len("lazy_") :]
        text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
        return text

    def _is_lazy_skill_group_dir_name(self, name: str) -> bool:
        return str(name or "").strip().startswith("lazy_")

    def _is_lazy_skill_group_active(self, name: str) -> bool:
        group = self._normalize_lazy_skill_group_name(name)
        return bool(group) and group in self._active_lazy_skill_groups

    def available_lazy_skill_groups(self) -> list[str]:
        if not self.skills_dir_path.is_dir():
            return []
        groups: list[str] = []
        for child in sorted(self.skills_dir_path.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not self._is_lazy_skill_group_dir_name(child.name):
                continue
            group = self._normalize_lazy_skill_group_name(child.name)
            if group:
                groups.append(group)
        return groups

    def active_lazy_skill_groups(self) -> list[str]:
        return sorted(self._active_lazy_skill_groups)

    def activate_lazy_skill_group(self, name: str) -> tuple[bool, str | None]:
        group = self._normalize_lazy_skill_group_name(name)
        if not group:
            return False, None
        available = set(self.available_lazy_skill_groups())
        if group not in available:
            return False, None
        changed = group not in self._active_lazy_skill_groups
        if changed:
            self._active_lazy_skill_groups.add(group)
            self._catalog.set_active_lazy_skill_groups(self._active_lazy_skill_groups)
            self._invalidate_skill_resolution_cache()
        return changed, group

    @staticmethod
    def _resolve_skills_paths() -> tuple[Path, Path]:
        default_root = Path(__file__).resolve().parents[2]

        env_skills_dir = (
            os.getenv("AGENT_SKILLS_DIR")
            or os.getenv("SKILLS_DIR")
            or os.getenv("CODEX_SKILLS_DIR")
        )
        env_skills_md = (
            os.getenv("AGENT_SKILLS_MD_PATH")
            or os.getenv("SKILLS_MD_PATH")
            or os.getenv("CODEX_SKILLS_MD_PATH")
        )

        if env_skills_dir:
            skills_dir = Path(env_skills_dir).expanduser().resolve()
            if skills_dir.is_dir():
                if env_skills_md:
                    return Path(env_skills_md).expanduser().resolve(), skills_dir
                return skills_dir.parent / "SKILLS.md", skills_dir

        if env_skills_md:
            md_path = Path(env_skills_md).expanduser().resolve()
            if md_path.is_dir():
                return md_path / "SKILLS.md", md_path
            return md_path, md_path.parent / "skills"

        candidates: list[Path] = [default_root]
        try:
            cwd = Path.cwd().resolve()
            candidates.append(cwd)
            candidates.extend(cwd.parents)
        except Exception:
            pass
        try:
            argv_root = Path(sys.argv[0]).expanduser().resolve().parent
            candidates.append(argv_root)
            candidates.extend(argv_root.parents)
        except Exception:
            pass

        seen: set[str] = set()
        for root in candidates:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            skills_dir = root / "skills"
            if skills_dir.is_dir():
                return root / "SKILLS.md", skills_dir

        return default_root / "SKILLS.md", default_root / "skills"

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(tools={len(self._tools)}, "
            f"skills_md_path={str(self.skills_md_path)!r}, "
            f"skills_dir_path={str(self.skills_dir_path)!r}, "
            f"bootstrapped={self._bootstrapped})"
        )

    def __str__(self) -> str:
        names = ", ".join(sorted(self._tools.keys()))
        return f"ToolRegistry[{len(self._tools)}]: {names}" if names else "ToolRegistry[0]"

    @property
    def version(self) -> int:
        return self._version

    @property
    def registry(self) -> dict[str, Tool]:
        return self._tools

    def install_context(self, ctx: Context, extra_globals: ExecutionGlobals | None = None) -> None:
        self._execution_globals["ctx"] = ctx
        if extra_globals:
            self._execution_globals.update(extra_globals)

    def register(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        function: Callable[..., Any],
        *,
        runtime: dict[str, Any] | None = None,
        doc: str | None = None,
        skill_id: str | None = None,
        source_path: str | None = None,
        skill_meta: dict[str, Any] | None = None,
    ) -> Self:
        self._log.info("Manually registering tool: %s", name)
        self._tools[name] = materialize_tool(
            function,
            name=name,
            description=description,
            parameters=parameters,
            runtime=runtime,
            doc=doc,
            skill_id=skill_id,
            source_path=source_path,
            skill_meta=skill_meta,
        )
        self._version += 1
        self._invalidate_skill_resolution_cache()
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list(self) -> list[Tool]:
        return self.list_tools()

    def call_tool(self, name: str, **kwargs: Any) -> str:
        call = ToolCall(id=f"internal_{time.time():.6f}", name=name, arguments=kwargs)
        return self.execute(call, use_toon=False)


__version__ = "4.0.0"
__all__ = [
    "HAS_TOON",
    "AppState",
    "Tool",
    "ToolCall",
    "ToolParameter",
    "ToolRegistry",
    "SkillCard",
    "SkillSelection",
    "parse_tool_calls",
    "parse_tool_call_strict",
    "Context",
    "ToolPermissionRule",
    "check_optional_deps",
]
