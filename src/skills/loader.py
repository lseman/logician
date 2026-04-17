# -*- coding: utf-8 -*-
"""
Skill loading utilities — parse SKILL.md files and return SkillDefinition instances.

This module handles the low-level parsing of skill markdown files into
SkillDefinition objects. Conditional activation and skill catalog management
are handled by ``SkillCatalog`` in ``src.tools.registry.catalog``.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.skills.skill_manifest import split_frontmatter

from ..tools.runtime import SkillDefinition

# --- Skill loading (returns SkillDefinition instances) ---


@functools.lru_cache(maxsize=32)
def load_skills_from_dir(base_path: str) -> Tuple[SkillDefinition, ...]:
    """
    Load all skills from a directory of SKILL.md files.

    Returns a tuple of SkillDefinition instances (cached with LRU).
    """
    from .skill_manifest import find_skill_markdown_files

    files = find_skill_markdown_files(base_path)
    skills: List[SkillDefinition] = []

    for fp in files:
        try:
            txt = Path(fp).read_text(encoding="utf-8")
        except Exception:
            continue
        manifest, body = split_frontmatter(txt)
        name = manifest.get("name") or Path(fp).parent.name
        desc = manifest.get("description") or ""
        paths = manifest.get("paths") if isinstance(manifest.get("paths"), list) else None
        source_path = str(Path(fp).resolve())
        base_dir = str(Path(fp).parent)

        skills.append(
            SkillDefinition(
                id=manifest.get("name") or Path(fp).parent.name,
                name=name,
                summary=desc,
                source_path=source_path,
                description=desc,
                frontmatter=manifest,
                body=body,
                base_dir=base_dir,
                paths=paths or [],
            )
        )
    return tuple(skills)


def load_skill_from_path(path: str) -> SkillDefinition:
    """
    Load a single skill from a SKILL.md file path.

    Returns a SkillDefinition with body/frontmatter for prompt rendering.
    """
    txt = Path(path).read_text(encoding="utf-8")
    manifest, body = split_frontmatter(txt)
    name = manifest.get("name") or Path(path).parent.name
    desc = manifest.get("description") or ""
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), list) else None
    source_path = str(Path(path).resolve())
    base_dir = str(Path(path).parent)

    return SkillDefinition(
        id=manifest.get("name") or Path(path).parent.name,
        name=name,
        summary=desc,
        source_path=source_path,
        description=desc,
        frontmatter=manifest,
        body=body,
        base_dir=base_dir,
        paths=paths or [],
    )


def load_all_skill_dirs(base_dirs: List[str]) -> List[SkillDefinition]:
    """Load skills from multiple base dirs and deduplicate by realpath (first-win)."""
    all_with_paths: List[SkillDefinition] = []
    for d in base_dirs:
        try:
            for s in load_skills_from_dir(d):
                all_with_paths.append(s)
        except Exception:
            continue

    seen: Dict[str, str] = {}
    deduped: List[SkillDefinition] = []
    for s in all_with_paths:
        try:
            rid = os.path.realpath(s.source_path)
        except Exception:
            rid = s.source_path
        if rid in seen:
            continue
        seen[rid] = s.source_path
        deduped.append(s)
    return deduped


__all__ = [
    "load_skills_from_dir",
    "load_skill_from_path",
    "load_all_skill_dirs",
    "execute_embedded_shell_commands",
    "_import_module_from_path",
]


# --- Embedded shell execution helper (minimal, opt-in) ---

BLOCK_PATTERN = re.compile(r"```!\s*\n?([\s\S]*?)\n?```")
INLINE_PATTERN = re.compile(r"(^|\s)!`([^`]+)`")


def execute_embedded_shell_commands(
    text: str,
    *,
    base_dir: Optional[str] = None,
    allow_shell: bool = False,
    executor: Optional[callable] = None,
) -> str:
    """Execute embedded shell commands in `text` and replace them with outputs.

    - `allow_shell` must be True to actually run commands; otherwise returns text unchanged.
    - `executor` is an optional callable(command: str, cwd: Optional[str]) -> str
      If not provided, uses subprocess.run with `shell=True`.
    This is intentionally minimal; callers should enforce permissions.
    """
    if not allow_shell:
        return text

    result = text
    matches = list(BLOCK_PATTERN.finditer(text))
    # inline only scan if cheap
    if "!`" in text:
        matches.extend(list(INLINE_PATTERN.finditer(text)))

    # process sequentially, replacing original spans
    out = []
    last_end = 0
    for m in matches:
        start, end = m.span()
        # append text before match
        out.append(result[last_end:start])
        prefix = ""
        cmd = m.group(1).strip() if m.lastindex == 1 else ""
        if len(m.groups()) > 1:
            prefix = m.group(1) or ""
            cmd = m.group(2).strip()
        output = ""
        try:
            if executor is not None:
                output = str(executor(cmd, base_dir) or "")
            else:
                completed = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=base_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                if stderr.strip():
                    output = stdout.strip() + "\n[stderr]\n" + stderr.strip()
                else:
                    output = stdout
        except Exception as e:
            output = f"[Error executing command: {e}]"
        out.append(prefix + output)
        last_end = end

    out.append(result[last_end:])
    return "".join(out)


# --- Import helpers for callable skill modules ---
def _import_module_from_path(path: str):
    path = str(path)
    try:
        spec = importlib.util.spec_from_file_location(
            f"skill_{os.path.basename(path)}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(mod)  # type: ignore
        return mod
    except Exception:
        raise
