from __future__ import annotations

import functools
import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.skills.skill_manifest import find_skill_markdown_files, split_frontmatter


@dataclass
class Skill:
    name: str
    description: str
    frontmatter: Dict[str, Any]
    body: str
    file_path: str
    base_dir: Optional[str] = None
    paths: Optional[List[str]] = None

    def is_callable(self) -> bool:
        """True when the skill is declared as callable (implementation exists in scripts).

        The loader will not import or execute implementation on discovery; callable
        skills are loaded by their `scripts` metadata when execution is requested.
        """
        fm = self.frontmatter or {}
        if isinstance(fm.get("callable"), bool):
            return fm.get("callable")
        # consider presence of `scripts` key as a hint
        return bool(fm.get("scripts") or fm.get("script"))

    def user_facing_name(self) -> str:
        return self.frontmatter.get("name") or self.name

    def get_prompt_for_command(
        self,
        args: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        content = self.body
        if args:
            for k, v in args.items():
                content = content.replace(f"${{{k}}}", str(v))
        # placeholders
        if self.base_dir:
            base_dir = self.base_dir
            if os.name == "nt":
                base_dir = base_dir.replace("\\", "/")
            content = content.replace("${CLAUDE_SKILL_DIR}", base_dir)
        if session_id:
            content = content.replace("${CLAUDE_SESSION_ID}", session_id)
        return content

    def render_prompt(
        self,
        args: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        *,
        executor: Optional[callable] = None,
    ) -> str:
        prompt = self.get_prompt_for_command(args=args, session_id=session_id)
        return execute_embedded_shell_commands(
            prompt,
            base_dir=self.base_dir,
            allow_shell=True,
            executor=executor,
        )


# --- Dynamic discovery / caching / conditional activation ---
_dynamic_skill_dirs: Set[str] = set()
_dynamic_skills: Dict[str, Skill] = {}
_conditional_skills: Dict[str, Skill] = {}
_activated_conditional_skill_names: Set[str] = set()


@functools.lru_cache(maxsize=32)
def load_skills_from_dir(base_path: str) -> Tuple[Skill, ...]:
    files = find_skill_markdown_files(base_path)
    skills: List[Skill] = []
    for fp in files:
        try:
            txt = Path(fp).read_text(encoding="utf-8")
        except Exception:
            continue
        manifest, body = split_frontmatter(txt)
        name = manifest.get("name") or Path(fp).parent.name
        desc = manifest.get("description") or ""
        paths = manifest.get("paths") if isinstance(manifest.get("paths"), list) else None
        # normalize scripts key to a list for convenience
        if manifest.get("script") and not manifest.get("scripts"):
            if isinstance(manifest.get("script"), str):
                manifest["scripts"] = [manifest.get("script")]
        if manifest.get("scripts") and isinstance(manifest.get("scripts"), str):
            manifest["scripts"] = [manifest.get("scripts")]
        skills.append(
            Skill(
                name=name,
                description=desc,
                frontmatter=manifest,
                body=body,
                file_path=fp,
                base_dir=str(Path(fp).parent),
                paths=paths,
            )
        )
    # return tuple for caching
    return tuple(skills)


def load_skill_from_path(path: str) -> Skill:
    txt = Path(path).read_text(encoding="utf-8")
    manifest, body = split_frontmatter(txt)
    name = manifest.get("name") or Path(path).parent.name
    desc = manifest.get("description") or ""
    paths = manifest.get("paths") if isinstance(manifest.get("paths"), list) else None
    if manifest.get("script") and not manifest.get("scripts"):
        if isinstance(manifest.get("script"), str):
            manifest["scripts"] = [manifest.get("script")]
    if manifest.get("scripts") and isinstance(manifest.get("scripts"), str):
        manifest["scripts"] = [manifest.get("scripts")]
    return Skill(
        name=name,
        description=desc,
        frontmatter=manifest,
        body=body,
        file_path=path,
        base_dir=str(Path(path).parent),
        paths=paths,
    )


def load_all_skill_dirs(base_dirs: List[str]) -> List[Skill]:
    """Load skills from multiple base dirs and deduplicate by realpath (first-win)."""
    all_with_paths: List[Skill] = []
    for d in base_dirs:
        for s in load_skills_from_dir(d):
            all_with_paths.append(s)

    seen: Dict[str, str] = {}
    deduped: List[Skill] = []
    for s in all_with_paths:
        try:
            rid = os.path.realpath(s.file_path)
        except Exception:
            rid = s.file_path
        if rid in seen:
            continue
        seen[rid] = s.file_path
        deduped.append(s)
    return deduped


def add_skill_directories(dirs: List[str]) -> None:
    """Dynamically load skills from directories and merge into dynamic skills map."""
    new_dirs = [d for d in dirs if d not in _dynamic_skill_dirs]
    if not new_dirs:
        return
    for d in new_dirs:
        _dynamic_skill_dirs.add(d)
    loaded: List[Skill] = []
    for d in dirs:
        loaded.extend(list(load_skills_from_dir(d)))
    # deeper directories should override shallower ones externally; here we do first-win
    for skill in loaded:
        if skill.name:
            if skill.paths:
                _conditional_skills[skill.name] = skill
            else:
                _dynamic_skills[skill.name] = skill


def get_dynamic_skills() -> List[Skill]:
    return list(_dynamic_skills.values())


def activate_conditional_skills_for_paths(file_paths: List[str], cwd: str) -> List[str]:
    if not _conditional_skills:
        return []
    activated: List[str] = []
    rel_cwd = cwd.rstrip(os.path.sep)
    for name, skill in list(_conditional_skills.items()):
        if not skill.paths:
            continue
        for fp in file_paths:
            try:
                rel = os.path.relpath(fp, rel_cwd)
            except Exception:
                rel = fp
            for pattern in skill.paths:
                # simple prefix match for now and glob-like ** behavior
                if pattern.endswith("/**"):
                    base = pattern[:-3]
                    if rel.startswith(base):
                        _dynamic_skills[name] = skill
                        del _conditional_skills[name]
                        _activated_conditional_skill_names.add(name)
                        activated.append(name)
                        break
                elif rel == pattern or rel.startswith(pattern + os.path.sep):
                    _dynamic_skills[name] = skill
                    del _conditional_skills[name]
                    _activated_conditional_skill_names.add(name)
                    activated.append(name)
                    break
            if name in _dynamic_skills:
                break
    return activated


__all__ = [
    "Skill",
    "find_skill_markdown_files",
    "load_skills_from_dir",
    "load_skill_from_path",
    "load_all_skill_dirs",
    "add_skill_directories",
    "get_dynamic_skills",
    "activate_conditional_skills_for_paths",
    "execute_embedded_shell_commands",
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
        spec = importlib.util.spec_from_file_location(f"skill_{os.path.basename(path)}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(mod)  # type: ignore
        return mod
    except Exception:
        raise
