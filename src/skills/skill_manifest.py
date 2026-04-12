from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore
    HAS_YAML = False

SkillManifest = dict[str, Any]

SkillManifest = dict[str, Any]


def is_skill_file(path: str) -> bool:
    return Path(path).name.lower() == "skill.md"


def find_skill_markdown_files(base_path: str) -> List[str]:
    base = Path(base_path)
    if not base.exists() or not base.is_dir():
        return []

    results: List[str] = []
    for root, dirs, files in os.walk(base, followlinks=True):
        if Path(root) == base:
            continue
        for fname in files:
            if is_skill_file(fname):
                results.append(str(Path(root) / fname))
    results.sort()
    return results


def _strip_skill_fence(content: str) -> str:
    lines = content.splitlines()
    if lines and re.match(r"^```skill\s*$", lines[0].strip()):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines)


def _parse_frontmatter_lines_fallback(lines: list[str]) -> SkillManifest:
    manifest: SkillManifest = {}
    current_key: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped and not stripped.startswith("- "):
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            current_key = key
            if val == "":
                manifest[key] = []
            else:
                manifest[key] = val
            continue
        if stripped.startswith("- ") and current_key:
            existing = manifest.get(current_key)
            if not isinstance(existing, list):
                existing = [] if existing in (None, "") else [str(existing)]
            existing.append(stripped[2:].strip())
            manifest[current_key] = existing
    return manifest


def _parse_frontmatter(raw_yaml: str) -> SkillManifest:
    if HAS_YAML and _yaml is not None:
        try:
            parsed = _yaml.safe_load(raw_yaml)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            pass
    return _parse_frontmatter_lines_fallback(raw_yaml.splitlines())


def split_frontmatter(content: str) -> tuple[SkillManifest, str]:
    stripped = _strip_skill_fence(content)
    if not stripped.startswith("---\n"):
        return {}, stripped
    lines = stripped.splitlines()
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, stripped
    raw_yaml = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return _parse_frontmatter(raw_yaml), body
