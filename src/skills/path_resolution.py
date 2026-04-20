from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

SkillRootKind = Literal["skills", "commands"]

_PLUGIN_SKILL_INDEX_FILENAME = "skill_index.json"
_PLUGIN_SKILL_INDEX_VERSION = 1


@dataclass(frozen=True)
class SkillSourceRoot:
    path: Path
    kind: SkillRootKind


def resolve_local_skill_paths(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    argv0: str | None = None,
) -> tuple[Path, Path]:
    effective_env = env if env is not None else os.environ
    default_root = Path(__file__).resolve().parents[2]

    env_skills_dir = (
        effective_env.get("AGENT_SKILLS_DIR")
        or effective_env.get("SKILLS_DIR")
        or effective_env.get("CODEX_SKILLS_DIR")
        or ""
    ).strip()
    env_skills_md = (
        effective_env.get("AGENT_SKILLS_MD_PATH")
        or effective_env.get("SKILLS_MD_PATH")
        or effective_env.get("CODEX_SKILLS_MD_PATH")
        or ""
    ).strip()

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
        resolved_cwd = (cwd or Path.cwd()).resolve()
        candidates.append(resolved_cwd)
        candidates.extend(resolved_cwd.parents)
    except Exception:
        pass
    try:
        argv_root = Path(argv0 or sys.argv[0]).expanduser().resolve().parent
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


def infer_skill_root_kind(path: Path) -> SkillRootKind:
    name = path.name.lower()
    if name == "commands":
        return "commands"
    if name == "skills":
        return "skills"
    if path.is_dir() and (path / "SKILL.md").is_file():
        return "skills"
    return "skills"


def _normalize_root_kind(kind: str | None, path: Path) -> SkillRootKind:
    value = str(kind or "").strip().lower()
    if value in {"skills", "commands"}:
        return value  # type: ignore[return-value]
    return infer_skill_root_kind(path)


def _normalize_skill_group_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("/").replace("-", "_").replace(" ", "_")
    if text.startswith("lazy_"):
        text = text[len("lazy_") :]
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
    return text


def _should_prune_lazy_group(
    *,
    root: Path,
    current_root: Path,
    active_lazy_skill_groups: set[str],
) -> bool:
    try:
        rel_parts = current_root.relative_to(root).parts
    except Exception:
        rel_parts = ()
    if not rel_parts:
        return False
    top_level = rel_parts[0]
    if not str(top_level).startswith("lazy_"):
        return False
    return _normalize_skill_group_name(top_level) not in active_lazy_skill_groups


def iter_entrypoint_markdown_files(
    root: Path,
    *,
    kind: str | None = None,
    active_lazy_skill_groups: Iterable[str] = (),
) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".md":
        return [root]
    if not root.is_dir():
        return []

    active = {_normalize_skill_group_name(item) for item in active_lazy_skill_groups if item}
    resolved_kind = _normalize_root_kind(kind, root)
    results: list[Path] = []

    for walk_root, dirs, files in os.walk(str(root), followlinks=True):
        current_root = Path(walk_root)

        if resolved_kind == "skills":
            if _should_prune_lazy_group(
                root=root,
                current_root=current_root,
                active_lazy_skill_groups=active,
            ):
                dirs[:] = []
                continue
            if current_root == root:
                dirs[:] = [
                    d
                    for d in dirs
                    if not str(d).startswith("lazy_")
                    or _normalize_skill_group_name(d) in active
                ]
            for fname in files:
                if fname.lower() == "skill.md":
                    results.append(current_root / fname)
            continue

        for fname in files:
            if fname.lower().endswith(".md"):
                results.append(current_root / fname)

    return sorted(results)


def plugin_skill_index_path(plugin_root: Path) -> Path:
    return plugin_root / _PLUGIN_SKILL_INDEX_FILENAME


def load_plugin_skill_index(plugin_root: Path) -> dict[str, object] | None:
    path = plugin_skill_index_path(plugin_root)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    if int(raw.get("version", 0) or 0) != _PLUGIN_SKILL_INDEX_VERSION:
        return None
    return raw


def _dedupe_roots(roots: Iterable[SkillSourceRoot]) -> list[SkillSourceRoot]:
    out: list[SkillSourceRoot] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        try:
            key = (str(root.path.resolve()), root.kind)
        except Exception:
            key = (str(root.path), root.kind)
        if key in seen or not root.path.is_dir():
            continue
        seen.add(key)
        out.append(root)
    return out


def _plugin_component_skill_roots(plugin_root: Path) -> list[SkillSourceRoot]:
    plugin_json = plugin_root / ".claude-plugin" / "plugin.json"
    try:
        raw = json.loads(plugin_json.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return []
    components = raw.get("components", {})
    if not isinstance(components, dict):
        return []
    skills = components.get("skills", [])
    if isinstance(skills, str):
        skills = [skills]
    if not isinstance(skills, list):
        return []

    roots: list[SkillSourceRoot] = []
    for item in skills:
        rel = str(item or "").strip()
        if not rel:
            continue
        candidate = (plugin_root / rel).resolve()
        if candidate.is_dir():
            roots.append(SkillSourceRoot(path=candidate, kind="skills"))
    return roots


def resolve_plugin_skill_roots(plugin_root: Path) -> list[SkillSourceRoot]:
    index = load_plugin_skill_index(plugin_root)
    roots: list[SkillSourceRoot] = []
    if index is not None:
        raw_roots = index.get("roots", [])
        if isinstance(raw_roots, list):
            for item in raw_roots:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("path") or "").strip()
                if not rel:
                    continue
                kind = _normalize_root_kind(str(item.get("kind") or ""), plugin_root / rel)
                path = (plugin_root / rel).resolve()
                if path.is_dir():
                    roots.append(SkillSourceRoot(path=path, kind=kind))
    if roots:
        return _dedupe_roots(roots)

    fallback: list[SkillSourceRoot] = []
    for subdir, kind in (("skills", "skills"), ("commands", "commands")):
        candidate = (plugin_root / subdir).resolve()
        if candidate.is_dir():
            fallback.append(SkillSourceRoot(path=candidate, kind=kind))
    fallback.extend(_plugin_component_skill_roots(plugin_root))
    return _dedupe_roots(fallback)


def _plugin_root_for_path(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".claude-plugin" / "plugin.json").is_file():
            return candidate
        if plugin_skill_index_path(candidate).is_file():
            return candidate
    return None


def plugin_index_entrypoints_for_root(
    root: Path,
    *,
    kind: str | None = None,
) -> list[Path]:
    plugin_root = _plugin_root_for_path(root)
    if plugin_root is None:
        return []
    index = load_plugin_skill_index(plugin_root)
    if index is None:
        return []

    try:
        rel_root = root.resolve().relative_to(plugin_root.resolve())
    except Exception:
        return []
    rel_root_str = str(rel_root).replace("\\", "/").strip("/")
    resolved_kind = _normalize_root_kind(kind, root)

    out: list[Path] = []
    seen: set[str] = set()
    entries = index.get("entrypoints", [])
    if not isinstance(entries, list):
        return []
    for item in entries:
        if not isinstance(item, dict):
            continue
        entry_kind = _normalize_root_kind(str(item.get("kind") or ""), root)
        if entry_kind != resolved_kind:
            continue
        rel = str(item.get("path") or "").strip().replace("\\", "/")
        if not rel:
            continue
        if rel_root_str and rel != rel_root_str and not rel.startswith(rel_root_str + "/"):
            continue
        candidate = (plugin_root / rel).resolve()
        if not candidate.is_file():
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return sorted(out)


def build_plugin_skill_index(plugin_root: Path) -> dict[str, object]:
    roots = resolve_plugin_skill_roots(plugin_root)
    root_payload = [
        {
            "path": str(root.path.resolve().relative_to(plugin_root.resolve())).replace("\\", "/"),
            "kind": root.kind,
        }
        for root in roots
        if root.path.exists()
    ]

    entries: list[dict[str, str]] = []
    seen_entry_paths: set[str] = set()
    for root in roots:
        for entrypoint in iter_entrypoint_markdown_files(root.path, kind=root.kind):
            try:
                rel = str(entrypoint.resolve().relative_to(plugin_root.resolve())).replace(
                    "\\", "/"
                )
            except Exception:
                continue
            if rel in seen_entry_paths:
                continue
            seen_entry_paths.add(rel)
            entries.append({"path": rel, "kind": root.kind})

    return {
        "version": _PLUGIN_SKILL_INDEX_VERSION,
        "roots": root_payload,
        "entrypoints": sorted(entries, key=lambda item: (item["kind"], item["path"])),
    }


def write_plugin_skill_index(plugin_root: Path) -> Path:
    index = build_plugin_skill_index(plugin_root)
    path = plugin_skill_index_path(plugin_root)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = [
    "SkillRootKind",
    "SkillSourceRoot",
    "build_plugin_skill_index",
    "infer_skill_root_kind",
    "iter_entrypoint_markdown_files",
    "load_plugin_skill_index",
    "plugin_index_entrypoints_for_root",
    "plugin_skill_index_path",
    "resolve_local_skill_paths",
    "resolve_plugin_skill_roots",
    "write_plugin_skill_index",
]
