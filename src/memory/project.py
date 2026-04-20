from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..aaak_dialect import compress_text_to_aaak

OBS_TYPE_EMOJI: dict[str, str] = {
    "bugfix": "🔴",
    "feature": "🟣",
    "refactor": "🔄",
    "change": "✅",
    "discovery": "🔵",
    "decision": "⚖️",
}
VALID_TYPES = set(OBS_TYPE_EMOJI)


def project_memory_enabled() -> bool:
    raw = str(os.getenv("LOGICIAN_PROJECT_MEMORY_ENABLED", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_memory_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / ".logician" / "memory"


def get_obs_dir(base_dir: str | Path | None = None) -> Path:
    return get_memory_dir(base_dir) / "obs"


def get_index_path(base_dir: str | Path | None = None) -> Path:
    return get_obs_dir(base_dir) / "index.json"


def load_index(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if not project_memory_enabled():
        return []
    index_path = get_index_path(base_dir)
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def save_index(index: list[dict[str, Any]], base_dir: str | Path | None = None) -> None:
    if not project_memory_enabled():
        return
    obs_dir = get_obs_dir(base_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)
    get_index_path(base_dir).write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_observation_content(obs_id: int, base_dir: str | Path | None = None) -> str:
    if not project_memory_enabled():
        return ""
    obs_path = get_obs_dir(base_dir) / f"{obs_id:04d}.md"
    if not obs_path.exists():
        return ""
    raw = obs_path.read_text(encoding="utf-8", errors="replace")
    return _strip_frontmatter(raw)


def compress_observation(
    obs_id: int,
    *,
    base_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    text = get_observation_content(obs_id, base_dir)
    if not text:
        return ""

    index = load_index(base_dir)
    entry = next((item for item in index if int(item.get("id", 0) or 0) == obs_id), None)
    if entry is None:
        entry = {}

    info = {
        "source_file": f"{obs_id:04d}.md",
        "session": str(entry.get("session") or ""),
        "title": str(entry.get("title") or ""),
        "date": str(entry.get("timestamp") or "")[:10],
    }
    if metadata:
        info.update(metadata)

    return compress_text_to_aaak(text, metadata=info)


def compress_observations(
    obs_ids: list[int], *, base_dir: str | Path | None = None
) -> dict[int, str]:
    return {obs_id: compress_observation(obs_id, base_dir=base_dir) for obs_id in obs_ids}


def format_observation_id(obs_id: int) -> str:
    return f"#{obs_id:03d}"


def format_observation_time(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(str(timestamp).rstrip("Z"))
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, TypeError):
        return str(timestamp or "")[:16]


def parse_frontmatter(content: str) -> tuple[str, str]:
    mem_type, description = "unknown", ""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return mem_type, description
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if stripped.startswith("type:"):
            mem_type = stripped[5:].strip().strip("\"'")
        elif stripped.startswith("description:"):
            description = stripped[12:].strip().strip("\"'")
    return mem_type, description


def _strip_frontmatter(content: str) -> str:
    text = str(content or "")
    if not text.startswith("---"):
        return text.strip()
    parts = text.split("---", 2)
    if len(parts) >= 3:
        return str(parts[2] or "").strip()
    return text.strip()


def _memory_fact_files(base_dir: str | Path | None = None) -> list[Path]:
    memory_dir = get_memory_dir(base_dir)
    files = [
        file_path for file_path in sorted(memory_dir.glob("*.md")) if file_path.name != "MEMORY.md"
    ]
    facts_dir = memory_dir / "facts"
    if facts_dir.exists():
        files.extend(sorted(facts_dir.glob("*.md")))
    return files


def list_fact_notes(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if not project_memory_enabled():
        return []
    rows: list[dict[str, Any]] = []
    for file_path in _memory_fact_files(base_dir):
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        mem_type, description = parse_frontmatter(raw)
        rows.append(
            {
                "name": file_path.stem,
                "type": mem_type,
                "description": description,
                "content": _strip_frontmatter(raw),
                "path": str(file_path),
                "is_preference": file_path.stem == "user_preferences",
            }
        )
    return rows


def _query_terms(query: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{1,}", str(query or ""))
        if len(token) >= 2
    ]


def _score_text_match(query: str, *parts: str) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0
    haystack = "\n".join(str(part or "") for part in parts).lower()
    score = 0
    for term in terms:
        if term in haystack:
            score += 6
        split_term = term.replace("/", " ").replace(".", " ").split()
        if len(split_term) > 1 and all(piece in haystack for piece in split_term):
            score += 3
    return score


def search_project_memory(
    query: str,
    *,
    base_dir: str | Path | None = None,
    fact_limit: int = 4,
    observation_limit: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    if not project_memory_enabled():
        return {"facts": [], "observations": []}
    normalized_query = str(query or "").strip().lower()
    facts: list[dict[str, Any]] = []
    for note in list_fact_notes(base_dir):
        score = 100 if note.get("is_preference") else 0
        score += _score_text_match(
            normalized_query,
            str(note.get("name") or ""),
            str(note.get("description") or ""),
            str(note.get("content") or ""),
        )
        if normalized_query and score <= 0 and not note.get("is_preference"):
            continue
        facts.append({**note, "score": score})

    observations: list[dict[str, Any]] = []
    obs_dir = get_obs_dir(base_dir)
    for entry in reversed(load_index(base_dir)):
        obs_id = int(entry.get("id", 0) or 0)
        obs_path = obs_dir / f"{obs_id:04d}.md"
        raw = obs_path.read_text(encoding="utf-8", errors="replace") if obs_path.exists() else ""
        content = _strip_frontmatter(raw)
        score = _score_text_match(
            normalized_query,
            str(entry.get("title") or ""),
            str(entry.get("preview") or ""),
            content,
            " ".join(str(path) for path in (entry.get("files") or [])),
        )
        if normalized_query and score <= 0:
            continue
        if not normalized_query:
            score = max(score, 1)
        observations.append(
            {
                "id": obs_id,
                "type": str(entry.get("type") or ""),
                "title": str(entry.get("title") or ""),
                "preview": str(entry.get("preview") or ""),
                "content": content,
                "files": [str(path) for path in (entry.get("files") or [])],
                "path": str(obs_path),
                "score": score,
            }
        )

    facts.sort(key=lambda item: (-int(item.get("score", 0) or 0), str(item.get("name") or "")))
    observations.sort(
        key=lambda item: (-int(item.get("score", 0) or 0), -int(item.get("id", 0) or 0))
    )
    return {
        "facts": facts[: max(0, int(fact_limit))],
        "observations": observations[: max(0, int(observation_limit))],
    }


def build_memory_context(
    query: str,
    *,
    base_dir: str | Path | None = None,
    max_chars: int = 1200,
    fact_limit: int = 3,
    observation_limit: int = 3,
) -> str:
    if not project_memory_enabled():
        return ""
    recalled = search_project_memory(
        query,
        base_dir=base_dir,
        fact_limit=fact_limit,
        observation_limit=observation_limit,
    )
    facts = list(recalled.get("facts") or [])
    observations = list(recalled.get("observations") or [])
    if not facts and not observations:
        return ""

    lines = ["Use this project memory if it helps:"]
    if facts:
        lines.append("Durable facts and preferences:")
        for item in facts:
            label = str(item.get("name") or "").replace("_", " ").strip()
            detail = str(item.get("description") or "").strip()
            if not detail:
                detail = compact_text(str(item.get("content") or ""), 180)
            lines.append(f"- {label}: {detail}")

    if observations:
        lines.append("Relevant past observations:")
        for item in observations:
            files = [str(path) for path in list(item.get("files") or []) if str(path).strip()]
            file_hint = (
                (f" [{', '.join(files)}]" if len(files) == 1 else f" [{', '.join(files)}]")
                if files
                else ""
            )
            preview = compact_text(
                str(item.get("preview") or item.get("content") or ""),
                180,
            )
            lines.append(
                f"- {format_observation_id(int(item.get('id', 0) or 0))}{file_hint}: {preview}"
            )

    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, int(max_chars) - 4)].rstrip() + " ..."


def record_observation(
    *,
    obs_type: str,
    title: str,
    content: str,
    files: list[str] | None = None,
    session_label: str | None = None,
    session_key: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not project_memory_enabled():
        return {
            "disabled": True,
            "id": 0,
            "formatted_id": "",
            "session": str(session_label or ""),
            "path": "",
            "timestamp": "",
            "preview": "",
            "memory_dir": str(get_memory_dir(base_dir)),
        }
    normalized_type = str(obs_type or "").strip().lower()
    if normalized_type not in VALID_TYPES:
        raise ValueError(f"Unknown observation type {obs_type!r}. Valid: {sorted(VALID_TYPES)}")

    title = str(title or "").strip()
    content = str(content or "").strip()
    if not title:
        raise ValueError("title is required")
    if not content:
        raise ValueError("content is required")

    memory_dir = get_memory_dir(base_dir)
    obs_dir = get_obs_dir(base_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(base_dir)
    obs_id = max((int(item.get("id", 0) or 0) for item in index), default=0) + 1
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    clean_files = [str(path).strip() for path in (files or []) if str(path).strip()]
    preview = compact_text(content, 240)
    session = session_label or _session_label_for_key(index, session_key)
    emoji = OBS_TYPE_EMOJI[normalized_type]

    obs_path = obs_dir / f"{obs_id:04d}.md"
    files_yaml = "\n".join(f"  - {path}" for path in clean_files) if clean_files else "  []"
    obs_path.write_text(
        "\n".join(
            [
                "---",
                f"id: {obs_id}",
                f"session: {session}",
                f"session_key: {session_key or ''}",
                f"type: {normalized_type}",
                f"emoji: {emoji}",
                f"timestamp: {timestamp}",
                f"title: {title}",
                "files:",
                files_yaml,
                "---",
                "",
                content,
                "",
            ]
        ),
        encoding="utf-8",
    )

    index.append(
        {
            "id": obs_id,
            "session": session,
            "session_key": session_key or "",
            "type": normalized_type,
            "emoji": emoji,
            "timestamp": timestamp,
            "title": title,
            "files": clean_files,
            "preview": preview,
        }
    )
    save_index(index, base_dir)
    rebuild_memory_md(base_dir)

    return {
        "id": obs_id,
        "formatted_id": format_observation_id(obs_id),
        "session": session,
        "path": str(obs_path),
        "timestamp": timestamp,
        "preview": preview,
        "memory_dir": str(memory_dir),
    }


def rebuild_memory_md(base_dir: str | Path | None = None) -> None:
    if not project_memory_enabled():
        return
    memory_dir = get_memory_dir(base_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(base_dir)

    sessions: dict[str, list[dict[str, Any]]] = {}
    for entry in index:
        session = str(entry.get("session") or "S000")
        sessions.setdefault(session, []).append(entry)

    lines = [
        "# Project Memory — Logician",
        "<!-- Auto-generated by project memory observation capture. -->",
        "",
        "## Observations",
        "",
    ]

    for session in sorted(sessions.keys()):
        entries = sessions[session]
        first_ts = entries[0].get("timestamp", "") if entries else ""
        lines.append(f"## {session} · {format_observation_time(str(first_ts))}")
        lines.append("")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            file_group = (entry.get("files") or ["General"])[0] or "General"
            grouped.setdefault(str(file_group), []).append(entry)

        for file_group, grouped_entries in grouped.items():
            lines.append(f"### {file_group}")
            lines.append("")
            for entry in grouped_entries:
                obs_id = int(entry.get("id", 0) or 0)
                title = str(entry.get("title", "")).replace("\n", " ").strip()
                preview = str(entry.get("preview", "")).replace("\n", " ").strip()
                emoji = OBS_TYPE_EMOJI.get(str(entry.get("type", "")).strip(), "•")
                files = [
                    str(path).strip() for path in (entry.get("files") or []) if str(path).strip()
                ]
                file_suffix = f" [{', '.join(files)}]" if files else ""
                if not title:
                    title = preview or "(no title)"
                lines.append(f"- {format_observation_id(obs_id)} {emoji} {title}{file_suffix}")
                if preview and preview != title:
                    lines.append(f"  - {preview}")
            lines.append("")

    fact_lines = []
    for file_path in sorted(memory_dir.glob("*.md")):
        if file_path.name == "MEMORY.md":
            continue
        _, description = parse_frontmatter(file_path.read_text(encoding="utf-8", errors="replace"))
        if description:
            fact_lines.append(f"- [{file_path.stem}]({file_path.name}): {description}")

    facts_dir = memory_dir / "facts"
    if facts_dir.exists():
        for file_path in sorted(facts_dir.glob("*.md")):
            _, description = parse_frontmatter(
                file_path.read_text(encoding="utf-8", errors="replace")
            )
            if description:
                fact_lines.append(f"- [{file_path.stem}](facts/{file_path.name}): {description}")

    if fact_lines:
        lines.append("## Facts")
        lines.extend(fact_lines)
        lines.append("")

    (memory_dir / "MEMORY.md").write_text("\n".join(lines), encoding="utf-8")


def compact_text(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + " ..."


def _session_label_for_key(index: list[dict[str, Any]], session_key: str | None) -> str:
    if session_key:
        for entry in index:
            if str(entry.get("session_key") or "") == session_key:
                existing = str(entry.get("session") or "").strip()
                if existing:
                    return existing

    seen_labels = {
        str(entry.get("session") or "").strip()
        for entry in index
        if str(entry.get("session") or "").strip()
    }
    next_number = 1
    while f"S{next_number:03d}" in seen_labels:
        next_number += 1
    return f"S{next_number:03d}"
