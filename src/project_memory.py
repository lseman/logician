from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OBS_TYPE_EMOJI: dict[str, str] = {
    "bugfix": "🔴",
    "feature": "🟣",
    "refactor": "🔄",
    "change": "✅",
    "discovery": "🔵",
    "decision": "⚖️",
}
VALID_TYPES = set(OBS_TYPE_EMOJI)


def get_memory_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / ".logician" / "memory"


def get_obs_dir(base_dir: str | Path | None = None) -> Path:
    return get_memory_dir(base_dir) / "obs"


def get_index_path(base_dir: str | Path | None = None) -> Path:
    return get_obs_dir(base_dir) / "index.json"


def load_index(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
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
    obs_dir = get_obs_dir(base_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)
    get_index_path(base_dir).write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
            mem_type = stripped[5:].strip().strip('"\'')
        elif stripped.startswith("description:"):
            description = stripped[12:].strip().strip('"\'')
    return mem_type, description


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
    normalized_type = str(obs_type or "").strip().lower()
    if normalized_type not in VALID_TYPES:
        raise ValueError(
            f"Unknown observation type {obs_type!r}. Valid: {sorted(VALID_TYPES)}"
        )

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
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
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
            lines.append("| ID | Time | T | Title |")
            lines.append("|----|------|---|-------|")
            for entry in grouped_entries:
                lines.append(
                    f"| {format_observation_id(int(entry.get('id', 0) or 0))} "
                    f"| {format_observation_time(str(entry.get('timestamp', '')))} "
                    f"| {OBS_TYPE_EMOJI.get(str(entry.get('type', '')), '•')} "
                    f"| {str(entry.get('title', '')).strip()} |"
                )
            lines.append("")

    fact_lines = []
    for file_path in sorted(memory_dir.glob("*.md")):
        if file_path.name == "MEMORY.md":
            continue
        _, description = parse_frontmatter(
            file_path.read_text(encoding="utf-8", errors="replace")
        )
        if description:
            fact_lines.append(f"- [{file_path.stem}]({file_path.name}): {description}")

    facts_dir = memory_dir / "facts"
    if facts_dir.exists():
        for file_path in sorted(facts_dir.glob("*.md")):
            _, description = parse_frontmatter(
                file_path.read_text(encoding="utf-8", errors="replace")
            )
            if description:
                fact_lines.append(
                    f"- [{file_path.stem}](facts/{file_path.name}): {description}"
                )

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
