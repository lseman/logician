"""Cross-session project memory tools (claude-mem style).

Storage layout in .logician/memory/:
  MEMORY.md         — session-indexed table, injected at startup
  obs/              — numbered observation files (0001.md, 0002.md, …)
  obs/index.json    — fast scan index (no need to read all obs files)
  <type_name>.md    — legacy static facts (user/feedback/project/reference)
  facts/            — optional subdirectory for static facts

Three-layer search pattern (minimise tokens):
  1. mem_search("topic")          → compact index table with IDs  (~50 t per result)
  2. mem_timeline("#42", depth=3) → context around an anchor      (optional)
  3. mem_get(["#42", "#43"])      → full content for chosen IDs   (~200-1000 t each)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.project_memory import record_observation

# Emoji markers — same convention as claude-mem
OBS_TYPE_EMOJI: dict[str, str] = {
    "bugfix":    "🔴",
    "feature":   "🟣",
    "refactor":  "🔄",
    "change":    "✅",
    "discovery": "🔵",
    "decision":  "⚖️",
}
VALID_TYPES = set(OBS_TYPE_EMOJI)


# ── Storage paths ──────────────────────────────────────────────────────────────

def _memory_dir() -> Path:
    return Path.cwd() / ".logician" / "memory"

def _obs_dir() -> Path:
    return _memory_dir() / "obs"

def _index_path() -> Path:
    return _obs_dir() / "index.json"


# ── Index I/O ──────────────────────────────────────────────────────────────────

def _load_index() -> list[dict[str, Any]]:
    idx = _index_path()
    if not idx.exists():
        return []
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def _save_index(index: list[dict[str, Any]]) -> None:
    _obs_dir().mkdir(parents=True, exist_ok=True)
    _index_path().write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _next_id(index: list[dict[str, Any]]) -> int:
    return max((e["id"] for e in index), default=0) + 1


# ── Public tools ───────────────────────────────────────────────────────────────

def mem_search(
    query: str,
    limit: int = 20,
    obs_type: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> str:
    """Search cross-session memory index. Returns a compact table with IDs — NOT full content.

    Use when: Recalling past work, checking if a topic was handled before.

    Three-layer pattern (token-efficient):
      1. mem_search("topic")          → get matching IDs (cheap, ~50 t/result)
      2. mem_timeline("#42")          → see what was happening around that point
      3. mem_get(["#42", "#43"])      → load full details only for IDs you need

    Triggers: did we solve this, do you remember, search memory, recall, check memory.
    Avoid when: The question is about the current session only — use scratch instead.

    Args:
      query:      Keyword or phrase (case-insensitive). Matched against title + files + type.
      limit:      Max results to return (default 20).
      obs_type:   Filter by type — bugfix|feature|refactor|change|discovery|decision.
      date_start: Inclusive ISO date "YYYY-MM-DD".
      date_end:   Inclusive ISO date "YYYY-MM-DD".

    Returns:
      Compact index table with ID, timestamp, type emoji, title.
      Use mem_get(ids) to load full observation content.
    """
    index = _load_index()
    query_lower = query.lower()

    results: list[dict[str, Any]] = []
    for entry in reversed(index):  # newest-first
        if obs_type and entry.get("type") != obs_type:
            continue
        ts = entry.get("timestamp", "")
        if date_start and ts[:10] < date_start:
            continue
        if date_end and ts[:10] > date_end:
            continue
        haystack = " ".join([
            entry.get("title", ""),
            entry.get("preview", ""),
            " ".join(entry.get("files", [])),
            entry.get("type", ""),
            entry.get("session", ""),
        ]).lower()
        if query_lower in haystack:
            results.append(entry)
            if len(results) >= limit:
                break

    if not results:
        # Fall back to legacy static-fact files
        return _search_facts(query)

    table = _render_table(results)
    return _safe_json({
        "status": "ok",
        "query": query,
        "count": len(results),
        "summary": f"{len(results)} result(s). Use mem_get([\"#001\"]) to load full details.",
        "table": table,
        "ids": [_fmt_id(e["id"]) for e in results],
    })


def mem_get(ids: list[str]) -> str:
    """Fetch full content for specific observation IDs.

    Use after mem_search() — load only the observations you actually need.
    Each observation is ~200-1000 tokens; be selective.

    Use when: You have IDs from mem_search and need the full details.
    Avoid when: You haven't searched yet — start with mem_search first.

    Args:
      ids: List of IDs like ["#001", "#042"]. The '#' prefix is optional.

    Returns:
      Full frontmatter + content for each requested observation ID.
    """
    obs_dir = _obs_dir()
    index = {e["id"]: e for e in _load_index()}

    results = []
    not_found = []

    for id_str in ids:
        try:
            obs_id = int(id_str.lstrip("#"))
        except ValueError:
            not_found.append(id_str)
            continue

        entry = index.get(obs_id)
        if not entry:
            not_found.append(id_str)
            continue

        obs_file = obs_dir / f"{obs_id:04d}.md"
        content = obs_file.read_text(encoding="utf-8", errors="replace") if obs_file.exists() else ""

        results.append({
            "id": _fmt_id(obs_id),
            "type": entry.get("type", ""),
            "emoji": OBS_TYPE_EMOJI.get(entry.get("type", ""), "•"),
            "timestamp": entry.get("timestamp", ""),
            "title": entry.get("title", ""),
            "files": entry.get("files", []),
            "session": entry.get("session", ""),
            "content": content,
        })

    return _safe_json({
        "status": "ok",
        "count": len(results),
        "not_found": not_found,
        "observations": results,
    })


def mem_timeline(anchor: str, depth: int = 5) -> str:
    """Show observations chronologically around an anchor ID or session.

    Use to understand context — what was happening just before/after a search result —
    without paying the token cost of loading all full content.

    Use when: You have an ID from mem_search and want surrounding context.
    Avoid when: You already know which IDs you want — use mem_get directly.

    Args:
      anchor: Observation ID "#042" or session label "S001".
      depth:  Observations to show on each side of the anchor (max 20, default 5).

    Returns:
      Compact index table of nearby observations.
    """
    index = _load_index()
    depth = min(depth, 20)

    # Session anchor — show all observations in that session
    if anchor.startswith("S") and not anchor.startswith("#"):
        session_entries = [e for e in index if e.get("session") == anchor]
        if not session_entries:
            return _safe_json({"status": "not_found", "message": f"Session {anchor!r} not found."})
        return _safe_json({
            "status": "ok",
            "session": anchor,
            "count": len(session_entries),
            "table": _render_table(session_entries, highlight=None),
        })

    # Numeric anchor
    try:
        anchor_id = int(anchor.lstrip("#"))
    except ValueError:
        return _safe_json({"status": "error", "message": f"Invalid anchor {anchor!r}. Use '#042' or 'S001'."})

    anchor_idx = next((i for i, e in enumerate(index) if e["id"] == anchor_id), None)
    if anchor_idx is None:
        return _safe_json({"status": "not_found", "message": f"Observation {anchor!r} not found."})

    start = max(0, anchor_idx - depth)
    end = min(len(index), anchor_idx + depth + 1)
    window = index[start:end]

    return _safe_json({
        "status": "ok",
        "anchor": anchor,
        "count": len(window),
        "table": _render_table(window, highlight=anchor_id),
    })


def mem_record(
    obs_type: str,
    title: str,
    content: str,
    files: list[str] | None = None,
    session: str | None = None,
) -> str:
    """Record a new observation in cross-session memory.

    Use when: Something significant was LEARNED, FIXED, BUILT, or DECIDED in this
    session that a future session should know about.

    Observation types:
      bugfix    (🔴) — something broken, now fixed
      feature   (🟣) — new capability added
      refactor  (🔄) — code restructured, behaviour unchanged
      change    (✅) — config, docs, or misc modification
      discovery (🔵) — learned something about the existing system
      decision  (⚖️) — architectural or design choice with rationale

    DO record:   Non-obvious learnings, bug fixes, key decisions, surprising patterns.
    NEVER record: Ephemeral task state, in-progress work, things obvious from git/code.

    Title style: WHAT was learned/fixed/built — not what *you* did.
      ✓ "write_file now normalises CRLF line endings"
      ✗ "Investigated write_file and found CRLF handling"

    Args:
      obs_type: One of bugfix|feature|refactor|change|discovery|decision.
      title:    One-line summary (max ~100 chars).
      content:  Full observation (include Why, trade-offs, How to apply).
      files:    Relevant file paths (relative to repo root). Optional.
      session:  Session label (e.g. "S042"). Auto-assigned if omitted.

    Returns:
      Confirmation with the new observation ID.
    """
    if obs_type not in VALID_TYPES:
        return _safe_json({
            "status": "error",
            "message": f"Unknown type {obs_type!r}. Valid: {' | '.join(sorted(VALID_TYPES))}",
        })

    recorded = record_observation(
        obs_type=obs_type,
        title=title,
        content=content,
        files=files or [],
        session_label=session,
    )

    return _safe_json({
        "status": "ok",
        "id": recorded["formatted_id"],
        "session": recorded["session"],
        "message": (
            f"Observation {recorded['formatted_id']} recorded "
            f"({OBS_TYPE_EMOJI[obs_type]} {obs_type}): {title}"
        ),
    })


def mem_list() -> str:
    """List all observations and static facts — compact summary, no full content.

    Use when: You want a broad overview of what's in memory before searching.
    Avoid when: You already know what you're looking for — use mem_search instead.

    Returns:
      Observation count, last 10 entries (compact), plus any legacy static facts.
    """
    memory_dir = _memory_dir()
    if not memory_dir.exists():
        return _safe_json({
            "status": "empty",
            "obs_count": 0,
            "facts_count": 0,
            "observations": [],
            "facts": [],
        })

    index = _load_index()

    # Legacy static facts (root *.md files + facts/ subdir)
    facts: list[dict[str, str]] = []
    for f in sorted(memory_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        mem_type, description = _parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
        facts.append({"file": f.name, "type": mem_type, "description": description})

    facts_subdir = memory_dir / "facts"
    if facts_subdir.exists():
        for f in sorted(facts_subdir.glob("*.md")):
            mem_type, description = _parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
            facts.append({"file": f"facts/{f.name}", "type": mem_type, "description": description})

    recent = index[-10:] if index else []

    return _safe_json({
        "status": "ok",
        "obs_count": len(index),
        "facts_count": len(facts),
        "recent_observations": [
            {
                "id": _fmt_id(e["id"]),
                "type": e.get("type", ""),
                "emoji": OBS_TYPE_EMOJI.get(e.get("type", ""), "•"),
                "timestamp": _fmt_time(e.get("timestamp", "")),
                "title": e.get("title", ""),
            }
            for e in recent
        ],
        "facts": facts,
    })


# ── Private helpers ────────────────────────────────────────────────────────────

def _fmt_id(obs_id: int) -> str:
    return f"#{obs_id:03d}"

def _fmt_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        return dt.strftime("%b %d, %H:%M")
    except (ValueError, AttributeError):
        return ts[:16] if ts else ""

def _render_table(entries: list[dict[str, Any]], highlight: int | None = None) -> str:
    rows = [
        f"| {'→' if e['id'] == highlight else ' '}{_fmt_id(e['id'])} "
        f"| {_fmt_time(e.get('timestamp', ''))} "
        f"| {OBS_TYPE_EMOJI.get(e.get('type', ''), '•')} "
        f"| {e.get('title', '')} |"
        for e in entries
    ]
    header = "| ID | Time | T | Title |\n|-----|------|---|-------|"
    return header + "\n" + "\n".join(rows)

def _search_facts(query: str) -> str:
    """Fallback: keyword search over legacy static-fact files."""
    memory_dir = _memory_dir()
    if not memory_dir.exists():
        return _safe_json({
            "status": "not_found",
            "query": query,
            "message": "No .logician/memory/ directory. Use mem_record() to start saving.",
        })

    query_lower = query.lower()
    results = []
    for f in sorted(memory_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        if query_lower in content.lower() or query_lower in f.stem.lower():
            results.append({"file": f.name, "content": content.strip()})

    if not results:
        return _safe_json({
            "status": "not_found",
            "query": query,
            "message": f"No observations or facts found matching {query!r}. Try mem_list() to see all.",
        })

    return _safe_json({
        "status": "ok",
        "query": query,
        "count": len(results),
        "source": "facts",
        "results": results,
    })

def _rebuild_memory_md(index: list[dict[str, Any]]) -> None:
    """Regenerate MEMORY.md in session-table format (mirrors claude-mem style)."""
    memory_dir = _memory_dir()
    memory_index = memory_dir / "MEMORY.md"

    # Group by session
    sessions: dict[str, list[dict]] = {}
    for entry in index:
        sess = entry.get("session", "S000")
        sessions.setdefault(sess, []).append(entry)

    lines = [
        "# Project Memory — Logician",
        "<!-- Auto-generated by mem_record(). Lines after 200 are truncated on load. -->",
        "",
    ]

    for sess in sorted(sessions.keys()):
        entries = sessions[sess]
        first_ts = entries[0].get("timestamp", "")
        date_str = _fmt_time(first_ts)

        lines.append(f"**#{sess}** ({date_str})")
        lines.append("")

        # Sub-group by file
        by_file: dict[str, list[dict]] = {}
        for e in entries:
            fs = e.get("files", [])
            key = fs[0] if fs else "General"
            by_file.setdefault(key, []).append(e)

        for file_key, file_entries in by_file.items():
            lines.append(f"**{file_key}**")
            lines.append("| ID | Time | T | Title |")
            lines.append("|----|------|---|-------|")
            for e in file_entries:
                lines.append(
                    f"| {_fmt_id(e['id'])} "
                    f"| {_fmt_time(e.get('timestamp', ''))} "
                    f"| {OBS_TYPE_EMOJI.get(e.get('type', ''), '•')} "
                    f"| {e.get('title', '')} |"
                )
            lines.append("")

    # Legacy static facts section
    fact_lines = []
    for f in sorted(memory_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        _, description = _parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
        if description:
            fact_lines.append(f"- [{f.stem}]({f.name}): {description}")

    facts_subdir = memory_dir / "facts"
    if facts_subdir.exists():
        for f in sorted(facts_subdir.glob("*.md")):
            _, description = _parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
            if description:
                fact_lines.append(f"- [{f.stem}](facts/{f.name}): {description}")

    if fact_lines:
        lines.append("## Facts")
        lines.extend(fact_lines)
        lines.append("")

    memory_index.write_text("\n".join(lines), encoding="utf-8")

def _parse_frontmatter(content: str) -> tuple[str, str]:
    """Extract (type, description) from YAML frontmatter block."""
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

def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ── Tool export ────────────────────────────────────────────────────────────────

__tools__ = [mem_search, mem_get, mem_timeline, mem_record, mem_list]
