"""Mutation-oriented core file tools."""

from __future__ import annotations

import ast
import difflib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ...text_normalization import normalize_text_for_matching, normalize_text_payload
from ..FileReadTool.state import (
    ensure_snapshot_allows_existing_file_write,
    parse_structured_patch,
    refresh_snapshot_after_write,
    resolve_tool_path,
)
from ..filesystem import DEFAULT_FILESYSTEM


def write_file(
    path: str,
    content: str,
    mode: str = "w",
    normalize_newlines: bool = True,
) -> dict[str, Any]:
    if mode not in ("w", "a"):
        return _err("mode must be 'w' or 'a'")

    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _err(f"Cannot create parent directory: {exc}")

    original = None
    original_newline = "\n"
    if p.exists():
        if not p.is_file():
            return _err(f"Path exists but is not a file: {path}")
        prepared = ensure_snapshot_allows_existing_file_write(
            globals().get("ctx"),
            p,
            operation="write",
        )
        if isinstance(prepared, dict):
            return prepared
        _, _, original, _ = prepared
        original_newline = _detect_newline_style(original)

    normalized, _ = _normalize_agent_content(
        content,
        language_hint=_language_hint_from_path(path),
    )
    final = normalized
    if mode == "a" and original is not None:
        final = original + normalized
    if normalize_newlines:
        final = _coerce_to_newline_style(final, original_newline)

    if original is not None and final == original:
        return {
            "status": "ok",
            "path": str(p),
            "mode": mode,
            "bytes_written": 0,
            "chars_written": 0,
            "newline": _newline_name(original_newline),
            "diff": "",
            "structured_patch": [],
            "snapshot": refresh_snapshot_after_write(
                globals().get("ctx"),
                p,
                content=final,
            ),
            "unchanged": True,
        }

    warning = _detect_truncation(final, path)
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    result: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "mode": mode,
        "bytes_written": len(final.encode("utf-8")),
        "chars_written": len(final),
        "newline": _newline_name(original_newline),
        "diff": _unified_diff(original or "", final, str(p)),
    }
    result["structured_patch"] = parse_structured_patch(result["diff"])
    result["snapshot"] = refresh_snapshot_after_write(
        globals().get("ctx"),
        p,
        content=final,
    )
    if warning:
        result["warning"] = warning
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        result["syntax_error"] = syntax_error
    return result


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    normalize_newlines: bool = True,
) -> dict[str, Any]:
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists():
        if old_string == "":
            return write_file(
                path,
                new_string,
                mode="w",
                normalize_newlines=normalize_newlines,
            )
        error = _err(f"File not found: {path}")
        suggestions = DEFAULT_FILESYSTEM._find_similar_paths(p)
        if suggestions:
            error["did_you_mean"] = suggestions
        return error
    if not p.is_file():
        return _err(f"File not found: {path}")

    prepared = ensure_snapshot_allows_existing_file_write(
        globals().get("ctx"),
        p,
        operation="edit",
    )
    if isinstance(prepared, dict):
        return prepared
    _, snapshot, original, _ = prepared

    file_newline = _detect_newline_style(original)
    view, boundaries = _normalized_view_boundaries(original)

    language_hint = _language_hint_from_path(path)
    old_normalized, _ = _normalize_agent_content(old_string, language_hint=language_hint)
    new_normalized, _ = _normalize_agent_content(new_string, language_hint=language_hint)
    old_norm = normalize_text_for_matching(old_normalized)
    if not old_norm:
        return _err("old_string is empty after normalization")

    occurrences = _find_all_occurrences(view, old_norm)
    count = len(occurrences)
    if count == 0:
        return {
            "status": "error",
            "error": (
                "old_string not found in file. "
                "Re-read the file and copy the exact block including indentation "
                "and blank lines. See closest_matches for bounded line-aware hints, or use read_edit_context()."
            ),
            "closest_matches": _find_closest_blocks(view, old_norm),
            "suggested_tool": "read_edit_context",
        }
    if count > 1:
        occurrence_lines = [_offset_to_line_number(view, idx) for idx in occurrences]
        if replace_all:
            patched = original
            for idx in reversed(occurrences):
                start_offset = boundaries[idx]
                end_offset = boundaries[idx + len(old_norm)]
                actual_old = patched[start_offset:end_offset]
                replacement = _preserve_quote_style(
                    old_normalized,
                    actual_old,
                    new_normalized,
                )
                if normalize_newlines:
                    replacement = _coerce_to_newline_style(replacement, file_newline)
                patched = patched[:start_offset] + replacement + patched[end_offset:]
            try:
                _atomic_write_text(p, patched)
            except OSError as exc:
                return _err(f"Cannot write file: {exc}")

            result: dict[str, Any] = {
                "status": "ok",
                "path": str(p),
                "lines_removed": old_norm.count("\n") + (1 if old_norm else 0),
                "lines_added": normalize_text_for_matching(new_normalized).count("\n")
                + (1 if new_normalized else 0),
                "newline": _newline_name(file_newline),
                "diff": _unified_diff(original, patched, str(p)),
                "replace_all": True,
                "matches_replaced": count,
                "occurrences_at_lines": occurrence_lines,
                "snapshot_before": {
                    "path": snapshot.get("path"),
                    "mtime_ns": snapshot.get("mtime_ns"),
                    "full_read": snapshot.get("full_read"),
                },
            }
            result["structured_patch"] = parse_structured_patch(result["diff"])
            result["snapshot"] = refresh_snapshot_after_write(
                globals().get("ctx"),
                p,
                content=patched,
            )
            syntax_error = _validate_syntax(p, patched)
            if syntax_error:
                result["syntax_error"] = syntax_error
            return result
        return {
            "status": "error",
            "error": (
                f"old_string found {count} times (lines {occurrence_lines}). "
                "Add more surrounding context to make the match unique, or set replace_all=true."
            ),
            "occurrences_at_lines": occurrence_lines,
            "replace_all_available": True,
        }

    idx = occurrences[0]
    start_offset = boundaries[idx]
    end_offset = boundaries[idx + len(old_norm)]
    actual_old = original[start_offset:end_offset]
    replacement = _preserve_quote_style(
        old_normalized,
        actual_old,
        new_normalized,
    )
    if normalize_newlines:
        replacement = _coerce_to_newline_style(replacement, file_newline)
    patched = original[:start_offset] + replacement + original[end_offset:]
    try:
        _atomic_write_text(p, patched)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    result: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "lines_removed": old_norm.count("\n") + (1 if old_norm else 0),
        "lines_added": normalize_text_for_matching(new_normalized).count("\n")
        + (1 if new_normalized else 0),
        "newline": _newline_name(file_newline),
        "diff": _unified_diff(original, patched, str(p)),
        "replace_all": False,
        "matches_replaced": 1,
        "snapshot_before": {
            "path": snapshot.get("path"),
            "mtime_ns": snapshot.get("mtime_ns"),
            "full_read": snapshot.get("full_read"),
        },
    }
    result["structured_patch"] = parse_structured_patch(result["diff"])
    result["snapshot"] = refresh_snapshot_after_write(
        globals().get("ctx"),
        p,
        content=patched,
    )
    syntax_error = _validate_syntax(p, patched)
    if syntax_error:
        result["syntax_error"] = syntax_error
    return result


def apply_edit_block(path: str, blocks: str) -> dict[str, Any]:
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    prepared = ensure_snapshot_allows_existing_file_write(
        globals().get("ctx"),
        p,
        operation="edit",
    )
    if isinstance(prepared, dict):
        return prepared
    _, _, original, _ = prepared

    file_newline = _detect_newline_style(original)
    parsed_blocks = _parse_edit_blocks(blocks)
    if isinstance(parsed_blocks, dict) and parsed_blocks.get("status") == "error":
        return parsed_blocks

    assert isinstance(parsed_blocks, list)
    result = original
    applied = 0
    errors: list[dict[str, Any]] = []
    for index, block in enumerate(parsed_blocks):
        search_text = normalize_text_for_matching(block["search"])
        view, boundaries = _normalized_view_boundaries(result)
        occurrences = _find_all_occurrences(view, search_text)
        count = len(occurrences)
        if count == 0:
            errors.append(
                {
                    "block": index,
                    "error": "SEARCH text not found in file",
                    "search_preview": search_text[:200],
                    "closest_matches": _find_closest_blocks(view, search_text),
                    "suggested_tool": "read_edit_context",
                }
            )
            continue
        if count > 1:
            lines = [_offset_to_line_number(view, idx) for idx in occurrences]
            errors.append(
                {
                    "block": index,
                    "error": (
                        f"SEARCH text found {count} times (lines {lines}). "
                        "Add more surrounding context to make it unique."
                    ),
                    "occurrences_at_lines": lines,
                }
            )
            continue
        idx = occurrences[0]
        start_offset = boundaries[idx]
        end_offset = boundaries[idx + len(search_text)]
        actual_old = result[start_offset:end_offset]
        replacement = _preserve_quote_style(
            block["search"],
            actual_old,
            block["replace"],
        )
        replacement = _coerce_to_newline_style(replacement, file_newline)
        result = result[:start_offset] + replacement + result[end_offset:]
        applied += 1

    if applied == 0 and errors:
        return {"status": "error", "path": str(p), "blocks_applied": 0, "errors": errors}

    final = _coerce_to_newline_style(result, file_newline)
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    out: dict[str, Any] = {
        "status": "ok" if not errors else "partial",
        "path": str(p),
        "blocks_applied": applied,
        "diff": _unified_diff(original, final, str(p)),
        "errors": errors,
    }
    out["structured_patch"] = parse_structured_patch(out["diff"])
    out["snapshot"] = refresh_snapshot_after_write(
        globals().get("ctx"),
        p,
        content=final,
    )
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        out["syntax_error"] = syntax_error
    return out


def preview_edit(path: str, blocks: str) -> dict[str, Any]:
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        original = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")

    parsed_blocks = _parse_edit_blocks(blocks)
    if isinstance(parsed_blocks, dict) and parsed_blocks.get("status") == "error":
        return parsed_blocks

    assert isinstance(parsed_blocks, list)
    file_newline = _detect_newline_style(original)
    simulated = original
    conflicts: list[dict[str, Any]] = []
    line_ranges: list[tuple[int, int]] = []
    for index, block in enumerate(parsed_blocks):
        search_text = normalize_text_for_matching(block["search"])
        view, boundaries = _normalized_view_boundaries(simulated)
        occurrences = _find_all_occurrences(view, search_text)
        count = len(occurrences)
        if count == 0:
            conflicts.append(
                {
                    "block_index": index,
                    "error": "SEARCH text not found",
                    "search_preview": search_text[:200],
                }
            )
            continue
        if count > 1:
            conflicts.append(
                {
                    "block_index": index,
                    "error": f"SEARCH text found {count} times — must be unique",
                    "search_preview": search_text[:200],
                }
            )
            continue
        idx = occurrences[0]
        start_line = _offset_to_line_number(view, idx)
        end_line = start_line + search_text.count("\n")
        line_ranges.append((start_line, end_line))
        start_offset = boundaries[idx]
        end_offset = boundaries[idx + len(search_text)]
        actual_old = simulated[start_offset:end_offset]
        replacement = _preserve_quote_style(
            block["search"],
            actual_old,
            block["replace"],
        )
        replacement = _coerce_to_newline_style(replacement, file_newline)
        simulated = simulated[:start_offset] + replacement + simulated[end_offset:]

    return {
        "status": "ok" if not conflicts else "conflict",
        "preview": _unified_diff(original, simulated, str(p)),
        "blocks_parsed": len(parsed_blocks),
        "conflicts": conflicts,
        "files_affected": [str(p)],
        "lines_modified": sum(max(0, end - start + 1) for start, end in line_ranges),
    }


def smart_edit(path: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(edits, list) or not edits:
        return _err("edits must be a non-empty list of dicts")

    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    prepared = ensure_snapshot_allows_existing_file_write(
        globals().get("ctx"),
        p,
        operation="edit",
    )
    if isinstance(prepared, dict):
        return prepared
    _, _, original, _ = prepared

    file_newline = _detect_newline_style(original)
    original_lines = original.splitlines(keepends=True)

    parsed: list[dict[str, Any]] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return _err(f"Edit #{index}: must be a dict")

        action = edit.get("action", "replace")
        if action not in ("replace", "insert", "delete"):
            return _err(f"Edit #{index}: invalid action '{action}' (use replace/insert/delete)")

        start = edit.get("start_line", edit.get("line"))
        end = edit.get("end_line", start)
        if not isinstance(start, int) or start < 1:
            return _err(f"Edit #{index}: start_line must be an int >= 1")
        if not isinstance(end, int) or end < start:
            return _err(f"Edit #{index}: end_line must be >= start_line ({start})")

        parsed.append(
            {
                "index": index,
                "action": action,
                "start_line": start,
                "end_line": end,
                "old_text": normalize_text_for_matching(
                    _normalize_agent_content(
                        edit.get("old_text", ""), language_hint=_language_hint_from_path(path)
                    )[0]
                ),
                "new_text": _normalize_agent_content(
                    edit.get("new_text", ""), language_hint=_language_hint_from_path(path)
                )[0],
            }
        )

    conflicts = _detect_edit_conflicts(parsed)
    if conflicts:
        return {
            "status": "error",
            "error": "Overlapping edit ranges detected — fix the ranges and retry",
            "conflicts": conflicts,
            "edits_applied": 0,
        }

    mismatch_errors = []
    for edit in parsed:
        if edit["action"] != "replace" or not edit["old_text"]:
            continue
        start = edit["start_line"] - 1
        end = edit["end_line"]
        actual = "".join(original_lines[start:end]).rstrip("\r\n")
        expected = edit["old_text"].rstrip("\n")
        actual_normalized = normalize_text_for_matching(actual)
        if actual_normalized == expected:
            continue
        ratio = difflib.SequenceMatcher(None, actual_normalized, expected).ratio()
        mismatch_errors.append(
            {
                "edit_index": edit["index"],
                "start_line": edit["start_line"],
                "end_line": edit["end_line"],
                "similarity": f"{ratio:.0%}",
                "expected_first_line": expected.splitlines()[0] if expected else "",
                "actual_first_line": actual.splitlines()[0] if actual else "",
                "actual_context": _build_context_chunk(
                    original_lines,
                    edit["start_line"],
                    edit["end_line"],
                ),
                "closest_matches": _find_closest_blocks(
                    normalize_text_for_matching(original),
                    expected,
                    n=3,
                ),
                "hint": (
                    "old_text does not match the file contents at the specified "
                    "line range. Re-read the file, use read_edit_context(), or correct the range/old_text."
                ),
            }
        )

    if mismatch_errors:
        return {
            "status": "error",
            "error": "old_text mismatch — edit(s) rejected before any write",
            "mismatch_errors": mismatch_errors,
            "edits_applied": 0,
        }

    result_lines = list(original_lines)
    applied = 0
    for edit in sorted(parsed, key=lambda item: item["start_line"], reverse=True):
        start = edit["start_line"] - 1
        end = min(edit["end_line"], len(result_lines))
        action = edit["action"]
        new_text = edit["new_text"]
        if action == "delete":
            result_lines = result_lines[:start] + result_lines[end:]
        elif action == "replace":
            replacement = _coerce_to_newline_style(new_text, file_newline)
            result_lines = result_lines[:start] + _string_to_lines(replacement) + result_lines[end:]
        else:
            insertion = _coerce_to_newline_style(new_text, file_newline)
            result_lines = result_lines[:start] + _string_to_lines(insertion) + result_lines[start:]
        applied += 1

    result = "".join(result_lines)
    final = _coerce_to_newline_style(result, file_newline)
    preview = _unified_diff(original, final, str(p))
    try:
        _atomic_write_text(p, final)
    except OSError as exc:
        return _err(f"Cannot write file: {exc}")

    out: dict[str, Any] = {
        "status": "ok",
        "path": str(p),
        "preview": preview,
        "edits_applied": applied,
        "conflicts": [],
        "message": f"Applied {applied} edit(s) successfully",
    }
    out["structured_patch"] = parse_structured_patch(preview)
    out["snapshot"] = refresh_snapshot_after_write(
        globals().get("ctx"),
        p,
        content=final,
    )
    syntax_error = _validate_syntax(p, final)
    if syntax_error:
        out["syntax_error"] = syntax_error
    return out


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        Path(tmp_name).replace(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_text_preserve_newlines(path: Path) -> str:
    return DEFAULT_FILESYSTEM.read_text(path, encoding="utf-8", errors="replace")


def _validate_syntax(path: Path, content: str) -> dict[str, Any] | None:
    if path.suffix.lower() != ".py" or not content.strip():
        return None
    try:
        ast.parse(content)
        return None
    except SyntaxError as exc:
        return {
            "line": exc.lineno,
            "offset": exc.offset,
            "message": str(exc),
            "hint": "File was written but contains a Python syntax error. Read the reported line and repair it.",
        }


def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _newline_name(newline: str) -> str:
    return {"\n": "LF", "\r\n": "CRLF", "\r": "CR"}.get(newline, "LF")


def _coerce_to_newline_style(text: str, newline: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\n":
        return normalized
    return normalized.replace("\n", newline)


def _language_hint_from_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        return "python"
    return None


def _normalize_agent_content(
    content: str,
    *,
    language_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    return normalize_text_payload(content, language_hint=language_hint)


def _detect_truncation(content: str, path: str) -> str | None:
    if not content:
        return None
    line_count = content.count("\n") + 1
    if line_count <= 2:
        return None
    stripped = content.rstrip()
    if stripped.endswith(("...", "…")):
        return f"Content for {path} may be truncated; verify the full file body before writing."
    return None


def _normalized_view_boundaries(text: str) -> tuple[str, list[int]]:
    i = 1 if text.startswith("\ufeff") else 0
    out: list[str] = []
    boundaries = [i]
    while i < len(text):
        if text.startswith("\r\n", i):
            out.append("\n")
            i += 2
            boundaries.append(i)
            continue
        ch = text[i]
        if ch == "\r":
            out.append("\n")
        else:
            out.append(normalize_text_for_matching(ch))
        i += 1
        boundaries.append(i)
    return "".join(out), boundaries


_LEFT_SINGLE_CURLY_QUOTE = "\u2018"
_RIGHT_SINGLE_CURLY_QUOTE = "\u2019"
_LEFT_DOUBLE_CURLY_QUOTE = "\u201c"
_RIGHT_DOUBLE_CURLY_QUOTE = "\u201d"


def _preserve_quote_style(old_string: str, actual_old_string: str, new_string: str) -> str:
    if old_string == actual_old_string:
        return new_string

    has_double_quotes = (
        _LEFT_DOUBLE_CURLY_QUOTE in actual_old_string
        or _RIGHT_DOUBLE_CURLY_QUOTE in actual_old_string
    )
    has_single_quotes = (
        _LEFT_SINGLE_CURLY_QUOTE in actual_old_string
        or _RIGHT_SINGLE_CURLY_QUOTE in actual_old_string
    )
    if not has_double_quotes and not has_single_quotes:
        return new_string

    result = new_string
    if has_double_quotes:
        result = _apply_curly_double_quotes(result)
    if has_single_quotes:
        result = _apply_curly_single_quotes(result)
    return result


def _is_opening_quote_context(chars: list[str], index: int) -> bool:
    if index == 0:
        return True
    prev = chars[index - 1]
    return prev in {" ", "\t", "\n", "\r", "(", "[", "{", "\u2014", "\u2013"}


def _apply_curly_double_quotes(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    for index, ch in enumerate(chars):
        if ch != '"':
            out.append(ch)
            continue
        out.append(
            _LEFT_DOUBLE_CURLY_QUOTE
            if _is_opening_quote_context(chars, index)
            else _RIGHT_DOUBLE_CURLY_QUOTE
        )
    return "".join(out)


def _apply_curly_single_quotes(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    for index, ch in enumerate(chars):
        if ch != "'":
            out.append(ch)
            continue
        prev = chars[index - 1] if index > 0 else None
        nxt = chars[index + 1] if index + 1 < len(chars) else None
        if prev and nxt and prev.isalpha() and nxt.isalpha():
            out.append(_RIGHT_SINGLE_CURLY_QUOTE)
            continue
        out.append(
            _LEFT_SINGLE_CURLY_QUOTE
            if _is_opening_quote_context(chars, index)
            else _RIGHT_SINGLE_CURLY_QUOTE
        )
    return "".join(out)


def _find_all_occurrences(haystack: str, needle: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def _offset_to_line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _find_closest_blocks(content: str, target: str, n: int = 5) -> list[dict[str, Any]]:
    normalized_target = target.strip("\n")
    target_lines = normalized_target.splitlines()
    if not target_lines:
        return []
    anchor = target_lines[0]
    lines = content.splitlines()
    if not lines:
        return []

    target_line_count = max(1, len(target_lines))
    candidate_lengths = sorted(
        {
            max(1, target_line_count - 1),
            target_line_count,
            target_line_count + 1,
            target_line_count + 2,
        }
    )
    scored: list[tuple[float, float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for candidate_length in candidate_lengths:
        max_start = max(1, len(lines) - candidate_length + 1)
        for start_index in range(max_start):
            end_index = min(len(lines), start_index + candidate_length)
            if end_index <= start_index:
                continue
            key = (start_index, end_index)
            if key in seen:
                continue
            seen.add(key)
            candidate = "\n".join(lines[start_index:end_index]).strip("\n")
            if not candidate:
                continue
            block_ratio = difflib.SequenceMatcher(None, candidate, normalized_target).ratio()
            anchor_ratio = difflib.SequenceMatcher(
                None,
                lines[start_index].strip(),
                anchor.strip(),
            ).ratio()
            if block_ratio <= 0 and anchor_ratio <= 0:
                continue
            scored.append((block_ratio, anchor_ratio, start_index, end_index))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    suggestions: list[dict[str, Any]] = []
    for block_ratio, anchor_ratio, start_index, end_index in scored[:n]:
        suggestions.append(
            _build_chunk_suggestion(
                lines,
                start_index + 1,
                end_index,
                block_ratio,
                anchor_ratio,
            )
        )
    return suggestions


def _build_chunk_suggestion(
    lines: list[str],
    start_line: int,
    end_line: int,
    similarity: float,
    anchor_similarity: float,
    *,
    context_lines: int = 2,
) -> dict[str, Any]:
    chunk = _build_context_chunk(lines, start_line, end_line, context_lines=context_lines)
    chunk["similarity"] = f"{similarity:.0%}"
    chunk["anchor_similarity"] = f"{anchor_similarity:.0%}"
    return chunk


def _build_context_chunk(
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    context_lines: int = 2,
) -> dict[str, Any]:
    total_lines = len(lines)
    bounded_start = max(1, start_line)
    bounded_end = min(total_lines, max(bounded_start, end_line))
    context_start = max(1, bounded_start - context_lines)
    context_end = min(total_lines, bounded_end + context_lines)
    return {
        "start_line": bounded_start,
        "end_line": bounded_end,
        "line_offset": context_start,
        "focus_start_line": bounded_start,
        "focus_end_line": bounded_end,
        "content": "".join(lines[context_start - 1 : context_end]),
        "matched_text": "".join(lines[bounded_start - 1 : bounded_end]),
        "context_before": lines[context_start - 1 : bounded_start - 1],
        "context_after": lines[bounded_end:context_end],
    }


def _parse_edit_blocks(blocks: str) -> list[dict[str, str]] | dict[str, Any]:
    normalized = normalize_text_for_matching(_normalize_agent_content(blocks)[0])
    pattern = re.compile(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.DOTALL,
    )
    parsed = [
        {"search": match.group(1), "replace": match.group(2)}
        for match in pattern.finditer(normalized)
    ]
    if parsed:
        return parsed
    return _err(
        "no valid SEARCH/REPLACE blocks found. Use the exact format:\n"
        "<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE"
    )


def _detect_edit_conflicts(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    ordered = sorted(edits, key=lambda item: (item["start_line"], item["end_line"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start_line"] <= left["end_line"]:
            conflicts.append(
                {
                    "first_edit": left["index"],
                    "second_edit": right["index"],
                    "overlap": [right["start_line"], left["end_line"]],
                }
            )
    return conflicts


def _string_to_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]
    if not text.endswith(("\n", "\r")):
        lines[-1] = lines[-1]
    return lines


def _unified_diff(original: str, updated: str, label: str = "file") -> str:
    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=f"{label} (before)",
            tofile=f"{label} (after)",
            n=3,
        )
    )
