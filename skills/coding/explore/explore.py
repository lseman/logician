"""Coding exploration skill metadata plus non-core search wrappers."""

from __future__ import annotations

import json as _json_mod
from typing import Any, Literal

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:  # type: ignore[misc]
        try:
            return _json_mod.dumps(obj, ensure_ascii=False)
        except Exception:
            return _json_mod.dumps({"status": "error", "error": repr(obj)})


import re
import shutil
import subprocess
from pathlib import Path

from skills.coding.bootstrap.runtime_access import get_coding_runtime, tool

__skill__ = {
    "name": "Explore",
    "description": "Use for codebase exploration, search, outlines, symbol lookup, and structural inspection.",
    "aliases": ["code search", "codebase map", "symbol search", "inspect structure"],
    "triggers": [
        "find where this is defined",
        "search the codebase",
        "inspect the project structure",
        "find references to this symbol",
    ],
    "preferred_tools": [
        "get_project_map",
        "find_symbol",
        "get_file_outline",
        "rg_search",
        "fd_find",
    ],
    "example_queries": [
        "where is this class defined",
        "find references to this helper",
        "map the project structure before editing",
    ],
    "when_not_to_use": ["the exact file and edit location are already known"],
    "next_skills": ["file_ops", "edit_block", "multi_edit", "search_replace"],
    "preferred_sequence": ["get_project_map", "find_symbol", "get_file_outline", "read_file"],
    "entry_criteria": [
        "The repo is unfamiliar or the edit target is still unclear.",
        "The user names a symbol, behavior, or error, but not the exact file and line.",
    ],
    "decision_rules": [
        "Start with a cheap map or search before opening long files.",
        "Use structural inspection before deep reads when the file is large.",
        "Once the target is clear, hand off quickly to the editing skill that fits the change.",
    ],
    "workflow": [
        "Start here when the task touches unfamiliar code.",
        "Prefer narrow search first, broader map second.",
        "Hand off to file_ops, edit_block, or multi_edit once the target is clear.",
    ],
    "failure_recovery": [
        "If search results are noisy, narrow by directory, extension, or symbol name.",
        "If a file is too large to read directly, use outlines and focused search to reduce context first.",
    ],
    "exit_criteria": [
        "A concrete target file, symbol, or call site has been identified.",
        "The next editing or verification step is unambiguous.",
    ],
    "anti_patterns": [
        "Reading large files end-to-end before trying outlines or symbol search.",
        "Staying in exploration after the edit target is already known.",
    ],
}


def _query_terms(text: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9_]+", str(text or "").lower()) if part]


def _path_relevance(path: str, query: str) -> int:
    rel = Path(path)
    path_text = rel.as_posix().lower()
    name = rel.name.lower()
    stem = rel.stem.lower()
    score = 0
    for term in _query_terms(query):
        if stem == term:
            score += 14
        elif name == term or name.startswith(f"{term}."):
            score += 12
        if f"/{term}/" in f"/{path_text}/":
            score += 6
        elif term in path_text:
            score += 3
    depth = len(rel.parts)
    score += max(0, 4 - min(depth, 4))
    if any(part.lower() in {"test", "tests", "__tests__", "spec"} for part in rel.parts):
        score -= 4
    return score


def _rollup_matches_by_file(matches: list[dict], *, limit: int = 8) -> list[dict]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in matches:
        file = str(item.get("file", ""))
        if not file:
            continue
        bucket = grouped.setdefault(
            file,
            {
                "file": file,
                "score": 0,
                "count": 0,
                "first_line": int(item.get("line", 0) or 0),
                "preview": "",
            },
        )
        bucket["score"] = max(int(bucket["score"]), int(item.get("score", 0)))
        if item.get("match", True):
            bucket["count"] += 1
        if int(item.get("line", 0) or 0) and (
            bucket["first_line"] == 0 or int(item.get("line", 0)) < int(bucket["first_line"])
        ):
            bucket["first_line"] = int(item.get("line", 0))
        if not bucket["preview"]:
            bucket["preview"] = str(item.get("text") or "").strip()[:160]

    out: list[dict[str, Any]] = []
    for item in grouped.values():
        out.append(
            {
                "file": item["file"],
                "count": item["count"],
                "first_line": item["first_line"],
                "preview": item["preview"],
                "score": item["score"],
            }
        )
    out.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            -int(item.get("count", 0)),
            str(item.get("file", "")),
        )
    )
    for item in out:
        item.pop("score", None)
    return out[:limit]


def _rank_search_matches(
    matches: list[dict],
    *,
    query: str,
    fixed_string: bool,
    case_sensitive: bool,
) -> list[dict]:
    ranked: list[dict] = []
    query_text = str(query or "")
    query_lower = query_text.lower()
    for item in matches:
        score = _path_relevance(str(item.get("file", "")), query_text)
        if item.get("match", True):
            score += 20
        text = str(item.get("text", ""))
        text_cmp = text if case_sensitive else text.lower()
        needle = query_text if case_sensitive else query_lower
        if fixed_string and needle and needle in text_cmp:
            score += 10
            if text.strip().startswith(query_text):
                score += 3
        elif not fixed_string and query_lower and query_lower in text.lower():
            score += 4
        ranked_item = dict(item)
        ranked_item["score"] = score
        ranked.append(ranked_item)
    ranked.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            str(item.get("file", "")),
            int(item.get("line", 0)),
            0 if item.get("match", True) else 1,
        )
    )
    return ranked


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _coerce_int(value: Any, *, name: str, default: int) -> tuple[int, str | None]:
    if value is None:
        return default, None
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        if value.is_integer():
            return int(value), None
        return 0, f"Invalid {name}: expected integer, got {value!r}"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default, None
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text), None
        return 0, f"Invalid {name}: expected integer, got {value!r}"
    return 0, f"Invalid {name}: expected integer, got {type(value).__name__}"


# ---------------------------------------------------------------------------
# Internal: run-command helper (reuse bootstrap's if available, else fallback)
# ---------------------------------------------------------------------------


def _explore_run(cmd: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a command using the shared coding runtime, else subprocess directly."""
    _rt = get_coding_runtime(globals())
    if _rt is not None:
        try:
            return _rt.run_cmd(cmd, cwd=cwd, timeout=timeout)
        except RuntimeError:
            pass
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "exit_code": -1, "stdout": "", "stderr": "timeout"}
    except Exception as exc:
        return {"status": "error", "exit_code": -1, "stdout": "", "stderr": str(exc)}


# ---------------------------------------------------------------------------
# rg_search / fd_find  (system-tool wrappers with pure-Python fallback)
# ---------------------------------------------------------------------------


@tool
def rg_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "",
    file_type: str = "",
    context_lines: int = 0,
    case_sensitive: bool = False,
    fixed_string: bool = False,
    max_results: int = 80,
) -> str:
    """Use when: Search for any text/pattern across a codebase quickly.

    Triggers: search code, find text, grep, rg, ripgrep, where is string, find pattern,
              find all uses, search files, look for, locate text, who calls.
    Avoid when: You want symbol definitions only — use find_symbol instead.
    Inputs:
      pattern (str, required): Regex or literal string to search for.
      directory (str, optional): Root directory (default ".").
      file_glob (str, optional): Glob pattern to restrict files, e.g. "*.py" or "src/**/*.ts".
      file_type (str, optional): ripgrep file type name, e.g. "py", "js", "ts", "rust", "md".
                                 Only used when rg is available. Ignored in fallback mode.
      context_lines (int, optional): Lines of context before/after each match (default 0).
      case_sensitive (bool, optional): Match case exactly (default False = case-insensitive).
      fixed_string (bool, optional): Treat pattern as literal string, not regex (default False).
      max_results (int, optional): Stop after this many matches (default 80).
    Returns: JSON with matches list {file, line, text} plus tool_used ("rg" or "python").
    Side effects: Read-only; spawns subprocess if rg is available.

    Examples:
      rg_search("ThinkingConfig")                    -- find all uses of a symbol
      rg_search("TODO", file_type="py")               -- find all Python TODOs
      rg_search("def run", context_lines=2)           -- show 2 lines around each match
      rg_search("import torch", fixed_string=True)    -- exact literal search
    """
    pattern = str(pattern)
    directory = str(directory or ".")
    file_glob = str(file_glob or "")
    file_type = str(file_type or "")
    case_sensitive = _coerce_bool(case_sensitive, default=False)
    fixed_string = _coerce_bool(fixed_string, default=False)
    context_lines_i, err = _coerce_int(context_lines, name="context_lines", default=0)
    if err:
        return _safe_json({"status": "error", "error": err})
    max_results_i, err = _coerce_int(max_results, name="max_results", default=80)
    if err:
        return _safe_json({"status": "error", "error": err})
    context_lines = max(0, context_lines_i)
    max_results = max(1, max_results_i)

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    # ---- Try ripgrep first ----
    if shutil.which("rg"):
        parts = ["rg", "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            parts.append("--ignore-case")
        if fixed_string:
            parts.append("--fixed-strings")
        if context_lines > 0:
            parts += ["-C", str(context_lines)]
        if file_type:
            parts += ["--type", file_type]
        if file_glob:
            parts += ["--glob", file_glob]
        parts += ["--max-count", str(max_results)]
        parts.append(pattern)
        parts.append(str(root))

        cmd = " ".join(p if re.match(r"^[\w./:@=+,-]+$", p) else f"'{p}'" for p in parts)
        r = _explore_run(cmd, timeout=30)

        raw_lines = r["stdout"].splitlines()
        matches: list[dict] = []
        # rg --no-heading format: "filepath:lineno:text"  or  "filepath-lineno-text" for context
        line_re = re.compile(r"^(.+?):(\d+):(.*)$")
        ctx_re = re.compile(r"^(.+?)-(\d+)-(.*)$")
        for raw in raw_lines:
            m = line_re.match(raw)
            if m:
                matches.append(
                    {
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "text": m.group(3),
                        "match": True,
                    }
                )
            elif context_lines > 0:
                mc = ctx_re.match(raw)
                if mc:
                    matches.append(
                        {
                            "file": mc.group(1),
                            "line": int(mc.group(2)),
                            "text": mc.group(3),
                            "match": False,
                        }
                    )

        # Make file paths relative to root when possible
        for entry in matches:
            try:
                entry["file"] = str(Path(entry["file"]).relative_to(root))
            except ValueError:
                pass

        matches = _rank_search_matches(
            matches,
            query=pattern,
            fixed_string=fixed_string,
            case_sensitive=case_sensitive,
        )
        real_count = len([m for m in matches if m.get("match", True)])
        truncated = real_count >= max_results
        return _safe_json(
            {
                "status": "ok",
                "tool_used": "rg",
                "pattern": pattern,
                "count": real_count,
                "truncated": truncated,
                "top_files": _rollup_matches_by_file(matches, limit=8),
                "matches": matches[:max_results],
            }
        )

    # ---- Pure-Python fallback ----
    glob_pat = file_glob if file_glob else "**/*"
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(re.escape(pattern) if fixed_string else pattern, flags)
    except re.error as exc:
        return _safe_json({"status": "error", "error": f"Invalid regex: {exc}"})

    matches = []
    for fpath in sorted(root.glob(glob_pat)):
        if not fpath.is_file():
            continue
        try:
            src_lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, text in enumerate(src_lines, 1):
            if rx.search(text):
                entry = {
                    "file": str(fpath.relative_to(root)),
                    "line": lineno,
                    "text": text.rstrip(),
                    "match": True,
                }
                matches.append(entry)
                if context_lines > 0:
                    for delta in range(1, context_lines + 1):
                        for sign, before in ((-delta, True), (delta, False)):
                            idx = lineno + sign - 1
                            if 0 <= idx < len(src_lines):
                                ctx_entry = {
                                    "file": str(fpath.relative_to(root)),
                                    "line": idx + 1,
                                    "text": src_lines[idx].rstrip(),
                                    "match": False,
                                }
                                if before:
                                    matches.insert(-1, ctx_entry)
                                else:
                                    matches.append(ctx_entry)
                if len([m for m in matches if m.get("match")]) >= max_results:
                    break
        if len([m for m in matches if m.get("match")]) >= max_results:
            break

    matches = _rank_search_matches(
        matches,
        query=pattern,
        fixed_string=fixed_string,
        case_sensitive=case_sensitive,
    )
    real_count = len([m for m in matches if m.get("match")])
    return _safe_json(
        {
            "status": "ok",
            "tool_used": "python",
            "pattern": pattern,
            "count": real_count,
            "truncated": real_count >= max_results,
            "top_files": _rollup_matches_by_file(matches, limit=8),
            "matches": matches,
        }
    )


@tool
def find_references(name: str, directory: str = ".", file_glob: str = "") -> str:
    """Use when: You need to locate where a specific symbol is used or called.

    Triggers: find references, usages of, who calls, where is used, find usages.
    Avoid when: You want to find the definition (use find_symbol instead).
    Inputs:
      name (str, required): The exact name of the symbol (e.g. 'compute_statistics').
      directory (str, optional): Root folder to search in (default '.').
      file_glob (str, optional): Glob to filter files (default '').
    Returns: JSON with matches and snippets.
    """
    pattern = rf"\b{name}\b"
    return rg_search(
        pattern=pattern, directory=directory, file_glob=file_glob, context_lines=1, max_results=30
    )


@tool
def fd_find(
    pattern: str,
    directory: str = ".",
    file_type: Literal["f", "d", ""] = "f",
    extension: str = "",
    max_depth: int = 8,
    max_results: int = 50,
    hidden: bool = False,
) -> str:
    """Use when: You want to find files or directories by name pattern across the project.

    Triggers: find file, locate file, where is file, search for file, find directory,
              which file is named, fd, find by name, file search, where lives.
    Avoid when: You want to search file *contents* — use rg_search or search_in_files.
    Inputs:
      pattern (str, required): Name pattern to match. fd uses smart-case regex, e.g.
                               "config", "test_.*\\.py", "__init__".
      directory (str, optional): Root directory to search (default ".").
      file_type (str, optional): "f" = files only (default), "d" = directories only, "" = both.
      extension (str, optional): Filter by extension, e.g. "py", "json", "md" (no dot).
      max_depth (int, optional): Maximum directory depth (default 8).
      max_results (int, optional): Stop after N results (default 50).
      hidden (bool, optional): Include hidden files/dirs (default False).
    Returns: JSON with list of matching paths relative to directory.
    Side effects: Read-only.

    Examples:
      fd_find("config")                     -- find all files with 'config' in name
      fd_find("test_", extension="py")      -- find all test_*.py files
      fd_find("__pycache__", file_type="d") -- find all __pycache__ dirs
      fd_find("README", extension="md")     -- find README.md files
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

    # ---- Try fd first ----
    if shutil.which("fd"):
        parts = ["fd", "--color=never", "--max-depth", str(max_depth)]
        if file_type in ("f", "d"):
            parts += ["--type", file_type]
        if extension:
            parts += ["--extension", extension.lstrip(".")]
        if hidden:
            parts.append("--hidden")
        parts.append(pattern)
        parts.append(str(root))

        cmd = " ".join(p if re.match(r"^[\w./:@=+,-]+$", p) else f"'{p}'" for p in parts)
        r = _explore_run(cmd, timeout=20)
        raw_paths = [ln.strip() for ln in r["stdout"].splitlines() if ln.strip()]

        # Make relative
        rel_paths = []
        for rp in raw_paths[:max_results]:
            try:
                rel_paths.append(str(Path(rp).relative_to(root)))
            except ValueError:
                rel_paths.append(rp)

        return _safe_json(
            {
                "status": "ok",
                "tool_used": "fd",
                "pattern": pattern,
                "count": len(rel_paths),
                "truncated": len(raw_paths) > max_results,
                "paths": rel_paths,
            }
        )

    # ---- Pure-Python fallback ----
    # Build a glob pattern from the extension + max_depth
    if extension:
        ext = extension.lstrip(".")
        glob_pats = [f"{'*/' * d}*.{ext}" for d in range(max_depth + 1)]
    else:
        glob_pats = [f"{'*/' * d}*" for d in range(max_depth + 1)]

    try:
        name_re = re.compile(pattern, re.IGNORECASE)
    except re.error:
        name_re = re.compile(re.escape(pattern), re.IGNORECASE)

    found: list[str] = []
    seen: set[str] = set()
    for gp in glob_pats:
        for fpath in sorted(root.glob(gp)):
            if not hidden and any(part.startswith(".") for part in fpath.parts):
                continue
            is_dir = fpath.is_dir()
            if file_type == "f" and is_dir:
                continue
            if file_type == "d" and not is_dir:
                continue
            if name_re.search(fpath.name):
                rel = str(fpath.relative_to(root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)
                    if len(found) >= max_results:
                        break
        if len(found) >= max_results:
            break

    return _safe_json(
        {
            "status": "ok",
            "tool_used": "python",
            "pattern": pattern,
            "count": len(found),
            "truncated": len(found) >= max_results,
            "paths": found,
        }
    )


__tools__ = [rg_search, find_references, fd_find]
