"""Core search tools backed by the shared filesystem backend."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from ..filesystem import DEFAULT_FILESYSTEM
from .inspection import (
    _err,
    _read_text_preserve_newlines,
    record_file_snapshot,
    resolve_tool_path,
)
from .inspection import (
    search_code as inspection_search_code,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _file_mtime(path: str, root: Path) -> float:
    """Return mtime for a file path (relative to root or absolute). 0.0 on error."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _rollup_matches_by_file(
    matches: list[dict], *, limit: int = 8, root: Path | None = None
) -> list[dict]:
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
                "mtime": 0.0,
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
        if bucket["mtime"] == 0.0 and root is not None:
            bucket["mtime"] = _file_mtime(file, root)

    out: list[dict[str, Any]] = []
    for item in grouped.values():
        out.append(
            {
                "file": item["file"],
                "count": item["count"],
                "first_line": item["first_line"],
                "preview": item["preview"],
                "score": item["score"],
                "mtime": item["mtime"],
            }
        )
    out.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            -float(item.get("mtime", 0.0)),
            -int(item.get("count", 0)),
            str(item.get("file", "")),
        )
    )
    for item in out:
        item.pop("score", None)
        item.pop("mtime", None)
    return out[:limit]


def _rank_search_matches(
    matches: list[dict],
    *,
    query: str,
    fixed_string: bool,
    case_sensitive: bool,
    root: Path | None = None,
) -> list[dict]:
    mtime_cache: dict[str, float] = {}
    ranked: list[dict] = []
    query_text = str(query or "")
    query_lower = query_text.lower()

    for item in matches:
        file = str(item.get("file", ""))
        score = _path_relevance(file, query_text)
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

        if file not in mtime_cache and root is not None:
            mtime_cache[file] = _file_mtime(file, root)
        mtime = mtime_cache.get(file, 0.0)

        ranked_item = dict(item)
        ranked_item["score"] = score
        ranked_item["_mtime"] = mtime
        ranked.append(ranked_item)

    ranked.sort(
        key=lambda item: (
            -int(item.get("score", 0)),
            -float(item.get("_mtime", 0.0)),
            str(item.get("file", "")),
            int(item.get("line", 0)),
            0 if item.get("match", True) else 1,
        )
    )
    for item in ranked:
        item.pop("_mtime", None)
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


def _run_cmd(cmd: str, cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a shell command, return {status, exit_code, stdout, stderr}."""
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


def _build_search_result(
    *,
    tool_used: str,
    pattern: str,
    output_mode: str,
    matches: list[dict],
    real_count: int,
    truncated: bool,
    top_files: list[dict],
    max_results: int,
    root: Path,
) -> dict:
    base: dict[str, Any] = {
        "status": "ok",
        "tool_used": tool_used,
        "pattern": pattern,
        "count": real_count,
        "truncated": truncated,
    }
    if truncated:
        base["hint"] = (
            f"Results truncated at {max_results}. "
            "Narrow with file_glob/file_type or increase max_results."
        )

    if output_mode == "files":
        seen: dict[str, float] = {}
        for m in matches:
            if not m.get("match", True):
                continue
            f = str(m.get("file", ""))
            if f and f not in seen:
                seen[f] = _file_mtime(f, root)
        sorted_files = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        base["files"] = [f for f, _ in sorted_files]
        base["file_count"] = len(base["files"])
        return base

    if output_mode == "count":
        counts: dict[str, int] = {}
        for m in matches:
            if not m.get("match", True):
                continue
            f = str(m.get("file", ""))
            if f:
                counts[f] = counts.get(f, 0) + 1
        sorted_counts = sorted(
            counts.items(),
            key=lambda kv: (-_file_mtime(kv[0], root), -kv[1], kv[0]),
        )
        base["per_file"] = [{"file": f, "count": c} for f, c in sorted_counts]
        return base

    base["top_files"] = top_files
    base["matches"] = matches[:max_results]
    return base


# ---------------------------------------------------------------------------
# rg_search
# ---------------------------------------------------------------------------


def rg_search(
    pattern: str,
    directory: str = ".",
    file_glob: str = "",
    file_type: str = "",
    context_lines: int = 0,
    case_sensitive: bool = False,
    fixed_string: bool = False,
    max_results: int = 80,
    output_mode: str = "content",
) -> dict:
    """Search for text/pattern across a codebase using ripgrep (pure-Python fallback).

    Args:
        pattern: Regex or literal string to search for.
        directory: Root directory to search (default ".").
        file_glob: Glob pattern to restrict files, e.g. "*.py" or "src/**/*.ts".
        file_type: ripgrep file-type name ("py", "js", "ts", "rust", "md" …).
            Only used when rg is available.
        context_lines: Lines of context before/after each match (default 0).
        case_sensitive: Match case exactly (default False).
        fixed_string: Treat pattern as literal string, not regex (default False).
        max_results: Stop after this many matches (default 80).
        output_mode: One of:
            "content" — full match lines + context (default)
            "files"   — unique file paths sorted by mtime; token-cheap
            "count"   — match count per file, sorted by mtime then count

    Returns:
        dict with matches and top_files sorted by modification time (newest first).
    """
    pattern = str(pattern)
    directory = str(directory or ".")
    file_glob = str(file_glob or "")
    file_type = str(file_type or "")
    output_mode = str(output_mode or "content").strip().lower()
    if output_mode not in {"content", "files", "count"}:
        output_mode = "content"
    case_sensitive = _coerce_bool(case_sensitive, default=False)
    fixed_string = _coerce_bool(fixed_string, default=False)

    context_lines_i, err = _coerce_int(context_lines, name="context_lines", default=0)
    if err:
        return {"status": "error", "error": err}
    max_results_i, err = _coerce_int(max_results, name="max_results", default=80)
    if err:
        return {"status": "error", "error": err}
    context_lines = max(0, context_lines_i)
    max_results = max(1, max_results_i)

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return {"status": "error", "error": f"Not a directory: {directory}"}

    if shutil.which("rg"):
        parts = ["rg", "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            parts.append("--ignore-case")
        if fixed_string:
            parts.append("--fixed-strings")
        if context_lines > 0 and output_mode == "content":
            parts += ["-C", str(context_lines)]
        if file_type:
            parts += ["--type", file_type]
        if file_glob:
            parts += ["--glob", file_glob]
        parts += ["--max-count", str(max_results)]
        parts.append(pattern)
        parts.append(str(root))

        cmd = " ".join(p if re.match(r"^[\w./:@=+,-]+$", p) else f"'{p}'" for p in parts)
        r = _run_cmd(cmd, timeout=30)

        raw_lines = r["stdout"].splitlines()
        matches: list[dict] = []
        line_re = re.compile(r"^(.+?):(\d+):(.*)$")
        ctx_re = re.compile(r"^(.+?)-(\d+)-(.*)$")
        for raw in raw_lines:
            m = line_re.match(raw)
            if m:
                matches.append(
                    {"file": m.group(1), "line": int(m.group(2)), "text": m.group(3), "match": True}
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
            root=root,
        )
        real_count = len([m for m in matches if m.get("match", True)])
        truncated = real_count >= max_results
        top_files = _rollup_matches_by_file(matches, limit=8, root=root)
        return _build_search_result(
            tool_used="rg",
            pattern=pattern,
            output_mode=output_mode,
            matches=matches,
            real_count=real_count,
            truncated=truncated,
            top_files=top_files,
            max_results=max_results,
            root=root,
        )

    glob_pat = file_glob if file_glob else "**/*"
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(re.escape(pattern) if fixed_string else pattern, flags)
    except re.error as exc:
        return {"status": "error", "error": f"Invalid regex: {exc}"}

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
                if context_lines > 0 and output_mode == "content":
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
                                matches.insert(-1, ctx_entry) if before else matches.append(
                                    ctx_entry
                                )
                if len([m for m in matches if m.get("match")]) >= max_results:
                    break
        if len([m for m in matches if m.get("match")]) >= max_results:
            break

    matches = _rank_search_matches(
        matches, query=pattern, fixed_string=fixed_string, case_sensitive=case_sensitive, root=root
    )
    real_count = len([m for m in matches if m.get("match")])
    top_files = _rollup_matches_by_file(matches, limit=8, root=root)
    return _build_search_result(
        tool_used="python",
        pattern=pattern,
        output_mode=output_mode,
        matches=matches,
        real_count=real_count,
        truncated=real_count >= max_results,
        top_files=top_files,
        max_results=max_results,
        root=root,
    )


# ---------------------------------------------------------------------------
# find_references
# ---------------------------------------------------------------------------


def find_references(name: str, directory: str = ".", file_glob: str = "") -> dict:
    """Find where a symbol is used (word-boundary match, 1 line context).

    Args:
        name: Exact symbol name to find usages of.
        directory: Root folder to search (default ".").
        file_glob: Glob to filter files (default "").

    Returns:
        Same dict format as rg_search.
    """
    return rg_search(
        pattern=rf"\b{name}\b",
        directory=directory,
        file_glob=file_glob,
        context_lines=1,
        max_results=30,
    )


# ---------------------------------------------------------------------------
# fd_find
# ---------------------------------------------------------------------------


def fd_find(
    pattern: str,
    directory: str = ".",
    file_type: Literal["f", "d", ""] = "f",
    extension: str = "",
    max_depth: int = 8,
    max_results: int = 50,
    hidden: bool = False,
) -> dict:
    """Find files or directories by name pattern (fd with pure-Python fallback).

    Results are sorted by modification time (newest first).

    Args:
        pattern: Name pattern to match. fd uses smart-case regex.
        directory: Root directory to search (default ".").
        file_type: "f" = files only (default), "d" = directories only, "" = both.
        extension: Filter by extension e.g. "py", "json" (no dot).
        max_depth: Maximum directory depth (default 8).
        max_results: Stop after N results (default 50).
        hidden: Include hidden files/dirs (default False).

    Returns:
        dict with paths sorted by mtime descending.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return {"status": "error", "error": f"Not a directory: {directory}"}

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
        r = _run_cmd(cmd, timeout=20)
        raw_paths = [ln.strip() for ln in r["stdout"].splitlines() if ln.strip()]

        rel_paths_with_mtime: list[tuple[str, float]] = []
        for rp in raw_paths:
            try:
                rel = str(Path(rp).relative_to(root))
            except ValueError:
                rel = rp
            rel_paths_with_mtime.append((rel, _file_mtime(rel, root)))

        rel_paths_with_mtime.sort(key=lambda x: (-x[1], x[0]))
        truncated = len(rel_paths_with_mtime) > max_results
        return {
            "status": "ok",
            "tool_used": "fd",
            "pattern": pattern,
            "count": min(len(rel_paths_with_mtime), max_results),
            "truncated": truncated,
            "paths": [p for p, _ in rel_paths_with_mtime[:max_results]],
        }

    if extension:
        ext = extension.lstrip(".")
        glob_pats = [f"{'*/' * d}*.{ext}" for d in range(max_depth + 1)]
    else:
        glob_pats = [f"{'*/' * d}*" for d in range(max_depth + 1)]

    try:
        name_re = re.compile(pattern, re.IGNORECASE)
    except re.error:
        name_re = re.compile(re.escape(pattern), re.IGNORECASE)

    collect_limit = max_results * 3
    found_with_mtime: list[tuple[str, float]] = []
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
                    found_with_mtime.append((rel, _file_mtime(rel, root)))
                    if len(found_with_mtime) >= collect_limit:
                        break
        if len(found_with_mtime) >= collect_limit:
            break

    found_with_mtime.sort(key=lambda x: (-x[1], x[0]))
    truncated = len(found_with_mtime) > max_results
    return {
        "status": "ok",
        "tool_used": "python",
        "pattern": pattern,
        "count": min(len(found_with_mtime), max_results),
        "truncated": truncated,
        "paths": [p for p, _ in found_with_mtime[:max_results]],
    }


# ---------------------------------------------------------------------------
# search_symbols — multi-language AST symbol search via tree-sitter
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".lua": "lua",
}

_SYMBOL_TYPES: dict[str, list[tuple[str, str]]] = {
    "python": [
        ("function_definition", "function"),
        ("async_function_definition", "async_function"),
        ("class_definition", "class"),
    ],
    "rust": [
        ("function_item", "function"),
        ("struct_item", "struct"),
        ("enum_item", "enum"),
        ("trait_item", "trait"),
        ("impl_item", "impl"),
        ("type_alias", "type"),
        ("static_item", "static"),
        ("const_item", "const"),
        ("macro_definition", "macro"),
    ],
    "javascript": [
        ("function_declaration", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("generator_function_declaration", "generator"),
        ("lexical_declaration", "const/let"),
        ("variable_declaration", "var"),
    ],
    "typescript": [
        ("function_declaration", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("generator_function_declaration", "generator"),
        ("interface_declaration", "interface"),
        ("type_alias_declaration", "type"),
        ("enum_declaration", "enum"),
        ("lexical_declaration", "const/let"),
        ("abstract_class_declaration", "abstract_class"),
    ],
    "go": [
        ("function_declaration", "function"),
        ("method_declaration", "method"),
        ("type_declaration", "type"),
        ("const_declaration", "const"),
        ("var_declaration", "var"),
    ],
    "ruby": [
        ("method", "method"),
        ("singleton_method", "method"),
        ("class", "class"),
        ("module", "module"),
    ],
    "java": [
        ("class_declaration", "class"),
        ("method_declaration", "method"),
        ("interface_declaration", "interface"),
        ("enum_declaration", "enum"),
        ("constructor_declaration", "constructor"),
    ],
    "c": [
        ("function_definition", "function"),
        ("struct_specifier", "struct"),
        ("enum_specifier", "enum"),
        ("type_definition", "typedef"),
    ],
    "cpp": [
        ("function_definition", "function"),
        ("class_specifier", "class"),
        ("struct_specifier", "struct"),
        ("enum_specifier", "enum"),
        ("template_declaration", "template"),
    ],
    "lua": [
        ("function_declaration", "function"),
        ("local_function", "function"),
        ("assignment_statement", "assignment"),
    ],
}

_ts_parser_cache: dict[str, Any] = {}


def _get_ts_parser(lang: str) -> Any:
    if lang in _ts_parser_cache:
        return _ts_parser_cache[lang]
    try:
        from tree_sitter_language_pack import get_parser as _get_parser

        parser = _get_parser(lang)
    except Exception:
        parser = None
    _ts_parser_cache[lang] = parser
    return parser


def _extract_name_from_node(node: Any, lang: str) -> str | None:
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier"):
            return child.text.decode("utf-8", errors="replace") if child.text else None
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        return (
                            grandchild.text.decode("utf-8", errors="replace")
                            if grandchild.text
                            else None
                        )
    if node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                for grandchild in child.children:
                    if grandchild.type == "type_identifier":
                        return (
                            grandchild.text.decode("utf-8", errors="replace")
                            if grandchild.text
                            else None
                        )
    return None


def _walk_ts_tree(
    node: Any, lang: str, symbol_type_set: set[str]
) -> list[tuple[str, str, int, int]]:
    results = []
    kind_map = {st: k for st, k in _SYMBOL_TYPES.get(lang, [])}
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in symbol_type_set:
            name = _extract_name_from_node(current, lang)
            if name:
                kind = kind_map.get(current.type, current.type)
                results.append((name, kind, current.start_point[0] + 1, current.end_point[0] + 1))
        stack.extend(reversed(current.children))
    return results


def search_symbols(
    name: str,
    directory: str = ".",
    file_glob: str = "",
    language: str = "",
    kind: str = "",
    case_sensitive: bool = False,
    max_results: int = 50,
) -> dict:
    """Find function/class/struct/method/type definitions by name using tree-sitter AST parsing.

    Supports: Python, Rust, JavaScript, TypeScript, Go, Ruby, Java, C, C++, Lua.
    Results sorted by file modification time (newest first).

    Args:
        name: Symbol name or substring to match (e.g. "parse", "MyClass").
        directory: Root directory to search (default ".").
        file_glob: Glob to restrict files, e.g. "src/**/*.rs", "*.py".
        language: Restrict to a language: "python", "rust", "typescript",
            "javascript", "go", "ruby", "java", "c", "cpp", "lua".
        kind: Filter by symbol kind: "function", "class", "struct", "method",
            "enum", "trait", "type", "impl", "const", "interface", etc.
        case_sensitive: Case-sensitive name matching (default False).
        max_results: Stop after N symbols (default 50).

    Returns:
        dict with symbols [{file, name, kind, line, end_line, language}].
    """
    name = str(name or "")
    if not name:
        return {"status": "error", "error": "name must not be empty"}
    directory = str(directory or ".")
    file_glob = str(file_glob or "")
    language = str(language or "").strip().lower()
    kind_filter = str(kind or "").strip().lower()
    case_sensitive = _coerce_bool(case_sensitive, default=False)

    max_results_i, err = _coerce_int(max_results, name="max_results", default=50)
    if err:
        return {"status": "error", "error": err}
    max_results = max(1, max_results_i)

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return {"status": "error", "error": f"Not a directory: {directory}"}

    if language:
        ext_set = {ext for ext, lang in _LANG_MAP.items() if lang == language}
        if not ext_set:
            return {
                "status": "error",
                "error": f"Unknown language: {language}. Supported: {sorted(set(_LANG_MAP.values()))}",
            }
    else:
        ext_set = set(_LANG_MAP.keys())

    candidates = sorted(root.glob(file_glob)) if file_glob else sorted(root.rglob("*"))
    needle = name if case_sensitive else name.lower()

    symbols: list[dict] = []
    files_searched = 0
    files_skipped_no_parser = 0

    for fpath in candidates:
        if not fpath.is_file():
            continue
        ext = fpath.suffix.lower()
        if ext not in ext_set:
            continue

        lang = _LANG_MAP[ext]
        parser = _get_ts_parser(lang)
        if parser is None:
            files_skipped_no_parser += 1
            continue

        try:
            raw = fpath.read_bytes()
        except OSError:
            continue
        if len(raw) > 2 * 1024 * 1024:
            continue

        try:
            tree = parser.parse(raw)
        except Exception:
            continue

        files_searched += 1
        file_rel = str(fpath.relative_to(root)) if fpath.is_relative_to(root) else str(fpath)

        lang_symbols = _SYMBOL_TYPES.get(lang, [])
        symbol_type_set = (
            {st for st, k in lang_symbols if kind_filter in k.lower()}
            if kind_filter
            else {st for st, _ in lang_symbols}
        )
        if not symbol_type_set:
            continue

        for sym_name, sym_kind, start_line, end_line in _walk_ts_tree(
            tree.root_node, lang, symbol_type_set
        ):
            haystack = sym_name if case_sensitive else sym_name.lower()
            if needle not in haystack:
                continue
            symbols.append(
                {
                    "file": file_rel,
                    "name": sym_name,
                    "kind": sym_kind,
                    "line": start_line,
                    "end_line": end_line,
                    "language": lang,
                }
            )
            if len(symbols) >= max_results * 3:
                break
        if len(symbols) >= max_results * 3:
            break

    mtime_cache: dict[str, float] = {}
    for s in symbols:
        f = s["file"]
        if f not in mtime_cache:
            mtime_cache[f] = _file_mtime(f, root)

    symbols.sort(key=lambda s: (-mtime_cache.get(s["file"], 0.0), s["file"], s["line"]))
    truncated = len(symbols) > max_results
    return {
        "status": "ok",
        "query": name,
        "kind_filter": kind_filter or None,
        "language_filter": language or None,
        "files_searched": files_searched,
        "files_skipped_no_parser": files_skipped_no_parser,
        "count": len(symbols[:max_results]),
        "truncated": truncated,
        "symbols": symbols[:max_results],
    }


def glob_files(
    pattern: str,
    path: str = ".",
    include_hidden: bool = False,
    offset: int = 0,
    max_results: int | None = None,
    output_mode: str = "entries",
) -> dict[str, Any]:
    """Find files matching a glob pattern under a directory.

    Returns structured metadata for each match instead of a plain newline string.

    Args:
        pattern: Glob pattern such as ``"**/*.py"`` or ``"*.md"``.
        path: Base directory to search.
    """
    return DEFAULT_FILESYSTEM.glob(
        pattern=pattern,
        path=path,
        include_hidden=include_hidden,
        offset=offset,
        max_results=max_results,
        output_mode=output_mode,
    )


def grep_files(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    literal: bool = True,
    case_sensitive: bool = True,
    context_lines: int = 1,
    max_matches: int = 50,
    include_hidden: bool = False,
    offset: int = 0,
    output_mode: str = "matches",
) -> dict[str, Any]:
    """Search for a pattern across files in a directory tree."""
    return DEFAULT_FILESYSTEM.grep(
        pattern,
        path=path,
        glob=glob,
        literal=literal,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_matches=max_matches,
        include_hidden=include_hidden,
        offset=offset,
        output_mode=output_mode,
    )


def search_file(
    path: str,
    pattern: str,
    literal: bool = True,
    case_sensitive: bool = True,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Search for text or regex matches in a single file."""
    try:
        p = resolve_tool_path(path)
    except ValueError as exc:
        return _err(str(exc))
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")

    try:
        text = _read_text_preserve_newlines(p)
    except OSError as exc:
        return _err(f"Cannot read file: {exc}")
    stat_result = p.stat()
    record_file_snapshot(
        globals().get("ctx"),
        p,
        content=text,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
        full_read=True,
        start_line=1,
        end_line=len(text.splitlines()),
        total_lines=len(text.splitlines()),
        truncated=False,
        source="search_file",
    )

    lines = text.splitlines()
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(re.escape(pattern) if literal else pattern, flags | re.MULTILINE)
    except re.error as exc:
        return _err(f"Invalid regex: {exc}")

    matches: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not rx.search(line):
            continue
        ctx_start = max(0, index - context_lines)
        ctx_end = min(len(lines), index + context_lines + 1)
        matches.append(
            {
                "line_number": index + 1,
                "line_content": line,
                "context_before": lines[ctx_start:index],
                "context_after": lines[index + 1 : ctx_end],
            }
        )
        if len(matches) >= 50:
            break

    return {
        "status": "ok",
        "path": str(p),
        "total_matches": len(matches),
        "matches": matches,
    }


def search_code(
    query: str,
    path: str = ".",
    glob: str = "**/*",
    mode: str = "literal",
    case_sensitive: bool = True,
    context_lines: int = 2,
    max_results: int = 50,
    include_hidden: bool = False,
    offset: int = 0,
) -> dict[str, Any]:
    """Search code using multiline text or Python symbol matching."""
    return inspection_search_code(
        query,
        path=path,
        glob=glob,
        mode=mode,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_results=max_results,
        include_hidden=include_hidden,
        offset=offset,
    )


__all__ = [
    "glob_files",
    "grep_files",
    "search_file",
    "search_code",
    "rg_search",
    "fd_find",
    "find_references",
    "search_symbols",
]
