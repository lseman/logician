"""Advanced search, chunked reading, and safe multi-file replace tools.

Complements the existing rg_search / fd_find / edit_file_replace with:
  read_file_smart   -- read any file size in semantic chunks (by function/class/N lines)
  rg_replace        -- regex search-and-replace across files with diff preview
  sed_replace       -- single-file sed-style replace (line-address or pattern)
  regex_replace     -- safe Python regex replace in a file, with preview
  show_diff         -- diff two files or a file vs proposed string
  count_lines       -- count lines / bytes / matches in files quickly
"""
from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

if "_safe_json" not in globals():
    import json as _j

    def _safe_json(obj):
        return _j.dumps(obj, ensure_ascii=False)


import difflib
import os
import re
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        base_cwd = (
            (_coding_config.get("default_cwd") or None)
            if "_coding_config" in globals()
            else None
        )
        if base_cwd:
            p = Path(base_cwd) / p
    return p.resolve()


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a subprocess and return {stdout, stderr, returncode}."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"command not found: {cmd[0]}", "returncode": 127}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


def _unified_diff(original: str, updated: str, label: str = "file") -> str:
    lines_a = original.splitlines(keepends=True)
    lines_b = updated.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(lines_a, lines_b, fromfile=f"a/{label}", tofile=f"b/{label}", n=3)
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@llm.tool(
    description=(
        "Read a file intelligently — auto-chunks large files by semantic unit "
        "(class/function) or by line-count window. Never truncates silently."
    )
)
def read_file_smart(
    path: str,
    start_line: int = 1,
    end_line: int = 0,
    symbol: str = "",
    chunk_size: int = 300,
) -> str:
    """Use when: Reading a large file where read_file would truncate, or jumping to a symbol.

    Triggers: read function, show class, read chunk, big file, large file, read method,
              show implementation, view function body.
    Avoid when: The file is small — just use read_file.
    Inputs:
      path (str, required): Path to the file.
      start_line (int, optional): First line (1-based). Default 1.
      end_line (int, optional): Last line inclusive. 0 = start_line + chunk_size.
      symbol (str, optional): If given, find the function/class named `symbol` and return it.
                              Overrides start_line/end_line.
      chunk_size (int, optional): Lines to return when end_line=0 (default 300).
    Returns: JSON with content, line range, total_lines, has_more.
    Side effects: Read-only.
    """
    try:
        p = _resolve(path)
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"Not a file: {path}"})

        raw = p.read_text(encoding="utf-8", errors="replace")
        all_lines = raw.splitlines(keepends=True)
        total = len(all_lines)

        if symbol:
            # Find the symbol using a simple def/class scanner
            sym_re = re.compile(
                rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b"
            )
            found_start = -1
            for i, line in enumerate(all_lines):
                if sym_re.match(line):
                    found_start = i
                    break
            if found_start < 0:
                return _safe_json(
                    {"status": "error", "error": f"Symbol '{symbol}' not found in {path}"}
                )
            # Find extent: collect until indent <= symbol's indent or EOF
            sym_indent = len(all_lines[found_start]) - len(all_lines[found_start].lstrip())
            end_idx = found_start + 1
            while end_idx < total:
                ln = all_lines[end_idx]
                if ln.strip() and (len(ln) - len(ln.lstrip())) <= sym_indent and end_idx > found_start + 1:
                    break
                end_idx += 1
            start_line = found_start + 1
            end_line = end_idx

        # Resolve line range
        s = max(1, start_line) - 1  # 0-based
        if end_line <= 0:
            e = min(total, s + chunk_size)
        else:
            e = min(total, end_line)

        selected = all_lines[s:e]
        content = "".join(selected)
        has_more = e < total

        return _safe_json(
            {
                "status": "ok",
                "path": str(p),
                "total_lines": total,
                "returned_lines": f"{s + 1}-{e}",
                "has_more": has_more,
                "next_start_line": e + 1 if has_more else None,
                "content": content,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description=(
        "Regex or literal search-and-replace across one or more files, "
        "with a unified diff preview. Does NOT write until confirmed."
    )
)
def regex_replace(
    path: str,
    pattern: str,
    replacement: str,
    flags: str = "MULTILINE",
    preview_only: bool = True,
    max_replacements: int = 0,
) -> str:
    """Use when: Replace a regex pattern in a file safely, with preview before committing.

    Triggers: regex replace, replace pattern, substitute, sed replace, find and replace,
              bulk edit, rename symbol in file, change all occurrences.
    Avoid when: You only need to replace one exact unique string — edit_file_replace is simpler.
    Inputs:
      path (str, required): File to modify.
      pattern (str, required): Python regex pattern.
      replacement (str, required): Replacement string (supports \\1, \\g<name> backreferences).
      flags (str, optional): Regex flags joined by | e.g. \"IGNORECASE|MULTILINE\" (default \"MULTILINE\").
      preview_only (bool, optional): If True (default), returns diff but does NOT write.
                                      Set False to apply the change.
      max_replacements (int, optional): Max occurrences to replace. 0 = all (default).
    Returns: JSON with diff preview, count of replacements, and write status.
    Side effects: Modifies file only when preview_only=False.

    Examples:
      regex_replace(\"/app/config.py\", r\"DEBUG\\s*=\\s*True\", \"DEBUG = False\")
      regex_replace(\"/app/api.py\", r\"def (old_name)\", \"def new_name\", preview_only=False)
    """
    try:
        p = _resolve(path)
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"Not a file: {path}"})

        # Parse flags
        flag_map = {
            "IGNORECASE": re.IGNORECASE, "I": re.IGNORECASE,
            "MULTILINE": re.MULTILINE, "M": re.MULTILINE,
            "DOTALL": re.DOTALL, "S": re.DOTALL,
            "VERBOSE": re.VERBOSE, "X": re.VERBOSE,
        }
        combined = 0
        for f in str(flags).replace(",", "|").split("|"):
            f = f.strip().upper()
            if f in flag_map:
                combined |= flag_map[f]

        try:
            rx = re.compile(pattern, combined)
        except re.error as exc:
            return _safe_json({"status": "error", "error": f"Invalid regex: {exc}"})

        original = p.read_text(encoding="utf-8")
        count = len(rx.findall(original))
        if count == 0:
            return _safe_json({"status": "ok", "matches": 0, "message": "Pattern not found — no changes."})

        n = int(max_replacements) if int(max_replacements) > 0 else 0
        updated = rx.sub(replacement, original, count=n)
        diff = _unified_diff(original, updated, label=p.name)

        if not preview_only:
            if isinstance(preview_only, str):
                preview_only = preview_only.lower() not in ("false", "0", "no")
            p.write_text(updated, encoding="utf-8")
            return _safe_json(
                {
                    "status": "ok",
                    "written": True,
                    "matches_replaced": count if n == 0 else min(n, count),
                    "diff": diff[:4000],
                }
            )

        return _safe_json(
            {
                "status": "ok",
                "written": False,
                "preview_only": True,
                "matches_found": count,
                "diff": diff[:4000],
                "hint": "Set preview_only=False to apply.",
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description=(
        "Regex search-and-replace across MULTIPLE files using ripgrep + Python. "
        "Shows a diff of proposed changes before writing."
    )
)
def rg_replace(
    pattern: str,
    replacement: str,
    directory: str = ".",
    file_glob: str = "",
    file_type: str = "",
    flags: str = "MULTILINE",
    preview_only: bool = True,
    max_files: int = 20,
) -> str:
    """Use when: Rename a symbol, fix a pattern, or bulk-edit across many files at once.

    Triggers: rename across project, replace everywhere, bulk rename, global replace,
              refactor symbol name, change all files, sed across files, rg replace.
    Avoid when: You only need to edit a single file — use regex_replace instead.
    Inputs:
      pattern (str, required): Python regex to search for.
      replacement (str, required): Replacement string (supports backreferences).
      directory (str, optional): Root directory (default \".\").
      file_glob (str, optional): Glob filter e.g. \"*.py\" or \"src/**/*.ts\".
      file_type (str, optional): ripgrep type e.g. \"py\", \"ts\".
      flags (str, optional): Regex flags e.g. \"IGNORECASE|MULTILINE\" (default \"MULTILINE\").
      preview_only (bool, optional): If True (default), shows diff but does NOT write.
      max_files (int, optional): Max files to touch (default 20, safety limit).
    Returns: JSON with per-file diffs, total matches, files affected.
    Side effects: Writes files only when preview_only=False.
    """
    try:
        root = _resolve(directory)
        if not root.is_dir():
            return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

        # Parse flags
        flag_map = {
            "IGNORECASE": re.IGNORECASE, "I": re.IGNORECASE,
            "MULTILINE": re.MULTILINE, "M": re.MULTILINE,
            "DOTALL": re.DOTALL, "S": re.DOTALL,
        }
        combined = 0
        for f in str(flags).replace(",", "|").split("|"):
            f = f.strip().upper()
            if f in flag_map:
                combined |= flag_map[f]

        try:
            rx = re.compile(pattern, combined)
        except re.error as exc:
            return _safe_json({"status": "error", "error": f"Invalid regex: {exc}"})

        # Discover matching files via rg or glob
        candidate_files: list[Path] = []
        if shutil.which("rg"):
            cmd = ["rg", "--files-with-matches", "--color=never"]
            if file_type:
                cmd += ["--type", file_type]
            if file_glob:
                cmd += ["--glob", file_glob]
            cmd += [pattern, str(root)]
            r = _run(cmd, timeout=20)
            for line in r["stdout"].splitlines():
                fp = Path(line.strip())
                if fp.is_file():
                    candidate_files.append(fp)
        else:
            glob_pat = file_glob if file_glob else "**/*"
            for fp in sorted(root.glob(glob_pat)):
                if fp.is_file():
                    try:
                        text = fp.read_text(encoding="utf-8", errors="replace")
                        if rx.search(text):
                            candidate_files.append(fp)
                    except Exception:
                        continue

        if not candidate_files:
            return _safe_json({"status": "ok", "matches": 0, "message": "No files matched."})

        max_files = max(1, int(max_files))
        results = []
        total_matches = 0
        for fp in candidate_files[:max_files]:
            try:
                original = fp.read_text(encoding="utf-8")
                hits = len(rx.findall(original))
                if hits == 0:
                    continue
                updated = rx.sub(replacement, original)
                diff = _unified_diff(original, updated, label=str(fp.relative_to(root)))
                entry = {
                    "file": str(fp.relative_to(root)),
                    "matches": hits,
                    "diff": diff[:2000],
                }
                if not preview_only:
                    fp.write_text(updated, encoding="utf-8")
                    entry["written"] = True
                results.append(entry)
                total_matches += hits
            except Exception as exc:
                results.append({"file": str(fp), "error": str(exc)})

        skipped = max(0, len(candidate_files) - max_files)
        return _safe_json(
            {
                "status": "ok",
                "written": not preview_only,
                "preview_only": bool(preview_only),
                "total_matches": total_matches,
                "files_affected": len(results),
                "files_skipped": skipped,
                "results": results,
                "hint": "Set preview_only=False to apply changes." if preview_only else None,
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description=(
        "Show a unified diff between two files, or between a file and a proposed new content string."
    )
)
def show_diff(
    path_a: str,
    path_b: str = "",
    proposed_content: str = "",
    context_lines: int = 5,
) -> str:
    """Use when: Preview what would change before applying an edit, or compare two files.

    Triggers: show diff, compare files, preview change, what changed, diff, before and after.
    Avoid when: You just want to read a file — use read_file instead.
    Inputs:
      path_a (str, required): The original file path.
      path_b (str, optional): Second file path (compare two files). Mutually exclusive with proposed_content.
      proposed_content (str, optional): New content to compare against path_a.
      context_lines (int, optional): Lines of context around changes (default 5).
    Returns: JSON with unified diff string.
    Side effects: Read-only.
    """
    try:
        pa = _resolve(path_a)
        if not pa.is_file():
            return _safe_json({"status": "error", "error": f"Not a file: {path_a}"})
        original = pa.read_text(encoding="utf-8", errors="replace")

        if path_b:
            pb = _resolve(path_b)
            if not pb.is_file():
                return _safe_json({"status": "error", "error": f"Not a file: {path_b}"})
            other = pb.read_text(encoding="utf-8", errors="replace")
            label_b = str(pb)
        elif proposed_content:
            other = proposed_content
            label_b = f"{path_a} (proposed)"
        else:
            return _safe_json({"status": "error", "error": "Provide either path_b or proposed_content."})

        a_lines = original.splitlines(keepends=True)
        b_lines = other.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                a_lines, b_lines,
                fromfile=path_a, tofile=label_b,
                n=int(context_lines),
            )
        )

        return _safe_json(
            {
                "status": "ok",
                "diff": diff[:8000] if diff else "(no differences)",
                "has_changes": bool(diff),
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description="Count lines, bytes, and pattern occurrences in a file — zero I/O overhead."
)
def count_in_file(
    path: str,
    pattern: str = "",
    case_sensitive: bool = False,
) -> str:
    """Use when: Quickly check file size, line count, or how many times a pattern appears.

    Triggers: how many lines, count occurrences, how many times, file size, line count,
              how often, count matches, how large.
    Avoid when: You need the actual matching lines — use rg_search instead.
    Inputs:
      path (str, required): File path.
      pattern (str, optional): Regex/string to count occurrences of (default: count lines only).
      case_sensitive (bool, optional): Match case (default False).
    Returns: JSON with total_lines, total_bytes, and match_count.
    Side effects: Read-only.
    """
    try:
        p = _resolve(path)
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"Not a file: {path}"})

        text = p.read_text(encoding="utf-8", errors="replace")
        total_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        total_bytes = p.stat().st_size
        match_count = 0

        if pattern:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                match_count = len(re.findall(pattern, text, flags))
            except re.error:
                match_count = text.lower().count(pattern.lower()) if not case_sensitive else text.count(pattern)

        result: dict = {
            "status": "ok",
            "path": str(p),
            "total_lines": total_lines,
            "total_bytes": total_bytes,
        }
        if pattern:
            result["pattern"] = pattern
            result["match_count"] = match_count
        return _safe_json(result)
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(
    description=(
        "Run ripgrep with multiline mode to find patterns that span multiple lines. "
        "Falls back to Python re.DOTALL search."
    )
)
def rg_multiline(
    pattern: str,
    directory: str = ".",
    file_glob: str = "",
    file_type: str = "",
    max_results: int = 30,
) -> str:
    """Use when: Searching for code patterns that span more than one line (e.g. function signatures).

    Triggers: multiline search, span lines, multi-line pattern, find across lines,
              find block, function signature regex, class body search.
    Avoid when: Your pattern fits on a single line — rg_search is faster.
    Inputs:
      pattern (str, required): Regex pattern (will be compiled with re.DOTALL).
      directory (str, optional): Root directory.
      file_glob (str, optional): Glob filter e.g. \"*.py\".
      file_type (str, optional): ripgrep file type e.g. \"py\".
      max_results (int, optional): Max matches to return (default 30).
    Returns: JSON with matches {file, start_line, end_line, text}.
    Side effects: Read-only.
    """
    try:
        root = _resolve(directory)
        if not root.is_dir():
            return _safe_json({"status": "error", "error": f"Not a directory: {directory}"})

        # Try rg --multiline first
        if shutil.which("rg"):
            cmd = [
                "rg", "--multiline", "--line-number", "--no-heading",
                "--color=never", "--max-count", str(max_results),
            ]
            if file_type:
                cmd += ["--type", file_type]
            if file_glob:
                cmd += ["--glob", file_glob]
            cmd += [pattern, str(root)]
            r = _run(cmd, timeout=30)
            if r["returncode"] in (0, 1):  # 1 = no matches
                raw_lines = r["stdout"].splitlines()
                matches = []
                line_re = re.compile(r"^(.+?):(\d+):(.*)$")
                for raw in raw_lines:
                    m = line_re.match(raw)
                    if m:
                        try:
                            rel = str(Path(m.group(1)).relative_to(root))
                        except ValueError:
                            rel = m.group(1)
                        matches.append({"file": rel, "line": int(m.group(2)), "text": m.group(3)})
                return _safe_json(
                    {"status": "ok", "tool_used": "rg --multiline", "count": len(matches), "matches": matches}
                )

        # Python fallback with re.DOTALL
        try:
            rx = re.compile(pattern, re.DOTALL)
        except re.error as exc:
            return _safe_json({"status": "error", "error": f"Invalid regex: {exc}"})

        glob_pat = file_glob if file_glob else "**/*"
        matches = []
        for fp in sorted(root.glob(glob_pat)):
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for m in rx.finditer(text):
                start_line = text[: m.start()].count("\n") + 1
                end_line = start_line + m.group(0).count("\n")
                rel = str(fp.relative_to(root))
                matches.append(
                    {
                        "file": rel,
                        "start_line": start_line,
                        "end_line": end_line,
                        "text": m.group(0)[:400],
                    }
                )
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        return _safe_json(
            {"status": "ok", "tool_used": "python re.DOTALL", "count": len(matches), "matches": matches}
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})
