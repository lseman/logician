"""Shared filesystem backend for core file and search tools.

This module centralizes path resolution, safer file reads, structured directory
metadata, globbing, and recursive search behavior. Public tool functions in
``files.py`` and ``search.py`` delegate to this backend so the agent-facing
tool names stay stable while the implementation gets more robust.
"""

from __future__ import annotations

import base64
import difflib
import fnmatch
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Device paths that must never be opened (infinite/blocking reads or security hazards).
_BLOCKED_DEVICE_PATHS: frozenset[str] = frozenset(
    {
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/null",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/tty",
        "/dev/full",
    }
)
# Patterns like /proc/<pid>/fd/0, /proc/<pid>/fd/1, /proc/<pid>/fd/2
_BLOCKED_PROC_FD_RE = re.compile(r"^/proc/[^/]+/fd/[012]$")


def _err(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}


class FilesystemBackend:
    """Filesystem helper with optional root enforcement and safer file access."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        enforce_root: bool = False,
        max_read_chars: int = 20_000,
        max_binary_bytes: int = 256_000,
        max_file_bytes: int = 1_048_576,
        max_grep_matches: int = 200,
    ) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.enforce_root = enforce_root
        self.max_read_chars = max_read_chars
        self.max_binary_bytes = max_binary_bytes
        self.max_file_bytes = max_file_bytes
        self.max_grep_matches = max_grep_matches

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a user path, optionally constraining it to ``self.root``."""
        candidate = Path(path).expanduser()
        if not self.enforce_root:
            if candidate.is_absolute():
                return Path(os.path.abspath(os.fspath(candidate)))
            return Path(os.path.abspath(os.fspath(Path.cwd() / candidate)))

        root = self.root or Path.cwd().resolve()
        candidate_text = str(path)
        if candidate_text.startswith("~"):
            raise ValueError("Path traversal not allowed")
        if candidate.is_absolute():
            relative = str(candidate).lstrip("/")
            full = Path(os.path.abspath(os.fspath(root / relative)))
        else:
            full = Path(os.path.abspath(os.fspath(root / candidate)))
        try:
            full.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes root: {path}") from exc
        return full

    def read_bytes(self, path: str | Path) -> bytes:
        """Read bytes from a file without following a final-path symlink when possible."""
        resolved = self.resolve_path(path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(resolved, flags)
        try:
            with os.fdopen(fd, "rb") as fh:
                return fh.read()
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def read_text(
        self, path: str | Path, *, encoding: str = "utf-8", errors: str = "replace"
    ) -> str:
        """Read text preserving raw newline bytes."""
        return self.read_bytes(path).decode(encoding, errors=errors)

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a text or binary file into a stable machine-readable payload."""
        try:
            resolved = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))

        # Block device paths that produce infinite/blocking reads.
        str_resolved = str(resolved)
        if str_resolved in _BLOCKED_DEVICE_PATHS or _BLOCKED_PROC_FD_RE.match(str_resolved):
            return _err(f"Reading from this device path is not allowed: {path}")

        if not resolved.exists():
            error = _err(f"File not found: {path}")
            suggestions = self._find_similar_paths(resolved)
            if suggestions:
                error["did_you_mean"] = suggestions
            return error

        try:
            stat_result = resolved.stat()
        except OSError as exc:
            return _err(f"Cannot stat file: {exc}")

        if resolved.is_dir():
            return _err(f"Path is a directory, not a file: {path}")
        if not resolved.is_file():
            return _err(f"Path is not a regular file: {path}")

        normalized_start, normalized_end, range_error = self._normalize_line_range(
            start_line,
            end_line,
        )
        if range_error is not None:
            return _err(range_error)

        try:
            raw = self.read_bytes(resolved)
        except OSError as exc:
            return _err(f"Cannot read file: {exc}")

        if self._is_probably_binary(raw):
            truncated = len(raw) > self.max_binary_bytes
            payload = raw[: self.max_binary_bytes] if truncated else raw
            return {
                "status": "ok",
                "path": str(resolved),
                "file_type": "binary",
                "total_lines": 0,
                "line_count": 0,
                "returned_lines": "0-0",
                "newline": None,
                "encoding": "base64",
                "content": base64.standard_b64encode(payload).decode("ascii"),
                "truncated": truncated,
                "warning": "Binary file returned as base64.",
                "size_bytes": len(raw),
                "read_bytes": len(payload),
            }

        if (
            stat_result.st_size > self.max_file_bytes
            and normalized_start is None
            and normalized_end is None
        ):
            return {
                "status": "error",
                "error": (
                    f"File content ({self._format_file_size(stat_result.st_size)}) exceeds the full-read limit "
                    f"({self._format_file_size(self.max_file_bytes)}). Use start_line/end_line or a search tool instead."
                ),
                "path": str(resolved),
                "reason": "file_too_large",
                "size_bytes": int(stat_result.st_size),
                "max_file_bytes": int(self.max_file_bytes),
                "requires_range": True,
            }

        if stat_result.st_size > self.max_file_bytes:
            return self._read_file_streaming(
                resolved,
                start_line=normalized_start,
                end_line=normalized_end,
                size_bytes=int(stat_result.st_size),
            )

        text = raw.decode("utf-8", errors="replace")
        if text.startswith("\ufeff"):
            text = text[1:]
        newline = self._detect_newline_style(text)
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        s = max(1, normalized_start or 1) - 1
        e = total_lines if normalized_end is None else min(total_lines, normalized_end)
        if s >= e:
            return {
                "status": "ok",
                "path": str(resolved),
                "file_type": "text",
                "total_lines": total_lines,
                "line_count": 0,
                "returned_lines": f"{s + 1}-{e}",
                "newline": self._newline_name(newline),
                "encoding": "utf-8",
                "content": "",
                "truncated": False,
                "size_bytes": len(raw),
                "read_bytes": 0,
            }

        selected_lines = lines[s:e]
        content, truncated = self._fit_text_output(selected_lines)

        return {
            "status": "ok",
            "path": str(resolved),
            "file_type": "text",
            "total_lines": total_lines,
            "line_count": len(selected_lines),
            "returned_lines": f"{s + 1}-{e}",
            "newline": self._newline_name(newline),
            "encoding": "utf-8",
            "content": content,
            "truncated": truncated,
            "size_bytes": len(raw),
            "read_bytes": len(content.encode("utf-8")),
        }

    def _read_file_streaming(
        self,
        path: Path,
        *,
        start_line: int | None,
        end_line: int | None,
        size_bytes: int,
    ) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                total_lines = 0
                selected_lines: list[str] = []
                first_line = True
                for raw_line in fh:
                    total_lines += 1
                    line = (
                        raw_line[1:] if first_line and raw_line.startswith("\ufeff") else raw_line
                    )
                    first_line = False
                    if start_line is not None and total_lines < start_line:
                        continue
                    if end_line is not None and total_lines > end_line:
                        continue
                    selected_lines.append(line)
        except OSError as exc:
            return _err(f"Cannot read file: {exc}")

        newline = self._detect_newline_style(
            "".join(selected_lines[: min(len(selected_lines), 32)])
        )
        start = start_line or 1
        end = total_lines if end_line is None else min(total_lines, end_line)
        if start > end:
            return {
                "status": "ok",
                "path": str(path),
                "file_type": "text",
                "total_lines": total_lines,
                "line_count": 0,
                "returned_lines": f"{start}-{end}",
                "newline": self._newline_name(newline),
                "encoding": "utf-8",
                "content": "",
                "truncated": False,
                "size_bytes": size_bytes,
                "read_bytes": 0,
            }

        content, truncated = self._fit_text_output(selected_lines)
        return {
            "status": "ok",
            "path": str(path),
            "file_type": "text",
            "total_lines": total_lines,
            "line_count": len(selected_lines),
            "returned_lines": f"{start}-{end}",
            "newline": self._newline_name(newline),
            "encoding": "utf-8",
            "content": content,
            "truncated": truncated,
            "size_bytes": size_bytes,
            "read_bytes": len(content.encode("utf-8")),
        }

    def _normalize_line_range(
        self,
        start_line: int | None,
        end_line: int | None,
    ) -> tuple[int | None, int | None, str | None]:
        if start_line is not None and start_line <= 0:
            return None, None, "start_line must be >= 1"
        if end_line is not None and end_line <= 0:
            return None, None, "end_line must be >= 1"
        if start_line is not None and end_line is not None and start_line > end_line:
            return None, None, "start_line must be <= end_line"
        return start_line, end_line, None

    def _fit_text_output(self, lines: list[str]) -> tuple[str, bool]:
        if not lines:
            return "", False

        total_chars = sum(len(line) for line in lines)
        if total_chars <= self.max_read_chars:
            return "".join(lines), False

        note = "\n...[truncated, use start_line/end_line to read more]"
        limit = max(0, self.max_read_chars - len(note))
        kept: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) > limit:
                break
            kept.append(line)
            used += len(line)

        if not kept and lines:
            kept.append(lines[0][:limit])
        return "".join(kept) + note, True

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size_bytes} B"

    def list_dir(self, path: str = ".", glob_pattern: str = "*") -> dict[str, Any]:
        """List directory entries with consistent metadata."""
        try:
            resolved = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))
        if not resolved.exists():
            return _err(f"Path not found: {path}")
        if not resolved.is_dir():
            return _err(f"Not a directory: {path}")

        entries: list[dict[str, Any]] = []
        try:
            for child in sorted(resolved.iterdir(), key=lambda item: item.name):
                if not fnmatch.fnmatch(child.name, glob_pattern):
                    continue
                entries.append(self._file_info(child, base=resolved))
        except OSError as exc:
            return _err(f"Cannot list directory: {exc}")

        return {
            "status": "ok",
            "path": str(resolved),
            "count": len(entries),
            "entries": entries,
        }

    def glob(
        self,
        pattern: str,
        path: str = ".",
        include_hidden: bool = False,
        offset: int = 0,
        max_results: int | None = None,
        output_mode: str = "entries",
    ) -> dict[str, Any]:
        """Find filesystem entries matching a glob pattern with metadata."""
        try:
            base = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))
        if not base.exists():
            return _err(f"Path not found: {path}")
        if not base.is_dir():
            return _err(f"Not a directory: {path}")
        if offset < 0:
            return _err("offset must be >= 0")
        if max_results is not None and max_results <= 0:
            return _err("max_results must be >= 1")
        if output_mode not in {"entries", "paths", "count"}:
            return _err("output_mode must be one of: entries, paths, count")

        try:
            matches = sorted(base.glob(pattern), key=lambda item: str(item.relative_to(base)))
        except (OSError, ValueError) as exc:
            return _err(f"Cannot glob files: {exc}")

        payload = [
            self._file_info(match, base=base)
            for match in matches
            if include_hidden or not self._path_is_hidden(match, base)
        ]
        total_count = len(payload)
        page_limit = max_results if max_results is not None else total_count
        paged = payload[offset : offset + page_limit]
        truncated = offset + len(paged) < total_count
        out = {
            "status": "ok",
            "path": str(base),
            "pattern": pattern,
            "include_hidden": include_hidden,
            "output_mode": output_mode,
            "offset": offset,
            "max_results": max_results,
            "count": len(paged),
            "total_count": total_count,
            "returned_count": len(paged),
            "truncated": truncated,
            "next_offset": (offset + len(paged)) if truncated else None,
        }
        if output_mode == "entries":
            out["matches"] = paged
        elif output_mode == "paths":
            out["paths"] = [
                str(item.get("relative_path") or item.get("path") or "") for item in paged
            ]
        return out

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob: str = "**/*",
        literal: bool = True,
        case_sensitive: bool = True,
        context_lines: int = 1,
        max_matches: int | None = None,
        include_hidden: bool = False,
        offset: int = 0,
        output_mode: str = "matches",
    ) -> dict[str, Any]:
        """Search for text across a directory tree with ripgrep fallback."""
        try:
            root = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))
        if not root.exists():
            return _err(f"Path not found: {path}")
        if context_lines < 0:
            return _err("context_lines must be >= 0")
        if offset < 0:
            return _err("offset must be >= 0")
        if output_mode not in {"matches", "files_with_matches", "count"}:
            return _err("output_mode must be one of: matches, files_with_matches, count")
        requested_limit = max_matches if max_matches is not None else self.max_grep_matches
        if requested_limit <= 0:
            return _err("max_matches must be >= 1")
        page_limit = min(requested_limit, self.max_grep_matches)
        raw_limit = self._grep_raw_limit(
            offset=offset,
            page_limit=page_limit,
            output_mode=output_mode,
        )
        search_root = root.parent if root.is_file() else root

        rg = self._try_rg(
            pattern=pattern,
            root=root,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_matches=raw_limit,
            include_hidden=include_hidden,
            offset=offset,
            output_mode=output_mode,
        )
        if rg is not None:
            return rg

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(re.escape(pattern) if literal else pattern, flags | re.MULTILINE)
        except re.error as exc:
            return _err(f"Invalid regex: {exc}")

        matches: list[dict[str, Any]] = []
        files_searched = 0
        files_with_matches: set[str] = set()
        skipped_binary = 0
        skipped_large = 0
        search_paths = [root] if root.is_file() else sorted(root.glob(glob))
        for fpath in search_paths:
            try:
                if not fpath.is_file():
                    continue
            except OSError:
                continue
            if not include_hidden and self._path_is_hidden(fpath, search_root):
                continue
            try:
                if fpath.stat().st_size > self.max_file_bytes:
                    skipped_large += 1
                    continue
                raw = self.read_bytes(fpath)
            except OSError:
                continue
            if self._is_probably_binary(raw):
                skipped_binary += 1
                continue

            try:
                content = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                continue

            files_searched += 1
            file_lines = content.splitlines()
            for index, line in enumerate(file_lines):
                submatches = [
                    {
                        "match": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    }
                    for match in rx.finditer(line)
                ]
                if not submatches:
                    continue
                ctx_start = max(0, index - context_lines)
                ctx_end = min(len(file_lines), index + context_lines + 1)
                relative_file = (
                    str(fpath.relative_to(search_root)) if fpath != search_root else fpath.name
                )
                files_with_matches.add(relative_file)
                matches.append(
                    {
                        "file": relative_file,
                        "line_number": index + 1,
                        "line_content": line,
                        "submatches": submatches,
                        "column_start": submatches[0]["start"] + 1,
                        "column_end": submatches[0]["end"],
                        "context_before": file_lines[ctx_start:index],
                        "context_after": file_lines[index + 1 : ctx_end],
                    }
                )
                if len(matches) >= raw_limit:
                    return self._render_grep_result(
                        pattern=pattern,
                        path=str(root),
                        glob=glob,
                        literal=literal,
                        case_sensitive=case_sensitive,
                        context_lines=context_lines,
                        include_hidden=include_hidden,
                        engine="python",
                        file_count=files_searched,
                        files_with_matches=files_with_matches,
                        requested_max_matches=requested_limit,
                        page_limit=page_limit,
                        raw_matches=matches,
                        skipped_binary=skipped_binary,
                        skipped_large=skipped_large,
                        raw_truncated=True,
                        offset=offset,
                        output_mode=output_mode,
                    )

        return self._render_grep_result(
            pattern=pattern,
            path=str(root),
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            include_hidden=include_hidden,
            engine="python",
            file_count=files_searched,
            files_with_matches=files_with_matches,
            requested_max_matches=requested_limit,
            page_limit=page_limit,
            raw_matches=matches,
            skipped_binary=skipped_binary,
            skipped_large=skipped_large,
            raw_truncated=False,
            offset=offset,
            output_mode=output_mode,
        )

    def _grep_raw_limit(self, *, offset: int, page_limit: int, output_mode: str) -> int:
        if output_mode == "matches":
            return min(self.max_grep_matches, offset + page_limit)
        return self.max_grep_matches

    def _render_grep_result(
        self,
        *,
        pattern: str,
        path: str,
        glob: str,
        literal: bool,
        case_sensitive: bool,
        context_lines: int,
        include_hidden: bool,
        engine: str,
        file_count: int,
        files_with_matches: set[str],
        requested_max_matches: int,
        page_limit: int,
        raw_matches: list[dict[str, Any]],
        skipped_binary: int,
        skipped_large: int,
        raw_truncated: bool,
        offset: int,
        output_mode: str,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "ok",
            "pattern": pattern,
            "path": path,
            "glob": glob,
            "literal": literal,
            "case_sensitive": case_sensitive,
            "context_lines": context_lines,
            "include_hidden": include_hidden,
            "engine": engine,
            "output_mode": output_mode,
            "file_count": file_count,
            "files_with_matches": len(files_with_matches),
            "requested_max_matches": requested_max_matches,
            "max_matches": page_limit,
            "files_skipped_binary": skipped_binary,
            "files_skipped_too_large": skipped_large,
            "offset": offset,
        }

        if output_mode == "matches":
            total_available = len(raw_matches)
            page = raw_matches[offset : offset + page_limit]
            truncated = raw_truncated or (offset + len(page) < total_available)
            out.update(
                {
                    "total_matches": total_available,
                    "returned_count": len(page),
                    "truncated": truncated,
                    "next_offset": (offset + len(page)) if truncated else None,
                    "matches": page,
                }
            )
            return out

        ordered_files = list(
            dict.fromkeys(match.get("file") for match in raw_matches if match.get("file"))
        )
        if output_mode == "files_with_matches":
            page = ordered_files[offset : offset + page_limit]
            truncated = raw_truncated or (offset + len(page) < len(ordered_files))
            out.update(
                {
                    "total_matches": len(raw_matches),
                    "total_files": len(ordered_files),
                    "returned_count": len(page),
                    "truncated": truncated,
                    "next_offset": (offset + len(page)) if truncated else None,
                    "paths": page,
                }
            )
            return out

        count_map: dict[str, int] = {}
        for match in raw_matches:
            file_name = str(match.get("file") or "")
            if not file_name:
                continue
            count_map[file_name] = count_map.get(file_name, 0) + 1
        count_entries = [
            {"file": file_name, "count": count_map[file_name]} for file_name in ordered_files
        ]
        page = count_entries[offset : offset + page_limit]
        truncated = raw_truncated or (offset + len(page) < len(count_entries))
        out.update(
            {
                "total_matches": len(raw_matches),
                "total_files": len(count_entries),
                "returned_count": len(page),
                "truncated": truncated,
                "next_offset": (offset + len(page)) if truncated else None,
                "counts": page,
            }
        )
        return out

    def _file_info(self, path: Path, *, base: Path) -> dict[str, Any]:
        is_symlink = path.is_symlink()
        info: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "relative_path": self._relative_path(path, base),
            "is_dir": False,
            "type": "file",
            "is_symlink": is_symlink,
        }
        try:
            stat_result = path.lstat() if is_symlink else path.stat()
            info["modified_at"] = datetime.fromtimestamp(
                stat_result.st_mtime, tz=timezone.utc
            ).isoformat()
            info["size_bytes"] = int(stat_result.st_size)
        except OSError:
            info["modified_at"] = ""
            info["size_bytes"] = 0

        try:
            if path.is_dir():
                info["is_dir"] = True
                info["type"] = "dir"
            elif is_symlink:
                info["type"] = "symlink"
        except OSError:
            if is_symlink:
                info["type"] = "symlink"

        return info

    def _relative_path(self, path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    def _path_is_hidden(self, path: Path, base: Path) -> bool:
        try:
            parts = path.relative_to(base).parts
        except ValueError:
            parts = path.parts
        return any(part.startswith(".") for part in parts if part not in {".", ".."})

    def _find_similar_paths(self, missing: Path) -> list[str]:
        """Return up to 3 close filename matches in the same directory."""
        parent = missing.parent
        if not parent.is_dir():
            return []
        try:
            siblings = [child.name for child in parent.iterdir()]
        except OSError:
            return []
        matches = difflib.get_close_matches(missing.name, siblings, n=3, cutoff=0.6)
        return [str(parent / m) for m in matches]

    def _try_rg(
        self,
        *,
        pattern: str,
        root: Path,
        glob: str,
        literal: bool,
        case_sensitive: bool,
        context_lines: int,
        max_matches: int,
        include_hidden: bool,
        offset: int,
        output_mode: str,
    ) -> dict[str, Any] | None:
        try:
            cmd = ["rg", "--json", "--max-count", str(max_matches), "--max-columns", "500"]
            if literal:
                cmd.append("--fixed-strings")
            if not case_sensitive:
                cmd.append("--ignore-case")
            if include_hidden:
                cmd.append("--hidden")
            if context_lines:
                cmd += ["-C", str(context_lines)]
            if root.is_dir():
                cmd += ["--glob", glob, pattern, str(root)]
            else:
                cmd += [pattern, str(root)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if proc.returncode not in (0, 1):
            return None

        matches: list[dict[str, Any]] = []
        files_seen: set[str] = set()
        files_with_matches: set[str] = set()
        base = root.parent if root.is_file() else root
        for line in proc.stdout.splitlines():
            if len(matches) >= max_matches:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj.get("data", {})
            file_text = data.get("path", {}).get("text")
            if not file_text:
                continue
            try:
                file_rel = str(Path(file_text).relative_to(base))
            except ValueError:
                file_rel = Path(file_text).name if root.is_file() else file_text
            file_path = Path(file_text)
            if not include_hidden and self._path_is_hidden(file_path, base):
                continue
            files_seen.add(file_rel)
            files_with_matches.add(file_rel)
            submatches = [
                {
                    "match": item.get("match", {}).get("text", ""),
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
                for item in data.get("submatches", [])
            ]
            matches.append(
                {
                    "file": file_rel,
                    "line_number": data.get("line_number"),
                    "line_content": data.get("lines", {}).get("text", "").rstrip("\n"),
                    "submatches": submatches,
                    "column_start": (submatches[0]["start"] + 1) if submatches else None,
                    "column_end": submatches[0]["end"] if submatches else None,
                    "context_before": [
                        item.get("lines", {}).get("text", "").rstrip("\n")
                        for item in data.get("context_before", [])
                    ],
                    "context_after": [
                        item.get("lines", {}).get("text", "").rstrip("\n")
                        for item in data.get("context_after", [])
                    ],
                }
            )

        return self._render_grep_result(
            pattern=pattern,
            path=str(root),
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            include_hidden=include_hidden,
            engine="ripgrep",
            file_count=len(files_seen),
            files_with_matches=files_with_matches,
            requested_max_matches=max_matches,
            page_limit=max_matches,
            raw_matches=matches,
            skipped_binary=0,
            skipped_large=0,
            raw_truncated=len(matches) >= max_matches,
            offset=offset,
            output_mode=output_mode,
        )

    @staticmethod
    def _detect_newline_style(text: str) -> str:
        if "\r\n" in text:
            return "\r\n"
        if "\r" in text:
            return "\r"
        return "\n"

    @staticmethod
    def _newline_name(newline: str) -> str:
        return {"\n": "LF", "\r\n": "CRLF", "\r": "CR"}.get(newline, "LF")

    @staticmethod
    def _is_probably_binary(raw: bytes) -> bool:
        if not raw:
            return False
        if b"\x00" in raw:
            return True

        sample = raw[:4096]
        text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
        non_text = sample.translate(None, text_bytes)
        return (len(non_text) / len(sample)) > 0.30


DEFAULT_FILESYSTEM = FilesystemBackend()
