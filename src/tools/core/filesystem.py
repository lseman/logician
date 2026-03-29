"""Shared filesystem backend for core file and search tools.

This module centralizes path resolution, safer file reads, structured directory
metadata, globbing, and recursive search behavior. Public tool functions in
``files.py`` and ``search.py`` delegate to this backend so the agent-facing
tool names stay stable while the implementation gets more robust.
"""

from __future__ import annotations

import base64
import fnmatch
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    def read_text(self, path: str | Path, *, encoding: str = "utf-8", errors: str = "replace") -> str:
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
        if not resolved.exists() or not resolved.is_file():
            return _err(f"File not found: {path}")

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
                "returned_lines": "0-0",
                "newline": None,
                "encoding": "base64",
                "content": base64.standard_b64encode(payload).decode("ascii"),
                "truncated": truncated,
                "warning": "Binary file returned as base64.",
                "size_bytes": len(raw),
            }

        text = raw.decode("utf-8", errors="replace")
        newline = self._detect_newline_style(text)
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        s = max(1, start_line or 1) - 1
        e = total_lines if (end_line is None or end_line <= 0) else min(total_lines, end_line)
        if s >= e:
            return {
                "status": "ok",
                "path": str(resolved),
                "file_type": "text",
                "total_lines": total_lines,
                "returned_lines": f"{s + 1}-{e}",
                "newline": self._newline_name(newline),
                "encoding": "utf-8",
                "content": "",
                "truncated": False,
                "size_bytes": len(raw),
            }

        content = "".join(lines[s:e])
        truncated = len(content) > self.max_read_chars
        if truncated:
            content = content[: self.max_read_chars] + "\n...[truncated, use start_line/end_line to read more]"

        return {
            "status": "ok",
            "path": str(resolved),
            "file_type": "text",
            "total_lines": total_lines,
            "returned_lines": f"{s + 1}-{e}",
            "newline": self._newline_name(newline),
            "encoding": "utf-8",
            "content": content,
            "truncated": truncated,
            "size_bytes": len(raw),
        }

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

    def glob(self, pattern: str, path: str = ".") -> dict[str, Any]:
        """Find filesystem entries matching a glob pattern with metadata."""
        try:
            base = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))
        if not base.exists():
            return _err(f"Path not found: {path}")
        if not base.is_dir():
            return _err(f"Not a directory: {path}")

        try:
            matches = sorted(base.glob(pattern), key=lambda item: str(item.relative_to(base)))
        except (OSError, ValueError) as exc:
            return _err(f"Cannot glob files: {exc}")

        payload = [self._file_info(match, base=base) for match in matches]
        return {
            "status": "ok",
            "path": str(base),
            "pattern": pattern,
            "count": len(payload),
            "matches": payload,
        }

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
    ) -> dict[str, Any]:
        """Search for text across a directory tree with ripgrep fallback."""
        try:
            root = self.resolve_path(path)
        except ValueError as exc:
            return _err(str(exc))
        if not root.exists():
            return _err(f"Path not found: {path}")
        if not root.is_dir():
            return _err(f"Not a directory: {path}")

        limit = max_matches if max_matches is not None else self.max_grep_matches
        rg = self._try_rg(
            pattern=pattern,
            root=root,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_matches=limit,
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
        for fpath in sorted(root.glob(glob)):
            try:
                if not fpath.is_file():
                    continue
            except OSError:
                continue
            try:
                if fpath.stat().st_size > self.max_file_bytes:
                    continue
                raw = self.read_bytes(fpath)
            except OSError:
                continue
            if self._is_probably_binary(raw):
                continue

            try:
                content = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                continue

            files_searched += 1
            file_lines = content.splitlines()
            for index, line in enumerate(file_lines):
                if not rx.search(line):
                    continue
                ctx_start = max(0, index - context_lines)
                ctx_end = min(len(file_lines), index + context_lines + 1)
                matches.append(
                    {
                        "file": str(fpath.relative_to(root)),
                        "line_number": index + 1,
                        "line_content": line,
                        "context_before": file_lines[ctx_start:index],
                        "context_after": file_lines[index + 1 : ctx_end],
                    }
                )
                if len(matches) >= limit:
                    return {
                        "status": "ok",
                        "pattern": pattern,
                        "path": str(root),
                        "engine": "python",
                        "file_count": files_searched,
                        "total_matches": len(matches),
                        "truncated": True,
                        "matches": matches,
                    }

        return {
            "status": "ok",
            "pattern": pattern,
            "path": str(root),
            "engine": "python",
            "file_count": files_searched,
            "total_matches": len(matches),
            "truncated": False,
            "matches": matches,
        }

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
    ) -> dict[str, Any] | None:
        try:
            cmd = ["rg", "--json", "--max-count", str(max_matches)]
            if literal:
                cmd.append("--fixed-strings")
            if not case_sensitive:
                cmd.append("--ignore-case")
            if context_lines:
                cmd += ["-C", str(context_lines)]
            cmd += ["--glob", glob, pattern, str(root)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if proc.returncode not in (0, 1):
            return None

        matches: list[dict[str, Any]] = []
        files_seen: set[str] = set()
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
                file_rel = str(Path(file_text).relative_to(root))
            except ValueError:
                file_rel = file_text
            files_seen.add(file_rel)
            matches.append(
                {
                    "file": file_rel,
                    "line_number": data.get("line_number"),
                    "line_content": data.get("lines", {}).get("text", "").rstrip("\n"),
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

        return {
            "status": "ok",
            "pattern": pattern,
            "path": str(root),
            "engine": "ripgrep",
            "file_count": len(files_seen),
            "total_matches": len(matches),
            "truncated": len(matches) >= max_matches,
            "matches": matches,
        }

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
