from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from src.tools.core.FileEditTool import write_file
from src.tools.core.FileReadTool import list_dir, read_edit_context, read_file
from src.tools.core.filesystem import DEFAULT_FILESYSTEM
from src.tools.core.SearchTool import glob_files, grep_files, search_code


def test_read_file_returns_base64_for_binary(tmp_path: Path) -> None:
    path = tmp_path / "image.bin"
    raw = b"\x00\x01hello\xff"
    path.write_bytes(raw)

    result = read_file(str(path))

    assert result["status"] == "ok"
    assert result["file_type"] == "binary"
    assert result["encoding"] == "base64"
    assert base64.standard_b64decode(result["content"]) == raw


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW not available")
def test_read_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("secret\n", encoding="utf-8")
    link.symlink_to(target)

    result = read_file(str(link))

    assert result["status"] == "error"
    assert "Cannot read file" in result["error"]


def test_read_file_rejects_invalid_line_range(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a\nb\n", encoding="utf-8")

    result = read_file(str(path), start_line=3, end_line=2)

    assert result["status"] == "error"
    assert "start_line must be <= end_line" in result["error"]


def test_read_file_requires_range_for_oversized_full_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.txt"
    path.write_text("alpha\n" * 200, encoding="utf-8")
    monkeypatch.setattr(DEFAULT_FILESYSTEM, "max_file_bytes", 64)

    result = read_file(str(path))

    assert result["status"] == "error"
    assert result["reason"] == "file_too_large"
    assert result["requires_range"] is True


def test_read_file_streams_requested_range_for_large_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large_range.txt"
    path.write_text("".join(f"line {i}\n" for i in range(1, 201)), encoding="utf-8")
    monkeypatch.setattr(DEFAULT_FILESYSTEM, "max_file_bytes", 64)

    result = read_file(str(path), start_line=50, end_line=52)

    assert result["status"] == "ok"
    assert result["returned_lines"] == "50-52"
    assert result["line_count"] == 3
    assert result["content"] == "line 50\nline 51\nline 52\n"


def test_read_edit_context_reports_truncated_scan_when_match_not_found_within_cap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "late_match.txt"
    path.write_text(("alpha\n" * 100) + "needle\n", encoding="utf-8")

    result = read_edit_context(str(path), "needle", context_lines=1, max_scan_bytes=64)

    assert result["status"] == "ok"
    assert result["found"] is False
    assert result["truncated"] is True


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW not available")
def test_write_file_rejects_existing_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("old\n", encoding="utf-8")
    link.symlink_to(target)

    result = write_file(str(link), "new\n")

    assert result["status"] == "error"
    assert "Read" in result["error"] or "read" in result["error"]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_list_dir_returns_structured_metadata(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("print('hi')\n", encoding="utf-8")

    result = list_dir(str(tmp_path))

    assert result["status"] == "ok"
    entries = {entry["name"]: entry for entry in result["entries"]}
    assert entries["pkg"]["type"] == "dir"
    assert entries["pkg"]["is_dir"] is True
    assert entries["pkg"]["relative_path"] == "pkg"
    assert "modified_at" in entries["pkg"]


def test_glob_files_returns_structured_matches(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "one.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b" / "two.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "b" / "note.txt").write_text("skip\n", encoding="utf-8")

    result = glob_files("**/*.py", str(tmp_path))

    assert result["status"] == "ok"
    assert result["count"] == 2
    rel_paths = {entry["relative_path"] for entry in result["matches"]}
    assert rel_paths == {"a/one.py", "b/two.py"}


def test_grep_files_python_fallback_skips_large_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = tmp_path / "small.txt"
    large = tmp_path / "large.txt"
    small.write_text("needle\n", encoding="utf-8")
    large.write_text("needle\n" + ("x" * (DEFAULT_FILESYSTEM.max_file_bytes + 1)), encoding="utf-8")

    monkeypatch.setattr(DEFAULT_FILESYSTEM, "_try_rg", lambda **_: None)

    result = grep_files("needle", path=str(tmp_path), glob="**/*.txt", max_matches=10)

    assert result["status"] == "ok"
    assert result["engine"] == "python"
    assert result["total_matches"] == 1
    assert result["matches"][0]["file"] == "small.txt"


def test_grep_files_python_fallback_returns_submatch_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "offsets.txt"
    path.write_text("needle and another needle\n", encoding="utf-8")

    monkeypatch.setattr(DEFAULT_FILESYSTEM, "_try_rg", lambda **_: None)

    result = grep_files("needle", path=str(tmp_path), glob="**/*.txt", max_matches=10)

    assert result["status"] == "ok"
    assert result["matches"][0]["submatches"] == [
        {"match": "needle", "start": 0, "end": 6},
        {"match": "needle", "start": 19, "end": 25},
    ]


def test_glob_files_excludes_hidden_by_default(tmp_path: Path) -> None:
    (tmp_path / ".secret.py").write_text("print('x')\n", encoding="utf-8")

    default_result = glob_files("**/*.py", str(tmp_path))
    included_result = glob_files("**/*.py", str(tmp_path), include_hidden=True)

    assert default_result["count"] == 0
    assert included_result["count"] == 1


def test_search_code_regex_mode_matches_across_lines(tmp_path: Path) -> None:
    path = tmp_path / "multiline.py"
    path.write_text(
        "def alpha():\n    first()\n    second()\n",
        encoding="utf-8",
    )

    result = search_code(
        r"def alpha\(\):\n\s+first\(\)\n\s+second\(\)",
        path=str(tmp_path),
        glob="**/*.py",
        mode="regex",
    )

    assert result["status"] == "ok"
    assert result["total_matches"] == 1
    assert result["matches"][0]["match_end_line"] == 3
