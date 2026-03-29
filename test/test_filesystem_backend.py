from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from src.tools.core.files import list_dir, read_file, write_file
from src.tools.core.search import glob_files, grep_files
from src.tools.core.filesystem import DEFAULT_FILESYSTEM


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


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW not available")
def test_write_file_rejects_existing_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("old\n", encoding="utf-8")
    link.symlink_to(target)

    result = write_file(str(link), "new\n")

    assert result["status"] == "error"
    assert "Cannot read existing file" in result["error"]
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


def test_grep_files_python_fallback_skips_large_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

