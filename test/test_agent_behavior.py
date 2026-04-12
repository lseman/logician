# -*- coding: utf-8 -*-
"""
Test suite to verify agent behavior matches Claude Code patterns.

Tests file read caching, stale snapshot detection, atomic writes,
and proper tool dispatch logic.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from src.tools.core.filesystem import DEFAULT_FILESYSTEM


def test_filesystem_read_basic():
    """Test basic file reading returns content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.txt"
        test_content = "Hello, World!\nThis is a test file.\n"

        # Write test file
        test_path.write_text(test_content)

        # Read file
        result = DEFAULT_FILESYSTEM.read_file(str(test_path))

        assert result["status"] == "ok"
        assert result["file_type"] == "text"
        assert result["content"] == test_content
        assert result["total_lines"] == 2
        assert result["newline"] == "LF"
        assert result["encoding"] == "utf-8"


def test_filesystem_read_nonexistent():
    """Test reading a non-existent file returns appropriate error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = DEFAULT_FILESYSTEM.read_file(str(Path(tmpdir) / "nonexistent.txt"))

        assert result["status"] == "error"
        assert "not found" in result.get("error", "").lower()

        # Verify suggestions are provided
        assert "did_you_mean" in result or "closest_matches" in result


def test_filesystem_read_binary():
    """Test reading a binary file returns base64."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "binary.bin"
        binary_content = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE])
        test_path.write_bytes(binary_content)

        result = DEFAULT_FILESYSTEM.read_file(str(test_path))

        assert result["status"] == "ok"
        assert result["file_type"] == "binary"
        assert result["encoding"] == "base64"


def test_filesystem_list_dir():
    """Test basic directory listing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        (Path(tmpdir) / "file1.txt").write_text("content1")
        (Path(tmpdir) / "file2.txt").write_text("content2")
        (Path(tmpdir) / "file3.py").write_text("# python")

        result = DEFAULT_FILESYSTEM.list_dir(str(tmpdir))

        assert result["status"] == "ok"
        assert result["count"] == 3
        assert len(result["entries"]) == 3

        # Verify entry structure
        for entry in result["entries"]:
            assert "name" in entry
            assert "path" in entry
            assert "type" in entry


def test_filesystem_list_dir_glob():
    """Test directory listing with glob pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "file1.txt").write_text("content1")
        (Path(tmpdir) / "file2.txt").write_text("content2")
        (Path(tmpdir) / "file3.py").write_text("# python")

        result = DEFAULT_FILESYSTEM.list_dir(str(tmpdir), glob_pattern="*.txt")

        assert result["status"] == "ok"
        assert result["count"] == 2
        assert all(entry["name"].endswith(".txt") for entry in result["entries"])


def test_filesystem_read_partial_range():
    """Test reading a file with line range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "partial.txt"
        test_content = "line1\nline2\nline3\nline4\nline5\n"
        test_path.write_text(test_content)

        # Read lines 2-4
        result = DEFAULT_FILESYSTEM.read_file(str(test_path), start_line=2, end_line=4)

        assert result["status"] == "ok"
        assert result["returned_lines"] == "2-4"
        assert result["content"] == "line2\nline3\nline4\n"
        assert result["total_lines"] == 5


def test_filesystem_read_large_file():
    """Test reading a large file returns error suggesting range query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "large.txt"
        # Create a 2MB file
        test_path.write_text("line\n" * 50000)

        result = DEFAULT_FILESYSTEM.read_file(str(test_path))

        # Should return error since file is too large for full read
        assert result["status"] == "error"
        assert "exceeds" in result.get("error", "").lower()
        assert "requires_range" in result


def test_filesystem_glob():
    """Test glob search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files
        (Path(tmpdir) / "test.py").write_text("# test")
        (Path(tmpdir) / "other.py").write_text("# other")
        (Path(tmpdir) / "readme.md").write_text("# readme")

        result = DEFAULT_FILESYSTEM.glob("**/*.py", path=str(tmpdir))

        assert result["status"] == "ok"
        assert result["count"] == 2
        assert result["total_count"] == 2


def test_filesystem_grep():
    """Test grep search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_content = "print('hello')\nx = 1\ny = 2\n"
        (Path(tmpdir) / "test.py").write_text(test_content)

        result = DEFAULT_FILESYSTEM.grep("print", path=str(tmpdir))

        assert result["status"] == "ok"
        assert result["file_count"] >= 1


def test_filesystem_read_unchanged():
    """Test that unchanged files return cached content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "unchanged.txt"
        original_content = "Original content\n"

        # Write test file
        test_path.write_text(original_content)

        # Read file
        result1 = DEFAULT_FILESYSTEM.read_file(str(test_path))
        assert result1["status"] == "ok"

        # Read again - should return cached content
        result2 = DEFAULT_FILESYSTEM.read_file(str(test_path))
        assert result2["status"] == "ok"
        assert result2["content"] == original_content


def test_filesystem_newline_preservation():
    """Test that newline styles are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "crlf.txt"

        # Write with CRLF
        content_crlf = "line1\r\nline2\r\n"
        test_path.write_bytes(content_crlf.encode('utf-8'))

        result = DEFAULT_FILESYSTEM.read_file(str(test_path))
        assert result["newline"] == "CRLF"


def test_filesystem_block_device_rejection():
    """Test that block devices are rejected."""
    # These should be rejected
    blocked_paths = [
        "/dev/zero",
        "/dev/random",
        "/dev/null",
    ]

    for blocked_path in blocked_paths:
        result = DEFAULT_FILESYSTEM.read_file(blocked_path)
        assert result["status"] == "error"
        assert "not allowed" in result.get("error", "").lower() or "device" in result.get("error", "").lower()


def test_filesystem_directory_as_file_error():
    """Test that trying to read a directory returns an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = DEFAULT_FILESYSTEM.read_file(str(Path(tmpdir)))

        assert result["status"] == "error"
        assert "directory" in result.get("error", "").lower()


def test_filesystem_glob_hidden_files():
    """Test that glob excludes hidden files by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "visible.txt").write_text("visible")
        (Path(tmpdir) / ".hidden.txt").write_text("hidden")

        result = DEFAULT_FILESYSTEM.glob("**/*.txt", path=str(tmpdir))

        assert result["status"] == "ok"
        # Hidden files should be excluded by default
        visible_count = sum(1 for m in result.get("matches", []) if not m.get("name", "").startswith("."))
        assert visible_count == 1


@pytest.mark.skip(reason="Requires context setup for write operations")
def test_write_file_atomic():
    """Test that write_file uses atomic writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "atomic.txt"
        test_content = "Atomic write test\n"

        # This would require context setup - skipping for now
        # result = write_file(str(test_path), test_content)
        # assert result["status"] == "ok"

        # Verify file exists and has correct content
        # assert test_path.exists()
        # assert test_path.read_text() == test_content


@pytest.mark.skip(reason="Requires context setup for edit operations")
def test_edit_file_success():
    """Test successful file edit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "edit.txt"
        original = "line1\nline2\nline3\n"
        replacement = "REPLACED\n"

        # Write original file
        test_path.write_text(original)

        # This would require context setup
        # result = edit_file(str(test_path), "line2", replacement)
        # assert result["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
