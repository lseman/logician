"""Tests for src/tools/core modules."""
from __future__ import annotations

import json
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from src.tools.core import (
    apply_edit_block,
    bash,
    edit_file,
    glob_files,
    grep_files,
    read_file,
    think,
    todo,
    write_file,
)


class TestReadFile:
    def test_read_existing_file(self) -> None:
        """Test reading an existing file returns content with line numbers."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("line 1\nline 2\nline 3\n")
            f.flush()
            try:
                result = read_file(f.name)
                assert "1\tline 1" in result
                assert "2\tline 2" in result
                assert "3\tline 3" in result
            finally:
                Path(f.name).unlink()

    def test_read_missing_file(self) -> None:
        """Test reading a missing file returns error."""
        result = read_file("/nonexistent/path/file.txt")
        assert "Error: file not found" in result

    def test_read_with_line_range(self) -> None:
        """Test reading a subset of lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            f.flush()
            try:
                result = read_file(f.name, start_line=2, end_line=4)
                assert "2\tline 2" in result
                assert "3\tline 3" in result
                assert "4\tline 4" in result
                assert "1\tline 1" not in result
                assert "5\tline 5" not in result
            finally:
                Path(f.name).unlink()

    def test_read_start_line_only(self) -> None:
        """Test reading from a start line to EOF."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("a\nb\nc\nd\n")
            f.flush()
            try:
                result = read_file(f.name, start_line=3)
                assert "3\tc" in result
                assert "4\td" in result
                assert "1\ta" not in result
            finally:
                Path(f.name).unlink()

    def test_read_end_line_only(self) -> None:
        """Test reading from start to an end line."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("a\nb\nc\nd\n")
            f.flush()
            try:
                result = read_file(f.name, end_line=2)
                assert "1\ta" in result
                assert "2\tb" in result
                assert "3\tc" not in result
            finally:
                Path(f.name).unlink()


class TestWriteFile:
    def test_write_new_file(self) -> None:
        """Test writing a new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "new_file.txt")
            result = write_file(path, "hello world")
            assert "Written:" in result
            assert Path(path).read_text() == "hello world"

    def test_write_creates_parent_dirs(self) -> None:
        """Test that write_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "a" / "b" / "c" / "file.txt")
            result = write_file(path, "nested content")
            assert "Written:" in result
            assert Path(path).read_text() == "nested content"

    def test_write_overwrites_existing(self) -> None:
        """Test that write_file overwrites existing files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("old content")
            f.flush()
            try:
                result = write_file(f.name, "new content")
                assert "Written:" in result
                assert Path(f.name).read_text() == "new content"
            finally:
                Path(f.name).unlink()

    def test_write_returns_byte_count(self) -> None:
        """Test that result includes byte count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "file.txt")
            content = "test"
            result = write_file(path, content)
            assert f"({len(content)} bytes)" in result


class TestEditFile:
    def test_edit_replaces_unique_string(self) -> None:
        """Test editing a file with a unique old_string."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("def foo():\n    pass\n")
            f.flush()
            try:
                result = edit_file(f.name, "def foo():", "def bar():")
                assert "Edited:" in result
                assert "def bar():" in Path(f.name).read_text()
            finally:
                Path(f.name).unlink()

    def test_edit_errors_on_missing_file(self) -> None:
        """Test editing a missing file returns error."""
        result = edit_file("/nonexistent/file.txt", "old", "new")
        assert "Error: file not found" in result

    def test_edit_errors_on_not_found(self) -> None:
        """Test editing with not-found old_string returns error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content")
            f.flush()
            try:
                result = edit_file(f.name, "not here", "replacement")
                assert "Error: old_string not found" in result
            finally:
                Path(f.name).unlink()

    def test_edit_errors_on_non_unique(self) -> None:
        """Test editing with non-unique old_string returns error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("same\nsame\nsame\n")
            f.flush()
            try:
                result = edit_file(f.name, "same", "different")
                assert "Error: old_string found" in result
                assert "must be unique" in result
            finally:
                Path(f.name).unlink()


class TestApplyEditBlock:
    def test_apply_single_block(self) -> None:
        """Test applying a single SEARCH/REPLACE block."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("def old_func():\n    return 42\n")
            f.flush()
            try:
                blocks = textwrap.dedent(
                    """
                    <<<<<<< SEARCH
                    def old_func():
                        return 42
                    =======
                    def new_func():
                        return 99
                    >>>>>>> REPLACE
                    """
                ).strip()
                result = apply_edit_block(f.name, blocks)
                assert "Applied 1 edit block(s)" in result
                content = Path(f.name).read_text()
                assert "def new_func():" in content
                assert "return 99" in content
            finally:
                Path(f.name).unlink()

    def test_apply_multiple_blocks(self) -> None:
        """Test applying multiple SEARCH/REPLACE blocks."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("def foo():\n    pass\n\ndef bar():\n    pass\n")
            f.flush()
            try:
                blocks = textwrap.dedent(
                    """
                    <<<<<<< SEARCH
                    def foo():
                        pass
                    =======
                    def foo_new():
                        pass
                    >>>>>>> REPLACE
                    <<<<<<< SEARCH
                    def bar():
                        pass
                    =======
                    def bar_new():
                        pass
                    >>>>>>> REPLACE
                    """
                ).strip()
                result = apply_edit_block(f.name, blocks)
                assert "Applied 2 edit block(s)" in result
                content = Path(f.name).read_text()
                assert "def foo_new():" in content
                assert "def bar_new():" in content
            finally:
                Path(f.name).unlink()

    def test_apply_block_errors_on_missing_file(self) -> None:
        """Test applying blocks to missing file returns error."""
        blocks = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        result = apply_edit_block("/nonexistent/file.txt", blocks)
        assert "Error: file not found" in result

    def test_apply_block_errors_on_no_blocks(self) -> None:
        """Test applying with no valid blocks returns error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content")
            f.flush()
            try:
                result = apply_edit_block(f.name, "not a valid block")
                assert "Error: no valid SEARCH/REPLACE blocks" in result
            finally:
                Path(f.name).unlink()

    def test_apply_block_errors_on_missing_search_text(self) -> None:
        """Test applying block where SEARCH text not in file returns error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("original content")
            f.flush()
            try:
                blocks = textwrap.dedent(
                    """
                    <<<<<<< SEARCH
                    not in file
                    =======
                    replacement
                    >>>>>>> REPLACE
                    """
                ).strip()
                result = apply_edit_block(f.name, blocks)
                assert "Error: SEARCH text not found" in result
            finally:
                Path(f.name).unlink()


class TestBash:
    def test_bash_simple_command(self) -> None:
        """Test running a simple bash command."""
        result = bash("echo hello")
        assert "hello" in result

    def test_bash_exit_code_in_output(self) -> None:
        """Test that non-zero exit codes appear in output."""
        result = bash("exit 42")
        assert "exit code: 42" in result

    def test_bash_stderr_included(self) -> None:
        """Test that stderr is included in output."""
        result = bash("echo error >&2 && echo stdout")
        assert "error" in result
        assert "stdout" in result

    def test_bash_timeout(self) -> None:
        """Test that timeout is honored."""
        result = bash("sleep 10", timeout=1)
        assert "Error: command timed out" in result

    def test_bash_zero_exit_success(self) -> None:
        """Test successful command doesn't include exit code."""
        result = bash("echo test")
        assert "exit code:" not in result
        assert "test" in result


class TestGlobFiles:
    def test_glob_finds_py_files(self) -> None:
        """Test finding Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").touch()
            Path(tmpdir, "file2.py").touch()
            Path(tmpdir, "file3.txt").touch()
            result = glob_files("*.py", tmpdir)
            assert "file1.py" in result
            assert "file2.py" in result
            assert "file3.txt" not in result

    def test_glob_no_matches(self) -> None:
        """Test when no files match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file.txt").touch()
            result = glob_files("*.py", tmpdir)
            assert "No files matching" in result

    def test_glob_missing_dir(self) -> None:
        """Test globbing a missing directory returns error."""
        result = glob_files("*.py", "/nonexistent/path")
        assert "Error: directory not found" in result

    def test_glob_recursive_pattern(self) -> None:
        """Test recursive glob patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a").mkdir()
            Path(tmpdir, "a", "file1.py").touch()
            Path(tmpdir, "b").mkdir()
            Path(tmpdir, "b", "file2.py").touch()
            result = glob_files("**/*.py", tmpdir)
            assert "file1.py" in result
            assert "file2.py" in result


class TestGrepFiles:
    def test_grep_files_with_matches(self) -> None:
        """Test finding files containing a pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").write_text("class MyClass:\n    pass\n")
            Path(tmpdir, "file2.py").write_text("def my_function():\n    pass\n")
            result = grep_files("class", tmpdir, "*.py")
            assert "file1.py" in result
            assert "file2.py" not in result

    def test_grep_files_content_mode(self) -> None:
        """Test content output mode with line numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("line 1\nmatch here\nline 3\n")
            result = grep_files("match", tmpdir, "*.py", output_mode="content")
            assert "test.py:2:" in result
            assert "match here" in result

    def test_grep_files_count_mode(self) -> None:
        """Test count output mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("match\nmatch\nno\n")
            result = grep_files("match", tmpdir, "*.py", output_mode="count")
            assert "test.py: 2" in result

    def test_grep_no_matches(self) -> None:
        """Test when no matches found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("some content\n")
            result = grep_files("nomatch", tmpdir, "*.py")
            assert "No matches" in result

    def test_grep_invalid_regex(self) -> None:
        """Test invalid regex returns error."""
        result = grep_files("[invalid(regex", ".", "*")
        assert "Error: invalid regex pattern" in result


class TestThink:
    def test_think_returns_thought_in_block(self) -> None:
        """Test that think() wraps thought in [thought] block."""
        thought = "This is my reasoning"
        result = think(thought)
        assert "[thought]" in result
        assert thought in result

    def test_think_multiline(self) -> None:
        """Test thinking with multiline content."""
        thought = "Line 1\nLine 2\nLine 3"
        result = think(thought)
        assert "[thought]" in result
        assert "Line 1" in result
        assert "Line 3" in result


class TestTodo:
    def test_todo_formats_list(self) -> None:
        """Test formatting a todo list."""
        todos = [
            {"content": "Task 1", "status": "pending", "activeForm": "Doing task 1"},
            {
                "content": "Task 2",
                "status": "in_progress",
                "activeForm": "Doing task 2",
            },
            {"content": "Task 3", "status": "completed", "activeForm": "Did task 3"},
        ]
        result = todo(todos)
        assert "## Task List" in result
        assert "[ ] Task 1" in result
        assert "[→] Task 2" in result
        assert "[x] Task 3" in result

    def test_todo_accepts_json_string(self) -> None:
        """Test that todo() accepts JSON string input."""
        todos_list = [
            {"content": "Fix bug", "status": "in_progress", "activeForm": "Fixing"}
        ]
        todos_json = json.dumps(todos_list)
        result = todo(todos_json)
        assert "[→] Fix bug" in result

    def test_todo_invalid_json(self) -> None:
        """Test error on invalid JSON."""
        result = todo("not valid json")
        assert "Error: invalid todos JSON" in result

    def test_todo_not_list(self) -> None:
        """Test error when todos is not a list."""
        result = todo({"content": "not a list"})
        assert "Error: todos must be a list" in result

    def test_todo_empty_list(self) -> None:
        """Test empty todo list."""
        result = todo([])
        assert "## Task List" in result

    def test_todo_ignores_non_dict_items(self) -> None:
        """Test that non-dict items in list are ignored."""
        todos = [
            {"content": "Task 1", "status": "pending", "activeForm": "Doing"},
            "not a dict",
            {"content": "Task 2", "status": "completed", "activeForm": "Done"},
        ]
        result = todo(todos)
        assert "Task 1" in result
        assert "Task 2" in result
        assert "not a dict" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
