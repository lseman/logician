"""Contract tests for core tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

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


def test_read_file_returns_structured_text_payload() -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as fh:
        fh.write("line 1\nline 2\nline 3\n")
        fh.flush()
        path = fh.name

    try:
        result = read_file(path, start_line=2, end_line=3)
        assert result["status"] == "ok"
        assert result["file_type"] == "text"
        assert result["returned_lines"] == "2-3"
        assert result["content"] == "line 2\nline 3\n"
    finally:
        Path(path).unlink()


def test_write_file_returns_diff_and_persists_text(tmp_path: Path) -> None:
    path = tmp_path / "new_file.txt"
    result = write_file(str(path), "hello world\n")

    assert result["status"] == "ok"
    assert result["bytes_written"] == len("hello world\n".encode("utf-8"))
    assert path.read_text(encoding="utf-8") == "hello world\n"


def test_edit_file_replaces_unique_string(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text("def foo():\n    return 1\n", encoding="utf-8")

    result = edit_file(str(path), "def foo():", "def bar():")

    assert result["status"] == "ok"
    assert "def bar():" in path.read_text(encoding="utf-8")


def test_apply_edit_block_applies_multiple_replacements(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8")
    blocks = (
        "<<<<<<< SEARCH\n"
        "def foo():\n    pass\n"
        "=======\n"
        "def foo_new():\n    pass\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "def bar():\n    pass\n"
        "=======\n"
        "def bar_new():\n    pass\n"
        ">>>>>>> REPLACE"
    )

    result = apply_edit_block(str(path), blocks)

    assert result["status"] == "ok"
    content = path.read_text(encoding="utf-8")
    assert "def foo_new():" in content
    assert "def bar_new():" in content


def test_bash_returns_structured_result() -> None:
    result = bash("echo hello")

    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["stdout"] == "hello\n"


def test_bash_nonzero_exit_sets_error_status() -> None:
    result = bash("exit 42")

    assert result["status"] == "error"
    assert result["returncode"] == 42


def test_glob_files_returns_structured_matches(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file1.py").touch()
    (tmp_path / "file2.py").touch()

    result = glob_files("**/*.py", str(tmp_path))

    assert result["status"] == "ok"
    assert {entry["relative_path"] for entry in result["matches"]} == {"a/file1.py", "file2.py"}


def test_grep_files_returns_structured_matches(tmp_path: Path) -> None:
    (tmp_path / "file1.py").write_text("class MyClass:\n    pass\n", encoding="utf-8")
    (tmp_path / "file2.py").write_text("def my_function():\n    pass\n", encoding="utf-8")

    result = grep_files("class", path=str(tmp_path), glob="**/*.py")

    assert result["status"] == "ok"
    assert result["total_matches"] == 1
    assert result["matches"][0]["file"] == "file1.py"


def test_think_returns_structured_payload() -> None:
    result = think("This is my reasoning")
    assert result["status"] == "ok"
    assert result["thought"] == "This is my reasoning"
    assert result["view"] == "[thought]\nThis is my reasoning"


def test_todo_returns_structured_payload_for_legacy_mode() -> None:
    result = todo(
        [
            {"content": "Task 1", "status": "pending", "activeForm": "Doing task 1"},
            {"content": "Task 2", "status": "completed", "activeForm": "Did task 2"},
        ]
    )

    assert result["status"] == "ok"
    assert len(result["todos"]) == 2
    assert "Task 1" in result["view"]


def test_todo_returns_error_payload_for_invalid_json() -> None:
    result = todo("not valid json")
    assert result["status"] == "error"
    assert "invalid todos JSON" in result["error"]
