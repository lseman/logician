"""Contract tests for core tools."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.tools import Context, ToolCall, ToolRegistry
from src.tools.core import (
    apply_edit_block,
    bash,
    delete_path,
    edit_file,
    fetch_url,
    find_symbol,
    get_git_diff,
    get_process_output,
    glob_files,
    grep_files,
    kill_process,
    list_processes,
    lsp_tool,
    mkdir,
    move_path,
    notebook_edit,
    read_edit_context,
    read_file,
    rg_search,
    run_python,
    run_rust,
    search_code,
    send_input_to_process,
    set_venv,
    set_working_directory,
    show_coding_config,
    start_background_process,
    think,
    todo,
    tool_search,
    write_file,
)
from src.tools.core.WebTool import _filter_web_search_results


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


def test_read_edit_context_returns_bounded_match_context(tmp_path: Path) -> None:
    path = tmp_path / "context.py"
    path.write_text(
        "line 1\nline 2\ndef render():\n    return old_value\nline 5\n",
        encoding="utf-8",
    )

    result = read_edit_context(
        str(path),
        "def render():\n    return old_value\n",
        context_lines=1,
    )

    assert result["status"] == "ok"
    assert result["found"] is True
    assert result["line_offset"] == 2
    assert result["match_start_line"] == 3
    assert result["match_end_line"] == 4
    assert result["content"] == "line 2\ndef render():\n    return old_value\nline 5\n"
    assert result["snapshot"]["full_read"] is False


def test_read_file_cached_range_respects_requested_window(tmp_path: Path) -> None:
    path = tmp_path / "window.py"
    path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    first = read_file(str(path))
    assert first["status"] == "ok"

    second = read_file(str(path), start_line=2, end_line=2)

    assert second["status"] == "ok"
    assert second["file_type"] == "file_unchanged"
    assert second["content"] == "line 2\n"


def test_write_file_returns_diff_and_persists_text(tmp_path: Path) -> None:
    path = tmp_path / "new_file.txt"
    result = write_file(str(path), "hello world\n")

    assert result["status"] == "ok"
    assert result["bytes_written"] == len("hello world\n".encode("utf-8"))
    assert path.read_text(encoding="utf-8") == "hello world\n"


def test_edit_file_replaces_unique_string(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    result = edit_file(str(path), "def foo():", "def bar():")

    assert result["status"] == "ok"
    assert "def bar():" in path.read_text(encoding="utf-8")


def test_notebook_edit_replaces_code_cell_and_clears_outputs(tmp_path: Path) -> None:
    path = tmp_path / "demo.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": 7,
                        "outputs": [{"output_type": "stream", "text": "old\n"}],
                        "source": "print('old')\n",
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    result = notebook_edit(
        str(path),
        action="replace",
        cell_index=0,
        source="print('new')\n",
    )

    assert result["status"] == "ok"
    assert result["edited_cell_index"] == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cells"][0]["source"] == "print('new')\n"
    assert payload["cells"][0]["outputs"] == []
    assert payload["cells"][0]["execution_count"] is None


def test_structured_filesystem_tools_create_move_and_delete(tmp_path: Path) -> None:
    created = mkdir(str(tmp_path / "nested" / "dir"))
    assert created["status"] == "ok"
    assert (tmp_path / "nested" / "dir").is_dir()

    source = tmp_path / "nested" / "dir" / "demo.txt"
    source.write_text("hello\n", encoding="utf-8")
    moved = move_path(str(source), str(tmp_path / "moved.txt"))
    assert moved["status"] == "ok"
    assert not source.exists()
    assert (tmp_path / "moved.txt").read_text(encoding="utf-8") == "hello\n"

    deleted = delete_path(str(tmp_path / "moved.txt"))
    assert deleted["status"] == "ok"
    assert not (tmp_path / "moved.txt").exists()


def test_tool_search_finds_registered_tools() -> None:
    registry = ToolRegistry(auto_load_from_skills=False)
    ctx = Context()
    registry.install_context(ctx)
    registry.register(
        name="demo_python_helper",
        description="Run a Python helper command.",
        parameters=[],
        function=lambda: {"status": "ok"},
    )
    registry._register_collected_python_tools(
        tool_entries=[(tool_search, getattr(tool_search, "__llm_tool_meta__", {}))],
        module_path=Path("src/tools/core/__init__.py"),
        skill_id="core",
        skill_meta={"always_on": True},
    )

    raw = registry.execute(
        ToolCall(id="tool_search_1", name="tool_search", arguments={"query": "python helper"}),
        use_toon=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "ok"
    assert any(match["name"] == "demo_python_helper" for match in payload["matches"])


def test_apply_edit_block_applies_multiple_replacements(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8")
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"
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


def test_edit_file_requires_fresh_read_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "stale.py"
    path.write_text("value = 1\n", encoding="utf-8")

    first_read = read_file(str(path))
    assert first_read["status"] == "ok"

    path.write_text("value = 2\n", encoding="utf-8")

    result = edit_file(str(path), "value = 1\n", "value = 3\n")

    assert result["status"] == "error"
    assert result["reason"] == "stale_snapshot"


def test_edit_file_returns_line_aware_closest_matches(tmp_path: Path) -> None:
    path = tmp_path / "closest.py"
    path.write_text(
        "def render_header():\n    return title\n\ndef render_body():\n    return body\n",
        encoding="utf-8",
    )
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    result = edit_file(
        str(path),
        "def render_boddy():\n    return body\n",
        "def render_body():\n    return body_text\n",
    )

    assert result["status"] == "error"
    assert result["suggested_tool"] == "read_edit_context"
    assert result["closest_matches"]
    first = result["closest_matches"][0]
    assert first["start_line"] == 4
    assert first["end_line"] >= 5
    assert "def render_body():" in first["content"]


def test_edit_file_replace_all_updates_every_match(tmp_path: Path) -> None:
    path = tmp_path / "replace_all.py"
    path.write_text("item = foo\nprint(foo)\nfoo = foo + 1\n", encoding="utf-8")
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    result = edit_file(str(path), "foo", "bar", replace_all=True)

    assert result["status"] == "ok"
    assert result["replace_all"] is True
    assert result["matches_replaced"] == 4
    assert "foo" not in path.read_text(encoding="utf-8")


def test_write_file_rejects_stale_snapshot_for_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "rewrite.py"
    path.write_text("alpha\n", encoding="utf-8")
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    path.write_text("beta\n", encoding="utf-8")

    result = write_file(str(path), "gamma\n")

    assert result["status"] == "error"
    assert result["reason"] == "stale_snapshot"


def test_write_file_reports_unchanged_when_content_matches(tmp_path: Path) -> None:
    path = tmp_path / "same.txt"
    path.write_text("alpha\n", encoding="utf-8")
    read_result = read_file(str(path))
    assert read_result["status"] == "ok"

    result = write_file(str(path), "alpha\n")

    assert result["status"] == "ok"
    assert result["unchanged"] is True
    assert result["bytes_written"] == 0
    assert result["diff"] == ""


def test_bash_returns_structured_result() -> None:
    result = bash("echo hello")

    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["stdout"] == "hello\n"
    assert result["validation"]["is_read_only"] is True


def test_bash_nonzero_exit_sets_error_status() -> None:
    result = bash("exit 42")

    assert result["status"] == "error"
    assert result["returncode"] == 42


def test_bash_grep_no_matches_is_informational_not_error(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    result = bash(f'grep "gamma" "{path}"')

    assert result["status"] == "ok"
    assert result["returncode"] == 1
    assert result["message"] == "No matches found"


def test_bash_diff_exit_one_is_informational_not_error(tmp_path: Path) -> None:
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("alpha\n", encoding="utf-8")
    right.write_text("beta\n", encoding="utf-8")

    result = bash(f'diff "{left}" "{right}"', require_read_only=True)

    assert result["status"] == "ok"
    assert result["returncode"] == 1
    assert result["message"] == "Files differ"
    assert result["validation"]["is_read_only"] is True


def test_bash_rejects_dangerous_commands() -> None:
    result = bash("rm -rf /tmp/logician-test")

    assert result["status"] == "error"
    assert result["reason"] == "dangerous_command"


def test_bash_require_read_only_rejects_mutating_commands() -> None:
    result = bash("touch created-by-bash.txt", require_read_only=True)

    assert result["status"] == "error"
    assert result["reason"] == "read_only_required"


def test_bash_require_read_only_allows_find_queries(tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("print('ok')\n", encoding="utf-8")

    result = bash(
        f'cd "{tmp_path}" && find . -name "*.py"',
        require_read_only=True,
    )

    assert result["status"] == "ok"
    assert result["validation"]["is_read_only"] is True


def test_bash_rejects_mutating_git_config_in_read_only_mode() -> None:
    result = bash("git config user.name example", require_read_only=True)

    assert result["status"] == "error"
    assert result["reason"] == "read_only_required"


def test_bash_rejects_find_exec_as_dangerous() -> None:
    result = bash("find . -name '*.py' -exec rm {} +")

    assert result["status"] == "error"
    assert result["reason"] == "dangerous_command"


def test_bash_background_mode_starts_process_and_returns_handle() -> None:
    result = bash("sleep 10", background=True)

    assert result["status"] == "ok"
    assert result["background"] is True
    assert "process_name" in result

    kill_result = kill_process(result["process_name"])
    assert kill_result["status"] == "ok"


def test_web_search_filters_allowed_and_blocked_domains() -> None:
    results = [
        {"title": "Example", "url": "https://example.com/page", "snippet": ""},
        {"title": "Subdomain", "url": "https://sub.example.org/page", "snippet": ""},
        {"title": "Other", "url": "https://other.com/page", "snippet": ""},
    ]

    filtered = _filter_web_search_results(
        results,
        allowed_domains=["example.com"],
        blocked_domains=["sub.example.org"],
    )

    assert filtered == [
        {"title": "Example", "url": "https://example.com/page", "snippet": ""},
    ]


def test_todo_validate_command_returns_warnings() -> None:
    result = todo(
        command="validate",
        items=[{"title": "", "status": "mystery"}],
    )

    assert result["status"] == "ok"
    assert isinstance(result["warnings"], list)
    assert any("unrecognized status" in warning for warning in result["warnings"])
    assert "verification_hint" in result


def test_todo_validate_warns_about_multiple_in_progress_items() -> None:
    result = todo(
        command="validate",
        items=[
            {"id": 1, "title": "Inspect logs", "status": "in-progress"},
            {"id": 2, "title": "Patch search tool", "status": "in-progress"},
        ],
    )

    assert result["status"] == "ok"
    assert any("Multiple tasks are marked in-progress" in warning for warning in result["warnings"])


def test_rg_search_treats_ripgrep_no_match_as_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr("src.tools.core.SearchTool.tool.shutil.which", lambda name: "/usr/bin/rg")
    monkeypatch.setattr("src.tools.core.SearchTool.tool.subprocess.run", lambda *args, **kwargs: _Proc())

    result = rg_search("needle", directory=str(tmp_path), fixed_string=True)

    assert result["status"] == "ok"
    assert result["tool_used"] == "rg"
    assert result["count"] == 0
    assert result["matches"] == []


def test_core_search_and_task_tools_expose_openclaude_metadata(tmp_path: Path) -> None:
    rg_meta = getattr(rg_search, "__llm_tool_meta__", {})
    rg_validate = rg_meta["validate_input"]
    rg_summary = rg_meta["get_tool_use_summary"]
    rg_activity = rg_meta["get_activity_description"]
    rg_flags = rg_meta["is_search_or_read_command"]

    assert rg_validate({"directory": str(tmp_path)}) is True
    rejected = rg_validate({"directory": str(tmp_path / "missing")})
    assert rejected["result"] is False
    assert "Directory does not exist" in rejected["message"]
    assert rg_summary({"pattern": "ServerConfig", "directory": "src"}) == "ServerConfig in src"
    assert rg_activity({"pattern": "ServerConfig"}) == "Searching for ServerConfig"
    assert rg_flags({}) == {"isSearch": True, "isRead": False, "isList": False}
    assert rg_meta["user_facing_name"]({}) == "Search"

    todo_meta = getattr(todo, "__llm_tool_meta__", {})
    todo_validate = todo_meta["validate_input"]
    assert todo_validate({"command": "explode"})["result"] is False
    assert todo_meta["get_tool_use_summary"]({"command": "mark", "id": 7}) == "mark task 7"
    assert todo_meta["get_activity_description"]({"command": "view"}) == "Reviewing task list"
    assert todo_meta["user_facing_name"]({}) == "Tasks"

    read_meta = getattr(read_file, "__llm_tool_meta__", {})
    assert read_meta["validate_input"]({"path": str(tmp_path / "missing.txt")})["result"] is False
    assert read_meta["is_search_or_read_command"]({}) == {
        "isSearch": False,
        "isRead": True,
        "isList": False,
    }
    assert read_meta["get_activity_description"]({}) == "Reading file"

    bash_meta = getattr(bash, "__llm_tool_meta__", {})
    assert bash_meta["validate_input"]({"command": "echo hi", "timeout": 5}) is True
    assert bash_meta["user_facing_name"]({}) == "Terminal"

    run_python_meta = getattr(run_python, "__llm_tool_meta__", {})
    assert run_python_meta["validate_input"]({"code": "print('hi')", "timeout": 5}) is True
    assert run_python_meta["user_facing_name"]({}) == "Python"


def test_run_python_rejects_invalid_cwd(tmp_path: Path) -> None:
    result = run_python("print('hi')", cwd=str(tmp_path / "missing"))

    assert result["status"] == "error"
    assert "Working directory not found" in result["error"]


def test_fetch_url_rejects_non_http_scheme() -> None:
    result = fetch_url("file:///tmp/example.txt")

    assert result["status"] == "error"
    assert "http or https" in result["error"]


def test_get_git_diff_uses_repo_relative_paths_for_directories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Logician"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "logician@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    tracked = nested / "demo.py"
    tracked.write_text("print('a')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

    tracked.write_text("print('b')\n", encoding="utf-8")

    result = get_git_diff(str(nested))

    assert result["status"] == "ok"
    assert result["repo_root"] == str(repo.resolve())
    assert "src/demo.py" in result["files_changed"]


def test_find_symbol_include_calls_still_honors_match_limit(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    body = ["def helper():\n    return 1\n"]
    body.extend("helper()\n" for _ in range(80))
    path.write_text("".join(body), encoding="utf-8")

    result = find_symbol("helper", directory=str(tmp_path), include_calls=True)

    assert result["status"] == "ok"
    assert result["count"] <= 60
    assert len(result["matches"]) <= 60


def test_rust_tools_are_registered_in_core_surface() -> None:
    from src.tools import core as core_mod

    assert callable(run_rust)
    assert "run_rust" in core_mod.CORE_TOOL_NAMES


def test_libcst_tools_only_surface_when_dependency_is_available() -> None:
    from src.tools import core as core_mod

    libcst_available = importlib.util.find_spec("libcst") is not None

    assert ("edit_file_libcst" in core_mod.CORE_TOOL_NAMES) is libcst_available
    assert ("find_function_by_name" in core_mod.CORE_TOOL_NAMES) is libcst_available


def test_set_working_directory_affects_core_python_execution(tmp_path: Path) -> None:
    original_cwd = os.getcwd()
    try:
        set_result = set_working_directory(str(tmp_path))
        assert set_result["status"] == "ok"

        run_result = run_python("import os\nprint(os.getcwd())\n")

        assert run_result["status"] == "ok"
        assert run_result["stdout"].strip() == str(tmp_path)
    finally:
        reset_result = set_working_directory(original_cwd)
        assert reset_result["status"] == "ok"


def test_set_venv_and_show_coding_config_share_runtime_state(tmp_path: Path) -> None:
    fake_venv = tmp_path / ".venv"
    bin_dir = fake_venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text("# mock activate\n", encoding="utf-8")
    (bin_dir / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    set_result = set_venv(str(fake_venv))
    config_result = show_coding_config()

    assert set_result["status"] == "ok"
    assert config_result["status"] == "ok"
    assert config_result["venv_path"] == str(fake_venv.resolve())


def test_background_process_tools_share_core_runtime_state(tmp_path: Path) -> None:
    name = "core-echo-process"
    command = (
        f"{sys.executable} -u -c \"import sys; print('ready', flush=True); "
        "[print(line, end='', flush=True) for line in sys.stdin]\""
    )
    original_cwd = os.getcwd()
    try:
        cwd_result = set_working_directory(str(tmp_path))
        assert cwd_result["status"] == "ok"

        started = start_background_process(command, name)
        assert started["status"] == "ok"
        assert started["cwd"] == str(tmp_path)

        listed = list_processes()
        assert listed["status"] == "ok"
        assert any(item["name"] == name for item in listed["processes"])

        sent = send_input_to_process(name, "hello from core\n")
        assert sent["status"] == "ok"

        time.sleep(0.2)
        output = get_process_output(name, tail_lines=10)
        assert output["status"] == "ok"
        assert "ready" in output["output"]
        assert "hello from core" in output["output"]
    finally:
        kill_process(name, force=True)
        reset_result = set_working_directory(original_cwd)
        assert reset_result["status"] == "ok"


def test_glob_files_returns_structured_matches(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file1.py").touch()
    (tmp_path / "file2.py").touch()

    result = glob_files("**/*.py", str(tmp_path))

    assert result["status"] == "ok"
    assert {entry["relative_path"] for entry in result["matches"]} == {"a/file1.py", "file2.py"}


def test_glob_files_supports_paths_mode_and_pagination(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).touch()

    result = glob_files(
        "*.py",
        str(tmp_path),
        output_mode="paths",
        offset=1,
        max_results=1,
    )

    assert result["status"] == "ok"
    assert result["paths"] == ["b.py"]
    assert result["truncated"] is True
    assert result["next_offset"] == 2


def test_grep_files_returns_structured_matches(tmp_path: Path) -> None:
    (tmp_path / "file1.py").write_text("class MyClass:\n    pass\n", encoding="utf-8")
    (tmp_path / "file2.py").write_text("def my_function():\n    pass\n", encoding="utf-8")

    result = grep_files("class", path=str(tmp_path), glob="**/*.py")

    assert result["status"] == "ok"
    assert result["total_matches"] == 1
    assert result["matches"][0]["file"] == "file1.py"
    assert result["matches"][0]["submatches"][0]["match"] == "class"


def test_grep_files_supports_files_mode_and_pagination(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("class B:\n    pass\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("class C:\n    pass\n", encoding="utf-8")

    result = grep_files(
        "class",
        path=str(tmp_path),
        glob="**/*.py",
        output_mode="files_with_matches",
        offset=1,
        max_matches=1,
    )

    assert result["status"] == "ok"
    assert result["paths"] == ["b.py"]
    assert result["truncated"] is True
    assert result["next_offset"] == 2


def test_grep_files_can_include_hidden_files(tmp_path: Path) -> None:
    (tmp_path / ".hidden.py").write_text("class Hidden:\n    pass\n", encoding="utf-8")

    hidden_default = grep_files("class", path=str(tmp_path), glob="**/*.py")
    hidden_included = grep_files(
        "class",
        path=str(tmp_path),
        glob="**/*.py",
        include_hidden=True,
    )

    assert hidden_default["total_matches"] == 0
    assert hidden_included["total_matches"] == 1


def test_search_code_literal_supports_multiline_matches(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "def render():\n    return old_value\n\nprint('done')\n",
        encoding="utf-8",
    )

    result = search_code(
        "def render():\n    return old_value\n",
        path=str(tmp_path),
        glob="**/*.py",
        mode="literal",
    )

    assert result["status"] == "ok"
    assert result["total_matches"] == 1
    assert result["matches"][0]["match_start_line"] == 1
    assert "return old_value" in result["matches"][0]["content"]


def test_search_code_symbol_finds_python_definitions(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "class RenderBox:\n    pass\n\ndef render_view():\n    return 1\n",
        encoding="utf-8",
    )

    result = search_code(
        "render", path=str(tmp_path), glob="**/*.py", mode="symbol", case_sensitive=False
    )

    assert result["status"] == "ok"
    assert result["total_matches"] == 2
    assert {item["symbol"] for item in result["matches"]} == {"RenderBox", "render_view"}


def test_lsp_tool_document_symbols_lists_python_definitions(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "class RenderBox:\n    pass\n\ndef render_view():\n    return 1\n",
        encoding="utf-8",
    )

    result = lsp_tool("document_symbols", str(path))

    assert result["status"] == "ok"
    assert {item["name"] for item in result["results"]} == {"RenderBox", "render_view"}


def test_lsp_tool_definition_and_references_follow_symbol(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "def render_view():\n    return 1\n\nvalue = render_view()\n",
        encoding="utf-8",
    )

    definition = lsp_tool(
        "go_to_definition",
        str(path),
        line=4,
        character=10,
    )
    references = lsp_tool(
        "find_references",
        str(path),
        query="render_view",
    )

    assert definition["status"] == "ok"
    assert definition["results"][0]["line"] == 1
    assert references["status"] == "ok"
    assert len(references["results"]) >= 2


def test_lsp_tool_implementation_and_call_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    path.write_text(
        "def helper():\n    return 1\n\n"
        "def render_view():\n    return helper()\n\n"
        "value = render_view()\n",
        encoding="utf-8",
    )

    implementation = lsp_tool(
        "go_to_implementation",
        str(path),
        line=4,
        character=10,
    )
    assert implementation["status"] == "ok"
    assert implementation["result_count"] >= 1
    assert implementation["results"][0]["line"] == 4

    prepare = lsp_tool(
        "prepare_call_hierarchy",
        str(path),
        line=4,
        character=10,
    )
    assert prepare["status"] == "ok"
    assert prepare["result_count"] >= 1

    incoming = lsp_tool(
        "incoming_calls",
        str(path),
        query="render_view",
    )
    assert incoming["status"] == "ok"
    assert incoming["result_count"] >= 1

    outgoing = lsp_tool(
        "outgoing_calls",
        str(path),
        query="render_view",
    )
    assert outgoing["status"] == "ok"


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
