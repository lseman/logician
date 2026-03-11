"""Tests for cc_tools skill."""
from __future__ import annotations
from pathlib import Path
import pytest

SKILL_MODULE = Path(__file__).resolve().parents[1] / "skills/coding/cc_tools/scripts/tools.py"

def _load_tools_module() -> dict:
    ns: dict = {}
    exec(SKILL_MODULE.read_text(), ns, ns)
    return ns

@pytest.fixture(scope="module")
def tools_ns() -> dict:
    return _load_tools_module()

# cc_glob
def test_cc_glob_returns_matching_files(tools_ns, tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    result = tools_ns["cc_glob"]("*.py", path=str(tmp_path))
    assert "a.py" in result
    assert "b.txt" not in result

def test_cc_glob_head_limit(tools_ns, tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("")
    result = tools_ns["cc_glob"]("*.py", path=str(tmp_path), head_limit=2)
    assert result.count(".py") == 2

def test_cc_glob_no_match(tools_ns, tmp_path):
    result = tools_ns["cc_glob"]("*.xyz", path=str(tmp_path))
    assert "(no matches)" in result

# cc_grep
def test_cc_grep_files_with_matches(tools_ns, tmp_path):
    (tmp_path / "yes.py").write_text("hello world\n")
    (tmp_path / "no.py").write_text("nothing here\n")
    result = tools_ns["cc_grep"]("hello", path=str(tmp_path), output_mode="files_with_matches")
    assert "yes.py" in result
    assert "no.py" not in result

def test_cc_grep_content_mode(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("line1\nhello\nline3\n")
    result = tools_ns["cc_grep"]("hello", path=str(tmp_path), output_mode="content")
    assert "hello" in result

def test_cc_grep_count_mode(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("x\nx\nx\n")
    result = tools_ns["cc_grep"]("x", path=str(tmp_path), output_mode="count")
    assert "3" in result

def test_cc_grep_no_match_returns_no_matches(tools_ns, tmp_path):
    (tmp_path / "f.py").write_text("nothing\n")
    result = tools_ns["cc_grep"]("zzznomatch", path=str(tmp_path))
    assert "(no matches)" in result or result.strip() == ""

# cc_read
def test_cc_read_full_file(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("line1\nline2\nline3\n")
    result = tools_ns["cc_read"](str(f))
    assert "     1\tline1" in result
    assert "     3\tline3" in result

def test_cc_read_with_offset_and_limit(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("\n".join(f"L{i}" for i in range(10)))
    result = tools_ns["cc_read"](str(f), offset=2, limit=3)
    assert "L2" in result
    assert "L5" not in result

def test_cc_read_truncates_long_lines(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("A" * 3000 + "\n")
    result = tools_ns["cc_read"](str(f))
    line_content = result.split("\t", 1)[1]
    assert len(line_content) <= 2000

# cc_edit
def test_cc_edit_basic_replacement(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("hello world\n")
    tools_ns["cc_edit"](str(f), "hello", "goodbye")
    assert f.read_text() == "goodbye world\n"

def test_cc_edit_not_found_raises(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("hello world\n")
    with pytest.raises(Exception, match="not found"):
        tools_ns["cc_edit"](str(f), "zzzmissing", "x")

def test_cc_edit_not_unique_raises(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x\nx\nx\n")
    with pytest.raises(Exception, match="not unique|matches"):
        tools_ns["cc_edit"](str(f), "x", "y")

def test_cc_edit_replace_all(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("x\nx\nx\n")
    tools_ns["cc_edit"](str(f), "x", "y", replace_all=True)
    assert f.read_text() == "y\ny\ny\n"

# cc_write
def test_cc_write_creates_file(tools_ns, tmp_path):
    f = tmp_path / "new.py"
    tools_ns["cc_write"](str(f), "print('hello')\n")
    assert f.read_text() == "print('hello')\n"

def test_cc_write_creates_parent_dirs(tools_ns, tmp_path):
    f = tmp_path / "sub" / "dir" / "new.py"
    tools_ns["cc_write"](str(f), "x\n")
    assert f.read_text() == "x\n"

# cc_multi_edit
def test_cc_multi_edit_sequential(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("alpha beta gamma\n")
    tools_ns["cc_multi_edit"](str(f), [
        {"old_string": "alpha", "new_string": "A"},
        {"old_string": "beta", "new_string": "B"},
    ])
    assert f.read_text() == "A B gamma\n"

def test_cc_multi_edit_second_sees_first_result(tools_ns, tmp_path):
    f = tmp_path / "f.py"
    f.write_text("foo\n")
    tools_ns["cc_multi_edit"](str(f), [
        {"old_string": "foo", "new_string": "bar"},
        {"old_string": "bar", "new_string": "baz"},
    ])
    assert f.read_text() == "baz\n"

# __grammars__ export
def test_grammars_exported(tools_ns):
    grammars = tools_ns.get("__grammars__", {})
    assert isinstance(grammars, dict)
    assert "cc_edit" in grammars
    assert "cc_multi_edit" in grammars
    for v in grammars.values():
        assert isinstance(v, str) and len(v) > 20

def test_registry_collects_grammars():
    """ToolRegistry collects __grammars__ from skill modules."""
    from src.tools import ToolRegistry
    registry = ToolRegistry(auto_load_from_skills=True)
    grammar = registry.get_grammar("cc_edit")
    assert grammar is not None
    assert len(grammar) > 50


def test_registry_get_grammar_unknown_returns_none():
    from src.tools import ToolRegistry
    registry = ToolRegistry(auto_load_from_skills=False)
    assert registry.get_grammar("nonexistent_tool_xyz") is None


# __tools__ export
def test_tools_exported(tools_ns):
    tools = tools_ns.get("__tools__", [])
    names = [getattr(t, "__name__", None) for t in tools]
    assert "cc_glob" in names
    assert "cc_grep" in names
    assert "cc_read" in names
    assert "cc_edit" in names
    assert "cc_write" in names
    assert "cc_multi_edit" in names
