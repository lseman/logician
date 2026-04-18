from __future__ import annotations

from src.tools.pool import (
    assemble_tool_pool,
    filter_mcp_tools,
    get_tools_for_default_preset,
)
from src.tools.runtime import PermissionContext, ToolParameter, build_tool, materialize_tool


def _make_tool(
    name: str,
    *,
    skill_id: str | None = None,
    should_defer: bool = False,
    always_load: bool = False,
):
    def _tool() -> dict[str, str]:
        return {"status": "ok", "tool": name}

    return build_tool(
        _tool,
        name=name,
        description=f"{name} tool",
        skill_id=skill_id,
        should_defer=should_defer,
        always_load=always_load,
    )


def test_materialize_tool_preserves_openclaude_metadata() -> None:
    def _demo(path: str) -> dict[str, str]:
        return {"status": "ok", "path": path}

    build_tool(
        _demo,
        name="demo_tool",
        description="Demo tool",
        parameters=[ToolParameter(name="path", type="string", required=True)],
        should_defer=True,
        always_load=True,
        max_result_size_chars=4321,
        get_tool_use_summary=lambda path: f"demo:{path}",
    )

    tool = materialize_tool(
        _demo,
        name="demo_tool",
        description="Demo tool",
        parameters=[ToolParameter(name="path", type="string", required=True)],
    )

    assert tool.should_defer is True
    assert tool.always_load is True
    assert tool.max_result_size_chars == 4321
    assert tool.get_tool_use_summary is not None
    assert tool.get_tool_use_summary("x.txt") == "demo:x.txt"


def test_filter_mcp_tools_supports_server_level_deny_rules() -> None:
    ctx = PermissionContext(always_deny_rules={"mcp__context7": []})
    kept = filter_mcp_tools(
        ctx,
        [
            _make_tool("read_file"),
            _make_tool("resolve_library_id", skill_id="mcp__context7"),
            _make_tool("fetch_docs", skill_id="mcp__other"),
        ],
    )

    assert [tool.name for tool in kept] == ["read_file", "fetch_docs"]


def test_assemble_tool_pool_deduplicates_and_sorts() -> None:
    ctx = PermissionContext()
    tools = assemble_tool_pool(
        ctx,
        built_in_tools=[_make_tool("bash"), _make_tool("read_file")],
        mcp_tools=[_make_tool("bash", skill_id="mcp__server"), _make_tool("z_tool")],
    )

    assert [tool.name for tool in tools] == ["bash", "read_file", "z_tool"]


def test_get_tools_for_default_preset_skips_deferred_tools() -> None:
    tools = [
        _make_tool("always", should_defer=True, always_load=True),
        _make_tool("search_tools", should_defer=True, always_load=False),
        _make_tool("read_file"),
    ]

    assert get_tools_for_default_preset(tools) == ["always", "read_file"]
