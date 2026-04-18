"""Tool pool assembly with deduplication and permission filtering."""

from __future__ import annotations

import fnmatch

from .runtime import PermissionContext, Tool


def assemble_tool_pool(
    permission_context: PermissionContext,
    built_in_tools: list[Tool],
    mcp_tools: list[Tool] | None = None,
) -> list[Tool]:
    """
    Assemble the complete tool pool for a given permission context.

    This is the single source of truth for combining built-in tools with MCP tools.

    The function:
    1. Filters built-in tools by permission context (deny rules)
    2. Filters MCP tools by permission context (deny rules)
    3. Deduplicates by tool name (built-in tools take precedence)
    4. Sorts by name for prompt-cache stability

    Args:
        permission_context: Permission context for filtering
        built_in_tools: Built-in tools to include
        mcp_tools: Optional MCP tools to include

    Returns:
        Combined, deduplicated array of tools
    """
    if mcp_tools is None:
        mcp_tools = []

    # Filter built-in tools by deny rules
    allowed_built_in = filter_built_in_tools(permission_context, built_in_tools)

    # Filter MCP tools by deny rules
    allowed_mcp = filter_mcp_tools(permission_context, mcp_tools)

    # Combine and deduplicate
    all_tools = allowed_built_in + allowed_mcp
    unique_tools = _deduplicate_tools(all_tools)

    # Sort by name for prompt-cache stability
    sorted_tools = _sort_tools_by_name(unique_tools)

    return sorted_tools


def filter_built_in_tools(permission_context: PermissionContext, tools: list[Tool]) -> list[Tool]:
    """
    Filter built-in tools by deny rules.

    A tool is filtered out if there's a deny rule matching its name.

    Args:
        permission_context: Permission context with deny rules
        tools: List of built-in tools to filter

    Returns:
        Filtered list of tools
    """
    return [tool for tool in tools if not _matches_deny_rules(permission_context, tool)]


def filter_mcp_tools(permission_context: PermissionContext, mcp_tools: list[Tool]) -> list[Tool]:
    """
    Filter MCP tools by deny rules.

    MCP tools are identified via their ``skill_id`` (``mcp__<server>``).
    Deny rules can target the tool name, the full skill_id, or the server name.

    Args:
        permission_context: Permission context with deny rules
        mcp_tools: List of MCP tools to filter

    Returns:
        Filtered list of MCP tools
    """
    return [tool for tool in mcp_tools if not _matches_deny_rules(permission_context, tool)]


def _matches_deny_rules(permission_context: PermissionContext, tool: Tool) -> bool:
    """
    Check if a tool matches any deny rules.

    Args:
        permission_context: Permission context with deny rules
        tool: Tool to check

    Returns:
        True if the tool should be denied
    """
    subjects = _tool_rule_subjects(tool)

    for pattern in permission_context.always_deny_rules:
        if any(_pattern_matches(pattern, subject) for subject in subjects):
            return True

    for pattern in permission_context.always_deny_rules.get("*", []):
        if any(_pattern_matches(pattern, subject) for subject in subjects):
            return True

    return False


def _tool_rule_subjects(tool: Tool) -> set[str]:
    subjects = {tool.name}
    skill_id = str(getattr(tool, "skill_id", "") or "").strip()
    if not skill_id:
        return subjects

    subjects.add(skill_id)
    if skill_id.startswith("mcp__"):
        server_name = skill_id[len("mcp__") :]
        if server_name:
            subjects.add(server_name)
            subjects.add(f"mcp__{server_name}")
    return subjects


def _pattern_matches(pattern: str, value: str) -> bool:
    """
    Check if a pattern matches a value.

    Supports:
    - Exact match: "tool" matches "tool"
    - Prefix match: "tool*" matches "tool_x"
    - Wildcard match: "*tool*" matches "my_tool"
    """
    pattern = str(pattern or "").strip()
    value = str(value or "").strip()
    if not pattern or not value:
        return False
    if pattern == "*":
        return True
    return fnmatch.fnmatchcase(value, pattern)


def _deduplicate_tools(tools: list[Tool]) -> list[Tool]:
    """
    Deduplicate tools by name.

    Built-in tools take precedence over MCP tools if there's a name conflict.

    Args:
        tools: List of tools (may contain duplicates)

    Returns:
        List of unique tools (built-in first, then MCP)
    """
    seen = set()
    unique = []

    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            unique.append(tool)

    return unique


def _sort_tools_by_name(tools: list[Tool]) -> list[Tool]:
    """
    Sort tools by name for prompt-cache stability.

    This ensures that the same set of tools always produces the same
    prompt, which is important for caching.

    Args:
        tools: List of tools to sort

    Returns:
        Sorted list of tools
    """
    return sorted(tools, key=lambda t: t.name)


def get_tools_for_default_preset(tools: list[Tool]) -> list[str]:
    """
    Get the list of tool names for the default preset.

    Default preset keeps always-loaded tools plus non-deferred tools.

    Args:
        tools: List of tools to filter

    Returns:
        List of tool names
    """
    return [tool.name for tool in tools if tool.always_load or not tool.should_defer]


def filter_tools_by_simple_mode(
    tools: list[Tool],
) -> list[Tool]:
    """
    Filter tools for simple mode (Bash, Read, Edit only).

    Args:
        tools: List of tools to filter

    Returns:
        Filtered list with only Bash, Read, and Edit tools
    """
    simple_tool_names = {"bash", "read_file", "edit_file", "write_file"}
    return [tool for tool in tools if tool.name in simple_tool_names]


def filter_tools_for_agent(
    tools: list[Tool],
    agent_type: str = "assistant",
) -> list[Tool]:
    """
    Filter tools allowed for a specific agent type.

    Args:
        tools: List of tools to filter
        agent_type: Type of agent ("assistant", "worker", etc.)

    Returns:
        Filtered list of tools allowed for the agent type
    """
    # Define which tools are allowed per agent type
    allowed_by_agent_type = {
        "assistant": set(),  # All tools
        "worker": {
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "fetch_url",
            "web_search",
            "todo",
            "think",
        },
        "coordinator": {
            "task_create",
            "task_get",
            "task_update",
            "task_list",
            "task_stop",
        },
    }

    allowed_names = allowed_by_agent_type.get(agent_type, set())
    if agent_type == "assistant":
        return tools

    return [tool for tool in tools if tool.name in allowed_names]
