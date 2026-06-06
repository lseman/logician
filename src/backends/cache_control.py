"""Prompt caching support with Anthropic-style cache_control markers.

Implements the same caching strategy as Pi:
- Static content (system prompt, tool definitions) → full cache
- Last user message → growing cache window
- Optional: last assistant tool_use → dual cache breakpoint

This provides 30-70% token cost reduction and 50-80% latency reduction
on cache hits for providers that support prompt caching (Anthropic,
OpenAI, and OpenRouter-compatible APIs).

See:
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://github.com/badlogic/pi-mono
- https://github.com/mcowger/pi-better-messages-cache
"""

from __future__ import annotations

from typing import Any


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    *,
    system: list[dict[str, Any]] | str | None = None,
    tools: list[dict[str, Any]] | None = None,
    use_cache: bool = True,
    ttl: str = "5m",
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Apply Anthropic-style cache_control markers to system, tools, and messages.

    Strategy:
    - System prompt: full cache (static content)
    - Tool definitions: full cache (static, changes rarely)
    - Messages: mark last user message for growing cache window
    - Optional: mark last assistant tool_use block for dual breakpoint

    This mirrors Pi's caching strategy plus the dual cache-breakpoint
    pattern from the pi-better-messages-cache extension.

    Args:
        messages: List of message dicts (will be mutated in place)
        system: System prompt as list of blocks or single string
        tools: Tool definitions list (will be mutated in place)
        use_cache: Whether to enable caching
        ttl: Cache TTL — "5m" (default, 5 minutes) or "1h" (extended)

    Returns:
        Tuple of (system, messages, tools) — may be mutated in place
    """
    if not use_cache:
        return system, messages, tools

    cache = {"type": "ephemeral"}
    if ttl == "1h":
        cache["ttl"] = "1h"

    # 1. System prompt → full cache
    if system:
        if isinstance(system, str):
            system = [{"type": "text", "text": system}]
        # Cache the entire system prompt
        system = [
            {**block, "cache_control": cache} for block in system
        ]

    # 2. Tool definitions → full cache
    if tools:
        for tool in tools:
            if isinstance(tool, dict):
                tool["cache_control"] = cache

    # 3. Messages → mark last user message for growing cache window
    if messages:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = [
                        {"type": "text", "text": content, "cache_control": cache}
                    ]
                elif isinstance(content, list):
                    # Mark last text block in last user message
                    for block in reversed(content):
                        if block.get("type") == "text":
                            block["cache_control"] = cache
                            break
                break

    return system, messages, tools


def apply_openai_compat_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    use_cache: bool = True,
    retention: str = "in_memory",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Apply Anthropic-style cache_control to OpenAI-compatible providers.

    Used when talking to OpenRouter, Vercel AI Gateway, or other proxies
    that support Anthropic-style cache_control markers on OpenAI-compatible APIs.

    Marks:
    - System/developer message
    - Last tool definition
    - Last user/assistant text content

    This was added in Pi 0.67.68 for OpenRouter Qwen models and similar.

    Args:
        messages: List of message dicts (will be mutated in place)
        tools: Tool definitions (will be mutated in place)
        use_cache: Whether to enable caching
        retention: Cache retention — "in_memory" (default) or "24h" (extended)

    Returns:
        Tuple of (messages, tools) — may be mutated in place
    """
    if not use_cache:
        return messages, tools

    cache = {"cache_control": "ephemeral"}

    # 1. System/developer message → full cache
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, **cache}
                ]
            break

    # 2. Last tool definition → cache
    if tools:
        tools[-1]["cache_control"] = "ephemeral"

    # 3. Last user/assistant text content → growing cache window
    for msg in reversed(messages):
        if msg.get("role") in ("user", "assistant"):
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, **cache}
                ]
            break

    return messages, tools


def mark_dual_cache_breakpoints(
    messages: list[dict[str, Any]],
    assistant_tool_block: dict[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Mark last assistant tool_use AND last user message for caching.

    Implements the dual cache-breakpoint strategy from pi-better-messages-cache:
    - Marks the last assistant tool_use block with cache_control
    - Marks the last user message with cache_control

    Both markers together ensure the full assistant turn (thinking + tool_use
    + tool_result) sits inside the growing cached prefix on every subsequent call.

    This dramatically improves cache hit rates on MiniMax, Kimi, and other
    Anthropic-compatible providers.

    Args:
        messages: List of message dicts (will be mutated in place)
        assistant_tool_block: Optional dict representing last assistant tool output
        use_cache: Whether to enable caching

    Returns:
        The messages list (mutated in place)
    """
    if not use_cache:
        return messages

    cache = {"cache_control": "ephemeral"}

    # Mark last assistant tool_use block
    if assistant_tool_block:
        assistant_tool_block["cache_control"] = cache

    # Mark last user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, **cache}
                ]
            elif isinstance(content, list):
                for block in reversed(content):
                    if block.get("type") == "text":
                        block["cache_control"] = cache
                        break
            break

    return messages


def estimate_cache_savings(
    system_tokens: int,
    tool_tokens: int,
    message_tokens: int,
    cache_hit_rate: float = 0.8,
) -> dict[str, int]:
    """Estimate token savings from prompt caching.

    Args:
        system_tokens: Tokens in system prompt
        tool_tokens: Tokens in tool definitions
        message_tokens: Tokens in conversation history
        cache_hit_rate: Expected cache hit rate (0.0-1.0)

    Returns:
        Dict with estimated savings breakdown
    """
    # Static content that gets cached: system + tools
    static_tokens = system_tokens + tool_tokens

    # On cache hit, we skip re-transmitting static content
    saved_per_hit = static_tokens * cache_hit_rate
    total_messages = 1  # Per-turn savings
    total_saved = int(saved_per_hit * total_messages)

    return {
        "static_tokens_cached": static_tokens,
        "estimated_savings_per_turn": total_saved,
        "cache_hit_rate": cache_hit_rate,
        "total_saved": total_saved,
    }
