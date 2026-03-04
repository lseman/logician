# agent_core/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------
# ThinkingConfig: controls prompt + reasoner combinations
# ---------------------------------------------------------------------
@dataclass
class ThinkingConfig:
    prompt: Optional[str] = None  # name from PROMPT_REGISTRY
    reasoner: Optional[str] = None  # name from REASONER_REGISTRY

    # Pipeline order:
    #   "prompt"
    #   "reasoner"
    #   "prompt->reasoner"
    #   "reasoner->prompt"
    #   "prompt->reasoner->prompt"
    order: str = "prompt->reasoner"

    max_rounds: int = 1

    prompt_temperature: float = 0.7
    reasoner_temperature: float = 0.7
    max_tokens: int = 2048

    # NEW: parameters forwarded directly into Reasoner constructor
    reasoner_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Agent Config
# ---------------------------------------------------------------------
@dataclass
class Config:
    # Core
    llama_cpp_url: str = "http://localhost:8080"
    timeout: float = 120.0
    temperature: float = 0.7
    max_tokens: int = 1024
    max_iterations: int = 8
    use_chat_api: bool = True
    chat_template: str = "chatml"
    stop: tuple[str, ...] = ("<|im_end|>", "</s>", "[INST]", "USER:", "<|user|>")
    stream: bool = False
    max_consecutive_tool_calls: int = 5
    retry_attempts: int = 2

    # Backend
    backend: str = "llama_cpp"  # or "vllm"

    # vLLM backend options
    vllm_model: str | None = None
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_dtype: str = "auto"

    # RAG
    rag_enabled: bool = True
    rag_top_k: int = 20
    vector_path: str = "message_history.vector"

    # Conversation context
    history_limit: int = 18
    history_recent_tail: int = 8
    tool_result_max_chars: int = 6000
    assistant_ctx_max_chars: int = 12000
    trace_context_max_messages: int = 8
    trace_context_max_chars: int = 500
    tool_memory_items: int = 8
    compact_summary_max_chars: int = 4000

    # Self-Reflection (light critique loop)
    enable_reflection: bool = False
    reflection_prompt: str = (
        "You are critiquing your own response. Review the conversation and "
        "final output: [FINAL]. If it's incomplete, unclear, or needs "
        "tools/refinement, output a tool call or improved text. Otherwise, "
        "say 'COMPLETE'."
    )

    # Logging
    log_level: str | int = field(
        default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "ERROR")
    )
    log_json: bool = field(
        default_factory=lambda: (
            os.getenv("AGENT_LOG_JSON", "0").lower() in ("1", "true", "yes")
        )
    )

    # Tools
    use_toon_for_tools: bool = True
    tool_schema_mode: Literal["rich", "compact", "json_schema"] = "rich"
    enable_skill_routing: bool = True
    dynamic_skill_routing: bool = True
    skill_top_k: int = 3
    skill_include_playbooks: bool = True
    skill_compact_fallback: bool = True

    # Auto-compact: keep session from growing unboundedly
    auto_compact: bool = True
    auto_compact_threshold: int = 2  # compact when messages > history_limit * threshold

    # Constrained decoding: use OpenAI function-calling protocol (use_chat_api=True only)
    # llama.cpp will enforce the grammar server-side, eliminating malformed tool-call parses.
    constrained_decoding: bool = False

    # Tool result cache: per-Agent in-memory cache of read-only tool results across turns.
    # Tools whose names contain any entry in tool_cache_write_patterns are never cached.
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 3600  # seconds; 0 = never expire
    tool_cache_max_size: int = 256  # max number of cached entries
    tool_cache_write_patterns: list = field(
        default_factory=lambda: [
            "write",
            "exec",
            "commit",
            "push",
            "apply_patch",
            "delete",
            "remove",
            "create_file",
            "mkdir",
        ]
    )

    # Token-accurate context budgeting: set to your model's context window (e.g. 4096)
    # to enforce a hard token cap before each generate() call via the /tokenize endpoint.
    # 0 = disabled — char-based limits (tool_result_max_chars, assistant_ctx_max_chars) apply.
    context_token_budget: int = 0

    # NEW — Thinking pipeline (Prompt + Reasoner)
    thinking: ThinkingConfig | None = None

    # MCP servers  {name: {"url": ..., "headers": {...}, "enabled": bool}}
    mcp_servers: dict = field(default_factory=dict)
