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
    # When True, max_iterations is treated as a strict budget for the full turn,
    # disabling extra post-loop passes (synthesis/reflection/thinking) once
    # the main loop has exhausted the iteration limit.
    strict_iteration_budget: bool = False

    # ── Pre-turn deliberation ("think before acting") ──────────────────────────
    # Before the first tool-call iteration, ask the LLM to produce a brief plan.
    # The plan is injected as a system message that anchors the whole turn.
    # Inspired by R1/o1 extended thinking and LATS deliberation.
    pre_turn_thinking: bool = False
    pre_turn_thinking_prompt: str = (
        "One sentence: what is the task and what is the first tool call?\n\n"
        "Rules:\n"
        "- If the path is obvious, name only the first action. Do not list all steps.\n"
        "- If multi-step and non-obvious, list up to 3 concrete steps max.\n"
        "- No tool_call blocks, no JSON, no code fences — plain text only.\n"
        "- Do NOT restate the user message. Do NOT explain what you are about to explain.\n"
        "- Shorter is better. One line is ideal."
    )
    # Optional post-tool reflection after each tool result.
    post_tool_thinking: bool = False
    post_tool_thinking_prompt: str = (
        "Briefly: what did this tool result tell you, and what is your precise next step?"
    )

    # ── Confidence-gated stopping ───────────────────────────────────────────────
    # After a candidate final answer is produced (no tool call emitted), score
    # it via a lightweight LLM judge call.  If the score falls below the
    # threshold the agent gets another iteration to improve, up to max_retries.
    # Inspired by process-reward models and self-evaluation stopping criteria.
    confidence_gate_enabled: bool = False
    confidence_gate_threshold: float = 7.0  # 0–10 scale
    confidence_gate_max_retries: int = 2

    # ── Extended ThinkingStrategy scope ────────────────────────────────────────
    # By default, the ThinkingStrategy post-pass only ran when no tools were
    # used.  Set this to True to also apply it after tool-using turns during
    # the synthesis phase — allows reasoners (Reflexion/SSR/ToT) to refine
    # tool-gathered evidence into a polished final answer.
    thinking_apply_after_tools: bool = True
    use_chat_api: bool = True
    chat_template: str = "chatml"
    stop: tuple[str, ...] = ("<|im_end|>", "</s>", "[INST]", "USER:", "<|user|>")
    stream: bool = True
    max_consecutive_tool_calls: int = 5
    allow_multi_tool_calls: bool = True
    multi_tool_call_max_calls: int = 4
    # Edit-heavy requests often require more chained tool calls/iterations.
    editing_min_iterations: int = 8
    editing_max_consecutive_tool_calls: int = 10
    editing_force_action_nudges: int = 2
    editing_require_inspection_nudges: int = 3
    editing_failure_repair_nudges: int = 3
    # Allow one corrective retry for a given guardrail, then stop the turn with
    # a blocker summary instead of repeating the same nudge indefinitely.
    guardrail_stall_nudge_limit: int = 1
    # If the model repeats materially the same no-tool draft after feedback,
    # stop the turn instead of asking for another rewrite of the same text.
    repeated_draft_stall_limit: int = 1
    repeated_draft_similarity_threshold: float = 0.88
    retry_attempts: int = 2
    lazy_mcp_init: bool = True
    default_use_semantic_retrieval: bool = True
    default_retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"

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
    rag_vector_path: str = "rag_docs.vector"
    # Vector index backend selector: "usearch" (default), "hnsw", or "chromadb".
    vector_backend: str = "usearch"
    rag_vector_backend: str = "usearch"
    rag_rerank_enabled: bool = True
    rag_rerank_fetch_k: int = 30
    rag_min_similarity: float = 0.20
    # Cap dominance from a single source file in retrieval output.
    rag_per_source_max_chunks: int = 4
    # Lightweight in-memory query cache for repeated RAG lookups in a session.
    rag_query_cache_enabled: bool = True
    rag_query_cache_ttl_sec: int = 90
    rag_query_cache_max_entries: int = 256
    # HNSW query/index knobs (affect speed/quality trade-off).
    rag_hnsw_ef_search: int = 128
    rag_hnsw_m: int = 16
    rag_hnsw_ef_construction: int = 200

    # Conversation context
    history_limit: int = 18
    history_recent_tail: int = 8
    tool_result_max_chars: int = 6000
    assistant_ctx_max_chars: int = 12000
    trace_context_max_messages: int = 8
    trace_context_max_chars: int = 500
    tool_memory_items: int = 8
    compact_summary_max_chars: int = 4000
    # Keep bulky read/search tool payloads out of vector history indexing.
    # Matching uses lowercase substring checks on tool names.
    tool_history_vector_exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "read_file",
            "read_file_smart",
            "rg_search",
            "get_file_outline",
            "get_project_map",
        ]
    )
    # Max chars kept when persisting compacted tool payload summaries.
    tool_history_summary_max_chars: int = 2200

    # Logging
    debug_trace: bool = True
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
    skill_include_on_demand_context: bool = True
    skill_on_demand_context_max_chars: int = 2600
    skill_on_demand_context_max_skills: int = 2
    skill_on_demand_context_max_files_per_skill: int = 5
    context7_docs_auto_nudge: bool = True
    # If the model claims it already ran tools without emitting a tool call in
    # this run, inject a corrective nudge and require a real tool call.
    tool_claim_guard_enabled: bool = True
    tool_claim_guard_max_nudges: int = 2
    # If a final answer still contains completed-action claims while zero tool
    # calls were actually executed, append a runtime transparency note.
    tool_claim_guard_append_runtime_note: bool = True
    # Guard against statements that contradict the latest filesystem/git
    # inspection tool result.
    inspection_result_guard_enabled: bool = True
    inspection_result_guard_max_nudges: int = 2
    inspection_result_guard_append_runtime_note: bool = True

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
            # Edit tools — must never return stale cached results
            "edit_file_replace",
            "apply_unified_diff",
            "apply_edit_block",
            "multi_edit",
            "rg_replace",
            "regex_replace",
        ]
    )
    # Read-only tools that still reflect live runtime state should bypass the
    # generic tool cache to avoid stale status/config snapshots.
    tool_cache_exclude_patterns: list = field(
        default_factory=lambda: [
            "rag_tuning_status",
            "rag_benchmark",
        ]
    )

    # Verification guardrail: after write-like tool calls, require at least one
    # verification-style tool call (tests/lint/check) before final answer.
    require_verification_after_writes: bool = True
    verification_tool_patterns: list = field(
        default_factory=lambda: [
            "test",
            "pytest",
            "unittest",
            "lint",
            "ruff",
            "mypy",
            "typecheck",
            "check",
            "verify",
            "validate",
            "build",
        ]
    )
    language_verification_matrix: dict = field(
        default_factory=lambda: {
            "python": [
                "run_ruff(path=...)",
                "run_pytest(path=...)",
                "run_mypy(path=...)",
            ],
            "rust": [
                'run_shell(cmd="cargo check")',
                'run_shell(cmd="cargo test")',
                'run_shell(cmd="cargo clippy")',
            ],
            "javascript": [
                'run_shell(cmd="npm test")',
                'run_shell(cmd="npm run lint")',
            ],
            "go": [
                'run_shell(cmd="go test ./...")',
                'run_shell(cmd="go vet ./...")',
            ],
        }
    )
    verification_auto_repair_max_attempts: int = 3
    enable_reflexion_repair: bool = True
    reflexion_repair_max_attempts: int = 1

    # Append a compact quality checklist to final answers after tool-driven runs.
    append_quality_checklist: bool = True
    quality_checklist_max_tools: int = 5

    # Token-accurate context budgeting: set to your model's context window (e.g. 4096)
    # to enforce a hard token cap before each generate() call via the /tokenize endpoint.
    # 0 = disabled — char-based limits (tool_result_max_chars, assistant_ctx_max_chars) apply.
    context_token_budget: int = 0

    # NEW — Thinking pipeline (Prompt + Reasoner)
    thinking: ThinkingConfig | None = None

    # MCP servers  {name: {"url": ..., "headers": {...}, "enabled": bool}}
    mcp_servers: dict = field(default_factory=dict)
