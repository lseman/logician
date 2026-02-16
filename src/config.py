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
    prompt: Optional[str] = None          # name from PROMPT_REGISTRY
    reasoner: Optional[str] = None        # name from REASONER_REGISTRY

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
        default_factory=lambda: os.getenv("AGENT_LOG_JSON", "0").lower()
        in ("1", "true", "yes")
    )

    # Tools
    use_toon_for_tools: bool = True
    tool_schema_mode: Literal["rich", "compact", "json_schema"] = "rich"

    # NEW — Thinking pipeline (Prompt + Reasoner)
    thinking: ThinkingConfig | None = None
