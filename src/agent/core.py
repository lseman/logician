from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ..config import Config
from ..logging_utils import get_logger
from ..memory import Memory
from ..messages import Message, MessageRole
from ..thinking import ThinkingStrategy
from ..tools import (
    HAS_TOON,
    Context,
    SkillSelection,
    ToolCall,
    ToolParameter,
    ToolRegistry,
    parse_tool_calls,
)
from .trace import (
    AgentResponse,
    _render_context_snapshot,
    _render_tool_progress_reminder,
    _tool_call_signature,
    _TraceCollector,
    _truncate_text,
    _vprint_block,
    render_trace_markdown,
)

if TYPE_CHECKING:
    pass


_FOLLOW_UP_PHRASES = (
    "continue",
    "go on",
    "next step",
    "what next",
    "what now",
    "carry on",
    "keep going",
    "proceed",
)


class Agent:
    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        system_prompt: str | None = None,
        config: Config | None = None,
        use_chat_api: bool = True,
        chat_template: str = "chatml",
        db_path: str = "agent_sessions.db",
        embedding_model: str | None = None,
        *,
        lazy_rag: bool = True,
    ) -> None:
        self.config = config or Config(
            llama_cpp_url=llm_url,
            use_chat_api=use_chat_api,
            chat_template=chat_template,
        )

        get_logger("agent.framework", self.config.log_level, self.config.log_json)
        self._log = get_logger("agent")
        self._log.info(
            "Initializing Agent backend=%s chat_api=%s template=%s",
            self.config.backend,
            self.config.use_chat_api,
            self.config.chat_template,
        )

        if self.config.use_toon_for_tools and not HAS_TOON:
            self._log.warning(
                "TOON enabled but toon_format not installed. Falling back to JSON."
            )
            self.config.use_toon_for_tools = False

        if self.config.backend == "vllm":
            from ..backends import VLLMClient

            if not self.config.vllm_model:
                raise ValueError(
                    "Config.vllm_model must be set when backend == 'vllm'."
                )
            self.llm = VLLMClient(
                model=self.config.vllm_model,
                chat_template=self.config.chat_template,
                stop=self.config.stop,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                dtype=self.config.vllm_dtype,
            )
        else:
            from ..backends import LlamaCppClient

            self.llm = LlamaCppClient(
                base_url=self.config.llama_cpp_url,
                timeout=self.config.timeout,
                use_chat_api=self.config.use_chat_api,
                chat_template=self.config.chat_template,
                stop=self.config.stop,
                retry_attempts=self.config.retry_attempts,
            )

        self.thinking: ThinkingStrategy | None = None
        if getattr(self.config, "thinking", None):
            self.thinking = ThinkingStrategy(self.llm, self.config.thinking)

        self.system_prompt = system_prompt or self._load_soul_or_default()

        self.ctx = Context()
        self.tools = ToolRegistry(auto_load_from_skills=False)
        self.tools.install_context(self.ctx)
        self.tools.load_tools_from_skills()
        self._mcp_clients: list[Any] = []
        self._load_mcp_servers()

        self._cached_sys_tools_prompt: str = ""
        self._cached_sys_prompt_base: str = ""
        self._cached_use_toon: bool = bool(self.config.use_toon_for_tools)
        self._cached_tools_version: int = -1

        self._embedding_model_name = embedding_model or "BAAI/bge-base-en-v1.5"
        self.memory = Memory(
            config=self.config,
            db_path=db_path,
            embedding_model=self._embedding_model_name,
            lazy_rag=lazy_rag,
        )
        self.current_session_id: str | None = None
        self._loaded_runtime_session_id: str | None = None
        self._last_context_messages: list[Message] = []
        # Per-agent in-memory tool result cache (persists across turns within the
        # same Agent instance; cleared on reset()).
        self._tool_result_cache: dict[str, tuple[float, str]] = {}

    def get_last_context_text(self) -> str:
        """Return a human-readable Markdown dump of the last context window sent to the LLM."""
        msgs = self._last_context_messages
        if not msgs:
            return "_No context captured yet — send at least one message first._"

        role_labels: dict[str, str] = {
            "system": "SYSTEM",
            "user": "USER",
            "assistant": "ASSISTANT",
            "tool": "TOOL",
        }
        parts: list[str] = [f"## Last context window ({len(msgs)} messages)\n"]
        for i, msg in enumerate(msgs, 1):
            role = getattr(msg.role, "value", str(msg.role))
            label = role_labels.get(role, role.upper())
            name_tag = f" · `{msg.name}`" if getattr(msg, "name", None) else ""
            content = str(msg.content or "")
            # Truncate very large messages (e.g. full tool results)
            if len(content) > 3000:
                content = (
                    content[:3000] + f"\n\n…_{len(content) - 3000} chars truncated_"
                )
            parts.append(f"### {i}. {label}{name_tag}\n\n```\n{content}\n```\n")
        return "\n".join(parts)

    def _load_mcp_servers(self) -> None:
        """Initialise every enabled MCP server from config and register their tools."""
        from ..mcp.client import MCPClient

        servers: dict = getattr(self.config, "mcp_servers", {}) or {}
        for name, cfg in servers.items():
            if not cfg.get("enabled", True):
                self._log.info("MCP server '%s' is disabled — skipping", name)
                continue
            url = cfg.get("url", "")
            if not url:
                self._log.warning("MCP server '%s' has no url — skipping", name)
                continue
            headers: dict[str, str] = {
                str(k): str(v) for k, v in (cfg.get("headers") or {}).items()
            }
            # Skip when any header value is empty — treat as unconfigured (no API key).
            missing = [k for k, v in headers.items() if not v.strip()]
            if missing:
                self._log.info(
                    "MCP '%s' skipped — header(s) %s are empty (API key not configured)",
                    name,
                    missing,
                )
                continue
            timeout = float(cfg.get("timeout", 30))
            client = MCPClient(name=name, url=url, headers=headers, timeout=timeout)
            try:
                client.initialize()
                n = self.tools.load_from_mcp_server(client)
                self._mcp_clients.append(client)
                self._log.info("MCP '%s' connected — %d tool(s) registered", name, n)
            except Exception as exc:
                self._log.warning(
                    "MCP '%s' failed to initialise: %s — skipping", name, exc
                )

    def _load_soul_or_default(self) -> str:
        soul_path = Path(__file__).resolve().parents[2] / "SOUL.md"
        if soul_path.exists():
            return soul_path.read_text(encoding="utf-8").strip()
        return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        if self.config.use_toon_for_tools:
            return (
                "You are a reliable assistant. If a TOOL is needed, output EXACT TOON:\n"
                "tool_call:\n"
                "  name: <tool>\n"
                "  arguments:\n"
                "    ...\n"
                "Return exactly one tool_call per response.\n"
                "If no tool is needed, answer normally and clearly."
            )
        return (
            "You are a reliable assistant. If a TOOL is needed, output EXACT JSON:\n"
            '{"tool_call":{"name":"<tool>","arguments":{...}}}\n'
            "Return exactly one tool_call per response.\n"
            "If no tool is needed, answer normally and clearly."
        )

    def _tools_version(self) -> int:
        v = getattr(self.tools, "_version", None)
        if isinstance(v, int):
            return v
        v2 = getattr(self.tools, "version", None)
        if isinstance(v2, int):
            return v2
        return 0

    def _system_plus_tools_prompt(self) -> str:
        tools_v = self._tools_version()
        use_toon = bool(self.config.use_toon_for_tools)

        if (
            self._cached_sys_tools_prompt
            and self._cached_sys_prompt_base == self.system_prompt
            and self._cached_tools_version == tools_v
            and self._cached_use_toon == use_toon
        ):
            return self._cached_sys_tools_prompt

        schema_mode = str(getattr(self.config, "tool_schema_mode", "rich"))
        schema = self.tools.tools_schema_prompt(use_toon, mode=schema_mode)
        self._cached_sys_tools_prompt = self.system_prompt + schema
        self._cached_sys_prompt_base = self.system_prompt
        self._cached_tools_version = tools_v
        self._cached_use_toon = use_toon
        return self._cached_sys_tools_prompt

    def _is_follow_up_message(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        if any(phrase in text for phrase in _FOLLOW_UP_PHRASES):
            return True
        return len(text.split()) <= 4

    def _reset_context_state(self) -> None:
        reset = getattr(self.ctx, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                self._log.exception("Failed to reset tool context")

    def _persist_runtime_state(self, session_id: str | None = None) -> None:
        sid = session_id or self.current_session_id
        if not sid:
            return
        try:
            self.memory.save_runtime_state(sid, self.ctx.to_state())
            self._loaded_runtime_session_id = sid
        except Exception:
            self._log.exception(
                "Failed to persist runtime state for session=%s", sid[:8]
            )

    def _load_runtime_state(self, session_id: str | None) -> None:
        sid = session_id
        if not sid:
            self._reset_context_state()
            self._loaded_runtime_session_id = None
            return
        try:
            state = self.memory.load_runtime_state(sid)
            self.ctx.load_state(state)
            self._loaded_runtime_session_id = sid
        except Exception:
            self._log.exception("Failed to load runtime state for session=%s", sid[:8])
            self._reset_context_state()
            self._loaded_runtime_session_id = sid

    def _clear_persisted_runtime_state(self, session_id: str | None) -> None:
        sid = session_id
        if not sid:
            return
        try:
            self.memory.clear_runtime_state(sid)
        except Exception:
            self._log.exception(
                "Failed to clear persisted runtime state for session=%s", sid[:8]
            )

    def _activate_session_runtime(
        self,
        session_id: str,
        *,
        clear_runtime: bool = False,
        force_reload: bool = False,
    ) -> None:
        prev_sid = self.current_session_id
        if prev_sid and prev_sid != session_id:
            self._persist_runtime_state(prev_sid)

        if clear_runtime:
            self._clear_persisted_runtime_state(session_id)
            self._reset_context_state()
            self._loaded_runtime_session_id = session_id
        elif force_reload or self._loaded_runtime_session_id != session_id:
            self._load_runtime_state(session_id)

        self.current_session_id = session_id

    def _compose_skill_routing_query(
        self,
        message: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_result_preview_by_sig: dict[tuple[str, str], str] | None = None,
    ) -> str:
        if not bool(getattr(self.config, "dynamic_skill_routing", True)):
            return message

        state_lines: list[str] = []
        if self.ctx.loaded:
            row_count = len(self.ctx.data) if self.ctx.data is not None else 0
            shape = "multivariate" if self.ctx.is_multivariate else "univariate"
            columns = ", ".join(self.ctx.value_columns[:4]) or "none"
            state_lines.append(
                f"Data is already loaded ({shape}, rows={row_count}, value columns={columns})."
            )
            if self.ctx.data_name:
                state_lines.append(f"Current dataset name: {self.ctx.data_name}.")
            if self.ctx.freq_cache:
                state_lines.append(f"Known series frequency: {self.ctx.freq_cache}.")
            if self.ctx.anomaly_store:
                state_lines.append("Anomaly detection results already exist in memory.")
            if self.ctx.nf_cv_full is not None:
                state_lines.append("Forecast cross-validation results already exist.")
            if self.ctx.nf_best_model:
                state_lines.append(
                    f"Current best forecast model in memory: {self.ctx.nf_best_model}."
                )
        else:
            state_lines.append(
                "No dataset is loaded yet, so data loading is the relevant skill."
            )

        recent_calls = tool_calls or []
        preview_by_sig = tool_result_preview_by_sig or {}
        if recent_calls:
            last_call = recent_calls[-1]
            state_lines.append(f"Most recent tool executed: {last_call.name}.")
            last_tool = self.tools.get(last_call.name)
            last_skill_id = last_tool.skill_id if last_tool else None
            if last_skill_id:
                state_lines.append(
                    "The most recent tool came from the "
                    f"{last_skill_id.replace('_', ' ')} skill."
                )
                if self._is_follow_up_message(message):
                    skill_card = self.tools._catalog.skills.get(last_skill_id)
                    next_skills = skill_card.next_skills if skill_card else []
                    if next_skills:
                        pretty = ", ".join(s.replace("_", " ") for s in next_skills)
                        state_lines.append(
                            f"For this follow-up, the likely next skills are: {pretty}."
                        )

            preview = preview_by_sig.get(_tool_call_signature(last_call), "").strip()
            if preview:
                preview_line = preview.replace("\n", " ")
                if len(preview_line) > 240:
                    preview_line = preview_line[:240] + " ..."
                state_lines.append(f"Latest tool result preview: {preview_line}")
        elif self.ctx.loaded and self._is_follow_up_message(message):
            state_lines.append(
                "For this follow-up, the likely next skills are: preprocessing, analysis, forecasting."
            )

        if not state_lines:
            return message

        bullets = "\n".join(f"- {line}" for line in state_lines)
        return f"{message.strip()}\n\nRuntime state for skill routing:\n{bullets}"

    def _resolve_system_prompt(
        self,
        message: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_result_preview_by_sig: dict[tuple[str, str], str] | None = None,
    ) -> tuple[str, SkillSelection | None, str]:
        if bool(getattr(self.config, "enable_skill_routing", True)):
            schema_mode = str(getattr(self.config, "tool_schema_mode", "rich"))
            routing_query = self._compose_skill_routing_query(
                message,
                tool_calls=tool_calls,
                tool_result_preview_by_sig=tool_result_preview_by_sig,
            )
            routed_prompt, selection = self.tools.skill_routing_prompt(
                routing_query,
                use_toon=bool(self.config.use_toon_for_tools),
                mode=schema_mode,
                top_k=int(getattr(self.config, "skill_top_k", 3)),
                include_playbooks=bool(
                    getattr(self.config, "skill_include_playbooks", True)
                ),
                include_compact_fallback=bool(
                    getattr(self.config, "skill_compact_fallback", True)
                ),
            )
            if selection.selected_skills:
                return self.system_prompt + routed_prompt, selection, routing_query
            return self._system_plus_tools_prompt(), selection, routing_query
        return self._system_plus_tools_prompt(), None, message

    def _build_system_prompt_for_message(
        self,
        message: str,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_result_preview_by_sig: dict[tuple[str, str], str] | None = None,
    ) -> str:
        prompt, _, _ = self._resolve_system_prompt(
            message,
            tool_calls=tool_calls,
            tool_result_preview_by_sig=tool_result_preview_by_sig,
        )
        return prompt

    def _ensure_doc_db(self) -> None:
        self.memory._ensure_doc_db()

    def _resolve_generation_settings(
        self, temperature: float | None, max_tokens: int | None
    ) -> tuple[float, int]:
        temp = temperature if temperature is not None else self.config.temperature
        n_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        return temp, n_tok

    def _create_base_conversation(self, message: str) -> list[Message]:
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=self._build_system_prompt_for_message(message),
            )
        ]

    def _load_and_append_history(
        self,
        convo: list[Message],
        sid: str,
        message: str,
        use_semantic_retrieval: bool,
        retrieval_mode: str = "vector",
    ) -> None:
        history = self.memory.load_history(
            sid, message, use_semantic_retrieval, retrieval_mode=retrieval_mode
        )
        convo.extend(history)

    def _append_rag_context(
        self,
        convo: list[Message],
        message: str,
        event_cb: Callable[..., None],
    ) -> None:
        rag_context = self.memory.get_rag_context(message, event_cb)
        if rag_context:
            convo.append(
                Message(
                    role=MessageRole.SYSTEM,
                    content=f"Use this document context if helpful:\n{rag_context}",
                )
            )

    def _append_user_message(
        self,
        convo: list[Message],
        sid: str,
        message: str,
        event_cb: Callable[..., None],
    ) -> None:
        user_msg = Message(role=MessageRole.USER, content=message)
        convo.append(user_msg)
        self.memory.save_message(sid, user_msg)
        event_cb("user_message", session=sid, message_preview=message[:120])

    def _append_assistant_message(
        self,
        convo: list[Message],
        sid: str,
        text: str,
        assistant_ctx_max_chars: int,
    ) -> None:
        asst_full = Message(role=MessageRole.ASSISTANT, content=text)
        self.memory.save_message(sid, asst_full)
        asst_ctx = _truncate_text(text, assistant_ctx_max_chars)
        convo.append(Message(role=MessageRole.ASSISTANT, content=asst_ctx))

    def _append_tool_message(
        self,
        convo: list[Message],
        sid: str,
        call: ToolCall,
        result_full: str,
        tool_result_max_chars: int,
    ) -> str:
        tool_msg_full = Message(
            role=MessageRole.TOOL,
            name=call.name,
            tool_call_id=call.id,
            content=result_full,
        )
        self.memory.save_message(sid, tool_msg_full)
        result_ctx = _truncate_text(result_full, tool_result_max_chars)
        convo.append(
            Message(
                role=MessageRole.TOOL,
                name=call.name,
                tool_call_id=call.id,
                content=result_ctx,
            )
        )
        return result_ctx

    def add_tool(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        parameters: list[ToolParameter] | None = None,
    ) -> Agent:
        self.tools.register(name, description, parameters or [], function)
        self._cached_sys_tools_prompt = ""
        return self

    def run_tool_direct(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        persist_to_history: bool = False,
        use_toon: bool | None = None,
    ) -> str:
        args = dict(arguments or {})
        sid = session_id or self.current_session_id
        if sid and sid != self.current_session_id:
            self._activate_session_runtime(sid, force_reload=True)
        call = ToolCall(
            id=f"direct_{uuid.uuid4().hex[:10]}",
            name=tool_name,
            arguments=args,
        )
        result_full = self.tools.execute(
            call,
            use_toon=self.config.use_toon_for_tools
            if use_toon is None
            else bool(use_toon),
        )

        sid_for_persistence = sid
        if persist_to_history:
            sid_for_persistence = sid_for_persistence or str(uuid.uuid4())
            assert sid_for_persistence is not None
            self.current_session_id = sid_for_persistence
            self.memory.save_message(
                sid_for_persistence,
                Message(
                    role=MessageRole.USER,
                    content=f"[direct_tool_call] {tool_name} {json.dumps(args, ensure_ascii=False)}",
                ),
            )
            self.memory.save_message(
                sid_for_persistence,
                Message(
                    role=MessageRole.TOOL,
                    name=tool_name,
                    tool_call_id=call.id,
                    content=result_full,
                ),
            )

        if sid_for_persistence:
            self._persist_runtime_state(sid_for_persistence)

        return result_full

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        verbose: bool = False,
        use_semantic_retrieval: bool = False,
        retrieval_mode: str = "vector",
        stream: Callable[[str], None] | None = None,
        fresh_session: bool = False,
        tool_callback: Callable[[str, dict], None] | None = None,
    ) -> str:
        return self.run(
            message,
            session_id=session_id,
            verbose=verbose,
            use_semantic_retrieval=use_semantic_retrieval,
            retrieval_mode=retrieval_mode,
            stream_callback=stream,
            fresh_session=fresh_session,
            tool_callback=tool_callback,
        ).final_response

    def reset(self, session_id: str | None = None) -> Agent:
        sid = session_id or self.current_session_id
        if sid:
            self.memory.clear_session(sid)
        self._reset_context_state()
        self._tool_result_cache.clear()
        self.current_session_id = None
        self._loaded_runtime_session_id = None
        return self

    # ── Tool result cache helpers ───────────────────────────────────────

    def _tool_cache_key(self, call: ToolCall) -> str:
        """Stable hash key for a tool call (name + sorted arguments)."""
        payload = (
            call.name
            + "\x00"
            + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _is_tool_cacheable(self, tool_name: str) -> bool:
        """Return True when a tool's result may be cached (read-only pattern check)."""
        if not bool(getattr(self.config, "tool_cache_enabled", True)):
            return False
        patterns: list[str] = list(
            getattr(self.config, "tool_cache_write_patterns", [])
        )
        name_lower = tool_name.lower()
        return not any(pat in name_lower for pat in patterns)

    def _trim_convo_to_budget(
        self,
        convo: list[Message],
        n_tok: int,
    ) -> list[Message]:
        """Trim *convo* so the total token count fits within
        ``context_token_budget - n_tok - 128``.

        A single ``llm.count_tokens()`` call is made per iteration; if the
        endpoint is unavailable the conversation is returned unchanged.
        ``context_token_budget == 0`` (default) disables trimming entirely.
        """
        budget = int(getattr(self.config, "context_token_budget", 0))
        if budget <= 0 or not hasattr(self.llm, "count_tokens"):
            return convo
        max_ctx_tokens = (
            budget - n_tok - 128
        )  # reserve headroom for response + overhead
        if max_ctx_tokens <= 64:
            return convo
        full_text = "\n".join(m.content or "" for m in convo)
        try:
            total = self.llm.count_tokens(full_text)
        except Exception:
            return convo  # tokenize endpoint unreachable — degrade gracefully
        if total <= max_ctx_tokens:
            return convo  # already fits; nothing to do
        # Estimate chars-per-token for this specific prompt.
        chars_per_token = max(1.0, len(full_text) / max(1, total))
        chars_to_cut = int((total - max_ctx_tokens) * chars_per_token)
        keep_tail = int(getattr(self.config, "history_recent_tail", 8))
        trimmed = list(convo)
        # Drop oldest tool/assistant turns, keeping the system header (index 0)
        # and the most recent `keep_tail` messages.
        trimmable = [
            i
            for i, m in enumerate(trimmed)
            if i > 0
            and i < len(trimmed) - keep_tail
            and m.role in (MessageRole.TOOL, MessageRole.ASSISTANT)
        ]
        chars_cut = 0
        to_remove: set[int] = set()
        for idx in trimmable:
            if chars_cut >= chars_to_cut:
                break
            chars_cut += len(trimmed[idx].content or "")
            to_remove.add(idx)
        result = [m for i, m in enumerate(trimmed) if i not in to_remove]
        self._log.info(
            "Token-budget trim: dropped %d messages (~%d chars) to fit %d-token budget",
            len(to_remove),
            chars_cut,
            budget,
        )
        return result

    def reset_runtime_state(
        self,
        session_id: str | None = None,
        *,
        clear_persisted: bool = True,
    ) -> Agent:
        sid = session_id or self.current_session_id
        if clear_persisted and sid:
            self._clear_persisted_runtime_state(sid)
        self._reset_context_state()
        self.current_session_id = None
        self._loaded_runtime_session_id = None
        return self

    def detach_runtime_state(self) -> Agent:
        self._reset_context_state()
        self.current_session_id = None
        self._loaded_runtime_session_id = None
        return self

    def _runtime_context_snapshot(self) -> dict[str, Any]:
        row_count = len(self.ctx.data) if self.ctx.data is not None else 0
        anomaly_points = sum(len(points) for points in self.ctx.anomaly_store.values())
        return {
            "loaded": bool(self.ctx.loaded),
            "data_name": self.ctx.data_name,
            "row_count": row_count,
            "value_columns": list(self.ctx.value_columns),
            "is_multivariate": bool(self.ctx.is_multivariate),
            "freq": self.ctx.freq_cache,
            "anomaly_series": len(self.ctx.anomaly_store),
            "anomaly_points": anomaly_points,
            "forecast_model": self.ctx.nf_best_model,
            "has_forecast_cv": self.ctx.nf_cv_full is not None,
            "forecast_prediction_column": self.ctx.nf_pred_col,
        }

    def describe_runtime_context(self, session_id: str | None = None) -> dict[str, Any]:
        sid = session_id or self.current_session_id
        persisted_messages = self.memory.count_session_messages(sid) if sid else 0
        history_limit = int(self.config.history_limit)
        history_recent_tail = int(self.config.history_recent_tail)
        if sid and sid == self.current_session_id:
            runtime_snapshot = self._runtime_context_snapshot()
        elif sid:
            persisted_state = self.memory.load_runtime_state(sid)
            persisted_ctx = Context()
            persisted_ctx.load_state(persisted_state)
            row_count = len(persisted_ctx.data) if persisted_ctx.data is not None else 0
            anomaly_points = sum(
                len(points) for points in persisted_ctx.anomaly_store.values()
            )
            runtime_snapshot = {
                "loaded": bool(persisted_ctx.loaded),
                "data_name": persisted_ctx.data_name,
                "row_count": row_count,
                "value_columns": list(persisted_ctx.value_columns),
                "is_multivariate": bool(persisted_ctx.is_multivariate),
                "freq": persisted_ctx.freq_cache,
                "anomaly_series": len(persisted_ctx.anomaly_store),
                "anomaly_points": anomaly_points,
                "forecast_model": persisted_ctx.nf_best_model,
                "has_forecast_cv": persisted_ctx.nf_cv_full is not None,
                "forecast_prediction_column": persisted_ctx.nf_pred_col,
            }
        else:
            runtime_snapshot = self._runtime_context_snapshot()
        return {
            "session_id": sid,
            "persisted_messages": persisted_messages,
            "history_limit": history_limit,
            "history_recent_tail": history_recent_tail,
            "loaded_message_budget": min(persisted_messages, history_limit),
            "history_over_budget": persisted_messages > history_limit,
            "runtime": runtime_snapshot,
        }

    @staticmethod
    def _compact_preview(text: str, limit: int = 160) -> str:
        compact = " ".join((text or "").split())
        return compact if len(compact) <= limit else compact[:limit].rstrip() + " ..."

    def _build_compaction_summary(
        self,
        older_messages: list[Message],
        *,
        kept_tail_count: int,
        max_chars: int,
    ) -> str:
        role_counts = Counter(m.role.value for m in older_messages)
        tool_counts = Counter(
            m.name for m in older_messages if m.role == MessageRole.TOOL and m.name
        )

        def recent_previews(role: MessageRole, limit: int) -> list[str]:
            previews: list[str] = []
            seen: set[str] = set()
            for msg in reversed(older_messages):
                if msg.role != role:
                    continue
                preview = self._compact_preview(msg.content)
                if not preview or preview in seen:
                    continue
                seen.add(preview)
                previews.append(preview)
                if len(previews) >= limit:
                    break
            previews.reverse()
            return previews

        runtime = self._runtime_context_snapshot()
        lines = [
            "[Session summary inserted by /compact]",
            (
                f"Compacted {len(older_messages)} earlier messages into this checkpoint. "
                f"The latest {kept_tail_count} messages remain verbatim after this summary."
            ),
            (
                "Role counts before compaction: "
                f"user={role_counts.get(MessageRole.USER.value, 0)}, "
                f"assistant={role_counts.get(MessageRole.ASSISTANT.value, 0)}, "
                f"tool={role_counts.get(MessageRole.TOOL.value, 0)}, "
                f"system={role_counts.get(MessageRole.SYSTEM.value, 0)}."
            ),
        ]

        user_previews = recent_previews(MessageRole.USER, 4)
        if user_previews:
            lines.append("Recent user requests before the kept tail:")
            lines.extend(f"- {preview}" for preview in user_previews)

        assistant_previews = recent_previews(MessageRole.ASSISTANT, 3)
        if assistant_previews:
            lines.append("Recent assistant outputs before the kept tail:")
            lines.extend(f"- {preview}" for preview in assistant_previews)

        if tool_counts:
            tool_summary = ", ".join(
                f"{name} x{count}" for name, count in tool_counts.most_common(6)
            )
            lines.append(f"Tools used earlier in the session: {tool_summary}.")

        if runtime["loaded"]:
            cols = ", ".join(runtime["value_columns"][:6]) or "none"
            dataset = runtime["data_name"] or "unnamed dataset"
            shape = "multivariate" if runtime["is_multivariate"] else "univariate"
            lines.append(
                f"Runtime dataset snapshot: {dataset} ({shape}, rows={runtime['row_count']}, value columns={cols})."
            )
            if runtime["freq"]:
                lines.append(f"Known frequency: {runtime['freq']}.")
        else:
            lines.append("Runtime dataset snapshot: no dataset currently loaded.")

        if runtime["anomaly_series"]:
            lines.append(
                "Anomaly memory snapshot: "
                f"{runtime['anomaly_series']} series / {runtime['anomaly_points']} points."
            )
        if runtime["forecast_model"]:
            lines.append(
                f"Forecast memory snapshot: best model={runtime['forecast_model']}."
            )
        if runtime["has_forecast_cv"]:
            lines.append(
                "Forecast cross-validation results are present in runtime memory."
            )

        return _truncate_text("\n".join(lines), max_chars)

    def compact_session(
        self,
        session_id: str | None = None,
        *,
        keep_last_messages: int | None = None,
        summary_max_chars: int | None = None,
    ) -> dict[str, Any]:
        sid = session_id or self.current_session_id
        if not sid:
            return {
                "status": "no_session",
                "session_id": None,
                "message": "No active session to compact.",
            }

        keep_last = max(
            1,
            int(
                keep_last_messages
                if keep_last_messages is not None
                else self.config.history_recent_tail
            ),
        )
        max_chars = max(
            400,
            int(
                summary_max_chars
                if summary_max_chars is not None
                else self.config.compact_summary_max_chars
            ),
        )

        messages = self.memory.get_session_messages(sid)
        total_messages = len(messages)
        if total_messages <= keep_last:
            return {
                "status": "noop",
                "session_id": sid,
                "message": "Session already fits inside the kept tail.",
                "old_message_count": total_messages,
                "new_message_count": total_messages,
                "kept_tail_count": total_messages,
            }

        older_messages = messages[:-keep_last]
        kept_tail = messages[-keep_last:]
        summary = self._build_compaction_summary(
            older_messages,
            kept_tail_count=len(kept_tail),
            max_chars=max_chars,
        )

        self.memory.clear_session_messages(sid)
        replacement = [Message(role=MessageRole.SYSTEM, content=summary), *kept_tail]
        for msg in replacement:
            self.memory.save_message(sid, msg)

        return {
            "status": "compacted",
            "session_id": sid,
            "message": (
                f"Compacted {len(older_messages)} messages into one summary and kept "
                f"the latest {len(kept_tail)} messages."
            ),
            "old_message_count": total_messages,
            "new_message_count": len(replacement),
            "kept_tail_count": len(kept_tail),
            "summary_chars": len(summary),
        }

    def list_sessions(self) -> list[tuple[str, str]]:
        return self.memory.list_sessions()

    def semantic_search(
        self,
        query: str,
        session_id: str,
        k: int = 8,
        retrieval_mode: str = "vector",
    ) -> list[Message]:
        return self.memory.semantic_search(
            query, session_id, k, retrieval_mode=retrieval_mode
        )

    def repl(self) -> None:
        print(f"--- Agent REPL :: {self.current_session_id or 'new session'} ---")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            try:
                user_input = input(">> ")
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                resp = self.chat(user_input, verbose=True)
                print(f"\n[Agent]: {resp}\n")
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"\n[Error]: {e}")

    def run(
        self,
        message: str,
        session_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        verbose: bool = False,
        use_semantic_retrieval: bool = False,
        retrieval_mode: str = "vector",
        stream_callback: Callable[[str], None] | None = None,
        fresh_session: bool = False,
        tool_callback: Callable[[str, dict], None] | None = None,
    ) -> AgentResponse:
        debug_on = bool(getattr(self.config, "debug_trace", True))
        tracer = _TraceCollector(enabled=debug_on)

        temp, n_tok = self._resolve_generation_settings(temperature, max_tokens)

        requested_sid = session_id or str(uuid.uuid4())
        if fresh_session:
            self.memory.clear_session(requested_sid)
            self._activate_session_runtime(
                requested_sid,
                clear_runtime=True,
                force_reload=True,
            )
        else:
            self._activate_session_runtime(requested_sid)
        sid = requested_sid
        self._log.info("Run session=%s msg_len=%d", sid[:8], len(message))

        # Auto-compact: keep session from growing unboundedly
        if bool(getattr(self.config, "auto_compact", True)) and not fresh_session:
            _ac_count = self.memory.count_session_messages(sid)
            _ac_threshold = self.config.history_limit * int(
                getattr(self.config, "auto_compact_threshold", 2)
            )
            if _ac_count > _ac_threshold:
                self._log.info(
                    "Auto-compact: %d messages > threshold %d — compacting",
                    _ac_count,
                    _ac_threshold,
                )
                self.compact_session(sid)
                tracer.emit(
                    "auto_compact",
                    message_count=_ac_count,
                    threshold=_ac_threshold,
                )

        convo = self._create_base_conversation(message)
        self._load_and_append_history(
            convo, sid, message, use_semantic_retrieval, retrieval_mode=retrieval_mode
        )
        self._append_rag_context(convo, message, tracer.emit)
        self._append_user_message(convo, sid, message, tracer.emit)

        tool_calls: list[ToolCall] = []
        seen_tool_signatures: set[tuple[str, str]] = set()
        tool_result_preview_by_sig: dict[tuple[str, str], str] = {}
        iterations = 0
        consecutive_tools = 0

        tool_result_max_chars = int(getattr(self.config, "tool_result_max_chars", 6000))
        assistant_ctx_max_chars = int(
            getattr(self.config, "assistant_ctx_max_chars", 12000)
        )
        trace_context_max_messages = int(
            getattr(self.config, "trace_context_max_messages", 8)
        )
        trace_context_max_chars = int(
            getattr(self.config, "trace_context_max_chars", 500)
        )

        while iterations < self.config.max_iterations:
            iterations += 1
            if verbose:
                print(f"\n--- Iter {iterations} • session={sid[:8]} ---")

            llm_convo = list(convo)
            system_prompt, selection, routing_query = self._resolve_system_prompt(
                message,
                tool_calls=tool_calls,
                tool_result_preview_by_sig=tool_result_preview_by_sig,
            )
            if llm_convo and llm_convo[0].role == MessageRole.SYSTEM:
                llm_convo[0] = Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt,
                )
            else:
                llm_convo.insert(
                    0,
                    Message(role=MessageRole.SYSTEM, content=system_prompt),
                )
            if selection is not None:
                tracer.emit(
                    "skill_routing",
                    iteration=iterations,
                    query_preview=routing_query[:240],
                    selected_skills=[skill.id for skill in selection.selected_skills],
                    selected_tools=selection.selected_tools[:12],
                )
            tool_progress_msg = _render_tool_progress_reminder(
                tool_calls,
                tool_result_preview_by_sig,
                max_items=int(getattr(self.config, "tool_memory_items", 8)),
            )
            llm_convo.append(
                Message(role=MessageRole.SYSTEM, content=tool_progress_msg)
            )
            tracer.emit(
                "tool_progress_prompt",
                iteration=iterations,
                preview=tool_progress_msg[:240],
            )

            llm_convo = self._trim_convo_to_budget(llm_convo, n_tok)
            self._last_context_messages = list(llm_convo)

            context_snapshot = _render_context_snapshot(
                llm_convo,
                max_messages=trace_context_max_messages,
                max_chars_per_message=trace_context_max_chars,
            )
            tracer.emit(
                "iter_context",
                iteration=iterations,
                total_messages=len(convo),
                context_snapshot=context_snapshot,
            )
            if verbose:
                _vprint_block(
                    f"context→llm (iter {iterations})",
                    context_snapshot,
                    limit=12000,
                )

            _do_stream = (stream_callback is not None) or self.config.stream

            # Constrained decoding: pass OpenAI tool schemas to llama.cpp so it
            # enforces valid JSON tool-call structure server-side.  Streaming is
            # automatically disabled inside generate() when tools are provided.
            _tools_payload: list | None = None
            if (
                bool(getattr(self.config, "constrained_decoding", False))
                and self.config.use_chat_api
            ):
                if selection is not None and selection.selected_tools:
                    _tools_payload = self.tools.openai_tool_schemas(
                        selection.selected_tools
                    )
                else:
                    _tools_payload = self.tools.openai_tool_schemas()

            tracer.emit(
                "llm_request_begin",
                iteration=iterations,
                temperature=temp,
                max_tokens=n_tok,
                stream=_do_stream,
                chat_api=self.config.use_chat_api,
                constrained=_tools_payload is not None,
            )

            gen_start = time.perf_counter()
            try:
                text = self.llm.generate(
                    llm_convo,
                    temperature=temp,
                    max_tokens=n_tok,
                    stream=_do_stream,
                    on_token=stream_callback,
                    tools=_tools_payload,
                )
            except Exception as e:
                gen_dur = time.perf_counter() - gen_start
                err_text = "I hit an LLM backend error while generating a response."
                self._log.exception(
                    "LLM generate failed for session=%s: %s", sid[:8], e
                )
                tracer.emit(
                    "llm_error",
                    iteration=iterations,
                    duration_s=round(gen_dur, 6),
                    error=str(e),
                )
                self._append_assistant_message(
                    convo, sid, err_text, assistant_ctx_max_chars
                )
                break
            gen_dur = time.perf_counter() - gen_start

            text = (text or "").strip()
            tracer.emit(
                "llm_response_raw",
                iteration=iterations,
                duration_s=round(gen_dur, 6),
                sample=text[:240],
            )

            if verbose:
                _vprint_block("assistant", text, limit=4000)

            parsed_calls = parse_tool_calls(
                text, use_toon=self.config.use_toon_for_tools
            )

            if len(parsed_calls) > 1:
                tracer.emit(
                    "multi_tool_calls_detected",
                    iteration=iterations,
                    count=len(parsed_calls),
                    names=[c.name for c in parsed_calls[:8]],
                )
                convo.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            "[Tool-calling rule] Return EXACTLY ONE tool_call per response. "
                            "Do not batch calls. Pick the single next-best tool."
                        ),
                    )
                )
                consecutive_tools = 0
                continue

            call = parsed_calls[0] if parsed_calls else None

            if not call:
                # Check whether the model emitted a plan/reasoning without acting.
                # Heuristics: response is short, contains imperative verbs, no question mark.
                _plan_signals = (
                    "i will",
                    "i'll",
                    "let me",
                    "first,",
                    "step 1",
                    "i need to",
                    "i should",
                    "to do this",
                )
                _looks_like_plan = (
                    len(text) < 800
                    and any(sig in text.lower() for sig in _plan_signals)
                    and not text.rstrip().endswith("?")
                    and iterations < self.config.max_iterations - 1
                )
                if _looks_like_plan:
                    # Nudge the model to act instead of just planning.
                    convo.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                "[Reasoning without action detected] "
                                "You described a plan but did not call any tool. "
                                "Proceed immediately: emit your first tool call now."
                            ),
                        )
                    )
                    tracer.emit(
                        "plan_without_action_nudge",
                        iteration=iterations,
                        preview=text[:120],
                    )
                    continue
                self._append_assistant_message(
                    convo, sid, text, assistant_ctx_max_chars
                )
                consecutive_tools = 0
                tracer.emit("final_answer", content_preview=text[:200])
                break

            consecutive_tools += 1
            if consecutive_tools > self.config.max_consecutive_tool_calls:
                msg = "[Tool-call limit reached] Provide your final answer succinctly."
                convo.append(Message(role=MessageRole.SYSTEM, content=msg))
                tracer.emit("guardrail_stop", reason="max_consecutive_tool_calls")
                break

            tool_calls.append(call)
            tracer.emit(
                "parsed_tool_call",
                iteration=iterations,
                name=call.name,
                arguments=call.arguments,
            )

            curr_sig = _tool_call_signature(call)
            if curr_sig in seen_tool_signatures:
                tracer.emit(
                    "duplicate_tool_call_blocked",
                    iteration=iterations,
                    name=call.name,
                    arguments=call.arguments,
                )
                prev_preview = tool_result_preview_by_sig.get(curr_sig, "")
                rule_msg = (
                    "[Tool-calling rule] You already called this exact tool with the same arguments "
                    "in this run. Reuse prior result, call a different tool, or provide the final answer."
                )
                if prev_preview:
                    rule_msg += f"\nPrevious result preview: {prev_preview[:300]}"
                convo.append(Message(role=MessageRole.SYSTEM, content=rule_msg))
                tool_calls.pop()
                consecutive_tools = 0
                continue

            exec_start = time.perf_counter()
            _cache_key = self._tool_cache_key(call)
            _cached_result: str | None = None
            if self._is_tool_cacheable(call.name):
                _entry = self._tool_result_cache.get(_cache_key)
                if _entry is not None:
                    _ts, _r = _entry
                    _ttl = float(getattr(self.config, "tool_cache_ttl", 3600))
                    if _ttl <= 0 or (time.time() - _ts) < _ttl:
                        _cached_result = _r

            # Notify the UI (or any observer) that a tool is about to run.
            if tool_callback is not None:
                try:
                    tool_callback(call.name, dict(call.arguments or {}))
                except Exception:
                    pass

            if _cached_result is not None:
                result_full = _cached_result
                exec_dur = 0.0
                tracer.emit("tool_cache_hit", name=call.name, key=_cache_key[:8])
            else:
                result_full = self.tools.execute(
                    call, use_toon=self.config.use_toon_for_tools
                )
                exec_dur = time.perf_counter() - exec_start
                # Store in cache: only read-only tools, only successful results
                _max_size = int(getattr(self.config, "tool_cache_max_size", 256))
                if (
                    self._is_tool_cacheable(call.name)
                    and not result_full.startswith("Error:")
                    and len(self._tool_result_cache) < _max_size
                ):
                    self._tool_result_cache[_cache_key] = (time.time(), result_full)

            # Structured tool error recovery: replace bare "Error:" strings with
            # diagnostic guidance so the LLM gets actionable context.
            if isinstance(result_full, str) and result_full.startswith("Error:"):
                result_full = (
                    f"[Tool '{call.name}' failed] {result_full}\n\n"
                    "Diagnose this failure: check argument types, file paths, or permissions. "
                    "Options: (a) retry with corrected arguments, "
                    "(b) use a different tool, or (c) explain the blocker clearly."
                )

            result_ctx = self._append_tool_message(
                convo=convo,
                sid=sid,
                call=call,
                result_full=result_full,
                tool_result_max_chars=tool_result_max_chars,
            )
            self._persist_runtime_state(sid)
            if verbose:
                print(result_ctx)
            tracer.emit(
                "tool_result",
                iteration=iterations,
                name=call.name,
                duration_s=round(exec_dur, 6),
                result_preview=result_ctx[:240],
                cache_hit=_cached_result is not None,
            )
            seen_tool_signatures.add(curr_sig)
            tool_result_preview_by_sig[curr_sig] = result_ctx[:240]

        final = next(
            (m.content for m in reversed(convo) if m.role == MessageRole.ASSISTANT), ""
        )

        # Synthesis pass: if the loop exhausted its iteration budget while still
        # executing tool calls (i.e. no natural-language answer was ever produced),
        # inject one extra LLM turn so the agent actually answers the user.
        _last_is_tool_call = bool(
            tool_calls
            and (
                not final.strip()
                or parse_tool_calls(final, use_toon=self.config.use_toon_for_tools)
            )
        )
        if _last_is_tool_call and iterations >= self.config.max_iterations:
            tracer.emit("synthesis_pass_begin", iterations_exhausted=iterations)
            _synth_nudge = (
                "You have now gathered all the information needed from the tools above. "
                "Stop calling tools and write your complete, final answer to the user's "
                "original request using only the information already collected."
            )
            _synth_convo = list(convo) + [
                Message(role=MessageRole.USER, content=_synth_nudge)
            ]
            try:
                _synth_text = (
                    self.llm.generate(
                        _synth_convo,
                        temperature=temp,
                        max_tokens=min(n_tok, 4096),
                        stream=False,
                        on_token=None,
                        tools=None,
                    )
                    or ""
                ).strip()
                if _synth_text:
                    # Strip any tool-call markup the model might still emit
                    _synth_calls = parse_tool_calls(
                        _synth_text, use_toon=self.config.use_toon_for_tools
                    )
                    if not _synth_calls:
                        final = _synth_text
                        self._append_assistant_message(
                            convo, sid, final, assistant_ctx_max_chars
                        )
                        tracer.emit("synthesis_pass_done", preview=final[:200])
                    else:
                        tracer.emit(
                            "synthesis_pass_skipped",
                            reason="model still emitting tool calls",
                        )
            except Exception as _exc:
                self._log.warning("Synthesis pass failed: %s", _exc)
                tracer.emit("synthesis_pass_error", error=str(_exc))

        # Self-reflection: single critique pass to catch incomplete/incorrect answers
        if self.config.enable_reflection and final.strip():
            tracer.emit("reflection_begin")
            reflection_prompt = self.config.reflection_prompt.replace(
                "[FINAL]", final[:2000]
            )
            reflect_convo = list(convo)
            reflect_convo.append(
                Message(role=MessageRole.USER, content=reflection_prompt)
            )
            try:
                reflect_text = (
                    self.llm.generate(
                        reflect_convo,
                        temperature=max(0.0, temp - 0.1),
                        max_tokens=min(n_tok, 1024),
                        stream=False,
                        on_token=None,
                    )
                    or ""
                ).strip()
                tracer.emit("reflection_raw", preview=reflect_text[:200])
                if reflect_text and "COMPLETE" not in reflect_text.upper():
                    reflect_calls = parse_tool_calls(
                        reflect_text, use_toon=self.config.use_toon_for_tools
                    )
                    if reflect_calls:
                        r_call = reflect_calls[0]
                        r_result = self.tools.execute(
                            r_call, use_toon=self.config.use_toon_for_tools
                        )
                        self.memory.save_message(
                            sid,
                            Message(
                                role=MessageRole.TOOL,
                                name=r_call.name,
                                tool_call_id=r_call.id,
                                content=r_result,
                            ),
                        )
                        reflect_convo.extend(
                            [
                                Message(
                                    role=MessageRole.ASSISTANT,
                                    content=reflect_text,
                                ),
                                Message(
                                    role=MessageRole.TOOL,
                                    name=r_call.name,
                                    tool_call_id=r_call.id,
                                    content=r_result[:tool_result_max_chars],
                                ),
                            ]
                        )
                        improved = (
                            self.llm.generate(
                                reflect_convo,
                                temperature=temp,
                                max_tokens=n_tok,
                                stream=False,
                                on_token=None,
                            )
                            or ""
                        ).strip()
                        if improved:
                            final = improved
                            self.memory.save_message(
                                sid,
                                Message(role=MessageRole.ASSISTANT, content=final),
                            )
                    else:
                        final = reflect_text
                        self.memory.save_message(
                            sid,
                            Message(role=MessageRole.ASSISTANT, content=final),
                        )
            except Exception as _exc:
                self._log.warning("Reflection pass failed: %s", _exc)
            tracer.emit("reflection_end", preview=final[:200])

        if self.thinking and final.strip() and not tool_calls:
            tracer.emit("thinking_begin", mode=self.config.thinking.order)
            improved = self.thinking.run(query=message, initial=final)
            final = improved
            tracer.emit("thinking_end", preview=final[:200])

        self._persist_runtime_state(sid)

        debug = tracer.build_debug_payload(
            sid=sid,
            iterations=iterations,
            tool_calls=tool_calls,
            temp=temp,
            n_tok=n_tok,
            config=self.config,
        )

        trace_md = render_trace_markdown(debug) if debug_on else ""

        return AgentResponse(
            messages=convo,
            tool_calls=tool_calls,
            iterations=iterations,
            final_response=final,
            debug=debug,
            trace_md=trace_md,
        )


__all__ = ["Agent"]
