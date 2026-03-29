from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import Config
from ..logging_utils import get_logger
from ..memory import Memory
from ..messages import Message, MessageRole
from ..repo_graph import related_repo_context
from ..repo_registry import load_repo_index
from ..tools import (
    HAS_TOON,
    Context,
    ToolCall,
    ToolParameter,
    ToolRegistry,
)
from .trace import (
    AgentResponse,
    _truncate_text,
)

from .loop import AgentLoop
from .classify import classify_turn
from .guardrails import GuardrailEngine, default_guards
from .prompt import default_prompt_builder
from .dispatcher import ToolDispatcher
from .state import TurnState
from .types import TurnResult


_OPEN_THINK_TAGS = ("<thinking>", "<think>")
_CLOSE_THINK_TAGS = ("</thinking>", "</think>")


def _find_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str] | None:
    lowered = text.lower()
    matches = [(lowered.find(tag), tag) for tag in tags if lowered.find(tag) >= 0]
    if not matches:
        return None
    pos, tag = min(matches, key=lambda item: item[0])
    return pos, tag


def _partial_tag_suffix_length(text: str, tags: tuple[str, ...]) -> int:
    lowered = text.lower()
    max_len = 0
    upper = min(len(text), max(len(tag) for tag in tags) - 1)
    for size in range(1, upper + 1):
        suffix = lowered[-size:]
        if any(tag.startswith(suffix) for tag in tags):
            max_len = size
    return max_len


class _ThinkingStreamRouter:
    def __init__(
        self,
        stream_callback: Callable[[str], None] | None,
        thinking_callback: Callable[[str], None] | None,
    ) -> None:
        self._stream_callback = stream_callback
        self._thinking_callback = thinking_callback
        self._buffer = ""
        self._in_think = False
        self._think_parts: list[str] = []
        self.visible_emitted = False

    def _emit_visible(self, text: str) -> None:
        if not text:
            return
        self.visible_emitted = True
        if self._stream_callback is not None:
            self._stream_callback(text)

    def _emit_thinking(self, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        if self._thinking_callback is not None:
            self._thinking_callback(content)

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk
        while self._buffer:
            if self._in_think:
                match = _find_tag(self._buffer, _CLOSE_THINK_TAGS)
                if match is None:
                    keep = _partial_tag_suffix_length(self._buffer, _CLOSE_THINK_TAGS)
                    if keep:
                        self._think_parts.append(self._buffer[:-keep])
                        self._buffer = self._buffer[-keep:]
                    else:
                        self._think_parts.append(self._buffer)
                        self._buffer = ""
                    break
                pos, tag = match
                self._think_parts.append(self._buffer[:pos])
                self._emit_thinking("".join(self._think_parts))
                self._think_parts.clear()
                self._buffer = self._buffer[pos + len(tag):]
                self._in_think = False
                continue

            match = _find_tag(self._buffer, _OPEN_THINK_TAGS)
            if match is None:
                keep = _partial_tag_suffix_length(self._buffer, _OPEN_THINK_TAGS)
                if keep:
                    self._emit_visible(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    self._emit_visible(self._buffer)
                    self._buffer = ""
                break

            pos, tag = match
            self._emit_visible(self._buffer[:pos])
            self._buffer = self._buffer[pos + len(tag):]
            self._in_think = True

    def flush(self) -> None:
        if self._in_think:
            self._think_parts.append(self._buffer)
            self._emit_thinking("".join(self._think_parts))
            self._think_parts.clear()
            self._buffer = ""
            self._in_think = False
            return
        self._emit_visible(self._buffer)
        self._buffer = ""


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

        self.system_prompt = system_prompt or self._load_soul_or_default()

        self.ctx = Context()
        self.tools = ToolRegistry(auto_load_from_skills=False)
        self.tools.install_context(self.ctx)
        self.tools.load_tools_from_skills()

        # Register the core tools as always-on runtime capabilities.
        from ..tools import core as _core_tools

        _core_entries = [
            (fn, getattr(fn, "__llm_tool_meta__", {}))
            for fn in _core_tools.CORE_TOOL_FUNCTIONS
        ]
        from pathlib import Path as _Path
        _core_src = _Path(__file__).resolve().parent.parent / "tools" / "core"
        self.tools._register_collected_python_tools(
            tool_entries=_core_entries,
            module_path=_core_src / "__init__.py",
            skill_id="core",
            skill_meta={"always_on": True},
        )

        self._mcp_clients: list[Any] = []
        self._mcp_loaded = False
        if not bool(getattr(self.config, "lazy_mcp_init", True)):
            self._load_mcp_servers()
            self._mcp_loaded = True

        self._embedding_model_name = embedding_model or (
            "BAAI/bge-m3|Snowflake/snowflake-arctic-embed-l-v2.0|"
            "Qwen/Qwen3-Embedding-0.6B|nomic-ai/nomic-embed-text-v1.5|"
            "intfloat/e5-mistral-7b-instruct|BAAI/bge-small-en-v1.5"
        )
        self.memory = Memory(
            config=self.config,
            db_path=db_path,
            embedding_model=self._embedding_model_name,
            lazy_rag=lazy_rag,
        )
        self.current_session_id: str | None = None
        self._loaded_runtime_session_id: str | None = None

        self._agent_loop = self._build_agent_loop()

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

    def _ensure_mcp_servers_loaded(self) -> None:
        if self._mcp_loaded:
            return
        self._load_mcp_servers()
        self._mcp_loaded = True

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
                "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.\n"
                "Never batch writes, edits, or verification commands.\n"
                "Prefer batching when the relevant read-only targets are already known.\n"
                "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.\n"
                "Never narrate internal policy checks or quote system instructions.\n"
                "If no tool is needed, answer normally and clearly."
            )
        return (
            "You are a reliable assistant. If a TOOL is needed, output EXACT JSON:\n"
            '{"tool_call":{"name":"<tool>","arguments":{...}}}\n'
            "Return one tool_call per response, or a small batch of 2-4 independent read-only tool_calls.\n"
            "Never batch writes, edits, or verification commands.\n"
            "Prefer batching when the relevant read-only targets are already known.\n"
            "For codebase review/architecture/improvement requests, inspect enough files before answering; do not stop at one listing or one file if evidence is still thin.\n"
            "Never narrate internal policy checks or quote system instructions.\n"
            "If no tool is needed, answer normally and clearly."
        )

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

    def add_tool(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        parameters: list[ToolParameter] | None = None,
    ) -> Agent:
        self.tools.register(name, description, parameters or [], function)
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
            persisted_result, vectorize = self._tool_result_for_persistence(
                call, result_full
            )
            self.memory.save_message(
                sid_for_persistence,
                Message(
                    role=MessageRole.TOOL,
                    name=tool_name,
                    tool_call_id=call.id,
                    content=persisted_result,
                    vectorize=vectorize,
                ),
            )

        if sid_for_persistence:
            self._persist_runtime_state(sid_for_persistence)

        return result_full

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        stream: Callable[[str], None] | None = None,
        fresh_session: bool = False,
        tool_callback: Callable[..., None] | None = None,
        repair_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        return self.run(
            message,
            session_id=session_id,
            stream_callback=stream,
            fresh_session=fresh_session,
            tool_callback=tool_callback,
            repair_callback=repair_callback,
        ).final_response

    def _parse_tool_result_payload(self, result_text: str) -> dict[str, Any] | None:
        text = str(result_text or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _tool_history_should_skip_vector(self, tool_name: str) -> bool:
        lname = str(tool_name or "").strip().lower()
        if not lname:
            return False
        patterns = list(
            getattr(self.config, "tool_history_vector_exclude_patterns", [])
        )
        for raw in patterns:
            token = str(raw or "").strip().lower()
            if token and token in lname:
                return True
        return False

    def _compact_tool_payload_for_history(
        self,
        *,
        call: ToolCall,
        payload: dict[str, Any],
        max_chars: int,
    ) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "status": str(payload.get("status", "ok") or "ok"),
            "tool": call.name,
        }
        for key in (
            "path",
            "returned_lines",
            "total_lines",
            "has_more",
            "next_start_line",
            "root",
            "file_count",
            "count",
            "matches_found",
            "message",
            "error",
            "error_type",
        ):
            if key in payload:
                compact[key] = payload.get(key)

        content = payload.get("content")
        if isinstance(content, str) and content:
            compact["content_chars"] = len(content)
            compact["content_sha1"] = hashlib.sha1(
                content.encode("utf-8", errors="replace")
            ).hexdigest()[:16]
            compact["content_preview"] = _truncate_text(
                content,
                min(900, max(240, max_chars // 2)),
            )

        matches = payload.get("matches")
        if isinstance(matches, list):
            compact["matches_total"] = len(matches)
            preview: list[dict[str, Any]] = []
            for item in matches[:12]:
                if not isinstance(item, dict):
                    continue
                preview.append(
                    {
                        "file": item.get("file"),
                        "line": item.get("line"),
                        "text": _truncate_text(str(item.get("text", "")), 140),
                        "match": item.get("match"),
                    }
                )
            if preview:
                compact["matches_preview"] = preview

        files = payload.get("files")
        if isinstance(files, list):
            compact["files_total"] = len(files)
            compact["files_preview"] = files[:20]

        entries = payload.get("entries")
        if isinstance(entries, list):
            compact["entries_total"] = len(entries)
            compact["entries_preview"] = entries[:24]

        return compact

    def _tool_result_for_persistence(
        self,
        call: ToolCall,
        result_full: str,
    ) -> tuple[str, bool]:
        """Return persisted tool text + whether it should be vector indexed."""
        text = str(result_full or "")
        if not self._tool_history_should_skip_vector(call.name):
            return text, True

        max_chars = max(
            400,
            int(getattr(self.config, "tool_history_summary_max_chars", 2200)),
        )
        payload = self._parse_tool_result_payload(text)
        if isinstance(payload, dict):
            compact = self._compact_tool_payload_for_history(
                call=call,
                payload=payload,
                max_chars=max_chars,
            )
            compact_text = json.dumps(compact, ensure_ascii=False)
            return _truncate_text(compact_text, max_chars), False
        return _truncate_text(text, max_chars), False

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
            "todo_items": [dict(item) for item in getattr(self.ctx, "todo_items", []) if isinstance(item, dict)],
            "mounted_paths": [
                dict(item)
                for item in getattr(self.ctx, "mounted_paths", [])
                if isinstance(item, dict)
            ],
            "active_repos": [
                dict(item)
                for item in getattr(self.ctx, "active_repos", [])
                if isinstance(item, dict)
            ],
            "retrieval_insights": [
                dict(item)
                for item in getattr(self.ctx, "retrieval_insights", [])
                if isinstance(item, dict)
            ],
            "rag_docs": [
                dict(item)
                for item in getattr(self.ctx, "rag_docs", [])
                if isinstance(item, dict)
            ],
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
                "todo_items": [
                    dict(item)
                    for item in getattr(persisted_ctx, "todo_items", [])
                    if isinstance(item, dict)
                ],
                "mounted_paths": [
                    dict(item)
                    for item in getattr(persisted_ctx, "mounted_paths", [])
                    if isinstance(item, dict)
                ],
                "active_repos": [
                    dict(item)
                    for item in getattr(persisted_ctx, "active_repos", [])
                    if isinstance(item, dict)
                ],
                "retrieval_insights": [
                    dict(item)
                    for item in getattr(persisted_ctx, "retrieval_insights", [])
                    if isinstance(item, dict)
                ],
                "rag_docs": [
                    dict(item)
                    for item in getattr(persisted_ctx, "rag_docs", [])
                    if isinstance(item, dict)
                ],
                "forecast_model": persisted_ctx.nf_best_model,
                "has_forecast_cv": persisted_ctx.nf_cv_full is not None,
                "forecast_prediction_column": persisted_ctx.nf_pred_col,
            }
        else:
            runtime_snapshot = self._runtime_context_snapshot()
        todo_summary = list(runtime_snapshot.get("todo_items") or [])

        return {
            "session_id": sid,
            "persisted_messages": persisted_messages,
            "history_limit": history_limit,
            "history_recent_tail": history_recent_tail,
            "loaded_message_budget": min(persisted_messages, history_limit),
            "history_over_budget": persisted_messages > history_limit,
            "runtime": runtime_snapshot,
            "todo": todo_summary,
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

    def preview_prompt_context(
        self,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sid = session_id or self.current_session_id
        query = str(message or "")
        if sid and self._loaded_runtime_session_id != sid:
            self._load_runtime_state(sid)

        classification = classify_turn(query)
        state = TurnState(
            turn_id=str(uuid.uuid4()),
            classified_as=classification.intent,
            domain_groups_activated=classification.domain_groups,
            user_query=query,
        )
        state.available_tool_names = self._agent_loop.dispatcher.available_tool_names()

        history: list[Message] = []
        if sid:
            try:
                history = self.memory.load_history(
                    sid,
                    message=query,
                    use_semantic_retrieval=bool(query)
                    and getattr(self.config, "default_use_semantic_retrieval", False),
                    retrieval_mode=getattr(
                        self.config, "default_retrieval_mode", "hybrid"
                    ),
                ) or []
            except Exception:
                history = []

        convo = list(history)
        if query.strip():
            convo.append(Message(role=MessageRole.USER, content=query))

        system = self._agent_loop.prompt_builder.build(state, self.config)
        untrimmed = self._agent_loop._with_system(convo, system)
        trimmed = self._agent_loop._trim_to_budget(untrimmed)

        return {
            "session_id": sid,
            "classified_as": classification.intent,
            "domain_groups": sorted(classification.domain_groups),
            "history_loaded_count": len(history),
            "untrimmed_message_count": len(untrimmed),
            "trimmed_message_count": len(trimmed),
            "history_limit": int(self.config.history_limit),
            "context_token_budget": int(getattr(self.config, "context_token_budget", 0)),
            "system_prompt": system,
            "messages": [
                {
                    "role": msg.role.value,
                    "name": msg.name,
                    "content": msg.content,
                }
                for msg in trimmed
            ],
        }

    def _prompt_runtime_context(self) -> str:
        runtime = self._runtime_context_snapshot()
        lines: list[str] = []

        mounted_paths = list(runtime.get("mounted_paths") or [])
        if mounted_paths:
            lines.append("Mounted paths available to the session:")
            for item in mounted_paths[:6]:
                path = str(item.get("path", "")).strip() or "(unknown path)"
                file_count = int(item.get("file_count", 0) or 0)
                token_count = int(item.get("token_count", 0) or 0)
                detail_parts = []
                if file_count > 0:
                    detail_parts.append(f"files={file_count}")
                if token_count > 0:
                    detail_parts.append(f"tokens≈{token_count}")
                details = f" ({', '.join(detail_parts)})" if detail_parts else ""
                lines.append(f"- {path}{details}")
            if len(mounted_paths) > 6:
                lines.append(f"- … {len(mounted_paths) - 6} more mounted path(s)")

        active_repos = list(runtime.get("active_repos") or [])
        if active_repos:
            lines.append("Active repos selected for this session:")
            for item in active_repos[:6]:
                name = str(item.get("name", "")).strip() or str(item.get("id", "")).strip() or "(unknown repo)"
                repo_id = str(item.get("id", "")).strip()
                chunks = int(item.get("chunks_added", 0) or 0)
                files_processed = int(item.get("files_processed", 0) or 0)
                detail_parts = []
                if repo_id:
                    detail_parts.append(f"id={repo_id}")
                if files_processed > 0:
                    detail_parts.append(f"files={files_processed}")
                if chunks > 0:
                    detail_parts.append(f"chunks={chunks}")
                details = f" ({', '.join(detail_parts)})" if detail_parts else ""
                lines.append(f"- {name}{details}")
            if len(active_repos) > 6:
                lines.append(f"- … {len(active_repos) - 6} more active repo(s)")

        rag_docs = list(runtime.get("rag_docs") or [])
        if rag_docs:
            lines.append("Active RAG docs available for retrieval:")
            for item in rag_docs[:6]:
                path = str(item.get("path", "")).strip() or "(unknown doc)"
                kind = str(item.get("kind", "")).strip()
                chunks = int(item.get("chunks", 0) or 0)
                label = str(item.get("label", "")).strip()
                detail_parts = []
                if kind:
                    detail_parts.append(kind)
                if chunks > 0:
                    detail_parts.append(f"chunks={chunks}")
                if label:
                    detail_parts.append(f"label={label}")
                details = f" ({', '.join(detail_parts)})" if detail_parts else ""
                lines.append(f"- {path}{details}")
            if len(rag_docs) > 6:
                lines.append(f"- … {len(rag_docs) - 6} more RAG doc(s)")

        todo_items = list(runtime.get("todo_items") or [])
        if todo_items:
            active_todos = []
            for item in todo_items:
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                status = str(item.get("status", "")).strip() or "pending"
                active_todos.append(f"- [{status}] {title}")
            if active_todos:
                lines.append("Current task checklist:")
                lines.extend(active_todos[:8])

        if runtime.get("loaded"):
            dataset = str(runtime.get("data_name", "")).strip() or "unnamed dataset"
            row_count = int(runtime.get("row_count", 0) or 0)
            cols = list(runtime.get("value_columns") or [])
            col_preview = ", ".join(str(col) for col in cols[:6]) or "none"
            shape = "multivariate" if runtime.get("is_multivariate") else "univariate"
            lines.append(
                f"Loaded dataset: {dataset} ({shape}, rows={row_count}, value_columns={col_preview})."
            )
            if runtime.get("freq"):
                lines.append(f"Known frequency: {runtime['freq']}.")

        return "\n".join(lines)

    @staticmethod
    def _merge_where(
        lhs: dict[str, Any] | None,
        rhs: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if lhs and rhs:
            return {"$and": [lhs, rhs]}
        return lhs or rhs

    @staticmethod
    def _paths_where(paths: list[str]) -> dict[str, Any] | None:
        clean = [str(path or "").strip() for path in paths if str(path or "").strip()]
        if not clean:
            return None
        if len(clean) == 1:
            return {"repo_rel_path": clean[0]}
        return {"$or": [{"repo_rel_path": path} for path in clean]}

    def _prompt_retrieval_context(self, state: TurnState) -> str:
        query = str(state.user_query or "").strip()
        self.ctx.retrieval_insights = []
        if not query or not getattr(self.config, "prompt_rag_context_enabled", True):
            return ""
        runtime = self._runtime_context_snapshot()
        active_repos = list(runtime.get("active_repos") or [])
        rag_docs = list(runtime.get("rag_docs") or [])
        repo_library = [
            dict(item)
            for item in load_repo_index()
            if isinstance(item, dict)
        ]
        if not active_repos and repo_library:
            active_repos = [
                item
                for item in repo_library
                if int(item.get("chunks_added", 0) or 0) > 0
            ][:3]
        repo_ids = [
            str(item.get("id") or "").strip()
            for item in active_repos
            if str(item.get("id") or "").strip()
        ]
        if not repo_ids and not rag_docs:
            return ""

        where: dict[str, Any] | None = None
        if repo_ids:
            if len(repo_ids) == 1:
                where = {"repo_id": repo_ids[0]}
            else:
                where = {"$or": [{"repo_id": repo_id} for repo_id in repo_ids]}

        rows = self.memory.search_rag(
            query,
            where=where,
            n_results=int(getattr(self.config, "prompt_rag_context_max_results", 4)),
            event_cb=None,
        )
        if not rows:
            return ""

        repo_lookup = {
            str(item.get("id") or "").strip(): dict(item)
            for item in load_repo_index()
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }

        lines = ["Use this retrieved project context if it helps answer the request:"]
        retrieval_insights: list[dict[str, Any]] = []
        for row in rows[: int(getattr(self.config, "prompt_rag_context_max_results", 4))]:
            meta = dict(row.get("metadata") or {})
            repo_id = str(meta.get("repo_id") or "").strip()
            repo_rel_path = str(meta.get("repo_rel_path") or "").strip()
            path = repo_rel_path or str(meta.get("path") or "").strip() or "(unknown path)"
            label = f"{repo_id}:{path}" if repo_id else path
            lines.append(
                f"- {label} — {self._compact_preview(str(row.get('content') or ''), 220)}"
            )

        if repo_ids:
            neighbor_limit = int(getattr(self.config, "prompt_rag_context_neighbor_results", 2))
            for repo_id in repo_ids[:3]:
                repo = repo_lookup.get(repo_id)
                if repo is None:
                    continue
                seed_paths = [
                    str((row.get("metadata") or {}).get("repo_rel_path") or "").strip()
                    for row in rows
                    if str((row.get("metadata") or {}).get("repo_id") or "").strip() == repo_id
                    and str((row.get("metadata") or {}).get("repo_rel_path") or "").strip()
                ]
                if not seed_paths:
                    continue
                related = related_repo_context(
                    repo,
                    rel_paths=seed_paths,
                    query=query,
                    limit=6,
                )
                related_file_rows = [
                    {
                        "rel_path": str(item.get("rel_path") or "").strip(),
                        "score": int(item.get("score", 0) or 0),
                    }
                    for item in list(related.get("related_files") or [])
                    if str(item.get("rel_path") or "").strip()
                ]
                related_files = [
                    str(item.get("rel_path") or "").strip()
                    for item in related_file_rows
                ][: max(0, neighbor_limit + 1)]
                if related_files:
                    lines.append(
                        f"Related files in repo {repo_id}: " + ", ".join(related_files[:6])
                    )
                    extra_rows = self.memory.search_rag(
                        query,
                        where=self._merge_where(
                            {"repo_id": repo_id},
                            self._paths_where(related_files[: max(0, neighbor_limit)]),
                        ),
                        n_results=max(1, neighbor_limit),
                        event_cb=None,
                    )
                    for extra in extra_rows[:neighbor_limit]:
                        meta = dict(extra.get("metadata") or {})
                        extra_path = (
                            str(meta.get("repo_rel_path") or "").strip()
                            or str(meta.get("path") or "").strip()
                            or "(unknown path)"
                        )
                        lines.append(
                            f"- neighbor {repo_id}:{extra_path} — "
                            f"{self._compact_preview(str(extra.get('content') or ''), 180)}"
                        )

                related_symbols = list(related.get("related_symbols") or [])[:4]
                if related_symbols:
                    symbol_preview = ", ".join(
                        (
                            f"{str(item.get('name') or '').strip()}@"
                            f"{str(item.get('rel_path') or '').strip()}"
                        )
                        for item in related_symbols
                        if str(item.get("name") or "").strip()
                    )
                    if symbol_preview:
                        lines.append(f"Related symbols in repo {repo_id}: {symbol_preview}")
                retrieval_insights.append(
                    {
                        "query": self._compact_preview(query, 140),
                        "repo_id": repo_id,
                        "repo_name": str(repo.get("name") or repo_id or "").strip(),
                        "seed_paths": seed_paths[:4],
                        "retrieved_paths": [
                            str((row.get("metadata") or {}).get("repo_rel_path") or "").strip()
                            for row in rows[:6]
                            if str((row.get("metadata") or {}).get("repo_id") or "").strip()
                            == repo_id
                            and str((row.get("metadata") or {}).get("repo_rel_path") or "").strip()
                        ][:4],
                        "related_files": related_file_rows[:4],
                        "related_symbols": [
                            {
                                "name": str(item.get("name") or "").strip(),
                                "symbol_kind": str(item.get("symbol_kind") or "").strip(),
                                "rel_path": str(item.get("rel_path") or "").strip(),
                                "line": int(item.get("line", 0) or 0),
                            }
                            for item in related_symbols
                            if str(item.get("name") or "").strip()
                        ][:4],
                    }
                )

        self.ctx.retrieval_insights = retrieval_insights[:4]

        return _truncate_text(
            "\n".join(lines),
            int(getattr(self.config, "prompt_rag_context_max_chars", 2200)),
        )

    def _build_agent_loop(self) -> AgentLoop:
        """Construct the AgentLoop with the default runtime components."""
        guards = default_guards(self.config)
        prompt_builder = default_prompt_builder(
            base_prompt_fn=lambda: self.system_prompt,
            tool_schema_fn=lambda: self.tools.tools_schema_prompt(
                use_toon=self.config.use_toon_for_tools,
                mode=self.config.tool_schema_mode,
                include_tool_names=self.tools.default_prompt_tool_names(),
            ),
            routing_fn=lambda query: self.tools.skill_routing_prompt(
                query,
                use_toon=self.config.use_toon_for_tools,
                mode=self.config.tool_schema_mode,
                top_k=1,
            )[0],
            domain_schema_fn=lambda groups: self.tools.skill_routing_prompt(
                " ".join(sorted(groups)),
                use_toon=self.config.use_toon_for_tools,
                mode=self.config.tool_schema_mode,
                top_k=2,
            )[0],
            retrieval_context_fn=self._prompt_retrieval_context,
            runtime_context_fn=self._prompt_runtime_context,
        )
        dispatcher = ToolDispatcher(self.tools)
        # guardrail_stall_nudge_limit: max corrective retries per guard (default 1).
        # A value of 1 means one nudge is allowed; the second triggers hard-stop.
        max_nudges = getattr(self.config, "guardrail_stall_nudge_limit", 1) + 1
        from ..thinking import ThinkingStrategy

        thinking = (
            ThinkingStrategy(self.llm, self.config.thinking)
            if getattr(self.config, "thinking", None)
            else None
        )
        return AgentLoop(
            llm=self.llm,
            guardrails=GuardrailEngine(guards, max_nudges_per_guard=max_nudges),
            prompt_builder=prompt_builder,
            dispatcher=dispatcher,
            config=self.config,
            use_toon=self.config.use_toon_for_tools,
            tool_schemas_fn=lambda: self.tools.openai_tool_schemas(),
            memory=self.memory,
            mcp_loader=self._ensure_mcp_servers_loaded,
            thinking=thinking,
        )

    def run(
        self,
        message: str,
        session_id: str | None = None,
        stream_callback: Callable[[str], None] | None = None,
        thinking_callback: Callable[[str], None] | None = None,
        fresh_session: bool = False,
        tool_callback: Callable[..., None] | None = None,
        repair_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResponse:
        sid = session_id or str(uuid.uuid4())
        self.current_session_id = sid

        if fresh_session:
            self.memory.clear_session(sid)
            self._reset_context_state()
            self._clear_persisted_runtime_state(sid)
        else:
            self._load_runtime_state(sid)
        self._loaded_runtime_session_id = sid

        self._ensure_mcp_servers_loaded()

        messages = [Message(role=MessageRole.USER, content=message)]

        # Keep the loop aligned with mutable Agent dependencies.
        self._agent_loop.llm = self.llm
        self._agent_loop.memory = self.memory

        token_emitted = False
        seen_thinking: set[str] = set()

        def _emit_thinking(content: str) -> None:
            normalized = re.sub(r"\s+", " ", str(content or "")).strip()
            if not normalized or normalized in seen_thinking:
                return
            seen_thinking.add(normalized)
            if thinking_callback is not None:
                thinking_callback(str(content or "").strip())

        stream_router = (
            _ThinkingStreamRouter(stream_callback=stream_callback, thinking_callback=_emit_thinking)
            if stream_callback is not None
            else None
        )

        def _emit_token(token: str) -> None:
            nonlocal token_emitted
            if stream_router is not None:
                stream_router.feed(token)
                token_emitted = token_emitted or stream_router.visible_emitted
                return
            token_emitted = True
            if stream_callback is not None:
                stream_callback(token)

        async def _run_loop() -> TurnResult:
            return await self._agent_loop.run(
                messages,
                session_id=sid,
                token_callback=_emit_token if stream_callback is not None else None,
                thinking_callback=_emit_thinking,
                tool_callback=tool_callback,
                repair_callback=repair_callback,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(_run_loop())
        else:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_loop())
                result = future.result()

        self._persist_runtime_state(sid)

        if stream_router is not None:
            stream_router.flush()
            token_emitted = token_emitted or stream_router.visible_emitted

        final_response = result.final_response or ""
        if stream_callback is not None and final_response and not token_emitted:
            try:
                stream_callback(final_response)
            except Exception:
                pass

        response_messages = list(result.messages)
        if final_response and (
            not response_messages
            or response_messages[-1].role != MessageRole.ASSISTANT
            or response_messages[-1].content != final_response
        ):
            response_messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=final_response,
                    thinking_log=list(result.thinking_log or []),
                )
            )

        debug = {
            "session_id": sid,
            "iterations": result.state.iteration,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in result.tool_calls
            ],
            "events": [],
        }

        return AgentResponse(
            messages=response_messages,
            tool_calls=result.tool_calls,
            iterations=result.state.iteration,
            final_response=final_response,
            debug=debug,
            trace_md="",
            thinking_log=result.thinking_log,
        )

__all__ = ["Agent"]
