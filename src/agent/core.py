from __future__ import annotations

import hashlib
import json
import re
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
from ..tools.parser import (
    detect_truncated_tool_call,
    extract_partial_write_from_truncated,
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

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.IGNORECASE | re.DOTALL)
_UNSET = object()
_SOCIAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "thanks",
    "thank you",
    "thx",
    "good morning",
    "good afternoon",
    "good evening",
    "morning",
    "afternoon",
    "evening",
    "how are you",
    "whats up",
    "what's up",
}
_SOCIAL_GREETING_TOKENS = {"hi", "hello", "hey", "yo"}
_TOOL_EXECUTION_CLAIM_PATTERNS = (
    r"\b(?:i|we)\s+(?:have\s+|has\s+|had\s+|already\s+|just\s+)?"
    r"(?:ran|run|executed|called|used|invoked|applied|checked|verified|searched|"
    r"inspected|opened|read|edited|modified|updated|created|wrote|deleted)\b",
    r"\b(?:i['’]?ve|we['’]?ve|it|data|dataset|sample)\s+"
    r"(?:already\s+|just\s+)?"
    r"(?:generated|created|loaded|saved|computed|prepared|produced|updated|applied)\b",
    r"\b(?:has|have)\s+been\s+"
    r"(?:generated|created|loaded|saved|computed|prepared|produced|updated|applied)\b",
    r"\b(?:is|are)\s+now\s+(?:available|loaded|ready|saved)\b",
    r"\b(?:tool|tools|command|commands)\b.{0,40}\b(?:ran|run|executed|called|used)\b",
    r"\b(?:done|completed|finished)\b.{0,40}\b(?:tool|command|task|change|update|edit)\b",
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

    def _is_social_message(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        normalized = re.sub(r"[^a-z0-9\s']", " ", text)
        normalized = " ".join(normalized.split())
        if not normalized:
            return False
        if normalized in _SOCIAL_MESSAGES:
            return True
        tokens = normalized.split()
        if len(tokens) <= 3 and tokens[0] in _SOCIAL_GREETING_TOKENS:
            trailing = [tok for tok in tokens[1:] if tok not in {"there", "team", "all"}]
            return not trailing
        return False

    def _is_follow_up_message(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        if self._is_social_message(text):
            return False
        if any(phrase in text for phrase in _FOLLOW_UP_PHRASES):
            return True
        return len(text.split()) <= 4

    def _is_editing_intent(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        edit_verbs = (
            "edit",
            "modify",
            "change",
            "update",
            "refactor",
            "fix",
            "patch",
            "implement",
            "add",
            "remove",
            "rename",
            "rewrite",
            "create",
            "write",
        )
        if not any(re.search(rf"\b{re.escape(verb)}\b", text) for verb in edit_verbs):
            return False
        # Explicit file path hint is a strong coding signal.
        if re.search(
            r"(?:^|[\s`\"'])[\w./\\-]+\.(?:py|rs|ts|tsx|js|jsx|go|java|c|cpp|h|hpp|md|json|toml|ya?ml)(?:$|[\s`\"'])",
            text,
        ):
            return True
        code_targets = (
            "file",
            "code",
            "function",
            "class",
            "module",
            "test",
            ".py",
            ".rs",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".go",
            ".java",
            ".c",
            ".cpp",
            ".md",
            ".json",
            ".toml",
            ".yaml",
            ".yml",
        )
        if any(token in text for token in code_targets):
            return True
        # Broader repository/implementation cues for feature requests.
        project_targets = (
            "feature",
            "bug",
            "issue",
            "endpoint",
            "command",
            "workflow",
            "pipeline",
            "plugin",
            "module",
            "repo",
            "repository",
            "project",
            "codebase",
            "cli",
            "agent",
        )
        return any(token in text for token in project_targets)

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
        if self._is_social_message(message):
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
            state_lines.append("No dataset is loaded yet.")

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
                    elif self.ctx.loaded:
                        state_lines.append(
                            "For this follow-up, the likely next skills are: preprocessing, analysis, forecasting."
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
        return f"{message.strip()}\n\nRuntime context:\n{bullets}"

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
                include_on_demand_context=bool(
                    getattr(self.config, "skill_include_on_demand_context", True)
                ),
                on_demand_context_max_chars=int(
                    getattr(self.config, "skill_on_demand_context_max_chars", 2600)
                ),
                on_demand_context_max_skills=int(
                    getattr(self.config, "skill_on_demand_context_max_skills", 2)
                ),
                on_demand_context_max_files_per_skill=int(
                    getattr(
                        self.config,
                        "skill_on_demand_context_max_files_per_skill",
                        5,
                    )
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

    @staticmethod
    def _strip_internal_reasoning_tags(text: str) -> str:
        """Remove <think>...</think> blocks that some reasoning models emit.

        These blocks are model-internal traces and can accidentally contain JSON-like
        snippets that confuse tool-call parsing.
        """
        if not text:
            return text
        stripped = _THINK_BLOCK_RE.sub("", text)
        return stripped.strip()

    def _svg_tool_call_nudge(
        self,
        message: str,
        *,
        selection: SkillSelection | None,
        tool_calls: list[ToolCall],
    ) -> str:
        """Return a strict instruction when the user explicitly asks for SVG output."""
        if tool_calls:
            return ""
        text = (message or "").strip().lower()
        if not text:
            return ""

        asks_svg = bool(re.search(r"\bsvg\b", text))
        asks_diagram = bool(
            re.search(
                r"\b(diagram|graph|chart|visuali[sz]e|visualization|visualisation)\b",
                text,
            )
        )
        creation_intent = bool(
            re.search(r"\b(create|generate|build|make|draw|render|produce)\b", text)
        )
        if not asks_svg and not (asks_diagram and creation_intent):
            return ""

        selected_tools = list(selection.selected_tools) if selection else []
        svg_tools = [name for name in selected_tools if name.startswith("svg_")]
        if not svg_tools:
            svg_tools = [
                tool.name for tool in self.tools.list_tools() if tool.name.startswith("svg_")
            ]
        if not svg_tools:
            return ""

        tool_preview = ", ".join(svg_tools[:8])
        return (
            "SVG/diagram generation requested. Return exactly one tool_call now "
            "(no prose). Use one of these tools: "
            f"{tool_preview}. Include concrete output_path/path arguments."
        )

    def _context7_tool_names(self) -> list[str]:
        names: list[str] = []
        for tool in self.tools.list_tools():
            tname = str(getattr(tool, "name", "") or "")
            if self._is_context7_tool_name(tname):
                names.append(tname)
        return list(dict.fromkeys(names))

    def _is_context7_tool_name(self, tool_name: str) -> bool:
        tname = str(tool_name or "").strip()
        if not tname:
            return False
        lname = tname.lower()
        if (
            lname.startswith("context7__")
            or "context7" in lname
            or lname
            in {
                "resolve_library_id",
                "get_library_docs",
                "query_docs",
            }
        ):
            return True
        tool = self.tools.get(tname)
        skill_id = str(getattr(tool, "skill_id", "") or "").lower() if tool else ""
        return skill_id == "mcp__context7"

    def _docs_intent(self, message: str) -> bool:
        text = (message or "").strip().lower()
        if not text:
            return False
        doc_terms = (
            r"\b(docs?|documentation|api reference|reference docs?|library docs?|"
            r"official docs?|latest docs?|how to use|usage guide)\b"
        )
        lib_terms = (
            r"\b(react|next\.?js|vue|svelte|typescript|javascript|node|python|pydantic|"
            r"langchain|openai|fastapi|django|flask|rust|tokio|axum|serde)\b"
        )
        asks_docs = bool(re.search(doc_terms, text))
        asks_library = bool(re.search(lib_terms, text))
        coding_with_lib = bool(
            re.search(
                r"\b(implement|integrate|migrate|upgrade|refactor|fix|build|rewrite)\b",
                text,
            )
        )
        return asks_docs or (
            asks_library
            and (
                "api" in text
                or "syntax" in text
                or "latest" in text
                or coding_with_lib
            )
        )

    def _context7_docs_nudge(
        self,
        message: str,
        *,
        tool_calls: list[ToolCall],
        selection: SkillSelection | None,
    ) -> str:
        if any(self._is_context7_tool_name(call.name) for call in tool_calls):
            return ""
        if not bool(getattr(self.config, "context7_docs_auto_nudge", True)):
            return ""
        if not self._docs_intent(message):
            return ""
        selected = list(selection.selected_tools) if selection else []
        candidates = [name for name in selected if self._is_context7_tool_name(name)]
        if not candidates:
            candidates = self._context7_tool_names()
        if not candidates:
            return ""
        preferred: list[str] = []
        for name in candidates:
            lname = name.lower()
            if "resolve" in lname or "library" in lname:
                preferred.append(name)
        for name in candidates:
            if name not in preferred:
                preferred.append(name)
        preview = ", ".join(preferred[:6])
        return (
            "Documentation request detected. Use Context7 MCP tools before coding from memory: "
            "first resolve the library ID, then fetch targeted docs. "
            f"Prefer tools such as: {preview}. Return exactly one tool_call now."
        )

    def _response_claims_tool_execution(
        self,
        text: str,
        *,
        preferred_tool_names: list[str] | None = None,
    ) -> bool:
        cleaned = self._strip_internal_reasoning_tags(text or "").strip()
        if not cleaned:
            return False
        lower = cleaned.lower()

        # Not an execution claim if clearly phrased as future intent.
        if re.search(
            r"\b(?:i|we)\s+(?:will|can|could|should|might|may|plan to|intend to|need to)\b",
            lower,
        ):
            return False

        if any(re.search(pattern, lower) for pattern in _TOOL_EXECUTION_CLAIM_PATTERNS):
            return True

        # Common hallucinated completion phrasing even without explicit "I ran X".
        if re.search(
            r"\b(?:in|into)\s+(?:the\s+)?(?:tool\s+)?(?:context|memory)\b",
            lower,
        ) and re.search(
            r"\b(?:generated|created|loaded|saved|updated|applied|available|ready)\b",
            lower,
        ):
            return True

        # Additional signal: mentions a known selected tool name in a past-tense context.
        for tool_name in preferred_tool_names or []:
            t = str(tool_name or "").strip().lower()
            if not t:
                continue
            if t in lower and re.search(
                r"\b(?:ran|executed|called|used|invoked|applied|finished|completed|done)\b",
                lower,
            ):
                return True
        return False

    def _tool_claim_guard_nudge(
        self,
        *,
        message: str,
        response_text: str,
        selection: SkillSelection | None,
        tool_calls: list[ToolCall],
        editing_intent: bool,
    ) -> str:
        if tool_calls:
            return ""

        expected_tools = list(selection.selected_tools[:8]) if selection else []
        likely_tool_task = bool(
            expected_tools
            or editing_intent
            or self._docs_intent(message)
        )
        if not likely_tool_task:
            return ""

        if not self._response_claims_tool_execution(
            response_text,
            preferred_tool_names=expected_tools,
        ):
            return ""

        tool_hint = ", ".join(expected_tools[:4]) if expected_tools else "the most relevant tool"
        return (
            "[Unverified tool execution claim] You said a tool/action already ran, "
            "but this run has no matching tool_call yet. Do not claim tool execution "
            "without actually calling a tool.\n"
            f"Now emit exactly one tool_call (no prose). Suggested tools: {tool_hint}."
        )

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
        persisted_result, vectorize = self._tool_result_for_persistence(call, result_full)
        tool_msg_full = Message(
            role=MessageRole.TOOL,
            name=call.name,
            tool_call_id=call.id,
            content=persisted_result,
            vectorize=vectorize,
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
        verbose: bool = False,
        use_semantic_retrieval: bool = False,
        retrieval_mode: str = "vector",
        stream: Callable[[str], None] | None = None,
        fresh_session: bool = False,
        tool_callback: Callable[[str, dict], None] | None = None,
        skill_callback: Callable[[list[str], list[str]], None] | None = None,
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
            skill_callback=skill_callback,
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

    def _is_write_tool_name(self, tool_name: str) -> bool:
        patterns: list[str] = list(
            getattr(self.config, "tool_cache_write_patterns", [])
        )
        name_lower = (tool_name or "").lower()
        if name_lower in {
            "write_file",
            "edit_file_replace",
            "multi_edit",
            "apply_unified_diff",
            "multi_patch",
            "apply_edit_block",
        }:
            return True
        if any(
            token in name_lower
            for token in ("edit_file", "multi_edit", "unified_diff", "multi_patch")
        ):
            return True
        return any(str(p).lower() in name_lower for p in patterns)

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if "/" in text or "\\" in text:
            return True
        return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", text))

    def _extract_paths_from_value(self, value: Any) -> set[str]:
        out: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                key_l = str(key).lower()
                if (
                    key_l in {"path", "file", "filename", "filepath", "output_path"}
                    and isinstance(item, str)
                    and self._looks_like_path(item)
                ):
                    out.add(item.strip())
                out.update(self._extract_paths_from_value(item))
            return out
        if isinstance(value, (list, tuple, set)):
            for item in value:
                out.update(self._extract_paths_from_value(item))
            return out
        if isinstance(value, str):
            text = value.strip()
            if self._looks_like_path(text):
                out.add(text)
            # Extract paths from unified diff headers.
            for line in text.splitlines():
                line_s = line.strip()
                if line_s.startswith("+++ b/") or line_s.startswith("--- a/"):
                    path = line_s[6:].strip()
                    if path and path != "/dev/null":
                        out.add(path)
                elif line_s.startswith("+++ ") or line_s.startswith("--- "):
                    path = line_s[4:].strip()
                    if path and path != "/dev/null":
                        out.add(path)
            return out
        return out

    def _extract_paths_from_tool_call(self, call: ToolCall) -> set[str]:
        return self._extract_paths_from_value(call.arguments or {})

    def _lang_from_path(self, path: str) -> str | None:
        p = str(path or "").strip().lower()
        if not p:
            return None
        if p.endswith(".rs") or p.endswith("cargo.toml") or p.endswith("cargo.lock"):
            return "rust"
        if (
            p.endswith(".js")
            or p.endswith(".jsx")
            or p.endswith(".ts")
            or p.endswith(".tsx")
            or p.endswith("package.json")
            or p.endswith("pnpm-lock.yaml")
            or p.endswith("yarn.lock")
        ):
            return "javascript"
        if p.endswith(".go") or p.endswith("go.mod") or p.endswith("go.sum"):
            return "go"
        if (
            p.endswith(".py")
            or p.endswith(".pyi")
            or p.endswith("pyproject.toml")
            or p.endswith("requirements.txt")
        ):
            return "python"
        return None

    def _infer_verification_profile(
        self, message: str, tool_calls: list[ToolCall]
    ) -> str:
        text = (message or "").lower()
        scores: dict[str, int] = {
            "rust": 0,
            "python": 0,
            "javascript": 0,
            "go": 0,
        }

        if re.search(r"\b(rust|cargo|clippy|rustfmt|crates?)\b", text):
            scores["rust"] += 3
        if re.search(r"\b(python|ruff|pytest|mypy|pyproject|pip)\b", text):
            scores["python"] += 3
        if re.search(r"\b(javascript|typescript|node|npm|pnpm|yarn|eslint|vitest|jest)\b", text):
            scores["javascript"] += 3
        if re.search(r"\b(go|golang|go test|go vet|golint)\b", text):
            scores["go"] += 3

        for call in tool_calls:
            for path in self._extract_paths_from_tool_call(call):
                lang = self._lang_from_path(path)
                if lang in scores:
                    scores[lang] += 2

        active = [lang for lang, score in scores.items() if score > 0]
        if len(active) > 1:
            return "mixed"
        if len(active) == 1:
            return active[0]
        return "generic"

    def _verification_guidance(self, profile: str) -> str:
        matrix = getattr(self.config, "language_verification_matrix", {}) or {}

        def _commands(lang: str, defaults: list[str]) -> list[str]:
            raw = matrix.get(lang, defaults)
            if not isinstance(raw, list):
                return defaults
            cleaned = [str(item).strip() for item in raw if str(item).strip()]
            return cleaned or defaults

        py_cmds = _commands(
            "python",
            ["run_ruff(path=...)", "run_pytest(path=...)", "run_mypy(path=...)"],
        )
        rust_cmds = _commands(
            "rust",
            [
                'run_shell(cmd="cargo check")',
                'run_shell(cmd="cargo test")',
                'run_shell(cmd="cargo clippy")',
            ],
        )
        js_cmds = _commands(
            "javascript",
            ['run_shell(cmd="npm run lint")', 'run_shell(cmd="npm test")'],
        )
        go_cmds = _commands(
            "go",
            ['run_shell(cmd="go test ./...")', 'run_shell(cmd="go vet ./...")'],
        )

        if profile == "rust":
            return (
                "Rust verification: "
                + ", ".join(f"`{cmd}`" for cmd in rust_cmds)
                + ". "
                "Do not use `run_ruff` for Rust-only edits."
            )
        if profile == "python":
            return (
                "Python verification: "
                + ", ".join(f"`{cmd}`" for cmd in py_cmds)
                + "."
            )
        if profile == "javascript":
            return (
                "JavaScript/TypeScript verification: "
                + ", ".join(f"`{cmd}`" for cmd in js_cmds)
                + "."
            )
        if profile == "go":
            return (
                "Go verification: "
                + ", ".join(f"`{cmd}`" for cmd in go_cmds)
                + "."
            )
        if profile == "mixed":
            return (
                "Mixed-language verification: run checks for each changed language "
                "(Python/Rust/JS/Go as applicable)."
                + "."
            )
        return (
            "Use project-appropriate verification (tests/lint/check/build) for the files you changed."
        )

    def _verification_failure_details(self, result_text: str) -> tuple[bool, str]:
        text = str(result_text or "").strip()
        if not text:
            return False, ""

        failed = False
        details: list[str] = []
        payload: Any = None
        try:
            payload = json.loads(text)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            status = str(payload.get("status", "")).lower()
            if status in {"error", "failed", "fail"}:
                failed = True
            for key in ("ok", "success", "passed"):
                if key in payload and payload.get(key) is False:
                    failed = True

            summary = payload.get("summary")
            if isinstance(summary, dict):
                for k in ("failed", "failures", "errors"):
                    v = summary.get(k, 0)
                    if isinstance(v, (int, float)) and v > 0:
                        failed = True

            for key in ("error", "stderr", "message"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    details.append(val.strip())

            for key in ("failures", "errors"):
                vals = payload.get(key)
                if isinstance(vals, list) and vals:
                    details.append(str(vals[0]))

        lower = text.lower()
        if re.search(
            r"\b(failed|traceback|could not compile|panic|assertionerror)\b",
            lower,
        ):
            failed = True
        if re.search(r"\b(failed|errors?)\s*[:=]\s*[1-9]\d*\b", lower):
            failed = True
        if "exit code" in lower and not re.search(r"exit code\s*[:=]?\s*0\b", lower):
            failed = True

        if not details:
            for line in text.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                if re.search(
                    r"(failed|error|traceback|could not compile|panic)",
                    line_s.lower(),
                ):
                    details.append(line_s)
                if len(details) >= 6:
                    break

        summary = "\n".join(details).strip()
        if len(summary) > 1500:
            summary = summary[:1500].rstrip() + " ..."
        return failed, summary

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

    def _tool_result_is_error(self, result_text: str) -> bool:
        text = str(result_text or "").strip()
        if not text:
            return False
        if text.lower().startswith("error:"):
            return True
        payload = self._parse_tool_result_payload(text)
        if isinstance(payload, dict):
            status = str(payload.get("status", "")).strip().lower()
            if status in {"error", "failed", "fail"}:
                return True
            if payload.get("ok") is False or payload.get("success") is False:
                return True
        return False

    def _tool_call_applied_write(self, call: ToolCall, result_text: str) -> bool:
        if not self._is_write_tool_name(call.name):
            return False
        text = str(result_text or "").strip()
        if not text:
            return False
        if self._tool_result_is_error(text):
            return False

        payload = self._parse_tool_result_payload(text)
        if isinstance(payload, dict):
            status = str(payload.get("status", "")).strip().lower()
            if status in {"error", "failed", "fail"}:
                return False
            if status == "partial":
                applied = payload.get("applied")
                if isinstance(applied, (int, float)) and applied > 0:
                    return True
                results = payload.get("results")
                if isinstance(results, list):
                    return any(
                        isinstance(item, dict)
                        and str(item.get("status", "")).lower() == "ok"
                        for item in results
                    )
                return False

            for key in (
                "bytes_written",
                "lines_added",
                "lines_removed",
                "applied",
                "updated",
                "modified",
                "changes",
            ):
                val = payload.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return True
            if status in {"ok", "success", "done"}:
                return True

        lower = text.lower()
        if re.search(r"\b(no changes|not found|failed|error)\b", lower):
            return False
        return True

    def _is_edit_tool_name(self, tool_name: str) -> bool:
        name = str(tool_name or "").strip().lower()
        if not name:
            return False
        if name in {
            "write_file",
            "edit_file_replace",
            "multi_edit",
            "apply_unified_diff",
            "multi_patch",
            "apply_edit_block",
        }:
            return True
        return any(token in name for token in ("edit", "patch", "write_file"))

    def _is_inspection_tool_name(self, tool_name: str) -> bool:
        name = str(tool_name or "").strip().lower()
        if not name:
            return False
        explicit = {
            "read_file",
            "get_file_outline",
            "get_project_map",
            "find_symbol",
            "rg_search",
            "fd_find",
            "list_directory",
            "git_status",
            "git_diff",
            "git_log",
        }
        if name in explicit:
            return True
        return any(
            token in name
            for token in (
                "read",
                "outline",
                "search",
                "find",
                "list_dir",
                "project_map",
                "symbol",
                "git_status",
                "git_diff",
            )
        )

    def _requires_prewrite_inspection(self, tool_name: str) -> bool:
        name = str(tool_name or "").strip().lower()
        if not name:
            return False
        # Allow write_file because it can be used for creating a new file.
        # For patch/edit-style mutations, require at least one inspection step first.
        if name in {
            "edit_file_replace",
            "multi_edit",
            "apply_unified_diff",
            "multi_patch",
            "apply_edit_block",
        }:
            return True
        return any(
            token in name
            for token in ("edit_file", "multi_edit", "unified_diff", "multi_patch", "apply_edit_block")
        )

    def _edit_tool_recovery_nudge(self, call: ToolCall, result_text: str) -> str:
        if not self._is_edit_tool_name(call.name):
            return ""
        text = str(result_text or "").strip()
        if not text:
            return ""

        payload = self._parse_tool_result_payload(text)
        status = str(payload.get("status", "")).strip().lower() if payload else ""
        partial = status == "partial"
        failed = self._tool_result_is_error(text) or partial
        if isinstance(payload, dict):
            failed_count = payload.get("failed")
            if isinstance(failed_count, (int, float)) and failed_count > 0:
                failed = True
                partial = partial or bool(payload.get("applied", 0))
        if not failed:
            return ""

        snippets: list[str] = []
        if isinstance(payload, dict):
            for key in ("error", "message", "stderr"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    snippets.append(value.strip())
            results = payload.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and item.get("status") != "ok":
                        err = item.get("error")
                        if isinstance(err, str) and err.strip():
                            snippets.append(err.strip())
                        if len(snippets) >= 4:
                            break
        if not snippets:
            snippets.append(text)
        err_summary = " | ".join(snippets)[:420]
        lower = " ".join(snippets).lower()

        if re.search(r"old_string not found|missing 'old_string'|not found in file", lower):
            action = (
                "Re-read the file and retry with `edit_file_replace` or `multi_edit`, "
                "including 3-5 unchanged lines of context around `old_string`."
            )
        elif re.search(r"matches \d+ locations|be more specific|unique|ambiguous", lower):
            action = (
                "Your target is ambiguous. Include more unchanged surrounding lines so the "
                "match is unique, then retry one precise edit call."
            )
        elif re.search(r"file not found|not a file|no such file", lower):
            action = (
                "Verify the path first (`list_directory`, `fd_find`, or `read_file`), then "
                "retry with the exact repository-relative path."
            )
        elif re.search(r"patch validation failed|hunk|diff has no hunks|failed to apply", lower):
            action = (
                "Refresh the file contents and regenerate a minimal diff against the current "
                "state, then retry `apply_unified_diff` (or switch to `edit_file_replace`)."
            )
        elif re.search(r"permission denied|operation not permitted|read-only", lower):
            action = (
                "Filesystem permissions blocked the write. Report the blocker clearly and "
                "request a writable path or permission change."
            )
        else:
            action = (
                "Diagnose from the latest file contents, then retry with one smaller targeted "
                "edit call."
            )

        prefix = (
            "Some edit hunks applied but at least one failed."
            if partial
            else "The last edit tool call failed."
        )
        return (
            f"[Edit recovery] {prefix} Do not finalize yet.\n"
            f"Recent error: {err_summary}\n"
            f"Next step: {action}\n"
            "Return exactly one repair tool_call now."
        )

    def _is_shell_verification_call(self, call: ToolCall, profile: str) -> bool:
        tool_name = (call.name or "").lower()
        if tool_name not in {"run_shell", "shell", "bash", "exec"}:
            return False

        args = call.arguments or {}
        cmd = (
            args.get("cmd")
            or args.get("command")
            or args.get("script")
            or args.get("shell_command")
            or ""
        )
        cmd_s = str(cmd).lower()
        if not cmd_s:
            return False

        rust_ok = bool(
            re.search(
                r"\bcargo\s+(check|test|clippy|fmt(?:\s+--\s+check)?)\b|\brustfmt\b.*--check",
                cmd_s,
            )
        )
        python_ok = bool(
            re.search(
                r"\bruff\b|\bpytest\b|\bmypy\b|\bpyright\b|\bpython\s+-m\s+pytest\b",
                cmd_s,
            )
        )
        js_ok = bool(
            re.search(
                r"\b(npm|pnpm|yarn)\s+(run\s+)?(test|lint)\b|\bjest\b|\bvitest\b|\beslint\b",
                cmd_s,
            )
        )
        go_ok = bool(
            re.search(
                r"\bgo\s+(test|vet|fmt)\b|\bgolint\b",
                cmd_s,
            )
        )

        if profile == "rust":
            return rust_ok
        if profile == "python":
            return python_ok
        if profile == "javascript":
            return js_ok
        if profile == "go":
            return go_ok
        if profile == "mixed":
            return rust_ok or python_ok or js_ok or go_ok
        return rust_ok or python_ok or js_ok or go_ok or bool(
            re.search(r"\b(test|lint|check|verify|validate|build)\b", cmd_s)
        )

    def _is_verification_tool_call(self, call: ToolCall, profile: str) -> bool:
        if self._is_shell_verification_call(call, profile):
            return True

        name_lower = (call.name or "").lower()
        patterns: list[str] = list(
            getattr(self.config, "verification_tool_patterns", [])
        )
        generic_match = any(str(p).lower() in name_lower for p in patterns)

        if profile == "rust":
            if "ruff" in name_lower or "pytest" in name_lower or "mypy" in name_lower:
                return False
            rust_name_match = bool(
                re.search(r"\b(cargo|clippy|rust|build|test|check|verify)\b", name_lower)
            )
            return rust_name_match or generic_match

        if profile == "python":
            py_name_match = bool(re.search(r"\b(ruff|pytest|mypy|pyright|lint|test)\b", name_lower))
            return py_name_match or generic_match
        if profile == "javascript":
            js_name_match = bool(re.search(r"\b(eslint|jest|vitest|lint|test|npm|pnpm|yarn)\b", name_lower))
            return js_name_match or generic_match
        if profile == "go":
            go_name_match = bool(re.search(r"\b(go|golang|vet|test|lint)\b", name_lower))
            return go_name_match or generic_match

        return generic_match

    def _build_quality_checklist(
        self,
        *,
        tool_calls: list[ToolCall],
        iterations: int,
        write_tool_called: bool,
        verification_after_write: bool,
        iteration_budget: int | None = None,
    ) -> str:
        if not tool_calls:
            return ""

        max_tools = max(1, int(getattr(self.config, "quality_checklist_max_tools", 5)))
        tool_names = [c.name for c in tool_calls]
        unique_names: list[str] = []
        seen: set[str] = set()
        for name in tool_names:
            if name in seen:
                continue
            seen.add(name)
            unique_names.append(name)
            if len(unique_names) >= max_tools:
                break

        if write_tool_called:
            verification_status = (
                "confirmed" if verification_after_write else "not confirmed"
            )
        else:
            verification_status = "not required"

        tools_preview = ", ".join(unique_names) if unique_names else "none"
        budget = int(iteration_budget or self.config.max_iterations)
        return (
            "**Quality Checklist**\n"
            f"- Iterations: {iterations}/{budget}\n"
            f"- Tools called: {len(tool_calls)}\n"
            f"- Tool summary: {tools_preview}\n"
            f"- Write operations detected: {'yes' if write_tool_called else 'no'}\n"
            f"- Verification after writes: {verification_status}"
        )

    def _score_answer_confidence(
        self,
        question: str,
        answer: str,
        temperature: float,
    ) -> float:
        """Call the LLM as a lightweight judge to score the given answer (0–10).

        Returns 5.0 on any failure so the gate is a no-op when the scoring
        call itself is unavailable.
        """
        prompt = (
            "[Question]\n"
            f"{question[:600]}\n\n"
            "[Answer]\n"
            f"{answer[:1200]}\n\n"
            "Rate the above answer on a scale from 0 to 10 for completeness, "
            "accuracy, and confidence.\n"
            "Output only a single number, e.g. '8.5'."
        )
        try:
            raw = (
                self._llm_generate(
                    [Message(role=MessageRole.USER, content=prompt)],
                    temperature=0.0,
                    max_tokens=16,
                    stream=False,
                    on_token=None,
                )
                or ""
            ).strip()
            m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.\d+)?)\b", raw)
            return float(m.group(0)) if m else 5.0
        except Exception:
            return 5.0

    def _llm_generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        on_token: Callable[[str], None] | None,
        tools: Any = _UNSET,
    ) -> str:
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "on_token": on_token,
        }
        if tools is not _UNSET:
            kwargs["tools"] = tools

        try:
            return self.llm.generate(messages, **kwargs)
        except TypeError as exc:
            # Compatibility fallback for test doubles or legacy backends that
            # don't accept a "tools" keyword argument.
            if "tools" in kwargs and "unexpected keyword argument 'tools'" in str(exc):
                kwargs.pop("tools", None)
                return self.llm.generate(messages, **kwargs)
            raise

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
        skill_callback: Callable[[list[str], list[str]], None] | None = None,
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
        editing_intent = self._is_editing_intent(message)
        max_iterations = int(self.config.max_iterations)
        max_consecutive_tool_calls = int(self.config.max_consecutive_tool_calls)
        if editing_intent:
            max_iterations = max(
                max_iterations,
                int(getattr(self.config, "editing_min_iterations", 12)),
            )
            max_consecutive_tool_calls = max(
                max_consecutive_tool_calls,
                int(
                    getattr(
                        self.config,
                        "editing_max_consecutive_tool_calls",
                        10,
                    )
                ),
            )
        edit_action_nudges = 0
        max_edit_action_nudges = max(
            1, int(getattr(self.config, "editing_force_action_nudges", 2))
        )
        edit_inspection_nudges = 0
        max_edit_inspection_nudges = max(
            1, int(getattr(self.config, "editing_require_inspection_nudges", 3))
        )
        inspection_tool_called = False
        edit_failure_repair_nudges = 0
        max_edit_failure_repair_nudges = max(
            1, int(getattr(self.config, "editing_failure_repair_nudges", 3))
        )
        verification_profile = self._infer_verification_profile(message, tool_calls)
        verification_blocking_failures = False
        last_verification_failure_summary = ""
        verification_repair_nudges = 0
        max_verification_repair_nudges = max(
            1,
            int(getattr(self.config, "verification_auto_repair_max_attempts", 3)),
        )
        write_tool_called = False
        verification_after_write = False
        verification_nudged_once = False
        tool_claim_guard_nudges = 0
        max_tool_claim_guard_nudges = max(
            1,
            int(getattr(self.config, "tool_claim_guard_max_nudges", 2)),
        )
        schema_validation_nudges: dict[str, int] = {}
        last_skill_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        svg_nudged_once = False
        docs_nudged_once = False
        _confidence_gate_retries = 0

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

        # ── Pre-turn deliberation ("think before acting") ────────────────────
        # Ask the LLM to produce a brief plan before any tool calls are made.
        # This anchors the whole turn and prevents premature / shallow actions.
        if bool(getattr(self.config, "pre_turn_thinking", False)):
            _plan_prompt = str(
                getattr(
                    self.config,
                    "pre_turn_thinking_prompt",
                    "Before taking any action, briefly plan your approach step by step.",
                )
            )
            _plan_max_tok = int(
                getattr(self.config, "pre_turn_thinking_max_tokens", 512)
            )
            _plan_convo = list(convo) + [
                Message(role=MessageRole.USER, content=_plan_prompt)
            ]
            try:
                _plan_text = (
                    self._llm_generate(
                        _plan_convo,
                        temperature=max(0.0, temp - 0.2),
                        max_tokens=_plan_max_tok,
                        stream=False,
                        on_token=None,
                    )
                    or ""
                ).strip()
                _plan_text = self._strip_internal_reasoning_tags(_plan_text)
                if _plan_text:
                    convo.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=f"[Pre-turn plan]\n{_plan_text}",
                        )
                    )
                    tracer.emit("pre_turn_thinking_done", preview=_plan_text[:200])
            except Exception as _exc:
                self._log.warning("Pre-turn thinking pass failed: %s", _exc)
                tracer.emit("pre_turn_thinking_error", error=str(_exc))

        while iterations < max_iterations:
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
                selected_skill_ids = [skill.id for skill in selection.selected_skills]
                selected_tool_names = selection.selected_tools[:12]
                tracer.emit(
                    "skill_routing",
                    iteration=iterations,
                    query_preview=routing_query[:240],
                    selected_skills=selected_skill_ids,
                    selected_tools=selected_tool_names,
                )
                if selected_skill_ids:
                    sig = (tuple(selected_skill_ids), tuple(selected_tool_names))
                    if sig != last_skill_signature and skill_callback is not None:
                        try:
                            skill_callback(selected_skill_ids, selected_tool_names)
                        except Exception as exc:
                            self._log.debug("skill_callback failed: %s", exc)
                    last_skill_signature = sig
            tool_progress_msg = _render_tool_progress_reminder(
                tool_calls,
                tool_result_preview_by_sig,
                max_items=int(getattr(self.config, "tool_memory_items", 8)),
            )
            if tool_progress_msg.strip():
                llm_convo.append(
                    Message(role=MessageRole.SYSTEM, content=tool_progress_msg)
                )
                tracer.emit(
                    "tool_progress_prompt",
                    iteration=iterations,
                    preview=tool_progress_msg[:240],
                )
            svg_nudge = self._svg_tool_call_nudge(
                message,
                selection=selection,
                tool_calls=tool_calls,
            )
            if svg_nudge and not svg_nudged_once:
                llm_convo.append(Message(role=MessageRole.SYSTEM, content=svg_nudge))
                svg_nudged_once = True
                tracer.emit(
                    "svg_tool_call_nudge",
                    iteration=iterations,
                    preview=svg_nudge[:240],
                )
            docs_nudge = self._context7_docs_nudge(
                message,
                tool_calls=tool_calls,
                selection=selection,
            )
            if docs_nudge and not docs_nudged_once:
                llm_convo.append(Message(role=MessageRole.SYSTEM, content=docs_nudge))
                docs_nudged_once = True
                tracer.emit(
                    "context7_docs_nudge",
                    iteration=iterations,
                    preview=docs_nudge[:240],
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
                text = self._llm_generate(
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
                self._strip_internal_reasoning_tags(text),
                use_toon=self.config.use_toon_for_tools,
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
                # ── Truncation guard ─────────────────────────────────────────
                # If the model hit max_tokens mid-JSON the brace-counting parser
                # returns nothing.  Detect this and, for write_file calls, try
                # to salvage whatever content was emitted before the cut-off.
                _stripped_text = self._strip_internal_reasoning_tags(text)
                if detect_truncated_tool_call(_stripped_text):
                    _partial = extract_partial_write_from_truncated(_stripped_text)
                    if _partial is not None:
                        _p_tool_name, _p_path, _p_content = _partial
                        _p_lines = _p_content.splitlines()
                        _partial_call = ToolCall(
                            id=f"call_{time.time():.6f}",
                            name=_p_tool_name,
                            arguments={
                                "path": _p_path,
                                "content": _p_content,
                                "mode": "w",
                            },
                        )
                        try:
                            _partial_result = self.tools.execute(
                                _partial_call,
                                use_toon=self.config.use_toon_for_tools,
                            )
                            _last_line = _p_lines[-1].strip() if _p_lines else ""
                            convo.append(
                                Message(
                                    role=MessageRole.SYSTEM,
                                    content=(
                                        f"[Partial write rescued] Your response was "
                                        f"truncated mid-content. {len(_p_lines)} lines "
                                        f"were saved to `{_p_path}` (write result: "
                                        f"{_partial_result[:120]}). "
                                        f"Last line written: {_last_line!r}. "
                                        f"Now APPEND the remaining content with "
                                        f"write_file using mode='a'. Do NOT repeat "
                                        f"lines already written."
                                    ),
                                )
                            )
                            write_tool_called = True
                            verification_after_write = False
                            verification_nudged_once = False
                            tracer.emit(
                                "truncated_partial_write_rescued",
                                iteration=iterations,
                                path=_p_path,
                                lines_written=len(_p_lines),
                            )
                        except Exception as _pe:
                            self._log.warning(
                                "Partial write rescue failed for %s: %s", _p_path, _pe
                            )
                            convo.append(
                                Message(
                                    role=MessageRole.SYSTEM,
                                    content=(
                                        "[Response truncated] Your last response was cut "
                                        "off before the tool call JSON was complete. "
                                        "Retry by splitting content across multiple "
                                        "write_file calls (≤300 lines each), or use "
                                        "edit_file_replace for targeted edits."
                                    ),
                                )
                            )
                    else:
                        convo.append(
                            Message(
                                role=MessageRole.SYSTEM,
                                content=(
                                    "[Response truncated] Your last response was cut off "
                                    "before the tool call JSON was complete (max_tokens "
                                    "limit reached). Please retry: if writing a large "
                                    "file split the content across multiple write_file "
                                    "calls (≤300 lines each), or use edit_file_replace "
                                    "for targeted edits."
                                ),
                            )
                        )
                    tracer.emit(
                        "truncated_tool_call_detected",
                        iteration=iterations,
                        rescued=_partial is not None,
                        preview=_stripped_text[:120],
                    )
                    continue
                # ─────────────────────────────────────────────────────────────

                if (
                    bool(getattr(self.config, "tool_claim_guard_enabled", True))
                    and tool_claim_guard_nudges < max_tool_claim_guard_nudges
                    and iterations < max_iterations
                ):
                    tool_claim_nudge = self._tool_claim_guard_nudge(
                        message=message,
                        response_text=text,
                        selection=selection,
                        tool_calls=tool_calls,
                        editing_intent=editing_intent,
                    )
                    if tool_claim_nudge:
                        tool_claim_guard_nudges += 1
                        convo.append(
                            Message(role=MessageRole.SYSTEM, content=tool_claim_nudge)
                        )
                        tracer.emit(
                            "tool_claim_guard_nudge",
                            iteration=iterations,
                            nudge=tool_claim_guard_nudges,
                            preview=self._strip_internal_reasoning_tags(text)[:180],
                        )
                        consecutive_tools = 0
                        continue

                if (
                    bool(
                        getattr(
                            self.config,
                            "require_verification_after_writes",
                            True,
                        )
                    )
                    and write_tool_called
                    and not verification_after_write
                    and not verification_nudged_once
                    and iterations < max_iterations
                ):
                    verify_hint = self._verification_guidance(verification_profile)
                    convo.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                "[Verification required] You made write/edit changes in this run. "
                                "Before finalizing, call at least one relevant verification tool. "
                                f"{verify_hint} "
                                "If verification cannot be run, explicitly explain why."
                            ),
                        )
                    )
                    verification_nudged_once = True
                    tracer.emit(
                        "verification_required_nudge",
                        iteration=iterations,
                    )
                    continue

                if (
                    write_tool_called
                    and verification_blocking_failures
                    and verification_repair_nudges < max_verification_repair_nudges
                    and iterations < max_iterations
                ):
                    verification_repair_nudges += 1
                    failure_note = (
                        f"\nRecent verification failures:\n{last_verification_failure_summary}\n"
                        if last_verification_failure_summary
                        else ""
                    )
                    convo.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                "[Verification failed] Verification checks still report failures, "
                                "so do not finalize yet. Apply targeted code fixes, then rerun the "
                                "relevant verification command(s)."
                                f"{failure_note}"
                            ),
                        )
                    )
                    tracer.emit(
                        "verification_failed_repair_nudge",
                        iteration=iterations,
                        nudge=verification_repair_nudges,
                        preview=last_verification_failure_summary[:200],
                    )
                    continue

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
                    len(self._strip_internal_reasoning_tags(text)) < 800
                    and any(
                        sig in self._strip_internal_reasoning_tags(text).lower()
                        for sig in _plan_signals
                    )
                    and not self._strip_internal_reasoning_tags(text)
                    .rstrip()
                    .endswith("?")
                    and iterations < max_iterations - 1
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

                if (
                    editing_intent
                    and not write_tool_called
                    and edit_action_nudges < max_edit_action_nudges
                    and iterations < max_iterations
                ):
                    _clean_text = self._strip_internal_reasoning_tags(text)
                    _clean_lower = _clean_text.lower()
                    _completion_claim = any(
                        phrase in _clean_lower
                        for phrase in (
                            "done",
                            "completed",
                            "finished",
                            "updated",
                            "edited",
                            "changed",
                            "implemented",
                            "fixed",
                            "all set",
                            "i have",
                            "i've",
                        )
                    )
                    attempted_write = any(
                        self._is_write_tool_name(c.name) for c in tool_calls
                    )
                    _needs_action_nudge = (
                        not tool_calls
                        or _completion_claim
                        or len(_clean_text) <= 600
                        or not attempted_write
                    )
                    if _needs_action_nudge:
                        edit_action_nudges += 1
                        convo.append(
                            Message(
                                role=MessageRole.SYSTEM,
                                content=(
                                    "[Editing action required] This request is code/file editing. "
                                    "Do not finalize yet. Emit the next concrete write/edit tool call now "
                                    "(for example edit_file_replace, multi_edit, apply_unified_diff, or write_file)."
                                ),
                            )
                        )
                        tracer.emit(
                            "editing_action_required_nudge",
                            iteration=iterations,
                            nudge=edit_action_nudges,
                            preview=text[:160],
                        )
                        consecutive_tools = 0
                        continue

                # ── Confidence-gated stopping ─────────────────────────────────
                # Score the candidate answer; if it falls below the threshold
                # and we still have retry budget, give the model another pass.
                _conf_enabled = bool(
                    getattr(self.config, "confidence_gate_enabled", False)
                )
                _conf_threshold = float(
                    getattr(self.config, "confidence_gate_threshold", 7.0)
                )
                _conf_max_retries = int(
                    getattr(self.config, "confidence_gate_max_retries", 2)
                )
                if (
                    _conf_enabled
                    and _confidence_gate_retries < _conf_max_retries
                    and iterations < max_iterations
                ):
                    _conf_score = self._score_answer_confidence(
                        message,
                        self._strip_internal_reasoning_tags(text),
                        temp,
                    )
                    tracer.emit(
                        "confidence_gate_check",
                        iteration=iterations,
                        score=_conf_score,
                        threshold=_conf_threshold,
                        retry=_confidence_gate_retries,
                    )
                    if _conf_score < _conf_threshold:
                        _confidence_gate_retries += 1
                        convo.append(
                            Message(
                                role=MessageRole.SYSTEM,
                                content=(
                                    f"[Confidence gate] Your answer scored "
                                    f"{_conf_score:.1f}/10 for completeness and "
                                    f"accuracy. Review your reasoning, address any "
                                    f"gaps or uncertainties, and provide a more "
                                    f"thorough and well-supported response."
                                ),
                            )
                        )
                        consecutive_tools = 0
                        continue

                self._append_assistant_message(
                    convo,
                    sid,
                    self._strip_internal_reasoning_tags(text),
                    assistant_ctx_max_chars,
                )
                consecutive_tools = 0
                tracer.emit(
                    "final_answer",
                    content_preview=self._strip_internal_reasoning_tags(text)[:200],
                )
                break

            if (
                editing_intent
                and not inspection_tool_called
                and self._requires_prewrite_inspection(call.name)
                and edit_inspection_nudges < max_edit_inspection_nudges
                and iterations < max_iterations
            ):
                edit_inspection_nudges += 1
                convo.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            "[Inspect before edit] Before patch/edit tools, inspect the target code first. "
                            "Call discovery/read tools now (for example: rg_search, get_file_outline, "
                            "read_file, find_symbol) and then emit the edit call."
                        ),
                    )
                )
                tracer.emit(
                    "editing_inspection_required_nudge",
                    iteration=iterations,
                    nudge=edit_inspection_nudges,
                    attempted_tool=call.name,
                )
                consecutive_tools = 0
                continue

            consecutive_tools += 1
            if consecutive_tools > max_consecutive_tool_calls:
                msg = (
                    "[Tool-call limit reached] Pause and decide the single highest-value next step. "
                    "If you already have enough evidence, provide the final answer now; otherwise emit exactly one next tool call."
                )
                convo.append(Message(role=MessageRole.SYSTEM, content=msg))
                tracer.emit(
                    "guardrail_stop",
                    reason="max_consecutive_tool_calls_nudge",
                    limit=max_consecutive_tool_calls,
                )
                consecutive_tools = 0
                if iterations < max_iterations:
                    continue
                break

            tool_calls.append(call)
            verification_profile = self._infer_verification_profile(message, tool_calls)
            tracer.emit(
                "parsed_tool_call",
                iteration=iterations,
                name=call.name,
                arguments=call.arguments,
            )

            is_write_tool = self._is_write_tool_name(call.name)
            is_verification_tool = self._is_verification_tool_call(
                call, verification_profile
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
                    and not self._tool_result_is_error(result_full)
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
            if self._is_inspection_tool_name(call.name) and not self._tool_result_is_error(
                result_full
            ):
                inspection_tool_called = True

            if is_write_tool and not is_verification_tool:
                write_applied = self._tool_call_applied_write(call, result_full)
                if write_applied:
                    write_tool_called = True
                    verification_after_write = False
                    verification_nudged_once = False
                else:
                    tracer.emit(
                        "write_tool_noop_or_failed",
                        iteration=iterations,
                        name=call.name,
                        preview=result_ctx[:220],
                    )

            edit_repair_nudge = self._edit_tool_recovery_nudge(call, result_full)
            if (
                edit_repair_nudge
                and edit_failure_repair_nudges < max_edit_failure_repair_nudges
                and iterations < max_iterations
            ):
                edit_failure_repair_nudges += 1
                convo.append(
                    Message(role=MessageRole.SYSTEM, content=edit_repair_nudge)
                )
                tracer.emit(
                    "edit_failure_recovery_nudge",
                    iteration=iterations,
                    nudge=edit_failure_repair_nudges,
                    preview=edit_repair_nudge[:240],
                )

            payload = self._parse_tool_result_payload(result_full)
            error_type = (
                str(payload.get("error_type", "")).strip().lower()
                if isinstance(payload, dict)
                else ""
            )
            if (
                error_type in {"schema_validation_failed", "schema_type_validation_failed"}
                and iterations < max_iterations
            ):
                seen_nudges = int(schema_validation_nudges.get(call.name, 0))
                if seen_nudges < 2:
                    schema_validation_nudges[call.name] = seen_nudges + 1
                    allowed = []
                    if isinstance(payload, dict):
                        raw_allowed = payload.get("allowed_arguments", [])
                        if isinstance(raw_allowed, list):
                            allowed = [str(item) for item in raw_allowed if str(item)]
                    allowed_text = (
                        f" Allowed arguments: {', '.join(allowed[:12])}."
                        if allowed
                        else ""
                    )
                    convo.append(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                f"[Schema validation failed] Tool `{call.name}` rejected the arguments."
                                f"{allowed_text} "
                                "Call `describe_tool` for this tool now, then retry with corrected arguments. "
                                "Do not repeat the same invalid argument keys."
                            ),
                        )
                    )
                    tracer.emit(
                        "schema_validation_nudge",
                        iteration=iterations,
                        tool=call.name,
                        error_type=error_type,
                        nudge=seen_nudges + 1,
                    )
                    consecutive_tools = 0
                    continue

            if write_tool_called and is_verification_tool:
                failed_verification, failure_summary = self._verification_failure_details(
                    result_full
                )
                if failed_verification:
                    verification_after_write = False
                    verification_blocking_failures = True
                    last_verification_failure_summary = failure_summary
                    tracer.emit(
                        "verification_failed",
                        iteration=iterations,
                        preview=failure_summary[:220],
                    )
                else:
                    verification_after_write = True
                    verification_blocking_failures = False
                    last_verification_failure_summary = ""
                    verification_repair_nudges = 0

            # ── Post-tool reflection ("afterthought") ────────────────────────
            # After each tool result land, ask the LLM to briefly synthesise what
            # it learnt and plan its next step. This mimics Anthropic's observation
            # loop and dramatically reduces redundant/hallucinated next tool calls.
            if bool(getattr(self.config, "post_tool_thinking", False)):
                _pt_max_tok = int(
                    getattr(self.config, "post_tool_thinking_max_tokens", 256)
                )
                _pt_prompt = str(
                    getattr(
                        self.config,
                        "post_tool_thinking_prompt",
                        "Briefly: what did this tool result tell you, and what is your precise next step?",
                    )
                )
                _pt_convo = list(convo) + [
                    Message(role=MessageRole.USER, content=_pt_prompt)
                ]
                try:
                    _pt_text = (
                        self._llm_generate(
                            _pt_convo,
                            temperature=max(0.0, temp - 0.3),
                            max_tokens=_pt_max_tok,
                            stream=False,
                            on_token=None,
                            tools=None,  # no tool calls during reflection
                        )
                        or ""
                    ).strip()
                    _pt_text = self._strip_internal_reasoning_tags(_pt_text)
                    if _pt_text:
                        convo.append(
                            Message(
                                role=MessageRole.SYSTEM,
                                content=f"[Post-tool reflection]\n{_pt_text}",
                            )
                        )
                        tracer.emit(
                            "post_tool_thinking_done",
                            iteration=iterations,
                            tool=call.name,
                            preview=_pt_text[:200],
                        )
                except Exception as _pt_exc:
                    self._log.debug("Post-tool thinking skipped: %s", _pt_exc)


        final = next(
            (m.content for m in reversed(convo) if m.role == MessageRole.ASSISTANT), ""
        )
        final = self._strip_internal_reasoning_tags(final)

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
        if (
            _last_is_tool_call
            and iterations >= max_iterations
            and not bool(getattr(self.config, "strict_iteration_budget", False))
        ):
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
                    self._llm_generate(
                        _synth_convo,
                        temperature=temp,
                        max_tokens=min(n_tok, 4096),
                        stream=False,
                        on_token=None,
                    )
                    or ""
                ).strip()
                if _synth_text:
                    # Strip any tool-call markup the model might still emit
                    _synth_calls = parse_tool_calls(
                        self._strip_internal_reasoning_tags(_synth_text),
                        use_toon=self.config.use_toon_for_tools,
                    )
                    if not _synth_calls:
                        final = self._strip_internal_reasoning_tags(_synth_text)
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
        _budget_exhausted = iterations >= max_iterations
        _strict_budget = bool(getattr(self.config, "strict_iteration_budget", False))
        _allow_post_passes = not (_strict_budget and _budget_exhausted)

        if self.config.enable_reflection and final.strip() and _allow_post_passes:
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
                    self._llm_generate(
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
                        persisted_result, vectorize = self._tool_result_for_persistence(
                            r_call, r_result
                        )
                        self.memory.save_message(
                            sid,
                            Message(
                                role=MessageRole.TOOL,
                                name=r_call.name,
                                tool_call_id=r_call.id,
                                content=persisted_result,
                                vectorize=vectorize,
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
                            self._llm_generate(
                                reflect_convo,
                                temperature=temp,
                                max_tokens=n_tok,
                                stream=False,
                                on_token=None,
                            )
                            or ""
                        ).strip()
                        if improved:
                            final = self._strip_internal_reasoning_tags(improved)
                            self.memory.save_message(
                                sid,
                                Message(role=MessageRole.ASSISTANT, content=final),
                            )
                    else:
                        final = self._strip_internal_reasoning_tags(reflect_text)
                        self.memory.save_message(
                            sid,
                            Message(role=MessageRole.ASSISTANT, content=final),
                        )
            except Exception as _exc:
                self._log.warning("Reflection pass failed: %s", _exc)
            tracer.emit("reflection_end", preview=final[:200])

        # Apply ThinkingStrategy when there were no tool calls (pure Q&A),
        # or when thinking_apply_after_tools=True (default) so reasoners like
        # SSR/Reflexion can polish tool-gathered evidence into a final answer.
        _apply_thinking = not tool_calls or bool(
            getattr(self.config, "thinking_apply_after_tools", True)
        )
        if self.thinking and final.strip() and _allow_post_passes and _apply_thinking:
            tracer.emit("thinking_begin", mode=self.config.thinking.order)
            try:
                improved = self.thinking.run(query=message, initial=final)
                improved_text = (
                    self._strip_internal_reasoning_tags(improved)
                    if isinstance(improved, str)
                    else ""
                )
                normalized_lines = [
                    ln.strip().lower()
                    for ln in improved_text.splitlines()
                    if ln.strip()
                ]
                single_placeholder = len(normalized_lines) == 1 and normalized_lines[
                    0
                ].rstrip(":") in {"reasoning", "final answer", "answer"}
                double_placeholder = (
                    len(normalized_lines) == 2
                    and normalized_lines[0].rstrip(":") == "reasoning"
                    and normalized_lines[1].rstrip(":") == "final answer"
                )
                if improved_text and not single_placeholder and not double_placeholder:
                    final = improved_text
                else:
                    tracer.emit(
                        "thinking_fallback",
                        reason="empty_or_invalid_thinking_output",
                    )
            except Exception as _exc:
                self._log.warning("Thinking pass failed: %s", _exc)
                tracer.emit("thinking_error", error=str(_exc))
            tracer.emit("thinking_end", preview=final[:200])

        if (
            bool(getattr(self.config, "tool_claim_guard_enabled", True))
            and bool(
                getattr(
                    self.config, "tool_claim_guard_append_runtime_note", True
                )
            )
            and final.strip()
            and not tool_calls
            and self._response_claims_tool_execution(final)
        ):
            runtime_note = (
                "[Runtime note] No tool call was executed in this turn. "
                "Treat completion claims above as unverified."
            )
            final = f"{final.rstrip()}\n\n{runtime_note}".strip()
            self.memory.save_message(
                sid,
                Message(role=MessageRole.ASSISTANT, content=final),
            )
            convo.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=_truncate_text(final, assistant_ctx_max_chars),
                )
            )
            tracer.emit("tool_claim_runtime_note_appended")

        if (
            bool(getattr(self.config, "append_quality_checklist", True))
            and final.strip()
        ):
            checklist = self._build_quality_checklist(
                tool_calls=tool_calls,
                iterations=iterations,
                write_tool_called=write_tool_called,
                verification_after_write=verification_after_write,
                iteration_budget=max_iterations,
            )
            if checklist:
                final = f"{final.rstrip()}\n\n{checklist}".strip()
                self.memory.save_message(
                    sid,
                    Message(role=MessageRole.ASSISTANT, content=final),
                )
                convo.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=_truncate_text(final, assistant_ctx_max_chars),
                    )
                )
                tracer.emit("quality_checklist_appended", preview=checklist[:220])

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
