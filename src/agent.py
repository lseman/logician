# agent_core/agent.py  (v2)
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .backends import LlamaCppClient, VLLMClient
from .config import Config
from .db import DocumentDB, MessageDB
from .logging_utils import get_logger
from .messages import Message, MessageRole
from .memory import Memory 

# Thinking pipeline (prompts + reasoners)
from .thinking import ThinkingStrategy

# Tool support
from .tools import (
    HAS_TOON,
    Context,
    ToolCall,
    ToolParameter,
    ToolRegistry,
    parse_tool_calls,
)

log = get_logger("agent.framework")


def _vprint_block(title: str, s: str, limit: int = 2000) -> None:
    s = s or ""
    if limit > 0 and len(s) > limit:
        s = s[:limit] + f"\n... [truncated {len(s) - limit} chars] ..."
    print(f"\n[{title}]\n{s}")


# ===========================================================================
# Response container
# ===========================================================================
@dataclass
class AgentResponse:
    messages: list[Message]
    tool_calls: list[ToolCall]
    iterations: int
    final_response: str
    debug: dict[str, Any] = field(default_factory=dict)
    trace_md: str = ""


def plot_tool_calls_by_iteration(
    response: AgentResponse | dict[str, Any],
    *,
    save_path: str | Path = "tool_calls_by_iteration.png",
    show: bool = False,
    dpi: int = 150,
    title: str = "Agent Tool Calls by Iteration",
) -> Path:
    """
    Plot which tool was called at each iteration from AgentResponse debug events.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    debug = response.get("debug", {}) if isinstance(response, dict) else response.debug
    if not isinstance(debug, dict):
        raise ValueError("Could not find debug payload on response.")

    events = debug.get("events", [])
    rows: list[tuple[int, str, float]] = []
    for ev in events:
        if ev.get("kind") != "parsed_tool_call":
            continue
        iter_id = ev.get("iteration")
        if not isinstance(iter_id, int):
            continue
        name = str(ev.get("name", "unknown_tool"))
        t = float(ev.get("t", 0.0))
        rows.append((iter_id, name, t))

    if not rows:
        raise ValueError("No parsed_tool_call events found. Enable debug_trace and rerun.")

    rows.sort(key=lambda x: (x[0], x[2]))
    iter_to_label: dict[int, str] = {}
    for iter_id, name, _ in rows:
        prev = iter_to_label.get(iter_id)
        iter_to_label[iter_id] = f"{prev}\n{name}" if prev else name

    iterations = sorted(iter_to_label.keys())
    labels = [iter_to_label[i] for i in iterations]
    max_lines = max(lbl.count("\n") + 1 for lbl in labels) if labels else 1

    fig_w = max(8.0, 1.0 * max(6, len(iterations)))
    fig_h = max(3.2, 1.1 + 0.45 * max_lines)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.scatter(iterations, [1] * len(iterations), marker="s", s=140, color="#2563EB")
    for x, lbl in zip(iterations, labels):
        ax.text(x, 1.04, lbl, ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_yticks([])
    ax.set_ylim(0.85, 1.3)
    ax.set_xticks(iterations)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()

    out = Path(save_path)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


@dataclass
class _TraceCollector:
    enabled: bool
    started_ts: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        self.events.append(
            {
                "t": round(time.perf_counter() - self.started_ts, 6),
                "kind": kind,
                **data,
            }
        )

    def total_duration_s(self) -> float:
        return round(time.perf_counter() - self.started_ts, 6)

    def build_debug_payload(
        self,
        *,
        sid: str,
        iterations: int,
        tool_calls: list[ToolCall],
        temp: float,
        n_tok: int,
        config: Config,
    ) -> dict[str, Any]:
        return {
            "session_id": sid,
            "iterations": iterations,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ],
            "events": self.events if self.enabled else [],
            "config": {
                "temperature": temp,
                "max_tokens": n_tok,
                "backend": config.backend,
                "use_chat_api": config.use_chat_api,
                "chat_template": config.chat_template,
                "stream": config.stream,
                "debug_trace": self.enabled,
            },
            "timings": {"total_duration_s": self.total_duration_s()},
        }


def _truncate_text(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"\n\n[... truncated {len(s) - limit} chars ...]"


def _render_context_snapshot(
    convo: list[Message],
    *,
    max_messages: int = 8,
    max_chars_per_message: int = 500,
) -> str:
    """
    Render the tail of the current context in a readable way before LLM call.
    """
    if max_messages < 1:
        max_messages = 1
    tail = convo[-max_messages:]
    lines: list[str] = []
    for i, m in enumerate(tail, start=1):
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        header = f"{i:02d}. {role}"
        if m.name:
            header += f"({m.name})"
        content = _truncate_text((m.content or "").strip(), max_chars_per_message)
        lines.append(header)
        lines.append(content if content else "<empty>")
    return "\n".join(lines)


def _tool_call_signature(call: ToolCall) -> tuple[str, str]:
    return (call.name, json.dumps(call.arguments, sort_keys=True))


def _render_tool_progress_reminder(
    tool_calls: list[ToolCall],
    tool_result_preview_by_sig: dict[tuple[str, str], str],
    *,
    max_items: int = 8,
) -> str:
    if not tool_calls:
        return (
            "[Iteration memory]\n"
            "Already executed tool calls in this run: none yet.\n"
            "Do NOT repeat identical tool calls once tools start being used.\n"
            "If enough evidence is already available, provide the final answer."
        )

    seen: set[tuple[str, str]] = set()
    ordered_unique: list[tuple[str, str]] = []
    for call in tool_calls:
        sig = _tool_call_signature(call)
        if sig in seen:
            continue
        seen.add(sig)
        ordered_unique.append(sig)

    recent = ordered_unique[-max(1, int(max_items)) :]
    lines = [
        "[Iteration memory]",
        "Already executed tool calls in this run (latest unique first):",
    ]
    for i, (name, args_json) in enumerate(reversed(recent), start=1):
        preview = tool_result_preview_by_sig.get((name, args_json), "")
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + " ..."
        lines.append(f"{i}. {name} args={args_json}")
        if preview:
            lines.append(f"   result={preview}")

    lines.append(
        "Do NOT call the exact same tool with the exact same arguments again."
    )
    lines.append(
        "If a prior tool returned empty/insufficient output, pick a different next-best tool or provide a final answer from current evidence."
    )
    return "\n".join(lines)


# ===========================================================================
# AGENT IMPLEMENTATION
# ===========================================================================
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
        # v2: optional lazy RAG init (startup speed)
        lazy_rag: bool = True,
    ) -> None:

        # --------------------------------------------------------------
        # CONFIG + LOGGING
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # TOON fallback
        # --------------------------------------------------------------
        if self.config.use_toon_for_tools and not HAS_TOON:
            self._log.warning(
                "TOON enabled but toon_format not installed. Falling back to JSON."
            )
            self.config.use_toon_for_tools = False

        # --------------------------------------------------------------
        # BACKEND SELECTION (LlamaCpp / vLLM)
        # --------------------------------------------------------------
        if self.config.backend == "vllm":
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
            self.llm = LlamaCppClient(
                base_url=self.config.llama_cpp_url,
                timeout=self.config.timeout,
                use_chat_api=self.config.use_chat_api,
                chat_template=self.config.chat_template,
                stop=self.config.stop,
                retry_attempts=self.config.retry_attempts,
            )

        # --------------------------------------------------------------
        # THINKING STRATEGY
        # --------------------------------------------------------------
        self.thinking: ThinkingStrategy | None = None
        if getattr(self.config, "thinking", None):
            self.thinking = ThinkingStrategy(self.llm, self.config.thinking)

        # --------------------------------------------------------------
        # SYSTEM PROMPT (SOUL.md)
        # --------------------------------------------------------------
        self.system_prompt = system_prompt or self._load_soul_or_default()

        # --------------------------------------------------------------
        # TOOL REGISTRY
        # --------------------------------------------------------------
        self.ctx = Context()

        # IMPORTANT: don't auto-load before injection
        self.tools = ToolRegistry(auto_load_from_skills=False)
        self.tools.install_context(self.ctx)

        # now load tools (they will see ctx/pd/np/helpers)
        self.tools.load_tools_from_skills()

        # invalidate cache since tools now loaded
        self._cached_sys_tools_prompt = ""

        # v2: cached system+tools schema prompt (rebuilt only if tools change)
        self._cached_sys_tools_prompt: str = ""
        self._cached_sys_prompt_base: str = ""
        self._cached_use_toon: bool = bool(self.config.use_toon_for_tools)
        self._cached_tools_version: int = -1  # reads ToolRegistry._version if present

        # --------------------------------------------------------------
        # MEMORY MODULE
        # --------------------------------------------------------------
        self._embedding_model_name = embedding_model or "BAAI/bge-base-en-v1.5"
        self.memory = Memory(
            config=self.config,
            db_path=db_path,
            embedding_model=self._embedding_model_name,
            lazy_rag=lazy_rag,
        )

        self.current_session_id: str | None = None

    def _load_soul_or_default(self) -> str:
        """Loads system prompt from SOUL.md if available, else default."""
        soul_path = Path(__file__).parent.parent / "SOUL.md"
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
        # Your ToolRegistry already tracks a version in other variants; keep compatible.
        v = getattr(self.tools, "_version", None)
        if isinstance(v, int):
            return v
        v2 = getattr(self.tools, "version", None)
        if isinstance(v2, int):
            return v2
        return 0

    def _system_plus_tools_prompt(self) -> str:
        """
        Cache system_prompt + tools schema prompt.
        Avoid rebuilding schema on every run when tools do not change.
        """
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

    def _ensure_doc_db(self) -> None:
        # Delegate to memory
        self.memory._ensure_doc_db()

    def _resolve_generation_settings(
        self, temperature: float | None, max_tokens: int | None
    ) -> tuple[float, int]:
        temp = temperature if temperature is not None else self.config.temperature
        n_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        return temp, n_tok

    def _create_base_conversation(self) -> list[Message]:
        return [
            Message(role=MessageRole.SYSTEM, content=self._system_plus_tools_prompt())
        ]

    def _load_and_append_history(
        self,
        convo: list[Message],
        sid: str,
        message: str,
        use_semantic_retrieval: bool,
        retrieval_mode: str = "vector",
    ) -> None:
        history_recent_tail = int(getattr(self.config, "history_recent_tail", 8))
        history = self.memory.load_history(
            sid,
            message,
            use_semantic_retrieval,
            retrieval_mode=retrieval_mode
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

    # ===========================================================================
    # PUBLIC API
    # ===========================================================================
    def add_tool(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        parameters: list[ToolParameter] | None = None,
    ) -> Agent:
        self.tools.register(name, description, parameters or [], function)
        # invalidate cache (tools version should change, but do it anyway)
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
        """
        Execute a tool directly without any LLM generation.

        Useful for validating skills/tool wiring deterministically.
        """
        args = dict(arguments or {})
        call = ToolCall(
            id=f"direct_{uuid.uuid4().hex[:10]}",
            name=tool_name,
            arguments=args,
        )
        result_full = self.tools.execute(
            call,
            use_toon=self.config.use_toon_for_tools if use_toon is None else bool(use_toon),
        )

        if persist_to_history:
            sid = session_id or self.current_session_id or str(uuid.uuid4())
            self.current_session_id = sid
            self.memory.save_message(
                sid,
                Message(
                    role=MessageRole.USER,
                    content=f"[direct_tool_call] {tool_name} {json.dumps(args, ensure_ascii=False)}",
                ),
            )
            self.memory.save_message(
                sid,
                Message(
                    role=MessageRole.TOOL,
                    name=tool_name,
                    tool_call_id=call.id,
                    content=result_full,
                ),
            )

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
    ) -> str:
        return self.run(
            message,
            session_id=session_id,
            verbose=verbose,
            use_semantic_retrieval=use_semantic_retrieval,
            retrieval_mode=retrieval_mode,
            stream_callback=stream,
            fresh_session=fresh_session,
        ).final_response

    def reset(self, session_id: str | None = None) -> Agent:
        sid = session_id or self.current_session_id
        if sid:
            self.memory.clear_session(sid)
        self.current_session_id = None
        return self

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
        """
        Starts an interactive Read-Eval-Print Loop (SOUL mode).
        """
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

    # ===========================================================================
    # MAIN EXECUTION LOOP
    # ===========================================================================
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
    ) -> AgentResponse:

        debug_on = bool(getattr(self.config, "debug_trace", True))
        tracer = _TraceCollector(enabled=debug_on)

        temp, n_tok = self._resolve_generation_settings(temperature, max_tokens)

        if fresh_session and session_id:
            self.memory.clear_session(session_id)
        self.current_session_id = session_id or str(uuid.uuid4())
        sid = self.current_session_id
        self._log.info("Run session=%s msg_len=%d", sid[:8], len(message))

        convo = self._create_base_conversation()
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

        # context budget knobs (safe defaults if Config lacks them)
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
            tool_progress_msg = _render_tool_progress_reminder(
                tool_calls,
                tool_result_preview_by_sig,
                max_items=int(getattr(self.config, "tool_memory_items", 8)),
            )
            llm_convo.append(Message(role=MessageRole.SYSTEM, content=tool_progress_msg))
            tracer.emit(
                "tool_progress_prompt",
                iteration=iterations,
                preview=tool_progress_msg[:240],
            )

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

            tracer.emit(
                "llm_request_begin",
                iteration=iterations,
                temperature=temp,
                max_tokens=n_tok,
                stream=self.config.stream and (stream_callback is not None),
                chat_api=self.config.use_chat_api,
            )

            gen_start = time.perf_counter()
            try:
                text = self.llm.generate(
                    llm_convo,
                    temperature=temp,
                    max_tokens=n_tok,
                    stream=self.config.stream and stream_callback is not None,
                    on_token=stream_callback,
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
            result_full = self.tools.execute(
                call, use_toon=self.config.use_toon_for_tools
            )

            exec_dur = time.perf_counter() - exec_start

            result_ctx = self._append_tool_message(
                convo=convo,
                sid=sid,
                call=call,
                result_full=result_full,
                tool_result_max_chars=tool_result_max_chars,
            )
            if verbose:
                print(result_ctx)
            tracer.emit(
                "tool_result",
                iteration=iterations,
                name=call.name,
                duration_s=round(exec_dur, 6),
                result_preview=result_ctx[:240],
            )
            seen_tool_signatures.add(curr_sig)
            tool_result_preview_by_sig[curr_sig] = result_ctx[:240]

        final = next(
            (m.content for m in reversed(convo) if m.role == MessageRole.ASSISTANT), ""
        )

        if self.thinking and final.strip():
            tracer.emit("thinking_begin", mode=self.config.thinking.order)
            improved = self.thinking.run(query=message, initial=final)
            final = improved
            tracer.emit("thinking_end", preview=final[:200])

        debug = tracer.build_debug_payload(
            sid=sid,
            iterations=iterations,
            tool_calls=tool_calls,
            temp=temp,
            n_tok=n_tok,
            config=self.config,
        )

        trace_md = self._render_trace_markdown(debug) if debug_on else ""

        return AgentResponse(
            messages=convo,
            tool_calls=tool_calls,
            iterations=iterations,
            final_response=final,
            debug=debug,
            trace_md=trace_md,
        )

    # ===========================================================================
    # TRACE MARKDOWN RENDERING
    # ===========================================================================
    def _render_trace_markdown(self, debug: dict[str, Any]) -> str:
        lines: list[str] = []

        lines.append(f"### Agent Trace — session `{debug['session_id'][:8]}`")
        lines.append("")
        lines.append(
            f"- **Iterations**: {debug['iterations']} | "
            f"**Total**: {debug['timings']['total_duration_s']} s"
        )

        cfg = debug.get("config", {})
        lines.append(
            f"- **Cfg**: temp={cfg.get('temperature')} max_tokens={cfg.get('max_tokens')} "
            f"chat_api={cfg.get('use_chat_api')} template=`{cfg.get('chat_template')}` "
            f"stream={cfg.get('stream')} backend={cfg.get('backend')} debug={cfg.get('debug_trace')}"
        )
        lines.append("")

        tool_calls = debug.get("tool_calls", [])
        if tool_calls:
            first = tool_calls[0]
            lines.append("**Agent command (first tool call):**")
            lines.append("```json")
            lines.append(
                json.dumps(
                    {
                        "tool_call": {
                            "name": first["name"],
                            "arguments": first["arguments"],
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            lines.append("```")
        else:
            lines.append("_No tool call emitted by model._")
        lines.append("")

        lines.append("**Timeline**")
        lines.append("")
        iters: dict[int, dict[str, Any]] = {}
        for ev in debug.get("events", []):
            t = f"{ev['t']:.3f}s"
            kind = ev["kind"]
            iter_id = ev.get("iteration")

            if isinstance(iter_id, int):
                bucket = iters.setdefault(iter_id, {})
                if kind == "iter_context":
                    bucket["context"] = ev.get("context_snapshot", "")
                    bucket["total_messages"] = ev.get("total_messages", 0)
                elif kind == "llm_response_raw":
                    bucket["assistant"] = ev.get("sample", "")
                elif kind == "parsed_tool_call":
                    bucket["tool_name"] = ev.get("name")
                    bucket["tool_args"] = ev.get("arguments", {})
                elif kind == "tool_result":
                    bucket["tool_result"] = ev.get("result_preview", "")

            if kind == "user_message":
                lines.append(f"- [{t}] user → “{ev.get('message_preview', '')[:80]}”")
            elif kind == "thinking_begin":
                lines.append(f"- [{t}] thinking_begin: order={ev.get('mode')}")
            elif kind == "thinking_end":
                lines.append(f"- [{t}] thinking_end: “{ev.get('preview', '')[:80]}”")
            elif kind == "rag_retrieval":
                lines.append(
                    f"- [{t}] rag_retrieval: {ev.get('n_results')} docs "
                    f"(preview: {ev.get('preview', '')[:80]}…)"
                )
            elif kind == "llm_request_begin":
                lines.append(
                    f"- [{t}] llm_request (temp={ev.get('temperature')}, max_tokens={ev.get('max_tokens')})"
                )
            elif kind == "llm_response_raw":
                sample = (ev.get("sample", "")[:80]).replace("\n", " ")
                lines.append(f"- [{t}] llm_response → “{sample}”")
            elif kind == "parsed_tool_call":
                ap = json.dumps(ev.get("arguments", {}), ensure_ascii=False)[:80]
                lines.append(f"- [{t}] parsed_tool_call: {ev.get('name')}({ap}…)")
            elif kind == "tool_result":
                rp = (ev.get("result_preview", "")[:80]).replace("\n", " ")
                lines.append(f"- [{t}] tool_result {ev.get('name')} → “{rp}”")
            elif kind == "final_answer":
                lines.append(
                    f"- [{t}] final_answer → “{ev.get('content_preview', '')[:80]}”"
                )
            elif kind == "guardrail_stop":
                lines.append(f"- [{t}] guardrail_stop: {ev.get('reason')}")

        if iters:
            lines.append("")
            lines.append("**Rendered Iterations**")
            lines.append("")
            for i in sorted(iters.keys()):
                row = iters[i]
                lines.append(f"#### Iteration {i}")
                lines.append("")
                lines.append(
                    f"- Context messages in prompt: {row.get('total_messages', 'n/a')}"
                )
                lines.append("")
                lines.append("Context passed to LLM:")
                lines.append("```text")
                lines.append((row.get("context") or "<none>").strip())
                lines.append("```")
                lines.append("")
                lines.append("Agent answer (raw):")
                lines.append("```text")
                lines.append((row.get("assistant") or "<none>").strip())
                lines.append("```")
                if row.get("tool_name"):
                    lines.append("")
                    lines.append("Tool call:")
                    lines.append("```json")
                    lines.append(
                        json.dumps(
                            {
                                "name": row.get("tool_name"),
                                "arguments": row.get("tool_args", {}),
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    lines.append("```")
                if row.get("tool_result"):
                    lines.append("")
                    lines.append("Tool result (context):")
                    lines.append("```text")
                    lines.append((row.get("tool_result") or "").strip())
                    lines.append("```")
                lines.append("")

        lines.append("")
        return "\n".join(lines)


# ===========================================================================
# AGENT FACTORY
# ===========================================================================
def create_agent(
    llm_url: str = "http://localhost:8080",
    system_prompt: str | None = None,
    use_chat_api: bool = True,
    chat_template: str = "chatml",
    db_path: str = "agent_sessions.db",
    embedding_model: str | None = "BAAI/bge-base-en-v1.5",
    *,
    config_overrides: dict[str, Any] | None = None,
) -> Agent:

    cfg = Config(
        llama_cpp_url=llm_url,
        use_chat_api=use_chat_api,
        chat_template=chat_template,
        use_toon_for_tools=True,
    )

    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return Agent(
        llm_url=cfg.llama_cpp_url,
        system_prompt=system_prompt,
        config=cfg,
        use_chat_api=cfg.use_chat_api,
        chat_template=cfg.chat_template,
        db_path=db_path,
        embedding_model=embedding_model,
        lazy_rag=True,  # v2 default: faster startup
    )
