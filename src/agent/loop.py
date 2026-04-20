"""AgentLoop: thin ReAct-style main loop for the refactored agent."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import uuid
from typing import Any, Callable

from ..backends.base import LLMBackend
from ..config import Config
from ..messages import Message, MessageRole
from ..thinking import ThinkingStrategy
from ..tools.parser import parse_tool_calls
from ..tools.runtime import ToolCall
from .classify import classify_turn
from .dispatcher import DispatchResult, ToolDispatcher
from .guardrails import GuardrailEngine
from .prompt import PromptBuilder
from .state import TurnState
from .types import TurnResult


def format_tool_results(results: list[DispatchResult]) -> list[Message]:
    """Convert dispatch results to tool-role messages for the conversation."""
    messages = []
    for r in results:
        content = r.error if r.error else r.output
        messages.append(
            Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=r.call_id,
                name=r.tool_name,
            )
        )
    return messages


def _sanitize_plan(plan: str) -> str:
    """Strip tool_call / JSON tool call blocks and meta-reasoning from a pre-turn plan.

    The planning step should produce prose.  If the LLM emits tool_call YAML
    blocks, JSON, or asterisk-style internal monologue, remove them so they
    don't bleed into the main conversation context.
    """
    # Strip TOON-style tool_call blocks (tool_call:\n  name: ...\n  arguments:\n ...)
    plan = re.sub(
        r"^tool_call:.*?(?=\n[^\s]|\Z)",
        "",
        plan,
        flags=re.MULTILINE | re.DOTALL,
    )
    # Strip JSON tool call objects { "name": ..., "arguments": ... }
    plan = re.sub(
        r"\{[^{}]*\"(?:name|tool)\"[^{}]*\"arguments\"[^{}]*\}",
        "",
        plan,
        flags=re.DOTALL,
    )
    # Strip lines that are asterisk-style internal monologue (* text or *   text)
    lines = plan.splitlines()
    lines = [ln for ln in lines if not re.match(r"^\s*\*[^*]", ln)]
    plan = "\n".join(lines)
    # Collapse excessive blank lines left behind
    plan = re.sub(r"\n{3,}", "\n\n", plan)
    return plan.strip()


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_think_blocks(text: str) -> tuple[str, str]:
    """Extract <think>/<thinking> blocks from an LLM response.

    Returns (think_content, clean_response) where:
    - think_content is all think-block contents joined by newlines
    - clean_response is the original text with think blocks removed
    """
    blocks = _THINK_BLOCK_RE.findall(text)
    think_content = "\n\n".join(b.strip() for b in blocks if b.strip())
    clean = _THINK_BLOCK_RE.sub("", text).strip()
    return think_content, clean


def _response_similarity(a: str, b: str) -> float:
    """Fast bigram-based similarity in [0, 1]. Avoids importing difflib/sklearn."""
    if not a or not b:
        return 0.0
    a_l, b_l = a.lower(), b.lower()
    if a_l == b_l:
        return 1.0

    def bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)}

    sa, sb = bigrams(a_l), bigrams(b_l)
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


class AgentLoop:
    def _llm_generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        grammar: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_reasoning_token: Callable[[str], None] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "on_token": on_token,
            "tools": tools,
            "grammar": grammar,
            "tool_choice": tool_choice,
        }
        if on_reasoning_token is not None:
            try:
                signature = inspect.signature(self.llm.generate)
            except (TypeError, ValueError):
                signature = None
            if signature is None:
                kwargs["on_reasoning_token"] = on_reasoning_token
            else:
                params = signature.parameters.values()
                supports_reasoning = (
                    "on_reasoning_token" in signature.parameters
                    or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
                )
                if supports_reasoning:
                    kwargs["on_reasoning_token"] = on_reasoning_token
        return self.llm.generate(messages, **kwargs)

    @staticmethod
    def _tool_call_summary(tool_calls: list[ToolCall]) -> str:
        names = [
            str(getattr(call, "name", "") or "").strip()
            for call in tool_calls
            if str(getattr(call, "name", "") or "").strip()
        ]
        if not names:
            return "a pending tool call"
        unique: list[str] = []
        for name in names:
            if name not in unique:
                unique.append(name)
        if len(unique) == 1:
            return f"`{unique[0]}`"
        if len(unique) == 2:
            return f"`{unique[0]}` and `{unique[1]}`"
        head = ", ".join(f"`{name}`" for name in unique[:2])
        return f"{head}, and {len(unique) - 2} more"

    @staticmethod
    def _tool_calls_match_tail(
        executed_calls: list[ToolCall],
        pending_calls: list[ToolCall],
    ) -> bool:
        if not pending_calls or len(executed_calls) < len(pending_calls):
            return False
        tail = executed_calls[-len(pending_calls) :]
        return all(
            str(lhs.name or "") == str(rhs.name or "")
            and dict(lhs.arguments or {}) == dict(rhs.arguments or {})
            for lhs, rhs in zip(tail, pending_calls)
        )

    def __init__(
        self,
        llm: LLMBackend,
        guardrails: GuardrailEngine,
        prompt_builder: PromptBuilder,
        dispatcher: ToolDispatcher,
        config: Config,
        use_toon: bool = False,
        tool_schemas_fn: Callable[[], list[dict[str, Any]]] | None = None,
        memory: Any | None = None,
        mcp_loader: Callable[[], None] | None = None,
        thinking: ThinkingStrategy | None = None,
    ) -> None:
        self.llm = llm
        self.guardrails = guardrails
        self.prompt_builder = prompt_builder
        self.dispatcher = dispatcher
        self.config = config
        self.use_toon = use_toon
        self._tool_schemas_fn = tool_schemas_fn
        self.memory = memory
        self._mcp_loader = mcp_loader
        self.thinking = thinking

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _load_history(self, session_id: str | None, query: str = "") -> list[Message]:
        """Load conversation history for a session from memory."""
        if not self.memory or not session_id:
            return []
        try:
            use_semantic = bool(query) and getattr(
                self.config, "default_use_semantic_retrieval", False
            )
            retrieval_mode = getattr(self.config, "default_retrieval_mode", "hybrid")
            result = self.memory.load_history(
                session_id,
                message=query,
                use_semantic_retrieval=use_semantic,
                retrieval_mode=retrieval_mode,
            )
            return result or []
        except Exception:
            return []

    def _save_message(self, msg: Message, session_id: str | None) -> None:
        """Persist a message to memory."""
        if not self.memory or not session_id:
            return
        try:
            self.memory.save_message(session_id, msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context trimming
    # ------------------------------------------------------------------

    def _trim_to_budget(self, messages: list[Message]) -> list[Message]:
        """Trim conversation to fit within context token budget.

        Strategy:
        1. Always keep the system message (index 0).
        2. Count tokens; if over budget, drop oldest non-system messages
           (oldest assistant/tool pairs first) until within budget.
        3. Fall back to message-count limit if token counting unavailable.
        """
        budget = getattr(self.config, "context_token_budget", 0)
        limit = getattr(self.config, "history_limit", 18)

        system = [m for m in messages[:1] if m.role == MessageRole.SYSTEM]
        rest = messages[1:]

        # Always enforce message-count limit
        if len(rest) > limit:
            rest = rest[-limit:]

        if not budget:
            return system + rest

        # Token-aware trimming: estimate ~4 chars/token
        def _approx_tokens(msgs: list[Message]) -> int:
            return sum(len(m.content or "") for m in msgs) // 4

        sys_tokens = _approx_tokens(system)
        available = budget - sys_tokens
        if available <= 0:
            return system + rest[-2:] if rest else system

        # Drop oldest messages until within token budget
        while rest and _approx_tokens(rest) > available:
            rest = rest[2:] if len(rest) >= 2 else rest[1:]

        return system + rest

    # ------------------------------------------------------------------
    # Post-tool reflection
    # ------------------------------------------------------------------

    async def _post_tool_think(self, convo: list[Message]) -> str:
        """One-shot LLM reflection immediately after a tool batch.

        Returns a short observation string (or "" if disabled / failed).
        The caller injects it as a system hint before the next main LLM call.
        """
        prompt = getattr(
            self.config,
            "post_tool_thinking_prompt",
            "Briefly: what did this tool result tell you, and what is your precise next step?",
        )
        reflection_convo = list(convo) + [Message(role=MessageRole.USER, content=prompt)]
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                lambda: self._llm_generate(
                    reflection_convo,
                    temperature=0.3,
                    max_tokens=256,
                ),
            )
            _, clean = _extract_think_blocks(raw)
            return (clean or raw).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Confidence gate
    # ------------------------------------------------------------------

    async def _score_response(self, query: str, response: str) -> float:
        """Ask the LLM to score a candidate final answer on a 0–10 scale.

        Returns 10.0 on failure so scoring errors never block the agent.
        """
        scoring_prompt = (
            f"Query: {query}\n\nResponse: {response}\n\n"
            "Score the response 0-10 for completeness and accuracy. "
            "Output ONLY a number, no explanation."
        )
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(
                None,
                lambda: self._llm_generate(
                    [Message(role=MessageRole.USER, content=scoring_prompt)],
                    temperature=0.0,
                    max_tokens=8,
                ),
            )
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
            return float(m.group(1)) if m else 10.0
        except Exception:
            return 10.0

    # ------------------------------------------------------------------
    # Pre-turn thinking
    # ------------------------------------------------------------------

    async def _pre_turn_think(self, convo: list[Message], session_id: str | None) -> str:
        """Generate a brief plan before the main tool loop."""
        return await self._pre_turn_think_with_callback(
            convo,
            session_id,
            token_callback=None,
        )

    async def _pre_turn_think_with_callback(
        self,
        convo: list[Message],
        session_id: str | None,
        token_callback: Callable[[str], None] | None,
    ) -> str:
        """Generate a brief plan before the main tool loop."""
        prompt = getattr(
            self.config,
            "pre_turn_thinking_prompt",
            "Before acting, briefly plan: what is the task, which tools will you use, what are the steps?",
        )
        planning_convo = list(convo) + [Message(role=MessageRole.USER, content=prompt)]
        loop = asyncio.get_running_loop()
        stream_enabled = token_callback is not None and bool(getattr(self.config, "stream", False))
        try:
            plan = await loop.run_in_executor(
                None,
                lambda: self._llm_generate(
                    planning_convo,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=stream_enabled,
                    on_token=token_callback if stream_enabled else None,
                ),
            )
            # Extract any embedded <think> blocks from the plan and emit them
            # via the callback before returning the clean plan text.
            think_content, clean_plan = _extract_think_blocks(plan)
            if think_content and token_callback is not None:
                token_callback(think_content)
            return (clean_plan or plan).strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _generation_tools(self) -> list[dict[str, Any]] | None:
        if not bool(getattr(self.config, "constrained_decoding", False)):
            return None
        if self._tool_schemas_fn is None:
            return None
        try:
            tools = self._tool_schemas_fn() or []
        except Exception:
            return None
        return tools or None

    @staticmethod
    def _tool_error_payload(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _repairable_tool_error(payload: dict[str, Any] | None) -> bool:
        if not isinstance(payload, dict):
            return False
        return str(payload.get("error_type", "")).strip().lower() in {
            "schema_validation_failed",
            "schema_type_validation_failed",
            "invalid_arguments",
            "tool_not_found",
        }

    def _preflight_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[list[ToolCall] | None, dict[str, Any] | None]:
        prepared: list[ToolCall] = []
        for call in tool_calls:
            prepared_call, error_text = self.dispatcher.prepare_call(call)
            if error_text is not None:
                return None, {
                    "call": {
                        "name": call.name,
                        "arguments": dict(call.arguments or {}),
                    },
                    "error": self._tool_error_payload(error_text)
                    or {
                        "status": "error",
                        "tool": call.name,
                        "error": error_text,
                    },
                }
            assert prepared_call is not None
            prepared.append(prepared_call)
        return prepared, None

    def _tool_repair_feedback(self, failure: dict[str, Any] | None) -> str:
        if not isinstance(failure, dict):
            return (
                "Your tool call could not be validated. "
                "Return only a corrected JSON tool_call object."
            )
        payload = failure.get("error")
        if not isinstance(payload, dict):
            return (
                "Your tool call could not be validated. "
                "Return only a corrected JSON tool_call object."
            )
        hint = str(payload.get("usage_hint") or "").strip()
        error = str(payload.get("error") or "Tool call validation failed.").strip()
        missing = payload.get("missing_required")
        unknown = payload.get("unknown_arguments")
        parts = [error]
        if isinstance(missing, list) and missing:
            parts.append("Missing: " + ", ".join(str(item) for item in missing))
        if isinstance(unknown, list) and unknown:
            parts.append("Unknown: " + ", ".join(str(item) for item in unknown))
        if hint:
            parts.append(hint)
        parts.append("Return only corrected JSON tool_call output.")
        return " ".join(part for part in parts if part)

    async def _attempt_tool_call_repair(
        self,
        *,
        state: TurnState,
        system_prompt: str,
        user_query: str,
        response: str,
        tool_calls: list[ToolCall],
        failure: dict[str, Any],
        repair_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, list[ToolCall]] | None:
        if not bool(getattr(self.config, "tool_call_repair_enabled", False)):
            return None
        max_attempts = max(0, int(getattr(self.config, "tool_call_repair_max_attempts", 1)))
        if state.tool_repair_attempts >= max_attempts:
            return None

        payload = failure.get("error")
        if not self._repairable_tool_error(payload):
            return None
        failing_call = failure.get("call", {})
        tool_name = (
            str(
                getattr(tool_calls[0], "name", "")
                or failing_call.get("name")
                or payload.get("tool")
                or "unknown"
            ).strip()
            or "unknown"
        )
        error_type = str(payload.get("error_type", "") or "").strip()
        attempt_no = state.tool_repair_attempts + 1
        if repair_callback is not None:
            try:
                repair_callback(
                    {
                        "stage": "attempt",
                        "attempt": attempt_no,
                        "tool": tool_name,
                        "error_type": error_type,
                        "message": self._tool_repair_feedback(failure),
                    }
                )
            except Exception:
                pass

        repair_prompt = "\n\n".join(
            [
                "You are repairing a malformed tool call.",
                "Return only corrected tool call JSON.",
                "Do not explain your reasoning.",
                "Do not answer the user directly.",
                "Preserve the user's intent and keep valid arguments when possible.",
                "If there were multiple tool calls, return the corrected full tool-call response only.",
            ]
        )
        parsed_calls = [
            {"name": call.name, "arguments": dict(call.arguments or {})} for call in tool_calls
        ]
        repair_messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.SYSTEM, content=repair_prompt),
            Message(
                role=MessageRole.USER,
                content=(
                    f"Original user request:\n{user_query}\n\n"
                    f"Previous tool-call response:\n{response}\n\n"
                    f"Parsed tool calls:\n{json.dumps(parsed_calls, ensure_ascii=False)}\n\n"
                    f"Failing tool call:\n{json.dumps(failing_call, ensure_ascii=False)}\n\n"
                    f"Validation error:\n{json.dumps(payload, ensure_ascii=False)}"
                ),
            ),
        ]

        loop = asyncio.get_running_loop()
        try:
            repaired_response = await loop.run_in_executor(
                None,
                lambda: self._llm_generate(
                    repair_messages,
                    temperature=0.0,
                    max_tokens=min(1200, self.config.max_tokens),
                    tools=self._generation_tools(),
                    stream=False,
                ),
            )
        except Exception:
            if repair_callback is not None:
                try:
                    repair_callback(
                        {
                            "stage": "failed",
                            "attempt": attempt_no,
                            "tool": tool_name,
                            "error_type": error_type or "repair_generation_failed",
                            "message": "tool-call repair generation failed",
                        }
                    )
                except Exception:
                    pass
            return None
        _, repaired_clean = _extract_think_blocks(repaired_response)
        repaired_text = (repaired_clean or repaired_response).strip()
        repaired_calls = parse_tool_calls(
            repaired_text,
            use_toon=self.use_toon,
            strict=bool(getattr(self.config, "strict_tool_call_parsing", False)),
        )
        if not repaired_calls:
            if repair_callback is not None:
                try:
                    repair_callback(
                        {
                            "stage": "failed",
                            "attempt": attempt_no,
                            "tool": tool_name,
                            "error_type": error_type or "repair_parse_failed",
                            "message": "tool-call repair did not yield a valid tool call",
                        }
                    )
                except Exception:
                    pass
            return None
        state.tool_repair_attempts += 1
        if repair_callback is not None:
            try:
                repair_callback(
                    {
                        "stage": "repaired",
                        "attempt": attempt_no,
                        "tool": tool_name,
                        "error_type": error_type,
                        "message": "tool call repaired successfully",
                    }
                )
            except Exception:
                pass
        return repaired_text, repaired_calls

    async def run(
        self,
        messages: list[Message],
        session_id: str | None = None,
        token_callback: Callable[[str], None] | None = None,
        thinking_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
        post_tool_callback: Callable[[ToolCall, DispatchResult], None] | None = None,
        repair_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        """Run one agent turn. Returns when the LLM produces a final response."""
        last_content = messages[-1].content or ""
        classification = classify_turn(last_content)
        state = TurnState(
            turn_id=str(uuid.uuid4()),
            classified_as=classification.intent,
            domain_groups_activated=classification.domain_groups,
            user_query=last_content,
        )
        state.available_tool_names = self.dispatcher.available_tool_names()

        # Persist user message before loading history so it's available for
        # subsequent sessions and doesn't vanish from the DB record.
        if messages and messages[-1].role == MessageRole.USER:
            self._save_message(messages[-1], session_id)

        # Load conversation history (semantic query anchored to current turn)
        history = self._load_history(session_id, query=last_content)
        # Avoid double-counting: history already contains the user message we
        # just saved, so strip the last message from `messages` if history ends
        # with a matching user turn.
        if history and history[-1].role == MessageRole.USER and history[-1].content == last_content:
            convo = history
        else:
            convo = history + list(messages)

        # Fast path for social/informational turns — single LLM call, no tools.
        if classification.intent in ("social", "informational"):
            return await self._fast_path(
                convo,
                state,
                token_callback=token_callback,
                thinking_callback=thinking_callback,
                session_id=session_id,
            )

        # Defer MCP startup until the turn actually needs the full tool loop.
        if self._mcp_loader is not None and callable(self._mcp_loader):
            self._mcp_loader()

        pre_turn_enabled = getattr(self.config, "pre_turn_thinking", False)

        # Pre-turn thinking: only fires for execution/design turns when enabled.
        # Produces a brief plan injected as a system hint before the tool loop.
        if pre_turn_enabled:
            cb = thinking_callback if thinking_callback is not None else None
            plan = await self._pre_turn_think_with_callback(
                convo,
                session_id,
                token_callback=cb,
            )
            sanitized = _sanitize_plan(plan)
            # Only inject if the plan ends on a sentence boundary — a truncated
            # mid-sentence plan confuses the LLM more than no plan at all.
            if sanitized and (sanitized[-1] in ".!?\n" or "\n" in sanitized):
                convo.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=f"[Pre-turn plan]\n{sanitized}",
                    )
                )

        response = ""
        _draft_stall_limit = getattr(self.config, "repeated_draft_stall_limit", 1)
        _draft_sim_threshold = getattr(self.config, "repeated_draft_similarity_threshold", 0.88)
        _last_no_tool_response: str = ""
        _repeated_draft_count: int = 0

        while state.iteration < self.config.max_iterations:
            # 1. Build system prompt
            system = self.prompt_builder.build(state, self.config)

            # 2. Inject system prompt as first message (or replace existing)
            llm_messages = self._with_system(convo, system)

            # 3. Apply context budget trimming
            llm_messages = self._trim_to_budget(llm_messages)

            stream_enabled = token_callback is not None and bool(
                getattr(self.config, "stream", False)
            )

            loop = asyncio.get_running_loop()
            # 4. LLM call (sync backend wrapped in executor)
            response = await loop.run_in_executor(
                None,
                lambda: self._llm_generate(
                    llm_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    tools=self._generation_tools(),
                    stream=stream_enabled,
                    on_token=token_callback if stream_enabled else None,
                    on_reasoning_token=thinking_callback,
                ),
            )

            # 4a. Extract <think> / <thinking> blocks from the response and
            #     route to thinking_callback. The clean response (without think
            #     blocks) is used for all subsequent processing so tool calls
            #     and guardrails don't see the thinking noise.
            think_content, response = _extract_think_blocks(response)
            if think_content:
                state.thinking_log.append(think_content)
                if thinking_callback is not None:
                    thinking_callback(think_content)

            # 5. Parse tool calls from response
            tool_calls: list[ToolCall] = parse_tool_calls(
                response,
                use_toon=self.use_toon,
                strict=bool(getattr(self.config, "strict_tool_call_parsing", False)),
            )

            # 6. Reset consecutive count when LLM produces no tool calls
            if not tool_calls:
                state.consecutive_tool_count = 0

            if tool_calls:
                prepared_calls, failure = self._preflight_tool_calls(tool_calls)
                if failure is not None:
                    if repair_callback is not None:
                        payload = failure.get("error") if isinstance(failure, dict) else None
                        call_info = failure.get("call") if isinstance(failure, dict) else {}
                        try:
                            repair_callback(
                                {
                                    "stage": "invalid",
                                    "attempt": state.tool_repair_attempts + 1,
                                    "tool": str(
                                        (call_info or {}).get("name")
                                        or (payload or {}).get("tool")
                                        or "unknown"
                                    ),
                                    "error_type": str((payload or {}).get("error_type", "") or ""),
                                    "message": self._tool_repair_feedback(failure),
                                }
                            )
                        except Exception:
                            pass
                    repair = await self._attempt_tool_call_repair(
                        state=state,
                        system_prompt=system,
                        user_query=last_content,
                        response=response,
                        tool_calls=tool_calls,
                        failure=failure,
                        repair_callback=repair_callback,
                    )
                    if repair is not None:
                        response, tool_calls = repair
                        prepared_calls, failure = self._preflight_tool_calls(tool_calls)

                if failure is not None:
                    if repair_callback is not None:
                        payload = failure.get("error") if isinstance(failure, dict) else None
                        call_info = failure.get("call") if isinstance(failure, dict) else {}
                        try:
                            repair_callback(
                                {
                                    "stage": "nudge",
                                    "attempt": state.tool_repair_attempts,
                                    "tool": str(
                                        (call_info or {}).get("name")
                                        or (payload or {}).get("tool")
                                        or "unknown"
                                    ),
                                    "error_type": str((payload or {}).get("error_type", "") or ""),
                                    "message": self._tool_repair_feedback(failure),
                                }
                            )
                        except Exception:
                            pass
                    convo.append(
                        Message(
                            role=MessageRole.USER,
                            content=self._tool_repair_feedback(failure),
                        )
                    )
                    state.iteration += 1
                    continue

                assert prepared_calls is not None
                tool_calls = prepared_calls

            # 7. Run guardrails (before executing tools)
            guard_result = self.guardrails.run(state, response, tool_calls)
            if guard_result.hard_stop:
                state.final_response = response
                break
            if guard_result.nudge:
                convo.append(Message(role=MessageRole.USER, content=guard_result.nudge))
                state.guardrail_nudges[guard_result.guard_name] = (
                    state.guardrail_nudges.get(guard_result.guard_name, 0) + 1
                )
                state.iteration += 1
                continue

            # 8. No tool calls → confidence gate → repeated-draft stall → accept as final
            if not tool_calls:
                # Confidence gate: score the candidate answer; retry if below threshold.
                if getattr(self.config, "confidence_gate_enabled", False):
                    _gate_max = getattr(self.config, "confidence_gate_max_retries", 2)
                    _gate_threshold = getattr(self.config, "confidence_gate_threshold", 7.0)
                    if state.confidence_retries < _gate_max:
                        score = await self._score_response(last_content, response)
                        if score < _gate_threshold:
                            state.confidence_retries += 1
                            convo.append(
                                Message(
                                    role=MessageRole.USER,
                                    content=(
                                        f"Your answer scored {score:.1f}/10 for completeness. "
                                        "Please expand or correct it."
                                    ),
                                )
                            )
                            state.iteration += 1
                            continue

                # Repeated-draft stall detection: if the model produces nearly
                # identical no-tool responses multiple times in a row, stop.
                if _last_no_tool_response:
                    sim = _response_similarity(response, _last_no_tool_response)
                    if sim >= _draft_sim_threshold:
                        _repeated_draft_count += 1
                        if _repeated_draft_count >= _draft_stall_limit:
                            state.final_response = response
                            break
                    else:
                        _repeated_draft_count = 0
                _last_no_tool_response = response

                state.final_response = response
                # Persist the final assistant message
                self._save_message(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response,
                        thinking_log=list(state.thinking_log or []),
                    ),
                    session_id,
                )
                break

            # 9. Execute tools (parallel reads, serial writes)
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=response,
                thinking_log=list(state.thinking_log or []),
            )
            convo.append(assistant_msg)
            self._save_message(assistant_msg, session_id)

            results = await self.dispatcher.dispatch(
                tool_calls,
                state,
                self.config,
                tool_callback=tool_callback,
            )
            if post_tool_callback is not None:
                call_by_id = {call.id: call for call in tool_calls}
                for dispatch_result in results:
                    tool_call = call_by_id.get(dispatch_result.call_id)
                    if tool_call is None:
                        continue
                    try:
                        post_tool_callback(tool_call, dispatch_result)
                    except Exception:
                        pass
            tool_msgs = format_tool_results(results)
            convo.extend(tool_msgs)
            for tm in tool_msgs:
                self._save_message(tm, session_id)

            # Update last_tool_output for InspectionGuard.
            if tool_msgs:
                state.last_tool_output = tool_msgs[-1].content or ""

            # Tool availability can change mid-turn (for example after invoke_skill
            # lazy-loads a skill's scripts). Refresh before the next guard/prompt pass.
            state.available_tool_names = self.dispatcher.available_tool_names()

            # Post-tool reflection: brief observation injected before the next LLM call.
            if getattr(self.config, "post_tool_thinking", False):
                observation = await self._post_tool_think(convo)
                if observation:
                    # Use USER role instead of SYSTEM to maintain proper message ordering
                    # (system messages must always be first)
                    convo.append(
                        Message(
                            role=MessageRole.USER,
                            content=f"[Post-tool observation]\n{observation}\n\nProceed with your next step.",
                        )
                    )
                    state.thinking_log.append(observation)
                    if thinking_callback is not None:
                        thinking_callback(observation)

            state.iteration += 1

        # If we exhausted iterations without a final response
        if state.final_response is None:
            final_tool_calls = parse_tool_calls(
                response,
                use_toon=self.use_toon,
                strict=bool(getattr(self.config, "strict_tool_call_parsing", False)),
            )
            if final_tool_calls:
                summary = self._tool_call_summary(final_tool_calls)
                if self._tool_calls_match_tail(state.tool_calls, final_tool_calls):
                    state.final_response = (
                        "I reached the turn iteration limit after executing "
                        f"{summary}, before I could write the final answer."
                    )
                else:
                    state.final_response = (
                        f"I reached the turn iteration limit before I could execute {summary}."
                    )
            else:
                state.final_response = response

        # Post-synthesis refinement: apply ThinkingStrategy when configured.
        # Runs after the tool loop so the reasoner can refine the gathered evidence
        # into a polished final answer.  Skipped when strict_iteration_budget is set
        # and the loop already hit max_iterations.
        _tools_were_used = bool(state.tool_calls)
        _thinking_after_tools = getattr(self.config, "thinking_apply_after_tools", True)
        if (
            self.thinking is not None
            and state.final_response
            and (not _tools_were_used or _thinking_after_tools)
            and not (
                getattr(self.config, "strict_iteration_budget", False)
                and state.iteration >= getattr(self.config, "max_iterations", 99)
            )
        ):
            try:
                refined = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.thinking.run(  # type: ignore[union-attr]
                        last_content, initial=state.final_response
                    ),
                )
                if refined and refined.strip():
                    state.final_response = refined
            except Exception:
                pass  # Thinking failure is non-fatal; keep original answer

        return TurnResult(state=state, messages=convo)

    async def _fast_path(
        self,
        messages: list[Message],
        state: TurnState,
        *,
        token_callback: Callable[[str], None] | None = None,
        thinking_callback: Callable[[str], None] | None = None,
        session_id: str | None = None,
    ) -> TurnResult:
        """Single LLM call for social/informational turns — no tools."""
        system = self.prompt_builder.build(state, self.config)
        llm_messages = self._with_system(messages, system)
        stream_enabled = token_callback is not None and bool(getattr(self.config, "stream", False))
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._llm_generate(
                llm_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=stream_enabled,
                on_token=token_callback if stream_enabled else None,
                on_reasoning_token=thinking_callback,
            ),
        )
        # Extract <think> blocks, log them, and route to thinking_callback
        think_content, response = _extract_think_blocks(response)
        if think_content:
            state.thinking_log.append(think_content)
            if thinking_callback is not None:
                thinking_callback(think_content)
        state.final_response = response
        self._save_message(
            Message(
                role=MessageRole.ASSISTANT,
                content=response,
                thinking_log=list(state.thinking_log or []),
            ),
            session_id,
        )
        return TurnResult(state=state, messages=list(messages))

    def _with_system(self, messages: list[Message], system: str) -> list[Message]:
        """Return messages with system prompt as first message."""
        sys_msg = Message(role=MessageRole.SYSTEM, content=system)
        if messages and messages[0].role == MessageRole.SYSTEM:
            return [sys_msg] + messages[1:]
        return [sys_msg] + list(messages)
