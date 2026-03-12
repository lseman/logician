"""AgentLoop: thin ReAct-style main loop for the refactored agent."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable

from ..messages import Message, MessageRole
from ..config import Config
from ..tools.runtime import ToolCall
from ..tools.parser import parse_tool_calls
from .state import TurnState
from .types import TurnResult
from .classify import classify_turn
from .guardrails import GuardrailEngine
from .prompt import PromptBuilder
from .dispatcher import ToolDispatcher, DispatchResult
from ..backends.base import LLMBackend


def format_tool_results(results: list[DispatchResult]) -> list[Message]:
    """Convert dispatch results to tool-role messages for the conversation."""
    messages = []
    for r in results:
        content = r.error if r.error else r.output
        messages.append(Message(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=r.call_id,
        ))
    return messages


class AgentLoop:
    def __init__(
        self,
        llm: LLMBackend,
        guardrails: GuardrailEngine,
        prompt_builder: PromptBuilder,
        dispatcher: ToolDispatcher,
        config: Config,
        use_toon: bool = False,
        memory: Any | None = None,
        mcp_loader: Callable[[], None] | None = None,
    ) -> None:
        self.llm = llm
        self.guardrails = guardrails
        self.prompt_builder = prompt_builder
        self.dispatcher = dispatcher
        self.config = config
        self.use_toon = use_toon
        self.memory = memory
        self._mcp_loader = mcp_loader

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _load_history(self, session_id: str | None) -> list[Message]:
        """Load conversation history for a session from memory."""
        if not self.memory or not session_id:
            return []
        try:
            use_semantic = getattr(self.config, "default_use_semantic_retrieval", False)
            retrieval_mode = getattr(self.config, "default_retrieval_mode", "hybrid")
            result = self.memory.load_history(
                session_id,
                message="",
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

        Strategy: always keep system message (index 0) and recent tail.
        Drop middle messages when over budget.
        """
        budget = getattr(self.config, "context_token_budget", 0)
        if not budget:
            return messages

        limit = getattr(self.config, "history_limit", 18)

        if len(messages) <= limit:
            return messages

        system = [m for m in messages[:1] if m.role == MessageRole.SYSTEM]
        rest = messages[1:]
        kept = rest[-limit:] if len(rest) > limit else rest
        return system + kept

    # ------------------------------------------------------------------
    # Pre-turn thinking
    # ------------------------------------------------------------------

    async def _pre_turn_think(self, convo: list[Message], session_id: str | None) -> str:
        """Generate a brief plan before the main tool loop."""
        prompt = getattr(
            self.config,
            "pre_turn_thinking_prompt",
            "Before acting, briefly plan: what is the task, which tools will you use, what are the steps?",
        )
        planning_convo = list(convo) + [Message(role=MessageRole.USER, content=prompt)]
        loop = asyncio.get_event_loop()
        try:
            plan = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    planning_convo,
                    temperature=self.config.temperature,
                    max_tokens=getattr(self.config, "pre_turn_thinking_max_tokens", 512),
                ),
            )
            return plan.strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[Message],
        session_id: str | None = None,
    ) -> TurnResult:
        """Run one agent turn. Returns when the LLM produces a final response."""
        # Ensure MCP servers are loaded (idempotent)
        if self._mcp_loader is not None and callable(self._mcp_loader):
            self._mcp_loader()

        classification = classify_turn(messages[-1].content)
        state = TurnState(
            turn_id=str(uuid.uuid4()),
            classified_as=classification.intent,
            domain_groups_activated=classification.domain_groups,
        )

        # Fast path for social/informational turns
        if classification.intent in ("social", "informational"):
            return await self._fast_path(messages, state)

        # Load conversation history and prepend to working copy
        history = self._load_history(session_id)
        convo = history + list(messages)

        # Pre-turn thinking: inject a planning hint before the main loop
        if (
            getattr(self.config, "pre_turn_thinking", False)
            and classification.intent == "execution"
        ):
            plan = await self._pre_turn_think(convo, session_id)
            if plan:
                hint = Message(
                    role=MessageRole.SYSTEM,
                    content=f"[Pre-turn plan]\n{plan}",
                )
                convo.append(hint)

        response = ""
        while state.iteration < self.config.max_iterations:
            # 1. Build system prompt
            system = self.prompt_builder.build(state, self.config)

            # 2. Inject system prompt as first message (or replace existing)
            llm_messages = self._with_system(convo, system)

            # 3. Apply context budget trimming
            llm_messages = self._trim_to_budget(llm_messages)

            # 4. LLM call (sync backend wrapped in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    llm_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
            )

            # 5. Parse tool calls from response
            tool_calls: list[ToolCall] = parse_tool_calls(response, use_toon=self.use_toon)

            # 6. Reset consecutive count when LLM produces no tool calls
            if not tool_calls:
                state.consecutive_tool_count = 0

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

            # 8. No tool calls → final response
            if not tool_calls:
                state.final_response = response
                # Persist the final assistant message
                self._save_message(
                    Message(role=MessageRole.ASSISTANT, content=response),
                    session_id,
                )
                break

            # 9. Execute tools (parallel reads, serial writes)
            assistant_msg = Message(role=MessageRole.ASSISTANT, content=response)
            convo.append(assistant_msg)
            self._save_message(assistant_msg, session_id)

            results = await self.dispatcher.dispatch(tool_calls, state)
            tool_msgs = format_tool_results(results)
            convo.extend(tool_msgs)
            for tm in tool_msgs:
                self._save_message(tm, session_id)

            state.iteration += 1

        # If we exhausted iterations without a final response
        if state.final_response is None:
            state.final_response = response

        return TurnResult(state=state, messages=convo)

    async def _fast_path(self, messages: list[Message], state: TurnState) -> TurnResult:
        """Single LLM call for social/informational turns — no tools."""
        system = self.prompt_builder.build(state, self.config)
        llm_messages = self._with_system(messages, system)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm.generate(
                llm_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
        )
        state.final_response = response
        return TurnResult(state=state, messages=list(messages))

    def _with_system(self, messages: list[Message], system: str) -> list[Message]:
        """Return messages with system prompt as first message."""
        sys_msg = Message(role=MessageRole.SYSTEM, content=system)
        if messages and messages[0].role == MessageRole.SYSTEM:
            return [sys_msg] + messages[1:]
        return [sys_msg] + list(messages)
