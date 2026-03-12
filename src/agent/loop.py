"""AgentLoop: thin ReAct-style main loop for the refactored agent."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

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
    ) -> None:
        self.llm = llm
        self.guardrails = guardrails
        self.prompt_builder = prompt_builder
        self.dispatcher = dispatcher
        self.config = config
        self.use_toon = use_toon

    async def run(self, messages: list[Message]) -> TurnResult:
        """Run one agent turn. Returns when the LLM produces a final response."""
        classification = classify_turn(messages[-1].content)
        state = TurnState(
            turn_id=str(uuid.uuid4()),
            classified_as=classification.intent,
            domain_groups_activated=classification.domain_groups,
        )

        # Fast path for social/informational turns
        if classification.intent in ("social", "informational"):
            return await self._fast_path(messages, state)

        # Build working copy of messages (don't mutate the caller's list)
        convo = list(messages)

        response = ""
        while state.iteration < self.config.max_iterations:
            # 1. Build system prompt
            system = self.prompt_builder.build(state, self.config)

            # 2. Inject system prompt as first message (or replace existing)
            llm_messages = self._with_system(convo, system)

            # 3. LLM call (sync backend wrapped in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    llm_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
            )

            # 4. Parse tool calls from response
            tool_calls: list[ToolCall] = parse_tool_calls(response, use_toon=self.use_toon)

            # 5. Reset consecutive count when LLM produces no tool calls
            if not tool_calls:
                state.consecutive_tool_count = 0

            # 6. Run guardrails (before executing tools)
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

            # 7. No tool calls → final response
            if not tool_calls:
                state.final_response = response
                break

            # 8. Execute tools (parallel reads, serial writes)
            convo.append(Message(role=MessageRole.ASSISTANT, content=response))
            results = await self.dispatcher.dispatch(tool_calls, state)
            convo.extend(format_tool_results(results))
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
