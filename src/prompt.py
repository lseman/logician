# agent_core/prompt.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .messages import Message, MessageRole

logger = logging.getLogger("prompt")


# ==============================================================================
# Base Prompt
# ==============================================================================

class Prompt(ABC):
    """
    Abstract base class for prompting strategies.
    Prompts are *single-shot* LLM calls with a template.
    """

    def __init__(self, llm_backend: Any, **config):
        self.llm = llm_backend
        self.config = config

    @staticmethod
    def _to_message(item: Dict[str, str]) -> Message:
        role_raw = str(item.get("role", "user")).lower()
        try:
            role = MessageRole(role_raw)
        except ValueError:
            role = MessageRole.USER
        return Message(role=role, content=str(item.get("content", "")))

    def _chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
        normalized = [self._to_message(m) for m in messages]
        return self.llm.generate(
            normalized,
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()

    @abstractmethod
    def run(self, query: str, initial: str | None = None) -> str:
        raise NotImplementedError


# ==============================================================================
# Prompt Implementations
# ==============================================================================

class CoTPrompt(Prompt):
    """Classical chain-of-thought prompting."""
    def run(self, query: str, initial: str | None = None) -> str:
        msg = [{
            "role": "user",
            "content": query + "\n\nThink step by step. End with 'Final answer: ...'."
        }]
        return self._chat(msg)


class ThinkShortPrompt(Prompt):
    """Minimal chain-of-thought, shortest valid reasoning."""
    def run(self, query: str, initial: str | None = None) -> str:
        msg = [{
            "role": "user",
            "content": (
                query +
                "\n\nProvide the SHORTEST logically complete chain-of-thought possible.\n"
                "End with 'Final answer: ...'."
            )
        }]
        return self._chat(msg, temperature=self.config.get("temperature", 0.4))


# ==============================================================================
# Registry
# ==============================================================================

PROMPT_REGISTRY: Dict[str, type[Prompt]] = {
    "cot": CoTPrompt,
    "think_short": ThinkShortPrompt,
}

def get_prompt(name: str, llm_backend: Any, **config) -> Prompt:
    cls = PROMPT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown prompt '{name}'. Registered: {list(PROMPT_REGISTRY.keys())}")
    return cls(llm_backend, **config)
