from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..messages import Message, MessageRole


@dataclass
class ReasoningTrace:
    reasoning: str
    answer: str
    metadata: Dict[str, Any]


class Reasoner(ABC):
    """
    Abstract base class for algorithmic multi-step reasoners.
    Prompts handle single-shot reasoning.
    """

    def __init__(self, llm_backend: Any, **config: Any):
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

    def _chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 2048))
        normalized = [self._to_message(m) for m in messages]
        return self.llm.generate(
            normalized,
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()

    @staticmethod
    def _extract_answer(text: str) -> str:
        if "Final answer:" in text:
            return text.split("Final answer:", 1)[-1].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else text.strip()

    def _split(self, text: str) -> Tuple[str, str]:
        if "Final answer:" in text:
            reasoning, answer = text.rsplit("Final answer:", 1)
            return reasoning.strip(), answer.strip()
        return text, self._extract_answer(text)

    @abstractmethod
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        raise NotImplementedError


__all__ = ["Reasoner", "ReasoningTrace"]
