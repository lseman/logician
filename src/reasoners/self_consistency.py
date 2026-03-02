from __future__ import annotations

from collections import Counter
from typing import Optional

from .base import Reasoner, ReasoningTrace


class SelfConsistencyReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        n = self.config.get("n_rollouts", 32)
        temp = self.config.get("temperature", 0.8)

        samples = [
            self._chat(
                [
                    {
                        "role": "user",
                        "content": query
                        + "\nThink step by step. End with 'Final answer: ...'.",
                    }
                ],
                temperature=temp,
            )
            for _ in range(n)
        ]

        answers = [self._extract_answer(sample) for sample in samples]
        majority, count = Counter(answers).most_common(1)[0]
        best = next(sample for sample in samples if self._extract_answer(sample) == majority)

        reasoning, answer = self._split(best)
        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "self_consistency", "votes": count, "n": n},
        )


__all__ = ["SelfConsistencyReasoner"]
