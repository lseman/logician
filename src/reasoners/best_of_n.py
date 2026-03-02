from __future__ import annotations

import re
from typing import Optional

from .base import Reasoner, ReasoningTrace


class BestOfNReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        n = self.config.get("n", 8)
        temp = self.config.get("temperature", 0.8)

        scored: list[tuple[str, float]] = []
        for _ in range(n):
            sample = self._chat(
                [
                    {
                        "role": "user",
                        "content": query
                        + "\nThink step by step. End with 'Final answer: ...'.",
                    }
                ],
                temperature=temp,
            )
            scored.append((sample, self._score(query, sample)))

        best_resp, best_score = max(scored, key=lambda item: item[1])
        reasoning, answer = self._split(best_resp)

        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "best_of_n", "score": best_score},
        )

    def _score(self, query: str, reasoning: str) -> float:
        prompt = f"""
[Problem]
{query}

[Candidate]
{reasoning}

Rate quality (0-1). Output only the number.
"""
        raw = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        match = re.search(r"[0-1](?:\.\d+)?", raw.strip())
        return float(match.group(0)) if match else 0.0


__all__ = ["BestOfNReasoner"]
