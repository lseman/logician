from __future__ import annotations

import heapq
import re
from typing import Optional

from .base import Reasoner, ReasoningTrace


class ToTReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        beam = self.config.get("beam_width", 6)
        max_depth = self.config.get("max_depth", 10)
        branch = self.config.get("branch_factor", 3)

        if initial_solution:
            frontier = [(initial_solution, self._score(query, initial_solution))]
        else:
            frontier = [("", 0.0)]

        best_reasoning = ""
        best_answer = ""
        best_score = float("-inf")
        evaluated = 0

        for _depth in range(max_depth):
            new = []
            for reasoning, score_prev in frontier:
                if "final answer" in reasoning.lower():
                    if score_prev > best_score:
                        best_score = score_prev
                        best_reasoning = reasoning
                        best_answer = self._extract_answer(reasoning)
                    continue

                prompt = f"""{query}

Reasoning so far:
{reasoning or "(empty)"}

Continue reasoning. If done, end with 'Final answer: ...'.
"""
                for _ in range(branch):
                    cont = self._chat(
                        [{"role": "user", "content": prompt}],
                        temperature=0.9,
                        max_tokens=512,
                    )
                    full = (reasoning + "\n" + cont).strip()
                    score = self._score(query, full)
                    evaluated += 1

                    if "final answer" in full.lower() and score > best_score:
                        best_score = score
                        best_reasoning = full
                        best_answer = self._extract_answer(full)

                    new.append((full, score))

            if not new:
                break

            frontier = heapq.nlargest(beam, new, key=lambda item: item[1])

        if best_score == float("-inf") and frontier:
            best_reasoning, best_score = max(frontier, key=lambda item: item[1])
            best_answer = self._extract_answer(best_reasoning)

        return ReasoningTrace(
            best_reasoning,
            best_answer,
            {"method": "tot", "states": evaluated},
        )

    def _score(self, query: str, reasoning: str) -> float:
        prompt = f"""
[Problem]
{query}

[Partial solution]
{reasoning}

Rate promise (0-1). Output only a number.
"""
        raw = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        match = re.search(r"[0-1](?:\.\d+)?", raw.strip())
        if match:
            try:
                return float(match.group(0))
            except Exception:
                pass
        length = min(len(reasoning) / 1000, 1.0)
        bonus = 0.2 if "final answer" in reasoning.lower() else 0.0
        return length + bonus


__all__ = ["ToTReasoner"]
