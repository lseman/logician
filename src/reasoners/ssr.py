from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .base import Reasoner, ReasoningTrace


@dataclass
class SocraticStep:
    index: int
    question: str
    answer: str
    confidence: float | None = None
    samples: List[str] | None = None


class SSRReasoner(Reasoner):
    """
    Socratic Self-Refinement:
    - initial solution
    - decompose into steps
    - verify each step via sampling
    - refine weakest step
    """

    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        max_iters = self.config.get("max_iterations", 3)
        m_samples = self.config.get("m_samples", 8)
        mode = self.config.get("mode", "plan")

        reasoning, answer = self._initial(query, initial_solution)

        if mode == "plan":
            reasoning, answer = self._refine_plan(query, reasoning, answer)

        history = []
        for it in range(max_iters):
            steps = self._decompose(query, reasoning, answer)
            steps = self._verify(query, steps, m_samples)
            updated = self._refine(query, reasoning, answer, steps)
            if updated is None:
                break
            reasoning, answer = updated
            history.append(
                {
                    "iteration": it + 1,
                    "steps": [{"index": step.index, "conf": step.confidence} for step in steps],
                }
            )

        return ReasoningTrace(reasoning, answer, {"method": "ssr", "history": history})

    def _initial(self, query: str, init: Optional[str]) -> Tuple[str, str]:
        if init:
            return self._split(init)
        resp = self._chat(
            [
                {
                    "role": "system",
                    "content": "Think step by step like a careful mathematician.",
                },
                {
                    "role": "user",
                    "content": query + "\n\nEnd with 'Final answer: ...'.",
                },
            ]
        )
        return self._split(resp)

    def _refine_plan(self, query: str, reasoning: str, answer: str) -> Tuple[str, str]:
        prompt = f"""
[Problem]
{query}

[Draft]
{reasoning}
Final answer: {answer}

1. Summarize a plan.
2. Improve the plan.
3. Rewrite cleanly.

End with 'Final answer: ...'.
"""
        resp = self._chat([{"role": "user", "content": prompt}])
        return self._split(resp)

    def _decompose(self, query: str, reasoning: str, answer: str) -> List[SocraticStep]:
        prompt = f"""
[Problem]
{query}

[Solution]
{reasoning}
Final answer: {answer}

Break this into <= 8 steps. Return JSON array only.
"""
        raw = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        try:
            data = json.loads(raw.strip("` \n"))
            return [
                SocraticStep(
                    index=int(step.get("index", i + 1)),
                    question=str(step["question"]),
                    answer=str(step["answer"]),
                )
                for i, step in enumerate(data)
            ]
        except Exception:
            return [
                SocraticStep(1, "Whole reasoning", reasoning + f"\nFinal answer: {answer}")
            ]

    def _verify(self, query: str, steps: List[SocraticStep], m: int) -> List[SocraticStep]:
        for idx, step in enumerate(steps):
            prev = "\n".join(f"Step {s.index}: {s.answer}" for s in steps[:idx])
            samples: list[str] = []
            for _ in range(m):
                prompt = f"""
[Problem]
{query}

Previous steps:
{prev or "(none)"}

Re-solve the sub-question:
{step.question}

Short answer only.
"""
                samples.append(
                    self._chat(
                        [{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=64,
                    )
                )
            step.samples = samples
            gold = self._norm(step.answer)
            matches = sum(1 for sample in samples if self._norm(sample) == gold)
            step.confidence = matches / m
        return steps

    def _refine(
        self,
        query: str,
        reasoning: str,
        answer: str,
        steps: List[SocraticStep],
    ) -> tuple[str, str] | None:
        bad = min(steps, key=lambda step: step.confidence)
        if bad.confidence is None or bad.confidence > 0.8:
            return None

        counts = Counter(self._norm(sample) for sample in (bad.samples or []))
        best_norm, _ = counts.most_common(1)[0]

        if best_norm == self._norm(bad.answer):
            return None

        best_raw = next(
            (sample for sample in (bad.samples or []) if self._norm(sample) == best_norm),
            bad.answer,
        )

        prompt = f"""
[Problem]
{query}

[Old solution]
{reasoning}
Final answer: {answer}

Bad step:
Q: {bad.question}
Old A: {bad.answer}
Better A: {best_raw}

Rewrite full solution consistently.
End with 'Final answer: ...'.
"""
        out = self._chat([{"role": "user", "content": prompt}], temperature=0.7)
        return self._split(out)

    @staticmethod
    def _norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[ \t\n\r]+", " ", s)
        return s.rstrip(".,;:!?")


__all__ = ["SocraticStep", "SSRReasoner"]
