from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasoningTrace


class AutoCoTReasoner(Reasoner):
    """
    - Generate diverse exemplars
    - Use them as in-context examples to solve the real query
    """

    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        exemplars_prompt = """
Generate 3 diverse example problems + solutions:

Q: ...
A: <reasoning>
Final answer: <answer>

Return EXACTLY 3 examples.
"""
        exemplars = self._chat(
            [{"role": "user", "content": exemplars_prompt}],
            temperature=0.7,
            max_tokens=1024,
        )

        solve_prompt = (
            exemplars.strip()
            + f"\n\nNow solve:\nQ: {query}\nA:\nEnd with 'Final answer: ...'."
        )

        out = self._chat(
            [{"role": "user", "content": solve_prompt}],
            temperature=0.3,
        )

        reasoning, answer = self._split(out)
        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "auto_cot", "exemplars": exemplars},
        )


__all__ = ["AutoCoTReasoner"]
