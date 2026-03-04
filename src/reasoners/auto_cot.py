from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasoningTrace


class AutoCoTReasoner(Reasoner):
    """
    Automatic Chain-of-Thought reasoner.

    Generates domain-relevant exemplars first, then uses them as in-context
    examples to guide the answer to the real query.  Only fires on pure Q&A
    turns (no tool calls made), so exemplars should target analytical/explanatory
    responses, not tool-dispatch decisions.
    """

    _EXEMPLAR_PROMPT = """\
You are an AI agent specializing in time series forecasting, data analysis, \
and software development.

Generate exactly 3 diverse example questions and answers that demonstrate \
clear, structured reasoning.  Cover different areas: one about time series / \
forecasting concepts, one about code / software architecture, one about data \
processing or statistics.

Format each example as:

Q: <concise technical question>
REASONING: <2-4 sentences showing step-by-step analysis>
Final answer: <direct, well-structured answer>

---

Return ONLY the 3 examples, no preamble.
"""

    def solve(
        self, query: str, initial_solution: Optional[str] = None
    ) -> ReasoningTrace:
        exemplars = self._chat(
            [{"role": "user", "content": self._EXEMPLAR_PROMPT}],
            temperature=0.6,
            max_tokens=self.config.get("max_tokens", 1024),
        )

        solve_prompt = (
            exemplars.strip()
            + f"\n\n---\n\nNow answer the following using the same structured reasoning:\n\n"
            f"Q: {query}\nREASONING:"
        )

        out = self._chat(
            [{"role": "user", "content": solve_prompt}],
            temperature=self.config.get("temperature", 0.3),
        )

        # Prepend "REASONING:" that we injected as part of the prompt
        full = "REASONING:" + out
        reasoning, answer = self._split(full)
        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "auto_cot", "exemplar_preview": exemplars[:200]},
        )


__all__ = ["AutoCoTReasoner"]
