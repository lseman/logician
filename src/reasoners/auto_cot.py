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
        ).strip()

        solve_prompt = (
            exemplars.strip()
            + f"\n\n---\n\nNow answer the following using the same structured reasoning:\n\n"
            f"Q: {query}\n"
            "REASONING: <explain briefly>\n"
            "Final answer: <your final answer>"
        )

        out = self._chat(
            [{"role": "user", "content": solve_prompt}],
            temperature=self.config.get("temperature", 0.3),
        ).strip()

        if not out:
            fallback = (initial_solution or "").strip()
            return ReasoningTrace(
                reasoning="",
                answer=fallback,
                metadata={
                    "method": "auto_cot",
                    "degraded": True,
                    "reason": "empty_generation",
                    "exemplar_preview": exemplars[:200],
                },
            )

        full = out if out.lstrip().lower().startswith("reasoning:") else f"REASONING: {out}"
        reasoning, answer = self._split(full)

        answer = (answer or "").strip()
        bad_answer_tokens = {"reasoning:", "reasoning", "final answer:", "final answer"}
        if not answer or answer.lower() in bad_answer_tokens:
            extracted = (self._extract_answer(out) or "").strip()
            if extracted.lower() not in bad_answer_tokens:
                answer = extracted
            else:
                answer = (initial_solution or out).strip()

        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "auto_cot", "exemplar_preview": exemplars[:200]},
        )


__all__ = ["AutoCoTReasoner"]
