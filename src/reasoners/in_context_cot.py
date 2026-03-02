from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasoningTrace


class InContextCoTReasoner(Reasoner):
    """
    Requires user-provided exemplars in config:
        config["exemplars"] = "<Q/A pairs ...>"
    """

    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        exemplars = self.config.get("exemplars")
        if not exemplars:
            raise ValueError("InContextCoT requires config['exemplars'].")

        prompt = exemplars.strip() + f"\n\nQ: {query}\nA: (step by step)\nEnd with 'Final answer: ...'."

        out = self._chat([{"role": "user", "content": prompt}], temperature=0.3)

        reasoning, answer = self._split(out)
        return ReasoningTrace(reasoning, answer, {"method": "in_context_cot"})


__all__ = ["InContextCoTReasoner"]
