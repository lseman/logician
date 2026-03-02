from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasoningTrace


class ReflexionReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        max_trials = self.config.get("max_trials", 3)

        attempt = initial_solution or ""
        reflections: list[str] = []

        for _ in range(max_trials):
            if not attempt:
                attempt = self._chat(
                    [
                        {
                            "role": "user",
                            "content": query
                            + "\nThink step by step. End with 'Final answer: ...'.",
                        }
                    ]
                )

            critique = self._chat(
                [
                    {
                        "role": "user",
                        "content": f"""
[Problem]
{query}

[Attempt]
{attempt}

Critique weaknesses or errors.
""",
                    }
                ]
            )
            reflections.append(critique)

            attempt = self._chat(
                [
                    {
                        "role": "user",
                        "content": f"""
[Problem]
{query}

Reflections:
{chr(10).join(reflections)}

Rewrite solution based on reflections.
End with 'Final answer: ...'.
""",
                    }
                ]
            )

        reasoning, answer = self._split(attempt)
        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "reflexion", "reflections": reflections},
        )


__all__ = ["ReflexionReasoner"]
