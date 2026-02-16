# agent_core/reasoner.py
from __future__ import annotations

import json
import heapq
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from .messages import Message, MessageRole

logger = logging.getLogger("reasoner")


# ==============================================================================
# Base: ReasoningTrace
# ==============================================================================

@dataclass
class ReasoningTrace:
    reasoning: str
    answer: str
    metadata: Dict[str, Any]


# ==============================================================================
# Base: Reasoner
# ==============================================================================

class Reasoner(ABC):
    """
    Abstract base class for *algorithmic* multi-step reasoners.
    Prompts (in prompt.py) handle single-shot reasoning.
    """

    def __init__(self, llm_backend: Any, **config):
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

    def _chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
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
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else text.strip()

    def _split(self, text: str) -> Tuple[str, str]:
        if "Final answer:" in text:
            r, a = text.rsplit("Final answer:", 1)
            return r.strip(), a.strip()
        return text, self._extract_answer(text)

    @abstractmethod
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        raise NotImplementedError


# ==============================================================================
# Socratic Self-Refinement (SSR)
# ==============================================================================

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

            history.append({
                "iteration": it + 1,
                "steps": [{"index": s.index, "conf": s.confidence} for s in steps]
            })

        return ReasoningTrace(reasoning, answer, {"method": "ssr", "history": history})

    # ---- helpers ----

    def _initial(self, query: str, init: Optional[str]) -> Tuple[str, str]:
        if init:
            return self._split(init)
        resp = self._chat([
            {"role": "system", "content": "Think step by step like a careful mathematician."},
            {"role": "user", "content": query + "\n\nEnd with 'Final answer: ...'."}
        ])
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
        raw = self._chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_tokens=512)
        try:
            data = json.loads(raw.strip("` \n"))
            return [
                SocraticStep(
                    index=int(s.get("index", i + 1)),
                    question=str(s["question"]),
                    answer=str(s["answer"]),
                )
                for i, s in enumerate(data)
            ]
        except Exception:
            return [
                SocraticStep(1, "Whole reasoning", reasoning + f"\nFinal answer: {answer}")
            ]

    def _verify(self, query: str, steps: List[SocraticStep], m: int):
        for idx, step in enumerate(steps):
            prev = "\n".join(f"Step {s.index}: {s.answer}" for s in steps[:idx])
            samples = []
            for _ in range(m):
                p = f"""
[Problem]
{query}

Previous steps:
{prev or "(none)"}

Re-solve the sub-question:
{step.question}

Short answer only.
"""
                samples.append(self._chat(
                    [{"role": "user", "content": p}],
                    temperature=0.8,
                    max_tokens=64
                ))
            step.samples = samples
            gold = self._norm(step.answer)
            matches = sum(1 for s in samples if self._norm(s) == gold)
            step.confidence = matches / m
        return steps

    def _refine(self, query: str, reasoning: str, answer: str, steps: List[SocraticStep]):
        bad = min(steps, key=lambda s: s.confidence)
        if bad.confidence is None or bad.confidence > 0.8:
            return None

        counts = Counter(self._norm(s) for s in bad.samples)
        best_norm, _ = counts.most_common(1)[0]

        if best_norm == self._norm(bad.answer):
            return None

        best_raw = next((s for s in bad.samples if self._norm(s) == best_norm),
                        bad.answer)

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


# ==============================================================================
# Tree-of-Thoughts (ToT)
# ==============================================================================

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

        for depth in range(max_depth):
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
                        max_tokens=512
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

            frontier = heapq.nlargest(beam, new, key=lambda x: x[1])

        if best_score == float("-inf") and frontier:
            best_reasoning, best_score = max(frontier, key=lambda x: x[1])
            best_answer = self._extract_answer(best_reasoning)

        return ReasoningTrace(best_reasoning, best_answer,
                              {"method": "tot", "states": evaluated})

    def _score(self, query: str, reasoning: str) -> float:
        p = f"""
[Problem]
{query}

[Partial solution]
{reasoning}

Rate promise (0-1). Output only a number.
"""
        raw = self._chat(
            [{"role": "user", "content": p}],
            temperature=0.0, max_tokens=16
        )
        m = re.search(r"[0-1](?:\.\d+)?", raw.strip())
        if m:
            try:
                return float(m.group(0))
            except:
                pass
        # fallback heuristic
        length = min(len(reasoning) / 1000, 1.0)
        bonus = 0.2 if "final answer" in reasoning.lower() else 0.0
        return length + bonus


# ==============================================================================
# Reflexion Reasoner
# ==============================================================================

class ReflexionReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        max_trials = self.config.get("max_trials", 3)

        attempt = initial_solution or ""
        reflections = []

        for _ in range(max_trials):
            if not attempt:
                attempt = self._chat([{
                    "role": "user",
                    "content": query +
                               "\nThink step by step. End with 'Final answer: ...'."
                }])

            critique = self._chat([{
                "role": "user",
                "content": f"""
[Problem]
{query}

[Attempt]
{attempt}

Critique weaknesses or errors.
"""
            }])
            reflections.append(critique)

            attempt = self._chat([{
                "role": "user",
                "content": f"""
[Problem]
{query}

Reflections:
{chr(10).join(reflections)}

Rewrite solution based on reflections.
End with 'Final answer: ...'.
"""
            }])

        reasoning, answer = self._split(attempt)
        return ReasoningTrace(reasoning, answer,
                              {"method": "reflexion", "reflections": reflections})


# ==============================================================================
# Self-Consistency Reasoner
# ==============================================================================

class SelfConsistencyReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        n = self.config.get("n_rollouts", 32)
        temp = self.config.get("temperature", 0.8)

        samples = [
            self._chat(
                [{"role": "user",
                  "content": query +
                             "\nThink step by step. End with 'Final answer: ...'."}],
                temperature=temp
            )
            for _ in range(n)
        ]

        answers = [self._extract_answer(s) for s in samples]
        maj, count = Counter(answers).most_common(1)[0]
        best = next(s for s in samples if self._extract_answer(s) == maj)

        reasoning, answer = self._split(best)
        return ReasoningTrace(
            reasoning,
            answer,
            {"method": "self_consistency", "votes": count, "n": n}
        )


# ==============================================================================
# Best-of-N
# ==============================================================================

class BestOfNReasoner(Reasoner):
    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        n = self.config.get("n", 8)
        temp = self.config.get("temperature", 0.8)

        scored = []
        for _ in range(n):
            s = self._chat(
                [{"role": "user",
                  "content": query +
                             "\nThink step by step. End with 'Final answer: ...'."}],
                temperature=temp
            )
            scored.append((s, self._score(query, s)))

        best_resp, best_score = max(scored, key=lambda x: x[1])
        reasoning, answer = self._split(best_resp)

        return ReasoningTrace(reasoning, answer,
                              {"method": "best_of_n", "score": best_score})

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
            temperature=0.0, max_tokens=16
        )
        m = re.search(r"[0-1](?:\.\d+)?", raw.strip())
        return float(m.group(0)) if m else 0.0


# ==============================================================================
# Auto-CoT (example generation + solve)
# ==============================================================================

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
            max_tokens=1024
        )

        solve_prompt = (
            exemplars.strip()
            + f"\n\nNow solve:\nQ: {query}\nA:\nEnd with 'Final answer: ...'."
        )

        out = self._chat(
            [{"role": "user", "content": solve_prompt}],
            temperature=0.3
        )

        reasoning, answer = self._split(out)
        return ReasoningTrace(reasoning, answer,
                              {"method": "auto_cot", "exemplars": exemplars})


# ==============================================================================
# In-Context-CoT
# ==============================================================================

class InContextCoTReasoner(Reasoner):
    """
    Requires user-provided exemplars in config:
        config["exemplars"] = "<Q/A pairs ...>"
    """

    def solve(self, query: str, initial_solution: Optional[str] = None) -> ReasoningTrace:
        exemplars = self.config.get("exemplars")
        if not exemplars:
            raise ValueError("InContextCoT requires config['exemplars'].")

        prompt = (
            exemplars.strip()
            + f"\n\nQ: {query}\nA: (step by step)\nEnd with 'Final answer: ...'."
        )

        out = self._chat([{"role": "user", "content": prompt}], temperature=0.3)

        reasoning, answer = self._split(out)
        return ReasoningTrace(reasoning, answer,
                              {"method": "in_context_cot"})


# ==============================================================================
# Registry
# ==============================================================================

REASONER_REGISTRY: Dict[str, type[Reasoner]] = {
    "ssr": SSRReasoner,
    "tot": ToTReasoner,
    "reflexion": ReflexionReasoner,
    "self_consistency": SelfConsistencyReasoner,
    "sc": SelfConsistencyReasoner,
    "best_of_n": BestOfNReasoner,
    "auto_cot": AutoCoTReasoner,
    "in_context_cot": InContextCoTReasoner,
}

def get_reasoner(name: str, llm_backend: Any, **config) -> Reasoner:
    cls = REASONER_REGISTRY.get(name.lower())
    if not cls:
        raise ValueError(
            f"Unknown reasoner '{name}'. Registered: {list(REASONER_REGISTRY.keys())}"
        )
    return cls(llm_backend, **config)
