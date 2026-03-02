"""Prompt templates, response parsing and prompt-builder functions for EoH."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import EoHConfig, Individual

# ---------------------------------------------------------------------------
# Marker specification (injected into every LLM prompt)
# ---------------------------------------------------------------------------

_MARKER_SPEC = """
You MUST output EXACTLY in this format:

[THOUGHT]
<concise heuristic policy description; mention which hooks you implement and why>

[CODE]
<pure python code; NO markdown fences; NO extra text outside markers>

Your code must define:
  def build_hooks() -> dict:
      # returns a dict mapping hook_name -> callable

Allowed hook names:
  - init_solution(problem, instance, rng, ctx=None) -> solution
  - propose_move(problem, instance, cur_solution, rng, ctx=None) -> move
  - apply_move(problem, instance, cur_solution, move, rng, ctx=None) -> new_solution
  - accept_move(problem, instance, cur_solution, cur_obj, cand_solution, cand_obj, temperature, rng, ctx=None) -> bool
  - update_params(problem, instance, it, cur_solution, cur_obj, best_solution, best_obj, temperature, rng, ctx=None) -> dict|None
  - restart(problem, instance, it, cur_solution, cur_obj, best_solution, best_obj, rng, ctx=None) -> bool

GA note:
- In GA, propose_move() may return a dict like:
    {"op":"mutate", ...}
    {"op":"crossover", "mate": <solution>, ...}
  If your apply_move() understands it, you can implement real crossover.
- If you do nothing GA-specific, the engine falls back to mutation moves using problem.move_neighborhood/apply_move.

Notes:
- ctx is optional; if you include it, use it only as a small dict for state.
- No imports (except math, random).
- No file/network/OS operations; no subprocess; no eval/exec; no dynamic imports.
- Keep code short, defensive, deterministic given rng.
- If you don't implement a hook, omit it from the dict (engine will fallback).
""".strip()

# ---------------------------------------------------------------------------
# System prompt (set once on the agent before an EoH run)
# ---------------------------------------------------------------------------

EOH_SYSTEM_PROMPT = (
    "You design compact, correct, scale-aware optimization heuristics as"
    " hook-based policies.\n"
    "Always follow the required output markers format exactly."
)

# For backward-compat
_EOH_SYSTEM_PROMPT = EOH_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_THOUGHT_CODE_RE = re.compile(
    r"\[THOUGHT\]\s*(.*?)\s*\[CODE\]\s*(.*)\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedHeuristic:
    thought: str
    code: str


def parse_thought_code(text: str) -> Optional[ParsedHeuristic]:
    """Parse the [THOUGHT] / [CODE] marker format from *text*."""
    m = _THOUGHT_CODE_RE.search(text or "")
    if not m:
        return None
    thought = (m.group(1) or "").strip()
    code = (m.group(2) or "").strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return ParsedHeuristic(thought=thought, code=code)


def parse_code_only_fallback(text: str) -> Optional[ParsedHeuristic]:
    """Fallback: accept a response that contains *build_hooks* without markers."""
    if not text or "def build_hooks(" not in text:
        return None
    code = text.replace("```python", "").replace("```", "").strip()
    return ParsedHeuristic(thought="Code-only policy (no thought provided).", code=code)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + f"\n[... truncated {len(s) - n} chars ...]"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def seed_prompt(task_desc: str, api_desc: str, baseline: str | None = None) -> str:
    """Initial-population prompt: ask the LLM to generate a seed heuristic."""
    baseline_section = f"[BASELINE]\n{baseline}" if baseline else ""
    return f"""
We are evolving heuristics for a generic optimization setup.

[TASK]
{task_desc}

[APIs]
{api_desc}

{baseline_section}

Create an INITIAL heuristic policy (hooks dict) that improves performance and robustness.

Emphasize:
- Generic behavior (no domain-specific assumptions about solution type).
- Scale-awareness (objective deltas can be huge/small).
- Feasibility: prioritize feasible solutions; use repair if available.
- Works under GA (default) and SA (optional).

{_MARKER_SPEC}
""".strip()


def e1_mutation_prompt(
    task_desc: str, api_desc: str, parent: Individual, cfg: EoHConfig
) -> str:
    """E1 mutation: ask the LLM to make 1-3 targeted improvements to *parent*."""
    p_th = _clip(parent.thought, cfg.max_parent_thought_in_prompt)
    p_cd = _clip(parent.code, cfg.max_parent_code_in_prompt)
    return f"""
We are evolving heuristics (hook policies).

[TASK]
{task_desc}

[APIs]
{api_desc}

[PARENT THOUGHT]
{p_th}

[PARENT CODE]
{p_cd}

Perform a MUTATION:
- Keep build_hooks() and hook signatures.
- Make 1–3 targeted improvements (move generation, acceptance, feasibility, adapt params, restarts).
- Keep it stable and fast; avoid complicated state.
- Update THOUGHT to match CODE.

{_MARKER_SPEC}
""".strip()


def e1_crossover_prompt(
    task_desc: str,
    api_desc: str,
    p1: Individual,
    p2: Individual,
    cfg: EoHConfig,
) -> str:
    """E1 crossover: combine the strongest ideas from two parents."""
    p1_th = _clip(p1.thought, cfg.max_parent_thought_in_prompt)
    p1_cd = _clip(p1.code, cfg.max_parent_code_in_prompt)
    p2_th = _clip(p2.thought, cfg.max_parent_thought_in_prompt)
    p2_cd = _clip(p2.code, cfg.max_parent_code_in_prompt)
    return f"""
We are evolving heuristics via CROSSOVER (policy-level crossover, not GA crossover).

[TASK]
{task_desc}

[APIs]
{api_desc}

[PARENT 1 THOUGHT]
{p1_th}

[PARENT 1 CODE]
{p1_cd}

[PARENT 2 THOUGHT]
{p2_th}

[PARENT 2 CODE]
{p2_cd}

Create a CHILD heuristic policy that combines the strongest ideas coherently.
- Prefer simplicity and correctness.
- Must remain generic (no domain assumptions about solution type).
- Update THOUGHT to match CODE.

{_MARKER_SPEC}
""".strip()


def e2_improve_prompt(
    task_desc: str,
    api_desc: str,
    exemplars: list[Individual],
    cfg: EoHConfig,
) -> str:
    """E2 exemplar-based improvement: propose a better heuristic from top examples."""
    ex_text = "\n\n".join(
        f"[EXEMPLAR id={e.id} fitness={e.fitness}]\n[THOUGHT]\n"
        f"{_clip(e.thought, cfg.max_exemplar_thought_in_prompt)}\n\n"
        f"[CODE]\n{_clip(e.code, cfg.max_exemplar_code_in_prompt)}"
        for e in exemplars
    )
    return f"""
We are evolving heuristics via EXEMPLAR-BASED IMPROVEMENT (E2).

[TASK]
{task_desc}

[APIs]
{api_desc}

Here are strong policies. Propose an improved one that is more robust and higher fitness.

{ex_text}

{_MARKER_SPEC}
""".strip()


def reformat_prompt(prev_text: str) -> str:
    """Ask the LLM to reformat a malformed response into the marker format."""
    return f"""
Reformat into EXACTLY the required markers format.
Do NOT change the algorithm unless required to satisfy build_hooks() and signatures.

CONTENT:
{prev_text}

{_MARKER_SPEC}
""".strip()


# ---------------------------------------------------------------------------
# Backward-compat aliases (private names used by the old flat module)
# ---------------------------------------------------------------------------

_seed_prompt = seed_prompt
_e1_mutation_prompt = e1_mutation_prompt
_e1_crossover_prompt = e1_crossover_prompt
_e2_improve_prompt = e2_improve_prompt
_reformat_prompt = reformat_prompt
