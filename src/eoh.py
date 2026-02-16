# eoh_generic.py (v3) — Generic Evolution of Heuristics (EoH) for hook-based policies
# =============================================================================
# Key upgrades vs v2:
# - Default metaheuristic is a Genetic Algorithm (GA), but SA is included.
# - More generic, cleaner separation: ProblemAPI / MetaheuristicAPI / EvalSpec.
# - GA supports optional crossover via propose_move returning {"op":"crossover","mate":...}.
# - Uses your Agent efficiently: sets system_prompt once, uses agent.run overrides.
# - Still safe-ish execution via AST validation + restricted builtins.
# =============================================================================

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

# -----------------------------------------------------------------------------
# Public Types
# -----------------------------------------------------------------------------

Representation = Literal["thought", "code", "thought+code"]
Strategy = Literal["e1", "e2"]

# Standard hooks contract (kept stable)
HOOKS = (
    "init_solution",
    "propose_move",
    "apply_move",
    "accept_move",
    "update_params",
    "restart",
)

# -----------------------------------------------------------------------------
# Problem + Metaheuristic Interfaces (generic)
# -----------------------------------------------------------------------------

@runtime_checkable
class ProblemAPI(Protocol):
    """
    Minimal contract:
      - sense: "min" or "max"
      - random_solution(instance, rng) -> solution
      - objective(solution, instance) -> float
      - is_feasible(solution, instance) -> bool

    Optional (recommended):
      - repair(solution, instance, rng) -> solution
      - instance_features(instance) -> dict
      - move_neighborhood(solution, instance, rng) -> move
      - apply_move(solution, move, instance, rng) -> solution
    """
    sense: str  # "min" or "max"

    def random_solution(self, instance: Any, rng: random.Random) -> Any: ...
    def objective(self, solution: Any, instance: Any) -> float: ...
    def is_feasible(self, solution: Any, instance: Any) -> bool: ...

    def repair(self, solution: Any, instance: Any, rng: random.Random) -> Any: ...
    def instance_features(self, instance: Any) -> dict[str, Any]: ...
    def move_neighborhood(self, solution: Any, instance: Any, rng: random.Random) -> Any: ...
    def apply_move(self, solution: Any, move: Any, instance: Any, rng: random.Random) -> Any: ...


@dataclass
class Budget:
    """
    Budget for a single optimizer run on a single instance.
    Interpreted by metaheuristics. For GA, max_iters = generations.
    """
    max_iters: int = 200
    max_time_s: float = 0.25


@dataclass
class RunTrace:
    best_obj: float
    best_solution: Any
    best_feasible: bool
    iters: int
    elapsed_s: float
    accepts: int
    proposals: int
    failures: int
    extra: dict[str, Any] = field(default_factory=dict)


class MetaheuristicAPI(Protocol):
    name: str

    def run(
        self,
        problem: ProblemAPI,
        instance: Any,
        hooks: "HeuristicHooks",
        budget: Budget,
        rng: random.Random,
    ) -> RunTrace: ...


# -----------------------------------------------------------------------------
# Hook adapters (precomputed, avoids signature inspection at runtime)
# -----------------------------------------------------------------------------

@dataclass
class HookFn:
    fn: Callable[..., Any]
    wants_ctx: bool

    def __call__(self, *args: Any, ctx: dict[str, Any]) -> Any:
        if self.wants_ctx:
            return self.fn(*args, ctx=ctx)
        return self.fn(*args)


def _wrap_hook(fn: Callable[..., Any]) -> HookFn:
    wants_ctx = False
    try:
        sig = inspect.signature(fn)
        wants_ctx = "ctx" in sig.parameters
    except Exception:
        wants_ctx = False
    return HookFn(fn=fn, wants_ctx=wants_ctx)


@dataclass
class HeuristicHooks:
    init_solution: HookFn | None = None
    propose_move: HookFn | None = None
    apply_move: HookFn | None = None
    accept_move: HookFn | None = None
    update_params: HookFn | None = None
    restart: HookFn | None = None

    def implemented(self) -> list[str]:
        out: list[str] = []
        for name in HOOKS:
            if getattr(self, name) is not None:
                out.append(name)
        return out


# -----------------------------------------------------------------------------
# Default metaheuristic: Genetic Algorithm (generic, robust fallbacks)
# -----------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size: int = 32
    elite_frac: float = 0.125
    tournament_k: int = 3
    crossover_rate: float = 0.20
    mutation_rate: float = 0.90
    # If many infeasible, keep pressure on feasibility
    infeas_penalty: float = 1e12
    # Soft cap on per-individual "restart" attempts
    restart_patience: int = 0
    # If objective scale is wild, you can dampen selection pressure by ranking (default)
    rank_selection: bool = True


class GeneticAlgorithm:
    """
    GA semantics with the SAME hook names:
      - init_solution: used to create individuals (else problem.random_solution)
      - propose_move: creates a move for mutation or crossover
          Suggested conventions:
            * {"op":"mutate", ...}          -> apply_move mutates parent
            * {"op":"crossover","mate":sol} -> apply_move uses mate to crossover
          If propose_move missing, GA falls back to:
            * mutate moves from problem.move_neighborhood(parent)
      - apply_move: builds child from (parent, move) (else problem.apply_move or random)
      - accept_move: decides replacement. Called as:
            accept_move(problem, instance,
                        cur_solution, cur_obj,
                        cand_solution, cand_obj,
                        temperature, rng, ctx=None) -> bool
        Here temperature is a small "selection temperature" (not SA), provided as ctx["sel_T"].
        If missing, GA uses default: accept if better by (sense + feasibility).
      - update_params: can update GA internal params via dict (e.g., {"mutation_rate":0.8})
      - restart: can request restarting a weak individual (diversity)
    """
    name = "ga"

    def __init__(self, cfg: GAConfig | None = None) -> None:
        self.cfg = cfg or GAConfig()

    def run(
        self,
        problem: ProblemAPI,
        instance: Any,
        hooks: HeuristicHooks,
        budget: Budget,
        rng: random.Random,
    ) -> RunTrace:
        started = time.perf_counter()
        failures = 0
        accepts = 0
        proposals = 0
        it = 0
        ctx: dict[str, Any] = {}

        sense = (problem.sense or "min").strip().lower()
        if sense not in ("min", "max"):
            sense = "min"

        def better(a: float, b: float) -> bool:
            return (a < b) if sense == "min" else (a > b)

        def quality(obj: float, feas: bool) -> float:
            q = -float(obj) if sense == "min" else float(obj)
            if not feas:
                q = -float(self.cfg.infeas_penalty)
            return q

        def maybe_repair(sol: Any) -> Any:
            if hasattr(problem, "repair"):
                try:
                    return problem.repair(sol, instance, rng)  # type: ignore[misc]
                except Exception:
                    nonlocal failures
                    failures += 1
            return sol

        def make_one() -> Any:
            nonlocal failures
            if hooks.init_solution is not None:
                try:
                    sol = hooks.init_solution(problem, instance, rng, ctx=ctx)
                except Exception:
                    failures += 1
                    sol = problem.random_solution(instance, rng)
            else:
                sol = problem.random_solution(instance, rng)
            sol = maybe_repair(sol)
            return sol

        def eval_one(sol: Any) -> tuple[float, bool]:
            try:
                feas = bool(problem.is_feasible(sol, instance))
            except Exception:
                nonlocal failures
                failures += 1
                feas = False
            try:
                obj = float(problem.objective(sol, instance))
            except Exception:
                failures += 1
                obj = float("inf") if sense == "min" else float("-inf")
            return obj, feas

        def tournament(pop: list[Any], objs: list[float], feas: list[bool]) -> Any:
            k = min(max(2, int(self.cfg.tournament_k)), len(pop))
            idxs = [rng.randrange(len(pop)) for _ in range(k)]
            # Prefer feasible, then better objective
            best_i = idxs[0]
            for i in idxs[1:]:
                if feas[i] and not feas[best_i]:
                    best_i = i
                elif feas[i] == feas[best_i] and better(objs[i], objs[best_i]):
                    best_i = i
            return pop[best_i]

        # --- init population ---
        pop_size = max(2, int(self.cfg.pop_size))
        pop = [make_one() for _ in range(pop_size)]
        objs, feas = zip(*(eval_one(s) for s in pop))
        objs = list(objs)
        feas = list(feas)

        # Track best
        best_idx = 0
        for i in range(1, len(pop)):
            if feas[i] and not feas[best_idx]:
                best_idx = i
            elif feas[i] == feas[best_idx] and better(objs[i], objs[best_idx]):
                best_idx = i

        best = pop[best_idx]
        best_obj = float(objs[best_idx])
        best_feas = bool(feas[best_idx])

        # selection temperature (small, mostly for hook usage)
        ctx["sel_T"] = 1.0

        elite_n = max(1, int(round(pop_size * max(0.0, min(0.5, float(self.cfg.elite_frac))))))

        # Main loop: generations
        while it < budget.max_iters and (time.perf_counter() - started) < budget.max_time_s:
            it += 1

            # rank indices for elites
            idxs = list(range(pop_size))
            # sort by (feasible first, then objective)
            idxs.sort(key=lambda i: (0 if feas[i] else 1, objs[i]), reverse=(sense == "max"))

            elites = [pop[i] for i in idxs[:elite_n]]

            new_pop: list[Any] = list(elites)
            new_objs: list[float] = [objs[i] for i in idxs[:elite_n]]
            new_feas: list[bool] = [feas[i] for i in idxs[:elite_n]]

            # offspring
            while len(new_pop) < pop_size and (time.perf_counter() - started) < budget.max_time_s:
                proposals += 1

                p1 = tournament(pop, objs, feas)
                mate = None
                do_cx = (rng.random() < float(self.cfg.crossover_rate))
                if do_cx:
                    mate = tournament(pop, objs, feas)

                # propose move
                try:
                    if hooks.propose_move is not None:
                        # allow access to mate through ctx (and/or via returned move)
                        ctx["_mate"] = mate
                        move = hooks.propose_move(problem, instance, p1, rng, ctx=ctx)
                    elif hasattr(problem, "move_neighborhood"):
                        move = problem.move_neighborhood(p1, instance, rng)  # type: ignore[misc]
                    else:
                        move = {"op": "restart"}
                except Exception:
                    failures += 1
                    move = {"op": "restart"}

                # If propose_move didn't include mate but we want crossover, wrap it
                if do_cx and mate is not None:
                    if isinstance(move, dict):
                        move = dict(move)
                        move.setdefault("op", "crossover")
                        move.setdefault("mate", mate)
                    else:
                        move = {"op": "crossover", "move": move, "mate": mate}

                # apply move -> child
                try:
                    if hooks.apply_move is not None:
                        child = hooks.apply_move(problem, instance, p1, move, rng, ctx=ctx)
                    elif hasattr(problem, "apply_move"):
                        child = problem.apply_move(p1, move, instance, rng)  # type: ignore[misc]
                    else:
                        # absolute fallback
                        child = problem.random_solution(instance, rng)
                except Exception:
                    failures += 1
                    child = problem.random_solution(instance, rng)

                child = maybe_repair(child)
                c_obj, c_feas = eval_one(child)

                # Decide accept/replacement into new population
                try:
                    if hooks.accept_move is not None:
                        ok = bool(
                            hooks.accept_move(
                                problem, instance, p1, float("nan"),
                                child, c_obj,
                                ctx.get("sel_T", 1.0), rng, ctx=ctx
                            )
                        )
                    else:
                        ok = True  # in GA we typically accept offspring; selection is by survival
                except Exception:
                    failures += 1
                    ok = False

                if ok:
                    new_pop.append(child)
                    new_objs.append(float(c_obj))
                    new_feas.append(bool(c_feas))
                    accepts += 1

                    # update global best
                    if c_feas and (not best_feas or better(c_obj, best_obj)):
                        best = child
                        best_obj = float(c_obj)
                        best_feas = bool(c_feas)

                # optional restart hook for diversity / escaping degeneracy
                try:
                    if hooks.restart is not None and len(new_pop) < pop_size:
                        # consider restarting if offspring is rejected or infeasible etc.
                        do_restart = bool(
                            hooks.restart(
                                problem, instance, it,
                                child, c_obj,
                                best, best_obj,
                                rng, ctx=ctx
                            )
                        )
                        if do_restart:
                            rs = make_one()
                            r_obj, r_feas = eval_one(rs)
                            new_pop.append(rs)
                            new_objs.append(float(r_obj))
                            new_feas.append(bool(r_feas))
                except Exception:
                    failures += 1

            # survival: if we overshot (rare), trim by feasibility + objective
            if len(new_pop) > pop_size:
                idx2 = list(range(len(new_pop)))
                idx2.sort(key=lambda i: (0 if new_feas[i] else 1, new_objs[i]), reverse=(sense == "max"))
                idx2 = idx2[:pop_size]
                pop = [new_pop[i] for i in idx2]
                objs = [new_objs[i] for i in idx2]
                feas = [new_feas[i] for i in idx2]
            else:
                pop = new_pop
                objs = new_objs
                feas = new_feas

            pop_size = len(pop)

            # update_params hook (can adjust GA rates)
            try:
                if hooks.update_params is not None:
                    upd = hooks.update_params(
                        problem, instance, it,
                        pop[0], objs[0],
                        best, best_obj,
                        ctx.get("sel_T", 1.0), rng, ctx=ctx
                    )
                    if isinstance(upd, dict):
                        if "mutation_rate" in upd:
                            self.cfg.mutation_rate = float(upd["mutation_rate"])
                        if "crossover_rate" in upd:
                            self.cfg.crossover_rate = float(upd["crossover_rate"])
                        if "elite_frac" in upd:
                            self.cfg.elite_frac = float(upd["elite_frac"])
                        if "sel_T" in upd:
                            ctx["sel_T"] = float(upd["sel_T"])
            except Exception:
                failures += 1

        return RunTrace(
            best_obj=float(best_obj),
            best_solution=best,
            best_feasible=bool(best_feas),
            iters=int(it),
            elapsed_s=float(time.perf_counter() - started),
            accepts=int(accepts),
            proposals=int(proposals),
            failures=int(failures),
            extra={
                "pop_size": int(self.cfg.pop_size),
                "elite_frac": float(self.cfg.elite_frac),
                "mutation_rate": float(self.cfg.mutation_rate),
                "crossover_rate": float(self.cfg.crossover_rate),
                "sel_T": float(ctx.get("sel_T", 1.0)),
            },
        )


# -----------------------------------------------------------------------------
# Alternative metaheuristic: Simulated Annealing-ish (kept from v2)
# -----------------------------------------------------------------------------

@dataclass
class SAConfig:
    t0: float = 1.0
    alpha: float = 0.98
    reheat_every: int = 0
    reheat_mult: float = 1.0
    scale_ema_beta: float = 0.90
    min_scale: float = 1e-9


class SimulatedAnnealing:
    name = "sa"

    def __init__(self, cfg: SAConfig | None = None) -> None:
        self.cfg = cfg or SAConfig()

    def run(
        self,
        problem: ProblemAPI,
        instance: Any,
        hooks: HeuristicHooks,
        budget: Budget,
        rng: random.Random,
    ) -> RunTrace:
        started = time.perf_counter()
        failures = 0
        accepts = 0
        proposals = 0
        it = 0
        ctx: dict[str, Any] = {}

        sense = (problem.sense or "min").strip().lower()
        if sense not in ("min", "max"):
            sense = "min"

        def better(a: float, b: float) -> bool:
            return (a < b) if sense == "min" else (a > b)

        scale = float(self.cfg.min_scale)

        def update_scale(delta: float) -> None:
            nonlocal scale
            ad = abs(float(delta))
            beta = float(self.cfg.scale_ema_beta)
            scale = max(float(self.cfg.min_scale), beta * scale + (1.0 - beta) * ad)

        def default_accept(delta: float, temp: float) -> bool:
            update_scale(delta)
            denom = max(1e-12, float(temp) * scale)
            if sense == "min":
                if delta <= 0:
                    return True
                return rng.random() < math.exp(-float(delta) / denom)
            else:
                if delta >= 0:
                    return True
                return rng.random() < math.exp(float(delta) / denom)

        # init
        if hooks.init_solution is not None:
            try:
                cur = hooks.init_solution(problem, instance, rng, ctx=ctx)
            except Exception:
                cur = problem.random_solution(instance, rng)
                failures += 1
        else:
            cur = problem.random_solution(instance, rng)

        if hasattr(problem, "repair"):
            try:
                cur = problem.repair(cur, instance, rng)  # type: ignore[misc]
            except Exception:
                failures += 1

        best = cur
        best_obj = float(problem.objective(cur, instance))
        best_feas = bool(problem.is_feasible(cur, instance))
        cur_obj = best_obj

        T = float(self.cfg.t0)

        while it < budget.max_iters and (time.perf_counter() - started) < budget.max_time_s:
            it += 1
            proposals += 1

            # propose
            try:
                if hooks.propose_move is not None:
                    move = hooks.propose_move(problem, instance, cur, rng, ctx=ctx)
                elif hasattr(problem, "move_neighborhood"):
                    move = problem.move_neighborhood(cur, instance, rng)  # type: ignore[misc]
                else:
                    move = {"op": "restart"}
            except Exception:
                failures += 1
                continue

            # apply
            try:
                if hooks.apply_move is not None:
                    cand = hooks.apply_move(problem, instance, cur, move, rng, ctx=ctx)
                elif hasattr(problem, "apply_move"):
                    cand = problem.apply_move(cur, move, instance, rng)  # type: ignore[misc]
                else:
                    cand = problem.random_solution(instance, rng)
            except Exception:
                failures += 1
                continue

            if hasattr(problem, "repair"):
                try:
                    cand = problem.repair(cand, instance, rng)  # type: ignore[misc]
                except Exception:
                    failures += 1

            cand_feas = bool(problem.is_feasible(cand, instance))
            cand_obj = float(problem.objective(cand, instance))

            # accept
            try:
                if hooks.accept_move is not None:
                    ok = bool(
                        hooks.accept_move(
                            problem, instance, cur, cur_obj, cand, cand_obj, T, rng, ctx=ctx
                        )
                    )
                else:
                    ok = default_accept(cand_obj - cur_obj, T)
            except Exception:
                failures += 1
                ok = False

            if ok:
                cur = cand
                cur_obj = cand_obj
                accepts += 1
                if cand_feas and (not best_feas or better(cand_obj, best_obj)):
                    best = cand
                    best_obj = cand_obj
                    best_feas = cand_feas

            # update params
            try:
                if hooks.update_params is not None:
                    upd = hooks.update_params(
                        problem, instance, it, cur, cur_obj, best, best_obj, T, rng, ctx=ctx
                    )
                    if isinstance(upd, dict):
                        if "T" in upd:
                            T = float(upd["T"])
                        if "scale" in upd:
                            scale = max(float(self.cfg.min_scale), float(upd["scale"]))
                else:
                    T *= float(self.cfg.alpha)
                    if self.cfg.reheat_every and (it % int(self.cfg.reheat_every) == 0):
                        T *= float(self.cfg.reheat_mult)
            except Exception:
                failures += 1
                T *= float(self.cfg.alpha)

            # restart
            try:
                if hooks.restart is not None:
                    do_restart = bool(
                        hooks.restart(problem, instance, it, cur, cur_obj, best, best_obj, rng, ctx=ctx)
                    )
                    if do_restart:
                        cur = problem.random_solution(instance, rng)
                        if hasattr(problem, "repair"):
                            try:
                                cur = problem.repair(cur, instance, rng)  # type: ignore[misc]
                            except Exception:
                                failures += 1
                        cur_obj = float(problem.objective(cur, instance))
            except Exception:
                failures += 1

        return RunTrace(
            best_obj=float(best_obj),
            best_solution=best,
            best_feasible=bool(best_feas),
            iters=int(it),
            elapsed_s=float(time.perf_counter() - started),
            accepts=int(accepts),
            proposals=int(proposals),
            failures=int(failures),
            extra={"final_T": float(T), "final_scale": float(scale)},
        )


# -----------------------------------------------------------------------------
# EoH Data Models
# -----------------------------------------------------------------------------

@dataclass
class EvalOutput:
    fitness: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Individual:
    id: str
    generation: int
    thought: str
    code: str
    parents: list[str] = field(default_factory=list)

    fitness: float = float("-inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    eval_error: str | None = None

    created_ts: float = field(default_factory=lambda: time.time())
    meta: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        h = hashlib.sha256()
        h.update((self.thought or "").encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
        h.update((self.code or "").encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]


@dataclass
class EoHConfig:
    pop_size: int = 16
    elite_k: int = 4
    offspring_per_gen: int = 16
    generations: int = 15

    strategy: Strategy = "e1"
    representation: Representation = "thought+code"

    tournament_k: int = 4
    crossover_rate: float = 0.35

    temperature: float = 0.7
    max_tokens: int = 1600

    enforce_markers: bool = True
    allow_code_only_fallback: bool = True

    max_code_chars: int = 16_000

    llm_retries: int = 2
    reformat_retries: int = 1

    # agent/system settings
    new_session_per_call: bool = True
    agent_verbose: bool = False
    agent_use_semantic: bool = False

    # prompt controls
    max_parent_code_in_prompt: int = 7000
    max_parent_thought_in_prompt: int = 1800
    max_exemplar_code_in_prompt: int = 4500
    max_exemplar_thought_in_prompt: int = 1200

    # caching
    cache_enabled: bool = True
    prompt_cache_enabled: bool = True

    # progressive evaluation
    progressive: bool = True
    stage_budgets: tuple[Budget, ...] = (
        Budget(max_iters=80, max_time_s=0.10),
        Budget(max_iters=220, max_time_s=0.28),
    )
    stage_instances: tuple[int, ...] = (3, 8)

    # optional parallel eval
    parallel_eval: bool = False
    max_workers: int = 0  # 0 -> choose automatically

    # robust aggregation
    w_mean: float = 1.0
    w_cvar: float = 0.3
    cvar_alpha: float = 0.25
    w_feas: float = 0.8
    w_time: float = 0.15

    infeas_quality_penalty: float = 1e12


@dataclass
class EoHResult:
    best: Individual
    hall_of_fame: list[Individual]
    population: list[Individual]
    history: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------

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

_EOH_SYSTEM_PROMPT = (
    "You design compact, correct, scale-aware optimization heuristics as hook-based policies.\n"
    "Always follow the required output markers format exactly."
)

_THOUGHT_CODE_RE = re.compile(
    r"\[THOUGHT\]\s*(.*?)\s*\[CODE\]\s*(.*)\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedHeuristic:
    thought: str
    code: str


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + f"\n[... truncated {len(s) - n} chars ...]"


def parse_thought_code(text: str) -> Optional[ParsedHeuristic]:
    m = _THOUGHT_CODE_RE.search(text or "")
    if not m:
        return None
    thought = (m.group(1) or "").strip()
    code = (m.group(2) or "").strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return ParsedHeuristic(thought=thought, code=code)


def parse_code_only_fallback(text: str) -> Optional[ParsedHeuristic]:
    if not text:
        return None
    if "def build_hooks(" not in text:
        return None
    code = text.replace("```python", "").replace("```", "").strip()
    return ParsedHeuristic(thought="Code-only policy (no thought provided).", code=code)


def _seed_prompt(task_desc: str, api_desc: str, baseline: str | None = None) -> str:
    return f"""
We are evolving heuristics for a generic optimization setup.

[TASK]
{task_desc}

[APIs]
{api_desc}

{("[BASELINE]\n" + baseline) if baseline else ""}

Create an INITIAL heuristic policy (hooks dict) that improves performance and robustness.

Emphasize:
- Generic behavior (no domain-specific assumptions about solution type).
- Scale-awareness (objective deltas can be huge/small).
- Feasibility: prioritize feasible solutions; use repair if available.
- Works under GA (default) and SA (optional).

{_MARKER_SPEC}
""".strip()


def _e1_mutation_prompt(task_desc: str, api_desc: str, parent: Individual, cfg: EoHConfig) -> str:
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


def _e1_crossover_prompt(task_desc: str, api_desc: str, p1: Individual, p2: Individual, cfg: EoHConfig) -> str:
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


def _e2_improve_prompt(task_desc: str, api_desc: str, exemplars: list[Individual], cfg: EoHConfig) -> str:
    ex_text = "\n\n".join(
        f"[EXEMPLAR id={e.id} fitness={e.fitness}]\n[THOUGHT]\n{_clip(e.thought, cfg.max_exemplar_thought_in_prompt)}\n\n"
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


def _reformat_prompt(prev_text: str) -> str:
    return f"""
Reformat into EXACTLY the required markers format.
Do NOT change the algorithm unless required to satisfy build_hooks() and signatures.

CONTENT:
{prev_text}

{_MARKER_SPEC}
""".strip()


# -----------------------------------------------------------------------------
# Compile generated code -> HeuristicHooks (AST validation + restricted builtins)
# -----------------------------------------------------------------------------

DENY_NAMES: set[str] = {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "input", "breakpoint", "help",
}

DENY_ATTR_PREFIX = "__"


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def fail(self, msg: str) -> None:
        if self.error is None:
            self.error = msg

    def visit_Import(self, node: ast.Import) -> None:
        self.fail("imports not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.fail("imports not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.attr, str) and node.attr.startswith(DENY_ATTR_PREFIX):
            self.fail(f"dunder attribute not allowed: {node.attr}")
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in DENY_NAMES:
            self.fail(f"forbidden name: {node.id}")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DENY_NAMES:
            self.fail(f"forbidden call: {node.func.id}(...)")
            return
        self.generic_visit(node)


def validate_code_ast(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    v = SafetyVisitor()
    v.visit(tree)
    return v.error


@dataclass
class ExecResult:
    hooks: HeuristicHooks | None
    error: str | None


def compile_hooks(code: str, *, max_chars: int) -> ExecResult:
    if len(code) > max_chars:
        return ExecResult(hooks=None, error="code too long")

    err = validate_code_ast(code)
    if err:
        return ExecResult(hooks=None, error=err)

    safe_builtins = {
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "float": float,
        "int": int,
        "bool": bool,
        "any": any,
        "all": all,
    }

    env: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "random": random,
        "math": math,
    }

    try:
        exec(code, env, env)  # noqa: S102 (intentional; restricted env)
    except Exception as e:
        return ExecResult(hooks=None, error=f"exec failed: {e}")

    build = env.get("build_hooks")
    if not callable(build):
        return ExecResult(hooks=None, error="missing callable build_hooks()")

    try:
        d = build()
    except Exception as e:
        return ExecResult(hooks=None, error=f"build_hooks() failed: {e}")

    if not isinstance(d, dict):
        return ExecResult(hooks=None, error="build_hooks() must return a dict")

    hh = HeuristicHooks()
    any_ok = False

    for k, v in d.items():
        if k not in HOOKS:
            continue
        if callable(v):
            setattr(hh, k, _wrap_hook(v))
            any_ok = True

    if not any_ok:
        return ExecResult(hooks=None, error="no valid hooks provided")

    return ExecResult(hooks=hh, error=None)


# -----------------------------------------------------------------------------
# Evaluation: progressive stages + caching + robust aggregation
# -----------------------------------------------------------------------------

@dataclass
class EvalSpec:
    problem: ProblemAPI
    instances_train: list[Any]
    instances_valid: list[Any] | None = None
    metaheuristic: MetaheuristicAPI | None = None  # default GA if None


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = int((len(ys) - 1) * max(0.0, min(1.0, q)))
    return float(ys[idx])


def _cvar_worst_quality(quality: list[float], alpha: float) -> float:
    if not quality:
        return float("nan")
    ys = sorted(quality)
    k = max(1, int(math.ceil(len(ys) * max(1e-9, min(1.0, alpha)))))
    tail = ys[:k]
    return float(sum(tail) / len(tail))


def _fitness_from_runs(
    objs: list[float],
    feas: list[bool],
    times: list[float],
    cfg: EoHConfig,
    sense: str,
) -> tuple[float, dict[str, Any]]:
    if not objs or not times or not feas:
        return float("-inf"), {"error": "empty evaluation"}

    sense = (sense or "min").strip().lower()
    if sense not in ("min", "max"):
        sense = "min"

    quality: list[float] = []
    for o, f in zip(objs, feas):
        q = (-float(o) if sense == "min" else float(o))
        if not f:
            q = -float(cfg.infeas_quality_penalty)
        quality.append(q)

    mean_q = float(sum(quality) / len(quality))
    cvar_q = _cvar_worst_quality(quality, cfg.cvar_alpha)

    feas_rate = float(sum(1.0 for f in feas if f) / len(feas))
    avg_time = float(sum(times) / len(times))

    fitness = (
        cfg.w_mean * mean_q +
        cfg.w_cvar * cvar_q +
        cfg.w_feas * feas_rate -
        cfg.w_time * avg_time
    )

    metrics = {
        "mean_obj": float(sum(objs) / len(objs)),
        "median_obj": _quantile(objs, 0.5),
        "best_obj": (min(objs) if sense == "min" else max(objs)),
        "feas_rate": feas_rate,
        "avg_time_s": avg_time,
        "mean_quality": mean_q,
        "cvar_quality": float(cvar_q),
    }
    return float(fitness), metrics


def _stable_rng_from_tag(tag: str) -> random.Random:
    h = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(h)


# -----------------------------------------------------------------------------
# EoH Engine
# -----------------------------------------------------------------------------

class EoHEngine:
    def __init__(
        self,
        agent: Any,
        task_desc: str,
        api_desc: str,
        eval_spec: EvalSpec,
        cfg: EoHConfig | None = None,
        rng_seed: int = 0,
        baseline: str | None = None,
    ) -> None:
        self.agent = agent
        self.task_desc = task_desc
        self.api_desc = api_desc
        self.eval_spec = eval_spec
        self.cfg = cfg or EoHConfig()
        self.rng = random.Random(rng_seed)
        self.baseline = baseline

        # default optimizer: GA
        if self.eval_spec.metaheuristic is None:
            self.eval_spec.metaheuristic = GeneticAlgorithm(GAConfig(pop_size=28))

        # (policy_hash, stage_idx, budget_key, n_inst) -> EvalOutput
        self._eval_cache: dict[tuple[str, int, str, int], EvalOutput] = {}

        # prompt cache for LLM calls
        self._prompt_cache: dict[str, str] = {}

        # precompute instance indices for each stage per policy hash lazily
        self._stage_index_cache: dict[tuple[str, int], list[int]] = {}

    # ---------------- agent wrapper ----------------
    def _agent_generate(self, user_prompt: str) -> str:
        """
        Uses agent.run if available to override temperature/max_tokens.
        Assumes agent.system_prompt already set for EoH (we do that once outside).
        """
        session_id = str(uuid.uuid4()) if self.cfg.new_session_per_call else None

        if hasattr(self.agent, "run"):
            resp = self.agent.run(
                user_prompt,
                session_id=session_id,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                verbose=self.cfg.agent_verbose,
                use_semantic_retrieval=self.cfg.agent_use_semantic,
                stream_callback=None,
            )
            return resp.final_response

        return self.agent.chat(
            user_prompt,
            session_id=session_id,
            verbose=self.cfg.agent_verbose,
            use_semantic_retrieval=self.cfg.agent_use_semantic,
        )

    def _llm_cached(self, prompt: str) -> str:
        if not self.cfg.prompt_cache_enabled:
            return self._agent_generate(prompt)
        key = hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()
        if key in self._prompt_cache:
            return self._prompt_cache[key]
        out = self._agent_generate(prompt)
        self._prompt_cache[key] = out
        return out

    def _ask_parse(self, user_prompt: str) -> Optional[ParsedHeuristic]:
        last_text = ""
        for _ in range(self.cfg.llm_retries + 1):
            text = self._llm_cached(user_prompt)
            last_text = text

            parsed = parse_thought_code(text)
            if parsed:
                return parsed

            for _ in range(self.cfg.reformat_retries):
                reform = _reformat_prompt(text)
                text2 = self._llm_cached(reform)
                last_text = text2
                parsed2 = parse_thought_code(text2)
                if parsed2:
                    return parsed2

            if self.cfg.allow_code_only_fallback:
                parsed3 = parse_code_only_fallback(last_text)
                if parsed3:
                    return parsed3

        return None

    # ---------------- selection ----------------
    def _tournament(self, pop: list[Individual]) -> Individual:
        k = min(self.cfg.tournament_k, len(pop))
        cand = self.rng.sample(pop, k=k)
        cand.sort(key=lambda x: x.fitness, reverse=True)
        return cand[0]

    def _select_elites(self, pop: list[Individual]) -> list[Individual]:
        return sorted(pop, key=lambda x: x.fitness, reverse=True)[: self.cfg.elite_k]

    # ---------------- evaluation ----------------
    def _budget_key(self, b: Budget) -> str:
        return f"{int(b.max_iters)}|{float(b.max_time_s):.6f}"

    def _stage_indices(self, policy_key: str, stage_idx: int, n: int, total: int) -> list[int]:
        cache_key = (policy_key, stage_idx)
        if cache_key in self._stage_index_cache:
            idxs = self._stage_index_cache[cache_key]
            return idxs[:n] if len(idxs) >= n else idxs

        srng = _stable_rng_from_tag(f"{policy_key}:stage:{stage_idx}")
        idxs = list(range(total))
        srng.shuffle(idxs)
        self._stage_index_cache[cache_key] = idxs
        return idxs[:n]

    def _eval_one_instance(
        self,
        policy_key: str,
        stage_idx: int,
        rep_idx: int,
        instance: Any,
        hooks_obj: HeuristicHooks,
        budget: Budget,
    ) -> tuple[float, bool, float, int]:
        rrng = _stable_rng_from_tag(f"{policy_key}:stage:{stage_idx}:rep:{rep_idx}")
        try:
            trace = self.eval_spec.metaheuristic.run(  # type: ignore[union-attr]
                problem=self.eval_spec.problem,
                instance=instance,
                hooks=hooks_obj,
                budget=budget,
                rng=rrng,
            )
            return float(trace.best_obj), bool(trace.best_feasible), float(trace.elapsed_s), int(trace.failures)
        except Exception:
            sense = (self.eval_spec.problem.sense or "min").strip().lower()
            if sense == "min":
                obj = float("inf")
            else:
                obj = float("-inf")
            return obj, False, float(budget.max_time_s), 1

    def _eval_individual(self, ind: Individual) -> Individual:
        if len(ind.code) > self.cfg.max_code_chars:
            ind.fitness = float("-inf")
            ind.eval_error = "code too long"
            ind.metrics = {"compile_error": "code too long"}
            return ind

        policy_key = ind.stable_hash()

        comp = compile_hooks(ind.code, max_chars=self.cfg.max_code_chars)
        if comp.error or comp.hooks is None:
            ind.fitness = float("-inf")
            ind.eval_error = comp.error or "compile failed"
            ind.metrics = {"compile_error": ind.eval_error}
            return ind

        hooks_obj = comp.hooks

        stages = self.cfg.stage_budgets if self.cfg.progressive else (Budget(max_iters=220, max_time_s=0.28),)
        stage_ns = self.cfg.stage_instances if self.cfg.progressive else (min(len(self.eval_spec.instances_train), 8),)

        insts = self.eval_spec.instances_train
        if not insts:
            ind.fitness = float("-inf")
            ind.eval_error = "no training instances"
            ind.metrics = {"eval_error": "no training instances"}
            return ind

        last_fit = float("-inf")
        last_metrics: dict[str, Any] = {}
        total_insts = len(insts)

        for stage_idx, (budget, n_inst) in enumerate(zip(stages, stage_ns)):
            n = min(int(n_inst), total_insts)
            bkey = self._budget_key(budget)
            cache_key = (policy_key, stage_idx, bkey, n)

            if self.cfg.cache_enabled and cache_key in self._eval_cache:
                cached = self._eval_cache[cache_key]
                last_fit, last_metrics = cached.fitness, dict(cached.metrics)
                continue

            idxs = self._stage_indices(policy_key, stage_idx, n=n, total=total_insts)

            objs: list[float] = []
            feas: list[bool] = []
            times: list[float] = []
            failures_total = 0

            if self.cfg.parallel_eval and n > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                mw = self.cfg.max_workers if self.cfg.max_workers > 0 else min(32, n)
                with ThreadPoolExecutor(max_workers=mw) as ex:
                    futs = []
                    for j, ii in enumerate(idxs):
                        futs.append(
                            ex.submit(
                                self._eval_one_instance,
                                policy_key, stage_idx, j, insts[ii], hooks_obj, budget
                            )
                        )
                    for f in as_completed(futs):
                        o, fe, tm, fail = f.result()
                        objs.append(o)
                        feas.append(fe)
                        times.append(tm)
                        failures_total += fail
            else:
                for j, ii in enumerate(idxs):
                    o, fe, tm, fail = self._eval_one_instance(
                        policy_key, stage_idx, j, insts[ii], hooks_obj, budget
                    )
                    objs.append(o)
                    feas.append(fe)
                    times.append(tm)
                    failures_total += fail

            fit, metrics = _fitness_from_runs(
                objs=objs,
                feas=feas,
                times=times,
                cfg=self.cfg,
                sense=self.eval_spec.problem.sense,
            )

            metrics.update(
                {
                    "stage": stage_idx,
                    "budget": {"max_iters": budget.max_iters, "max_time_s": budget.max_time_s},
                    "n_instances": n,
                    "hook_impl": hooks_obj.implemented(),
                    "failures": int(failures_total),
                    "metaheuristic": getattr(self.eval_spec.metaheuristic, "name", "unknown"),
                }
            )

            out = EvalOutput(fitness=float(fit), metrics=dict(metrics))
            if self.cfg.cache_enabled:
                self._eval_cache[cache_key] = out

            last_fit, last_metrics = float(fit), dict(metrics)

            if (not math.isfinite(last_fit)) or (last_metrics.get("feas_rate", 0.0) < 0.05):
                break

        ind.fitness = float(last_fit)
        ind.metrics = dict(last_metrics)
        ind.eval_error = None if math.isfinite(ind.fitness) else "non-finite fitness"
        return ind

    # ---------------- individual factory ----------------
    def _make_ind(self, generation: int, thought: str, code: str, parents: list[str]) -> Individual:
        return Individual(
            id=str(uuid.uuid4())[:8],
            generation=generation,
            thought=thought.strip(),
            code=code.strip(),
            parents=list(parents),
            meta={},
        )

    # ---------------- initialization ----------------
    def initialize(self) -> list[Individual]:
        pop: list[Individual] = []
        while len(pop) < self.cfg.pop_size:
            usr = _seed_prompt(self.task_desc, self.api_desc, baseline=self.baseline)
            parsed = self._ask_parse(usr)
            if not parsed:
                continue
            ind = self._make_ind(0, parsed.thought, parsed.code, parents=[])
            pop.append(self._eval_individual(ind))
        return pop

    # ---------------- evolution ----------------
    def evolve(self) -> EoHResult:
        pop = self.initialize()
        hof: list[Individual] = []
        history: list[dict[str, Any]] = []

        def snapshot(gen: int, population: list[Individual]) -> None:
            best = max(population, key=lambda x: x.fitness)
            finite = [x.fitness for x in population if math.isfinite(x.fitness)]
            avg = float(sum(finite) / max(1, len(finite))) if finite else float("-inf")
            history.append(
                {
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "best_id": best.id,
                    "avg_fitness": avg,
                    "top_fitness": sorted((x.fitness for x in population), reverse=True)[: min(10, len(population))],
                    "best_hooks": best.metrics.get("hook_impl", []),
                    "best_feas": best.metrics.get("feas_rate", None),
                    "best_obj": best.metrics.get("mean_obj", None),
                    "metaheuristic": best.metrics.get("metaheuristic", None),
                }
            )

        snapshot(0, pop)

        for gen in range(1, self.cfg.generations + 1):
            elites = self._select_elites(pop)
            hof.extend(elites)
            hof = sorted(hof, key=lambda x: x.fitness, reverse=True)[: max(15, self.cfg.elite_k * 6)]

            children: list[Individual] = []
            while len(children) < self.cfg.offspring_per_gen:
                if self.cfg.strategy == "e2":
                    usr = _e2_improve_prompt(self.task_desc, self.api_desc, exemplars=elites, cfg=self.cfg)
                    parsed = self._ask_parse(usr)
                    if not parsed:
                        continue
                    child = self._make_ind(gen, parsed.thought, parsed.code, parents=[e.id for e in elites])
                    children.append(self._eval_individual(child))
                    continue

                do_cx = (self.rng.random() < self.cfg.crossover_rate) and (len(pop) >= 2)
                if do_cx:
                    p1 = self._tournament(pop)
                    p2 = self._tournament(pop)
                    if p2.id == p1.id:
                        p2 = self.rng.choice(pop)
                    usr = _e1_crossover_prompt(self.task_desc, self.api_desc, p1, p2, cfg=self.cfg)
                    parsed = self._ask_parse(usr)
                    if not parsed:
                        continue
                    child = self._make_ind(gen, parsed.thought, parsed.code, parents=[p1.id, p2.id])
                    children.append(self._eval_individual(child))
                else:
                    p = self._tournament(pop)
                    usr = _e1_mutation_prompt(self.task_desc, self.api_desc, p, cfg=self.cfg)
                    parsed = self._ask_parse(usr)
                    if not parsed:
                        continue
                    child = self._make_ind(gen, parsed.thought, parsed.code, parents=[p.id])
                    children.append(self._eval_individual(child))

            children_sorted = sorted(children, key=lambda x: x.fitness, reverse=True)
            pop = elites + children_sorted[: max(0, self.cfg.pop_size - len(elites))]
            snapshot(gen, pop)

        best = max(hof + pop, key=lambda x: x.fitness)
        return EoHResult(best=best, hall_of_fame=hof, population=pop, history=history)


# -----------------------------------------------------------------------------
# Convenience runner (sets agent.system_prompt once)
# -----------------------------------------------------------------------------

def run_eoh(
    agent: Any,
    *,
    task_desc: str,
    api_desc: str,
    eval_spec: EvalSpec,
    cfg: EoHConfig | None = None,
    rng_seed: int = 0,
    baseline: str | None = None,
) -> EoHResult:
    cfg = cfg or EoHConfig()
    engine = EoHEngine(
        agent=agent,
        task_desc=task_desc,
        api_desc=api_desc,
        eval_spec=eval_spec,
        cfg=cfg,
        rng_seed=rng_seed,
        baseline=baseline,
    )

    old_sys = getattr(agent, "system_prompt", None)
    try:
        if old_sys is not None:
            agent.system_prompt = _EOH_SYSTEM_PROMPT
        return engine.evolve()
    finally:
        if old_sys is not None:
            agent.system_prompt = old_sys


# =============================================================================
# Example: Continuous Sphere minimization (works with GA default)
# =============================================================================

class SphereProblem:
    sense = "min"

    def __init__(self, dim: int = 10, bound: float = 5.0) -> None:
        self.dim = int(dim)
        self.bound = float(bound)

    def random_solution(self, instance: dict[str, Any], rng: random.Random) -> list[float]:
        d = int(instance["dim"])
        b = float(instance["bound"])
        return [rng.uniform(-b, b) for _ in range(d)]

    def objective(self, solution: list[float], instance: dict[str, Any]) -> float:
        return float(sum(float(x) * float(x) for x in solution))

    def is_feasible(self, solution: list[float], instance: dict[str, Any]) -> bool:
        b = float(instance["bound"])
        return all((-b <= float(x) <= b) for x in solution)

    def repair(self, solution: list[float], instance: dict[str, Any], rng: random.Random) -> list[float]:
        b = float(instance["bound"])
        return [min(b, max(-b, float(x))) for x in solution]

    def move_neighborhood(self, solution: list[float], instance: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        # mutation move: tweak one coordinate
        i = rng.randrange(len(solution))
        delta = rng.uniform(-0.6, 0.6)
        return {"op": "mutate", "kind": "coord", "i": int(i), "delta": float(delta)}

    def apply_move(self, solution: list[float], move: Any, instance: dict[str, Any], rng: random.Random) -> list[float]:
        # supports both mutation and a simple crossover fallback if move contains mate
        if isinstance(move, dict) and move.get("op") == "crossover" and "mate" in move:
            mate = move["mate"]
            if isinstance(mate, list) and len(mate) == len(solution):
                cut = rng.randrange(1, len(solution))
                child = list(solution[:cut]) + list(mate[cut:])
                # optional small mutation
                if rng.random() < 0.4:
                    j = rng.randrange(len(child))
                    child[j] = float(child[j] + rng.gauss(0.0, 0.2))
                return child

        if isinstance(move, dict) and move.get("kind") == "coord":
            out = list(solution)
            out[int(move["i"])] = float(out[int(move["i"])] + float(move["delta"]))
            return out

        # ultimate fallback
        return list(solution)


def make_sphere_instances(n: int, dim: int = 10, bound: float = 5.0) -> list[dict[str, Any]]:
    return [{"dim": int(dim), "bound": float(bound)} for _ in range(int(n))]


def example_api_desc() -> str:
    return textwrap.dedent(
        """
        ProblemAPI:
          - problem.sense is "min" or "max"
          - problem.random_solution(instance, rng) -> solution
          - problem.objective(solution, instance) -> float
          - problem.is_feasible(solution, instance) -> bool
          - optional: problem.repair(solution, instance, rng) -> solution
          - optional: problem.move_neighborhood(solution, instance, rng) -> move
          - optional: problem.apply_move(solution, move, instance, rng) -> solution

        MetaheuristicAPI (used in evaluation):
          - metaheuristic.run(problem, instance, hooks, budget, rng) -> RunTrace

        Your heuristic must output build_hooks()->dict with any subset of:
          init_solution / propose_move / apply_move / accept_move / update_params / restart

        GA conventions:
          - propose_move() may return {"op":"mutate", ...} or {"op":"crossover","mate":sol,...}.
          - apply_move() should understand the move object you emit.
        """
    ).strip()


# =============================================================================
# Example usage (edit for your project)
# =============================================================================

if __name__ == "__main__":
    # Provide your Agent instance here.
    # Example:
    #   from agent_core.agent import create_agent
    #   agent = create_agent(llm_url="http://localhost:8080", config_overrides={...})
    agent = None  # replace with your Agent instance

    problem = SphereProblem(dim=12, bound=5.0)
    train_instances = make_sphere_instances(n=12, dim=12, bound=5.0)

    # metaheuristic defaults to GA if None
    eval_spec = EvalSpec(
        problem=problem,
        instances_train=train_instances,
        metaheuristic=None,
    )

    task_desc = (
        "Evolve a generic heuristic policy (hook dict) to improve the metaheuristic's "
        "performance on the given ProblemAPI. Optimize robustness: feasible solutions, "
        "fast convergence, stable improvements."
    )

    cfg = EoHConfig(
        pop_size=10,
        elite_k=3,
        offspring_per_gen=10,
        generations=8,
        strategy="e1",
        temperature=0.7,
        max_tokens=1400,
        progressive=True,
        stage_instances=(3, 8),
        stage_budgets=(Budget(60, 0.10), Budget(140, 0.22)),
        w_feas=0.8,
        prompt_cache_enabled=True,
        parallel_eval=False,
    )

    if agent is None:
        raise RuntimeError("Set `agent` to your Agent instance before running this demo.")

    result = run_eoh(
        agent,
        task_desc=task_desc,
        api_desc=example_api_desc(),
        eval_spec=eval_spec,
        cfg=cfg,
        rng_seed=0,
        baseline=None,
    )

    best = result.best
    print("\n=== BEST INDIVIDUAL ===")
    print("fitness:", best.fitness)
    print("hooks:", best.metrics.get("hook_impl"))
    print("metrics:", json.dumps(best.metrics, indent=2))
    print("\n[THOUGHT]\n", best.thought)
    print("\n[CODE]\n", best.code)

    print("\n=== HISTORY ===")
    print(json.dumps(result.history, indent=2))
