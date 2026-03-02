"""Genetic Algorithm metaheuristic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..hooks import HeuristicHooks
from ..types import Budget, ProblemAPI, RunTrace

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    pop_size: int = 32
    elite_frac: float = 0.125
    tournament_k: int = 3
    crossover_rate: float = 0.20
    mutation_rate: float = 0.90
    infeas_penalty: float = 1e12
    restart_patience: int = 0
    rank_selection: bool = True


# ---------------------------------------------------------------------------
# GeneticAlgorithm
# ---------------------------------------------------------------------------


class GeneticAlgorithm:
    """
    Genetic Algorithm with hook-based customisation.

    Hook conventions
    ----------------
    - **init_solution** — used to create individuals (falls back to
      ``problem.random_solution`` if absent).
    - **propose_move** — returns a move dict for mutation or crossover:
        ``{"op":"mutate", ...}`` or ``{"op":"crossover","mate":sol}``.
      Falls back to ``problem.move_neighborhood``.
    - **apply_move** — builds a child from (parent, move).
      Falls back to ``problem.apply_move`` or random solution.
    - **accept_move** — decides replacement.  Called with
      ``(problem, instance, cur, cur_obj, cand, cand_obj, sel_T, rng, ctx)``.
      Defaults to always-accept offspring (selection happens via survival).
    - **update_params** — can return a dict with updated rates/temperature.
    - **restart** — can trigger diversity injection for a weak individual.
    """

    name = "ga"

    def __init__(self, cfg: GAConfig | None = None) -> None:
        self.cfg = cfg or GAConfig()

    # ------------------------------------------------------------------
    def run(
        self,
        problem: ProblemAPI,
        instance: Any,
        hooks: HeuristicHooks,
        budget: Budget,
        rng: Any,
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
            return q if feas else -float(self.cfg.infeas_penalty)

        def maybe_repair(sol: Any) -> Any:
            nonlocal failures
            if hasattr(problem, "repair"):
                try:
                    return problem.repair(sol, instance, rng)
                except Exception:
                    failures += 1
            return sol

        def make_one() -> Any:
            nonlocal failures
            if hooks.init_solution is not None:
                try:
                    return maybe_repair(
                        hooks.init_solution(problem, instance, rng, ctx=ctx)
                    )
                except Exception:
                    failures += 1
            return maybe_repair(problem.random_solution(instance, rng))

        def eval_one(sol: Any) -> tuple[float, bool]:
            nonlocal failures
            try:
                feas = bool(problem.is_feasible(sol, instance))
            except Exception:
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
            best_i = idxs[0]
            for i in idxs[1:]:
                if feas[i] and not feas[best_i]:
                    best_i = i
                elif feas[i] == feas[best_i] and better(objs[i], objs[best_i]):
                    best_i = i
            return pop[best_i]

        # ---- initialise population ----
        pop_size = max(2, int(self.cfg.pop_size))
        pop = [make_one() for _ in range(pop_size)]
        objs_list, feas_list = zip(*(eval_one(s) for s in pop))
        objs: list[float] = list(objs_list)
        feas: list[bool] = list(feas_list)

        best_idx = max(range(len(pop)), key=lambda i: quality(objs[i], feas[i]))
        best = pop[best_idx]
        best_obj = float(objs[best_idx])
        best_feas = bool(feas[best_idx])

        ctx["sel_T"] = 1.0
        elite_n = max(
            1, int(round(pop_size * max(0.0, min(0.5, float(self.cfg.elite_frac)))))
        )

        # ---- main loop ----
        while (
            it < budget.max_iters
            and (time.perf_counter() - started) < budget.max_time_s
        ):
            it += 1

            # elites
            idxs = sorted(
                range(pop_size),
                key=lambda i: (0 if feas[i] else 1, objs[i]),
                reverse=(sense == "max"),
            )
            elites = [pop[i] for i in idxs[:elite_n]]
            new_pop: list[Any] = list(elites)
            new_objs: list[float] = [objs[i] for i in idxs[:elite_n]]
            new_feas: list[bool] = [feas[i] for i in idxs[:elite_n]]

            # offspring
            while (
                len(new_pop) < pop_size
                and (time.perf_counter() - started) < budget.max_time_s
            ):
                proposals += 1
                p1 = tournament(pop, objs, feas)
                mate = (
                    tournament(pop, objs, feas)
                    if rng.random() < float(self.cfg.crossover_rate)
                    else None
                )

                # --- propose move ---
                try:
                    if hooks.propose_move is not None:
                        ctx["_mate"] = mate
                        move = hooks.propose_move(problem, instance, p1, rng, ctx=ctx)
                    elif hasattr(problem, "move_neighborhood"):
                        move = problem.move_neighborhood(p1, instance, rng)
                    else:
                        move = {"op": "restart"}
                except Exception:
                    failures += 1
                    move = {"op": "restart"}

                if mate is not None:
                    if isinstance(move, dict):
                        move = dict(move)
                        move.setdefault("op", "crossover")
                        move.setdefault("mate", mate)
                    else:
                        move = {"op": "crossover", "move": move, "mate": mate}

                # --- apply move → child ---
                try:
                    if hooks.apply_move is not None:
                        child = hooks.apply_move(
                            problem, instance, p1, move, rng, ctx=ctx
                        )
                    elif hasattr(problem, "apply_move"):
                        child = problem.apply_move(p1, move, instance, rng)
                    else:
                        child = problem.random_solution(instance, rng)
                except Exception:
                    failures += 1
                    child = problem.random_solution(instance, rng)

                child = maybe_repair(child)
                c_obj, c_feas = eval_one(child)

                # --- accept ---
                try:
                    if hooks.accept_move is not None:
                        ok = bool(
                            hooks.accept_move(
                                problem,
                                instance,
                                p1,
                                float("nan"),
                                child,
                                c_obj,
                                ctx.get("sel_T", 1.0),
                                rng,
                                ctx=ctx,
                            )
                        )
                    else:
                        ok = True
                except Exception:
                    failures += 1
                    ok = False

                if ok:
                    new_pop.append(child)
                    new_objs.append(float(c_obj))
                    new_feas.append(bool(c_feas))
                    accepts += 1
                    if c_feas and (not best_feas or better(c_obj, best_obj)):
                        best = child
                        best_obj = float(c_obj)
                        best_feas = bool(c_feas)

                # --- optional restart for diversity ---
                try:
                    if hooks.restart is not None and len(new_pop) < pop_size:
                        if bool(
                            hooks.restart(
                                problem,
                                instance,
                                it,
                                child,
                                c_obj,
                                best,
                                best_obj,
                                rng,
                                ctx=ctx,
                            )
                        ):
                            rs = make_one()
                            r_obj, r_feas = eval_one(rs)
                            new_pop.append(rs)
                            new_objs.append(float(r_obj))
                            new_feas.append(bool(r_feas))
                except Exception:
                    failures += 1

            # survival trim
            if len(new_pop) > pop_size:
                keep = sorted(
                    range(len(new_pop)),
                    key=lambda i: (0 if new_feas[i] else 1, new_objs[i]),
                    reverse=(sense == "max"),
                )[:pop_size]
                pop = [new_pop[i] for i in keep]
                objs = [new_objs[i] for i in keep]
                feas = [new_feas[i] for i in keep]
            else:
                pop, objs, feas = new_pop, new_objs, new_feas

            pop_size = len(pop)

            # update_params hook
            try:
                if hooks.update_params is not None:
                    upd = hooks.update_params(
                        problem,
                        instance,
                        it,
                        pop[0],
                        objs[0],
                        best,
                        best_obj,
                        ctx.get("sel_T", 1.0),
                        rng,
                        ctx=ctx,
                    )
                    if isinstance(upd, dict):
                        for attr in ("mutation_rate", "crossover_rate", "elite_frac"):
                            if attr in upd:
                                setattr(self.cfg, attr, float(upd[attr]))
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
