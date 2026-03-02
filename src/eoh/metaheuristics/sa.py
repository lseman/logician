"""Simulated Annealing metaheuristic."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from ..hooks import HeuristicHooks
from ..types import Budget, ProblemAPI, RunTrace


@dataclass
class SAConfig:
    t0: float = 1.0
    alpha: float = 0.98
    reheat_every: int = 0
    reheat_mult: float = 1.0
    scale_ema_beta: float = 0.90
    min_scale: float = 1e-9


class SimulatedAnnealing:
    """
    Simulated Annealing with hook-based customisation.

    Temperature is cooled geometrically at each iteration unless the
    ``update_params`` hook returns ``{"T": <new_temp>}``.
    An optional ``reheat_every`` parameter triggers multiplicative reheating.
    """

    name = "sa"

    def __init__(self, cfg: SAConfig | None = None) -> None:
        self.cfg = cfg or SAConfig()

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

        scale = float(self.cfg.min_scale)

        def update_scale(delta: float) -> None:
            nonlocal scale
            beta = float(self.cfg.scale_ema_beta)
            scale = max(
                float(self.cfg.min_scale),
                beta * scale + (1.0 - beta) * abs(float(delta)),
            )

        def default_accept(delta: float, temp: float) -> bool:
            update_scale(delta)
            denom = max(1e-12, float(temp) * scale)
            if sense == "min":
                return delta <= 0 or rng.random() < math.exp(-float(delta) / denom)
            return delta >= 0 or rng.random() < math.exp(float(delta) / denom)

        # ---- init ----
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
                cur = problem.repair(cur, instance, rng)
            except Exception:
                failures += 1

        best = cur
        best_obj = float(problem.objective(cur, instance))
        best_feas = bool(problem.is_feasible(cur, instance))
        cur_obj = best_obj
        T = float(self.cfg.t0)

        # ---- main loop ----
        while (
            it < budget.max_iters
            and (time.perf_counter() - started) < budget.max_time_s
        ):
            it += 1
            proposals += 1

            # propose
            try:
                if hooks.propose_move is not None:
                    move = hooks.propose_move(problem, instance, cur, rng, ctx=ctx)
                elif hasattr(problem, "move_neighborhood"):
                    move = problem.move_neighborhood(cur, instance, rng)
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
                    cand = problem.apply_move(cur, move, instance, rng)
                else:
                    cand = problem.random_solution(instance, rng)
            except Exception:
                failures += 1
                continue

            if hasattr(problem, "repair"):
                try:
                    cand = problem.repair(cand, instance, rng)
                except Exception:
                    failures += 1

            cand_feas = bool(problem.is_feasible(cand, instance))
            cand_obj = float(problem.objective(cand, instance))

            # accept
            try:
                if hooks.accept_move is not None:
                    ok = bool(
                        hooks.accept_move(
                            problem,
                            instance,
                            cur,
                            cur_obj,
                            cand,
                            cand_obj,
                            T,
                            rng,
                            ctx=ctx,
                        )
                    )
                else:
                    ok = default_accept(cand_obj - cur_obj, T)
            except Exception:
                failures += 1
                ok = False

            if ok:
                cur, cur_obj = cand, cand_obj
                accepts += 1
                if cand_feas and (not best_feas or better(cand_obj, best_obj)):
                    best, best_obj, best_feas = cand, cand_obj, cand_feas

            # update_params / cool
            try:
                if hooks.update_params is not None:
                    upd = hooks.update_params(
                        problem,
                        instance,
                        it,
                        cur,
                        cur_obj,
                        best,
                        best_obj,
                        T,
                        rng,
                        ctx=ctx,
                    )
                    if isinstance(upd, dict):
                        if "T" in upd:
                            T = float(upd["T"])
                        if "scale" in upd:
                            scale = max(float(self.cfg.min_scale), float(upd["scale"]))
                else:
                    T *= float(self.cfg.alpha)
                    if self.cfg.reheat_every and it % int(self.cfg.reheat_every) == 0:
                        T *= float(self.cfg.reheat_mult)
            except Exception:
                failures += 1
                T *= float(self.cfg.alpha)

            # restart
            try:
                if hooks.restart is not None:
                    if bool(
                        hooks.restart(
                            problem,
                            instance,
                            it,
                            cur,
                            cur_obj,
                            best,
                            best_obj,
                            rng,
                            ctx=ctx,
                        )
                    ):
                        cur = problem.random_solution(instance, rng)
                        if hasattr(problem, "repair"):
                            try:
                                cur = problem.repair(cur, instance, rng)
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
