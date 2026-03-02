"""Sphere minimisation example problem and demo runner."""

from __future__ import annotations

import json
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# SphereProblem
# ---------------------------------------------------------------------------


class SphereProblem:
    """
    Continuous sphere function: minimise :math:`\\sum x_i^2`.

    Supports both mutation (coord-tweak) and real crossover moves so the GA
    can exercise its full hook contract.
    """

    sense = "min"

    def __init__(self, dim: int = 10, bound: float = 5.0) -> None:
        self.dim = int(dim)
        self.bound = float(bound)

    def random_solution(self, instance: dict[str, Any], rng: Any) -> list[float]:
        d, b = int(instance["dim"]), float(instance["bound"])
        return [rng.uniform(-b, b) for _ in range(d)]

    def objective(self, solution: list[float], instance: dict[str, Any]) -> float:
        return float(sum(x * x for x in solution))

    def is_feasible(self, solution: list[float], instance: dict[str, Any]) -> bool:
        b = float(instance["bound"])
        return all(-b <= float(x) <= b for x in solution)

    def repair(
        self, solution: list[float], instance: dict[str, Any], rng: Any
    ) -> list[float]:
        b = float(instance["bound"])
        return [min(b, max(-b, float(x))) for x in solution]

    def move_neighborhood(
        self, solution: list[float], instance: dict[str, Any], rng: Any
    ) -> dict[str, Any]:
        i = rng.randrange(len(solution))
        delta = rng.uniform(-0.6, 0.6)
        return {"op": "mutate", "kind": "coord", "i": int(i), "delta": float(delta)}

    def apply_move(
        self, solution: list[float], move: Any, instance: dict[str, Any], rng: Any
    ) -> list[float]:
        if isinstance(move, dict) and move.get("op") == "crossover" and "mate" in move:
            mate = move["mate"]
            if isinstance(mate, list) and len(mate) == len(solution):
                cut = rng.randrange(1, len(solution))
                child = list(solution[:cut]) + list(mate[cut:])
                if rng.random() < 0.4:
                    j = rng.randrange(len(child))
                    child[j] = float(child[j]) + rng.gauss(0.0, 0.2)
                return child

        if isinstance(move, dict) and move.get("kind") == "coord":
            out = list(solution)
            out[int(move["i"])] = float(out[int(move["i"])]) + float(move["delta"])
            return out

        return list(solution)


# ---------------------------------------------------------------------------
# Instance factory
# ---------------------------------------------------------------------------


def make_sphere_instances(
    n: int, dim: int = 10, bound: float = 5.0
) -> list[dict[str, Any]]:
    """Return *n* identical sphere instances (dim and bound parametrised)."""
    return [{"dim": int(dim), "bound": float(bound)} for _ in range(int(n))]


# ---------------------------------------------------------------------------
# API description string for use in prompts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Demo __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace *agent* with your Agent instance before running.
    #
    #   from src.agent.factory import create_agent
    #   agent = create_agent(...)

    import sys

    # Local imports so this script is also runnable standalone
    try:
        from src.eoh.evaluation import EvalSpec
        from src.eoh.models import EoHConfig
        from src.eoh.runner import run_eoh
        from src.eoh.types import Budget
    except ImportError:
        print(
            "Run from the project root: python -m src.eoh.examples.sphere",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = None  # ← set your agent here
    if agent is None:
        raise RuntimeError("Assign your Agent instance to `agent` before running.")

    problem = SphereProblem(dim=12, bound=5.0)
    train_instances = make_sphere_instances(n=12, dim=12, bound=5.0)
    eval_spec = EvalSpec(problem=problem, instances_train=train_instances)

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

    result = run_eoh(
        agent,
        task_desc=(
            "Evolve a generic heuristic policy (hook dict) to improve the metaheuristic's "
            "performance on the given ProblemAPI. Optimise robustness: feasible solutions, "
            "fast convergence, stable improvements."
        ),
        api_desc=example_api_desc(),
        eval_spec=eval_spec,
        cfg=cfg,
        rng_seed=0,
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
