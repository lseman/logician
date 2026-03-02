"""Core types, protocols and value objects used across the EoH package."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Scalar type aliases
# ---------------------------------------------------------------------------

Representation = Literal["thought", "code", "thought+code"]
Strategy = Literal["e1", "e2"]

# ---------------------------------------------------------------------------
# Hook name registry (contract)
# ---------------------------------------------------------------------------

HOOKS: tuple[str, ...] = (
    "init_solution",
    "propose_move",
    "apply_move",
    "accept_move",
    "update_params",
    "restart",
)

# ---------------------------------------------------------------------------
# ProblemAPI – minimal contract for any optimisation problem
# ---------------------------------------------------------------------------


@runtime_checkable
class ProblemAPI(Protocol):
    """
    Minimal interface every problem must satisfy.

    Required
    --------
    sense : "min" | "max"
    random_solution(instance, rng) -> solution
    objective(solution, instance) -> float
    is_feasible(solution, instance) -> bool

    Recommended (optional)
    ----------------------
    repair(solution, instance, rng) -> solution
    instance_features(instance) -> dict
    move_neighborhood(solution, instance, rng) -> move
    apply_move(solution, move, instance, rng) -> solution
    """

    sense: str  # "min" or "max"

    def random_solution(self, instance: Any, rng: random.Random) -> Any: ...
    def objective(self, solution: Any, instance: Any) -> float: ...
    def is_feasible(self, solution: Any, instance: Any) -> bool: ...

    # optional but strongly recommended
    def repair(self, solution: Any, instance: Any, rng: random.Random) -> Any: ...
    def instance_features(self, instance: Any) -> dict[str, Any]: ...
    def move_neighborhood(
        self, solution: Any, instance: Any, rng: random.Random
    ) -> Any: ...
    def apply_move(
        self, solution: Any, move: Any, instance: Any, rng: random.Random
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Budget – per-run resource limits
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Budget:
    """
    Resource limits for a single optimiser run on a single instance.
    For GA, *max_iters* is the number of generations.
    """

    max_iters: int = 200
    max_time_s: float = 0.25


# ---------------------------------------------------------------------------
# RunTrace – structured result returned by any metaheuristic
# ---------------------------------------------------------------------------


@dataclass(slots=True)
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


# ---------------------------------------------------------------------------
# MetaheuristicAPI – anything with a .run() method
# ---------------------------------------------------------------------------


class MetaheuristicAPI(Protocol):
    name: str

    def run(
        self,
        problem: ProblemAPI,
        instance: Any,
        hooks: Any,  # HeuristicHooks — forward-reference kept loose to avoid circular import
        budget: Budget,
        rng: random.Random,
    ) -> RunTrace: ...
