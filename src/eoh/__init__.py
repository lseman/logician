"""
Evolution of Heuristics (EoH) — hook-based meta-learning for optimisation.

Package layout
--------------
::

    eoh/
    ├── types.py            — ProblemAPI, MetaheuristicAPI, Budget, RunTrace
    ├── hooks.py            — HookFn, HeuristicHooks
    ├── models.py           — Individual, EoHConfig, EoHResult, EvalOutput
    ├── prompts.py          — Prompt builders, ParsedHeuristic, parsers
    ├── safety.py           — AST validator, compile_hooks, ExecResult
    ├── evaluation.py       — EvalSpec, fitness_from_runs, stable_rng
    ├── engine.py           — EoHEngine (evolution loop)
    ├── runner.py           — run_eoh (convenience entry point)
    ├── metaheuristics/
    │   ├── ga.py           — GAConfig, GeneticAlgorithm
    │   └── sa.py           — SAConfig, SimulatedAnnealing
    └── examples/
        └── sphere.py       — SphereProblem demo

Quick start
-----------
::

    from src.eoh import run_eoh, EoHConfig, EvalSpec, Budget
    from src.eoh.examples import SphereProblem, make_sphere_instances

    problem = SphereProblem(dim=10)
    result = run_eoh(
        agent,
        task_desc="...",
        api_desc="...",
        eval_spec=EvalSpec(problem=problem, instances_train=make_sphere_instances(10)),
        cfg=EoHConfig(pop_size=8, generations=5),
    )
    print(result.best.fitness)
"""

from .engine import EoHEngine
from .evaluation import EvalSpec, fitness_from_runs, stable_rng
from .hooks import HeuristicHooks, HookFn
from .metaheuristics.ga import GAConfig, GeneticAlgorithm
from .metaheuristics.sa import SAConfig, SimulatedAnnealing
from .models import EoHConfig, EoHResult, EvalOutput, Individual
from .prompts import (
    EOH_SYSTEM_PROMPT,
    ParsedHeuristic,
    e1_crossover_prompt,
    e1_mutation_prompt,
    e2_improve_prompt,
    parse_code_only_fallback,
    parse_thought_code,
    reformat_prompt,
    seed_prompt,
)
from .runner import run_eoh
from .safety import ExecResult, SafetyVisitor, compile_hooks, validate_code_ast
from .types import Budget, MetaheuristicAPI, ProblemAPI, RunTrace

__all__ = [
    # ---- entry points ----
    "run_eoh",
    "EoHEngine",
    # ---- config / data models ----
    "EoHConfig",
    "EoHResult",
    "EvalOutput",
    "Individual",
    # ---- evaluation ----
    "EvalSpec",
    "fitness_from_runs",
    "stable_rng",
    # ---- types / protocols ----
    "Budget",
    "RunTrace",
    "ProblemAPI",
    "MetaheuristicAPI",
    # ---- hooks ----
    "HookFn",
    "HeuristicHooks",
    # ---- metaheuristics ----
    "GAConfig",
    "GeneticAlgorithm",
    "SAConfig",
    "SimulatedAnnealing",
    # ---- prompts ----
    "EOH_SYSTEM_PROMPT",
    "ParsedHeuristic",
    "seed_prompt",
    "e1_mutation_prompt",
    "e1_crossover_prompt",
    "e2_improve_prompt",
    "reformat_prompt",
    "parse_thought_code",
    "parse_code_only_fallback",
    # ---- safety ----
    "compile_hooks",
    "validate_code_ast",
    "ExecResult",
    "SafetyVisitor",
]
