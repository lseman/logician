"""EoH data models: Individual, EoHConfig, EoHResult, EvalOutput."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from .types import Budget, Representation, Strategy

# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalOutput:
    fitness: float
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual – one evolved heuristic policy
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """
    A single candidate heuristic, identified by a UUID-prefix id.

    The *stable_hash* method produces a content-based fingerprint suitable
    for cache keys.
    """

    id: str
    generation: int
    thought: str
    code: str
    parents: list[str] = field(default_factory=list)

    fitness: float = float("-inf")
    metrics: dict[str, Any] = field(default_factory=dict)
    eval_error: str | None = None

    created_ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        h = hashlib.sha256()
        h.update((self.thought or "").encode("utf-8", errors="ignore"))
        h.update(b"\n---\n")
        h.update((self.code or "").encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# EoHConfig – all knobs for an EoH run
# ---------------------------------------------------------------------------


@dataclass
class EoHConfig:
    # ----- population -----
    pop_size: int = 16
    elite_k: int = 4
    offspring_per_gen: int = 16
    generations: int = 15

    # ----- search strategy -----
    strategy: Strategy = "e1"
    representation: Representation = "thought+code"
    tournament_k: int = 4
    crossover_rate: float = 0.35

    # ----- LLM sampling -----
    temperature: float = 0.7
    max_tokens: int = 1600
    enforce_markers: bool = True
    allow_code_only_fallback: bool = True
    max_code_chars: int = 16_000
    llm_retries: int = 2
    reformat_retries: int = 1

    # ----- agent session -----
    new_session_per_call: bool = True
    agent_verbose: bool = False
    agent_use_semantic: bool = False

    # ----- prompt budget -----
    max_parent_code_in_prompt: int = 7_000
    max_parent_thought_in_prompt: int = 1_800
    max_exemplar_code_in_prompt: int = 4_500
    max_exemplar_thought_in_prompt: int = 1_200

    # ----- caching -----
    cache_enabled: bool = True
    prompt_cache_enabled: bool = True

    # ----- progressive evaluation -----
    progressive: bool = True
    stage_budgets: tuple[Budget, ...] = (
        Budget(max_iters=80, max_time_s=0.10),
        Budget(max_iters=220, max_time_s=0.28),
    )
    stage_instances: tuple[int, ...] = (3, 8)

    # ----- parallel evaluation -----
    parallel_eval: bool = False
    max_workers: int = 0  # 0 → choose automatically

    # ----- fitness aggregation weights -----
    w_mean: float = 1.0
    w_cvar: float = 0.3
    cvar_alpha: float = 0.25
    w_feas: float = 0.8
    w_time: float = 0.15
    infeas_quality_penalty: float = 1e12


# ---------------------------------------------------------------------------
# EoHResult – final outcome of EoHEngine.evolve()
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EoHResult:
    best: Individual
    hall_of_fame: list[Individual]
    population: list[Individual]
    history: list[dict[str, Any]]
