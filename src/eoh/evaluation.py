"""EvalSpec, fitness aggregation helpers and deterministic RNG utilities."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

from .models import EoHConfig
from .types import MetaheuristicAPI, ProblemAPI

# ---------------------------------------------------------------------------
# EvalSpec – bundles a problem, instances and metaheuristic for evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalSpec:
    """
    Everything needed to evaluate a heuristic policy.

    Parameters
    ----------
    problem:
        Object satisfying :class:`~eoh.types.ProblemAPI`.
    instances_train:
        List of problem instances used during evolution.
    instances_valid:
        Optional hold-out instances for final validation.
    metaheuristic:
        Engine to run heuristics against.  Defaults to
        :class:`~eoh.metaheuristics.ga.GeneticAlgorithm` when ``None``.
    """

    problem: ProblemAPI
    instances_train: list[Any]
    instances_valid: list[Any] | None = None
    metaheuristic: MetaheuristicAPI | None = None


# ---------------------------------------------------------------------------
# Deterministic per-policy RNGs (reproducible across runs)
# ---------------------------------------------------------------------------


def stable_rng(tag: str) -> random.Random:
    """Return a *deterministic* :class:`random.Random` seeded from *tag*."""
    seed = int(hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


# backward-compat alias used inside the engine
_stable_rng_from_tag = stable_rng


# ---------------------------------------------------------------------------
# Robust fitness aggregation
# ---------------------------------------------------------------------------


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = int((len(ys) - 1) * max(0.0, min(1.0, q)))
    return float(ys[idx])


def _cvar_worst_quality(quality: list[float], alpha: float) -> float:
    """Mean of the *alpha* worst-quality fraction (Conditional Value-at-Risk)."""
    if not quality:
        return float("nan")
    ys = sorted(quality)
    k = max(1, int(math.ceil(len(ys) * max(1e-9, min(1.0, alpha)))))
    tail = ys[:k]
    return float(sum(tail) / len(tail))


def fitness_from_runs(
    objs: list[float],
    feas: list[bool],
    times: list[float],
    cfg: EoHConfig,
    sense: str,
) -> tuple[float, dict[str, Any]]:
    """
    Aggregate per-run observations into a single scalar *fitness* and a
    metrics dict.

    The formula is::

        fitness = w_mean * mean_quality
                + w_cvar * cvar_quality
                + w_feas * feas_rate
                - w_time * avg_time_s
    """
    if not objs or not times or not feas:
        return float("-inf"), {"error": "empty evaluation"}

    sense = (sense or "min").strip().lower()
    if sense not in ("min", "max"):
        sense = "min"

    quality: list[float] = []
    for o, f in zip(objs, feas):
        q = -float(o) if sense == "min" else float(o)
        if not f:
            q = -float(cfg.infeas_quality_penalty)
        quality.append(q)

    mean_q = float(sum(quality) / len(quality))
    cvar_q = _cvar_worst_quality(quality, cfg.cvar_alpha)
    feas_rate = float(sum(1.0 for f in feas if f) / len(feas))
    avg_time = float(sum(times) / len(times))

    fitness = (
        cfg.w_mean * mean_q
        + cfg.w_cvar * cvar_q
        + cfg.w_feas * feas_rate
        - cfg.w_time * avg_time
    )

    metrics: dict[str, Any] = {
        "mean_obj": float(sum(objs) / len(objs)),
        "median_obj": _quantile(objs, 0.5),
        "best_obj": (min(objs) if sense == "min" else max(objs)),
        "feas_rate": feas_rate,
        "avg_time_s": avg_time,
        "mean_quality": mean_q,
        "cvar_quality": float(cvar_q),
    }
    return float(fitness), metrics


# backward-compat alias
_fitness_from_runs = fitness_from_runs
