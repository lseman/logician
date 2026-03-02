"""Top-level convenience runner: run_eoh."""

from __future__ import annotations

from typing import Any

from .engine import EoHEngine
from .evaluation import EvalSpec
from .models import EoHConfig, EoHResult
from .prompts import EOH_SYSTEM_PROMPT


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
    """
    Run an EoH evolution session and return the :class:`~eoh.models.EoHResult`.

    The agent's ``system_prompt`` is temporarily overwritten for the duration
    of the run and restored afterwards.

    Parameters
    ----------
    agent:
        Any object that exposes ``agent.run(prompt, ...)`` or ``agent.chat()``.
    task_desc:
        Plain-text description of the optimisation task.
    api_desc:
        Description of the :class:`~eoh.types.ProblemAPI` the hooks will
        receive (typically from :func:`~eoh.examples.sphere.example_api_desc`).
    eval_spec:
        Problem instance + training instances + optional metaheuristic.
    cfg:
        EoH hyper-parameters.  Defaults to :class:`~eoh.models.EoHConfig`.
    rng_seed:
        Seed for the population-level RNG.
    baseline:
        Optional existing heuristic code pasted into the seed prompt.
    """
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
            agent.system_prompt = EOH_SYSTEM_PROMPT
        return engine.evolve()
    finally:
        if old_sys is not None:
            agent.system_prompt = old_sys
