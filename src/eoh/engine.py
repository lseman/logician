"""EoH evolution engine: initialisation, evaluation and generational loop."""

from __future__ import annotations

import hashlib
import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from .evaluation import EvalOutput, EvalSpec, _fitness_from_runs, _stable_rng_from_tag
from .metaheuristics.ga import GAConfig, GeneticAlgorithm
from .models import EoHConfig, EoHResult, Individual
from .prompts import (
    ParsedHeuristic,
    _e1_crossover_prompt,
    _e1_mutation_prompt,
    _e2_improve_prompt,
    _reformat_prompt,
    _seed_prompt,
    parse_code_only_fallback,
    parse_thought_code,
)
from .safety import compile_hooks
from .types import Budget


class EoHEngine:
    """
    Evolution-of-Heuristics engine.

    Manages the population, drives LLM-based mutation/crossover and
    evaluates individuals using progressive staged budgets.

    Usage::

        engine = EoHEngine(agent, task_desc, api_desc, eval_spec, cfg)
        result = engine.evolve()
    """

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

        # Default optimizer: GA
        if self.eval_spec.metaheuristic is None:
            self.eval_spec.metaheuristic = GeneticAlgorithm(GAConfig(pop_size=28))

        # (policy_hash, stage_idx, budget_key, n_inst) → EvalOutput
        self._eval_cache: dict[tuple[str, int, str, int], EvalOutput] = {}

        # prompt-hash → LLM response
        self._prompt_cache: dict[str, str] = {}

        # policy_hash + stage_idx → shuffled instance indices
        self._stage_index_cache: dict[tuple[str, int], list[int]] = {}

    # ------------------------------------------------------------------
    # Agent wrapper
    # ------------------------------------------------------------------

    def _agent_generate(self, user_prompt: str) -> str:
        """Send *user_prompt* to the agent and return the raw text reply."""
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
        if key not in self._prompt_cache:
            self._prompt_cache[key] = self._agent_generate(prompt)
        return self._prompt_cache[key]

    def _ask_parse(self, user_prompt: str) -> Optional[ParsedHeuristic]:
        """Send prompt to LLM, retry + reformat, return parsed heuristic or None."""
        last_text = ""
        for _ in range(self.cfg.llm_retries + 1):
            text = self._llm_cached(user_prompt)
            last_text = text

            parsed = parse_thought_code(text)
            if parsed:
                return parsed

            for _ in range(self.cfg.reformat_retries):
                text2 = self._llm_cached(_reformat_prompt(text))
                last_text = text2
                parsed2 = parse_thought_code(text2)
                if parsed2:
                    return parsed2

            if self.cfg.allow_code_only_fallback:
                parsed3 = parse_code_only_fallback(last_text)
                if parsed3:
                    return parsed3

        return None

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _tournament(self, pop: list[Individual]) -> Individual:
        k = min(self.cfg.tournament_k, len(pop))
        return max(self.rng.sample(pop, k=k), key=lambda x: x.fitness)

    def _select_elites(self, pop: list[Individual]) -> list[Individual]:
        return sorted(pop, key=lambda x: x.fitness, reverse=True)[: self.cfg.elite_k]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _budget_key(self, b: Budget) -> str:
        return f"{int(b.max_iters)}|{float(b.max_time_s):.6f}"

    def _stage_indices(
        self, policy_key: str, stage_idx: int, n: int, total: int
    ) -> list[int]:
        cache_key = (policy_key, stage_idx)
        if cache_key in self._stage_index_cache:
            cached = self._stage_index_cache[cache_key]
            return cached[:n] if len(cached) >= n else cached

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
        hooks_obj: Any,
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
            return (
                float(trace.best_obj),
                bool(trace.best_feasible),
                float(trace.elapsed_s),
                int(trace.failures),
            )
        except Exception:
            sense = (self.eval_spec.problem.sense or "min").strip().lower()
            fallback_obj = float("inf") if sense == "min" else float("-inf")
            return fallback_obj, False, float(budget.max_time_s), 1

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
        insts = self.eval_spec.instances_train
        if not insts:
            ind.fitness = float("-inf")
            ind.eval_error = "no training instances"
            ind.metrics = {"eval_error": "no training instances"}
            return ind

        stages = (
            self.cfg.stage_budgets
            if self.cfg.progressive
            else (Budget(max_iters=220, max_time_s=0.28),)
        )
        stage_ns = (
            self.cfg.stage_instances if self.cfg.progressive else (min(len(insts), 8),)
        )

        last_fit = float("-inf")
        last_metrics: dict[str, Any] = {}
        total_insts = len(insts)

        for stage_idx, (budget, n_inst) in enumerate(zip(stages, stage_ns)):
            n = min(int(n_inst), total_insts)
            cache_key = (policy_key, stage_idx, self._budget_key(budget), n)

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
                mw = self.cfg.max_workers if self.cfg.max_workers > 0 else min(32, n)
                with ThreadPoolExecutor(max_workers=mw) as ex:
                    futs = [
                        ex.submit(
                            self._eval_one_instance,
                            policy_key,
                            stage_idx,
                            j,
                            insts[ii],
                            hooks_obj,
                            budget,
                        )
                        for j, ii in enumerate(idxs)
                    ]
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
                    "budget": {
                        "max_iters": budget.max_iters,
                        "max_time_s": budget.max_time_s,
                    },
                    "n_instances": n,
                    "hook_impl": hooks_obj.implemented(),
                    "failures": int(failures_total),
                    "metaheuristic": getattr(
                        self.eval_spec.metaheuristic, "name", "unknown"
                    ),
                }
            )

            out = EvalOutput(fitness=float(fit), metrics=dict(metrics))
            if self.cfg.cache_enabled:
                self._eval_cache[cache_key] = out

            last_fit, last_metrics = float(fit), dict(metrics)

            if (not math.isfinite(last_fit)) or (
                last_metrics.get("feas_rate", 0.0) < 0.05
            ):
                break  # prune failing individuals early

        ind.fitness = float(last_fit)
        ind.metrics = dict(last_metrics)
        ind.eval_error = None if math.isfinite(ind.fitness) else "non-finite fitness"
        return ind

    # ------------------------------------------------------------------
    # Individual factory
    # ------------------------------------------------------------------

    def _make_ind(
        self, generation: int, thought: str, code: str, parents: list[str]
    ) -> Individual:
        return Individual(
            id=str(uuid.uuid4())[:8],
            generation=generation,
            thought=thought.strip(),
            code=code.strip(),
            parents=list(parents),
            meta={},
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> list[Individual]:
        """Generate and evaluate the initial population."""
        pop: list[Individual] = []
        while len(pop) < self.cfg.pop_size:
            usr = _seed_prompt(self.task_desc, self.api_desc, baseline=self.baseline)
            parsed = self._ask_parse(usr)
            if not parsed:
                continue
            ind = self._make_ind(0, parsed.thought, parsed.code, parents=[])
            pop.append(self._eval_individual(ind))
        return pop

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def evolve(self) -> EoHResult:
        """Run the full EoH loop and return the best result."""
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
                    "top_fitness": sorted(
                        (x.fitness for x in population), reverse=True
                    )[: min(10, len(population))],
                    "best_hooks": best.metrics.get("hook_impl", []),
                    "best_feas": best.metrics.get("feas_rate"),
                    "best_obj": best.metrics.get("mean_obj"),
                    "metaheuristic": best.metrics.get("metaheuristic"),
                }
            )

        snapshot(0, pop)

        for gen in range(1, self.cfg.generations + 1):
            elites = self._select_elites(pop)
            hof.extend(elites)
            hof = sorted(hof, key=lambda x: x.fitness, reverse=True)[
                : max(15, self.cfg.elite_k * 6)
            ]

            children: list[Individual] = []
            while len(children) < self.cfg.offspring_per_gen:
                if self.cfg.strategy == "e2":
                    parsed = self._ask_parse(
                        _e2_improve_prompt(
                            self.task_desc,
                            self.api_desc,
                            exemplars=elites,
                            cfg=self.cfg,
                        )
                    )
                    if not parsed:
                        continue
                    child = self._make_ind(
                        gen, parsed.thought, parsed.code, parents=[e.id for e in elites]
                    )
                    children.append(self._eval_individual(child))
                    continue

                if self.rng.random() < self.cfg.crossover_rate and len(pop) >= 2:
                    p1 = self._tournament(pop)
                    p2 = self._tournament(pop)
                    if p2.id == p1.id:
                        p2 = self.rng.choice(pop)
                    parsed = self._ask_parse(
                        _e1_crossover_prompt(
                            self.task_desc, self.api_desc, p1, p2, cfg=self.cfg
                        )
                    )
                    if not parsed:
                        continue
                    child = self._make_ind(
                        gen, parsed.thought, parsed.code, parents=[p1.id, p2.id]
                    )
                else:
                    p = self._tournament(pop)
                    parsed = self._ask_parse(
                        _e1_mutation_prompt(
                            self.task_desc, self.api_desc, p, cfg=self.cfg
                        )
                    )
                    if not parsed:
                        continue
                    child = self._make_ind(
                        gen, parsed.thought, parsed.code, parents=[p.id]
                    )

                children.append(self._eval_individual(child))

            children_sorted = sorted(children, key=lambda x: x.fitness, reverse=True)
            pop = elites + children_sorted[: max(0, self.cfg.pop_size - len(elites))]
            snapshot(gen, pop)

        best = max(hof + pop, key=lambda x: x.fitness)
        return EoHResult(best=best, hall_of_fame=hof, population=pop, history=history)
