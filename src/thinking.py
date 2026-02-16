# agent_core/thinking.py
from __future__ import annotations

from typing import Any, Optional

from .config import ThinkingConfig

# FIXED: import prompt utilities from prompt.py
from .prompt import get_prompt, Prompt

# FIXED: import reasoner utilities from reasoner.py
from .reasoner import get_reasoner, Reasoner


# ==============================================================================
# ThinkingStrategy — orchestrates (Prompt, Reasoner) pipelines
# ==============================================================================

class ThinkingStrategy:
    """
    Orchestrates:
        - prompt-only
        - reasoner-only
        - prompt → reasoner
        - reasoner → prompt
        - prompt → reasoner → prompt
    across multiple refinement rounds.
    """

    _SUPPORTED_ORDERS = {
        "prompt",
        "reasoner",
        "prompt->reasoner",
        "reasoner->prompt",
        "prompt->reasoner->prompt",
    }

    def __init__(self, llm_backend: Any, config: ThinkingConfig):
        self.llm = llm_backend
        self.cfg = config

        # -----------------------------------------------------------
        # Instantiate prompt if configured
        # -----------------------------------------------------------
        self.prompt: Prompt | None = None
        if self.cfg.prompt:
            self.prompt = get_prompt(
                self.cfg.prompt,
                llm_backend,
                temperature=self.cfg.prompt_temperature,
                max_tokens=self.cfg.max_tokens,
            )

        # -----------------------------------------------------------
        # Instantiate reasoner if configured
        # -----------------------------------------------------------
        self.reasoner: Reasoner | None = None
        if self.cfg.reasoner:
            reasoner_kwargs = dict(self.cfg.reasoner_kwargs or {})
            self.reasoner = get_reasoner(
                self.cfg.reasoner,
                llm_backend,
                temperature=self.cfg.reasoner_temperature,
                max_tokens=self.cfg.max_tokens,
                **reasoner_kwargs,
            )

        # -----------------------------------------------------------
        # Validate pipeline order
        # -----------------------------------------------------------
        if self.cfg.order not in self._SUPPORTED_ORDERS:
            raise ValueError(
                f"ThinkingConfig.order={self.cfg.order!r} "
                f"is not supported. Must be one of {sorted(self._SUPPORTED_ORDERS)}"
            )

    # --------------------------------------------------------------------------
    # Main public entry
    # --------------------------------------------------------------------------
    def run(self, query: str, initial: Optional[str] = None) -> str:
        output = initial
        rounds = max(1, self.cfg.max_rounds)

        for _ in range(rounds):
            output = self._run_once(query, output)

        return output

    # --------------------------------------------------------------------------
    # Run a single configured pipeline pass
    # --------------------------------------------------------------------------
    def _run_once(self, query: str, initial: Optional[str]) -> str:
        order = self.cfg.order

        if order == "prompt":
            return self._do_prompt(query, initial)

        elif order == "reasoner":
            return self._do_reasoner(query, initial)

        elif order == "prompt->reasoner":
            x = self._do_prompt(query, initial)
            return self._do_reasoner(query, x)

        elif order == "reasoner->prompt":
            x = self._do_reasoner(query, initial)
            return self._do_prompt(query, x)

        elif order == "prompt->reasoner->prompt":
            x1 = self._do_prompt(query, initial)
            x2 = self._do_reasoner(query, x1)
            return self._do_prompt(query, x2)

        else:
            raise ValueError(f"Invalid ThinkingConfig.order: {order}")

    # --------------------------------------------------------------------------
    # Sub-steps
    # --------------------------------------------------------------------------
    def _do_prompt(self, query: str, initial: Optional[str]) -> str:
        if not self.prompt:
            raise RuntimeError(
                "ThinkingConfig.order requires a prompt, but no prompt is configured."
            )
        return self.prompt.run(query, initial)

    def _do_reasoner(self, query: str, initial: Optional[str]) -> str:
        if not self.reasoner:
            raise RuntimeError(
                "ThinkingConfig.order requires a reasoner, but no reasoner is configured."
            )
        trace = self.reasoner.solve(query, initial_solution=initial)
        return trace.answer or trace.reasoning
