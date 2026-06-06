# agent_core/thinking.py
from __future__ import annotations

from typing import Any, Optional

from .config import ThinkingConfig

# FIXED: import prompt utilities from prompt.py
from .prompt import Prompt, get_prompt

# Reasoners are opt-in — only imported when actually needed to keep
# the common path (native thinking) fast and dependency-light.
_REASONERS_MODULE: Any = None
_REASONERS_AVAILABLE = False


def _get_reasoners_module() -> Any:
    global _REASONERS_MODULE, _REASONERS_AVAILABLE
    if _REASONERS_MODULE is None and _REASONERS_AVAILABLE is False:
        try:
            import src.reasoners as _reasoners
            _REASONERS_MODULE = _reasoners
            _REASONERS_AVAILABLE = True
        except ImportError:
            _REASONERS_MODULE = None
            _REASONERS_AVAILABLE = False
    return _REASONERS_MODULE

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
        # Instantiate reasoner if configured (lazy — only loads when
        # a reasoner is explicitly requested).
        # -----------------------------------------------------------
        self.reasoner: Reasoner | None = None
        if self.cfg.reasoner:
            reasoners_mod = _get_reasoners_module()
            if reasoners_mod is None:
                raise ImportError(
                    f"Reasoner '{self.cfg.reasoner}' requested but the "
                    "reasoners package is not installed. "
                    "Install it to enable structured reasoning."
                )
            reasoner_kwargs = dict(self.cfg.reasoner_kwargs or {})
            self.reasoner = reasoners_mod.get_reasoner(
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
            # Reasoner is opt-in; if not configured, pass through unchanged
            if self.reasoner:
                return self._do_reasoner(query, initial)
            return initial or ""

        elif order == "prompt->reasoner":
            x = self._do_prompt(query, initial)
            if self.reasoner:
                return self._do_reasoner(query, x)
            return x

        elif order == "reasoner->prompt":
            x = self._do_reasoner(query, initial) if self.reasoner else initial or ""
            return self._do_prompt(query, x)

        elif order == "prompt->reasoner->prompt":
            x1 = self._do_prompt(query, initial)
            x2 = self._do_reasoner(query, x1) if self.reasoner else x1
            return self._do_prompt(query, x2)

        else:
            raise ValueError(f"Invalid ThinkingConfig.order: {order}")

    # --------------------------------------------------------------------------
    # Sub-steps
    # --------------------------------------------------------------------------
    def _do_prompt(self, query: str, initial: Optional[str]) -> str:
        if not self.prompt:
            # No prompt configured — pass through unchanged
            return initial or ""
        return self.prompt.run(query, initial)

    def _do_reasoner(self, query: str, initial: Optional[str]) -> str:
        if not self.reasoner:
            # Reasoner is opt-in; return initial unchanged
            return initial or ""
        from .reasoners import Reasoner

        trace = self.reasoner.solve(query, initial_solution=initial)
        return trace.answer or trace.reasoning
