from __future__ import annotations

from typing import Any

from .auto_cot import AutoCoTReasoner
from .base import Reasoner
from .best_of_n import BestOfNReasoner
from .in_context_cot import InContextCoTReasoner
from .reflexion import ReflexionReasoner
from .self_consistency import SelfConsistencyReasoner
from .ssr import SSRReasoner
from .tot import ToTReasoner

REASONER_REGISTRY: dict[str, type[Reasoner]] = {
    "ssr": SSRReasoner,
    "tot": ToTReasoner,
    "reflexion": ReflexionReasoner,
    "self_consistency": SelfConsistencyReasoner,
    "sc": SelfConsistencyReasoner,
    "best_of_n": BestOfNReasoner,
    "auto_cot": AutoCoTReasoner,
    "in_context_cot": InContextCoTReasoner,
}


def get_reasoner(name: str, llm_backend: Any, **config: Any) -> Reasoner:
    cls = REASONER_REGISTRY.get(name.lower())
    if not cls:
        raise ValueError(
            f"Unknown reasoner '{name}'. Registered: {list(REASONER_REGISTRY.keys())}"
        )
    return cls(llm_backend, **config)


__all__ = ["REASONER_REGISTRY", "get_reasoner"]
