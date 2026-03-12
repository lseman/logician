"""Turn classification: intent + domain group detection (no LLM calls)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnClassification:
    intent: str  # "social" | "informational" | "execution" | "design"
    domain_groups: set[str]  # e.g. {"timeseries", "academic"}


# Intent: first match wins (order matters)
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "social",
        [
            "hello",
            "hi ",
            "hey ",
            "thanks",
            "thank you",
            "thx",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what's up",
            "whats up",
        ],
    ),
    (
        "design",
        [
            "design ",
            "architect",
            "how should i structure",
            "how should we structure",
            "propose ",
            "trade-off",
            "tradeoff",
            "which approach",
            "what approach",
        ],
    ),
    (
        "informational",
        [
            "explain ",
            "what is ",
            "what are ",
            "how does ",
            "how do ",
            "describe ",
            "why is ",
            "why does ",
            "tell me about",
        ],
    ),
    # "execution" is the default — no keyword needed
]

_DOMAIN_PATTERNS: dict[str, list[str]] = {
    "timeseries": [
        "reservoir",
        "forecast",
        "ons",
        "time series",
        "hydroelectric",
        "energy data",
        "timeseries",
        "time-series",
    ],
    "academic": [
        "paper",
        "citation",
        "s2 ",
        "ieee",
        "openalex",
        "literature",
        "systematic review",
        "related work",
        "arxiv",
    ],
    "rag": [
        "ingest",
        "retrieve",
        "embed",
        "knowledge base",
        "vector store",
        "chromadb",
        "rag ",
    ],
    "svg": [" svg", "diagram", " chart", "visualize", "visualise"],
}


def classify_turn(content: str) -> TurnClassification:
    """Classify a user message. Pure function — no LLM calls."""
    lower = content.lower()
    intent = _match_intent(lower)
    domain_groups = _match_domains(lower)
    return TurnClassification(intent=intent, domain_groups=domain_groups)


def _match_intent(lower: str) -> str:
    for intent, keywords in _INTENT_PATTERNS:
        if any(kw in lower for kw in keywords):
            return intent
    return "execution"


def _match_domains(lower: str) -> set[str]:
    return {
        group
        for group, keywords in _DOMAIN_PATTERNS.items()
        if any(kw in lower for kw in keywords)
    }
