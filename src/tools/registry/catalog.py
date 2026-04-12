from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Sequence, TypedDict

from ..runtime import SkillCard, SkillSelection, ToolParameter
from src.skills.skill_manifest import split_frontmatter

try:
    from markdown_it import MarkdownIt

    HAS_MARKDOWN_IT = True
except ImportError:
    MarkdownIt = None  # type: ignore
    HAS_MARKDOWN_IT = False

try:
    from rapidfuzz import fuzz as _rf_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    _rf_fuzz = None  # type: ignore
    HAS_RAPIDFUZZ = False

_SENTENCE_TRANSFORMER_TYPE: Any | None = None
_SENTENCE_TRANSFORMER_IMPORT_ATTEMPTED = False
_SENTENCE_TRANSFORMER_IMPORT_ERROR = ""


def _lazy_import_sentence_transformer() -> Any | None:
    global _SENTENCE_TRANSFORMER_TYPE
    global _SENTENCE_TRANSFORMER_IMPORT_ATTEMPTED
    global _SENTENCE_TRANSFORMER_IMPORT_ERROR

    if _SENTENCE_TRANSFORMER_IMPORT_ATTEMPTED:
        return _SENTENCE_TRANSFORMER_TYPE

    _SENTENCE_TRANSFORMER_IMPORT_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        _SENTENCE_TRANSFORMER_TYPE = None
        _SENTENCE_TRANSFORMER_IMPORT_ERROR = str(exc) or exc.__class__.__name__
        return None

    _SENTENCE_TRANSFORMER_TYPE = SentenceTransformer
    _SENTENCE_TRANSFORMER_IMPORT_ERROR = ""
    return _SENTENCE_TRANSFORMER_TYPE


_SKILL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
    "your",
}

_TOOL_BACKED_DEFAULT_TRIGGERS: dict[str, list[str]] = {
    "data_loading": ["load csv", "load dataset", "read data file"],
    "preprocessing": ["clean data", "transform series", "prepare dataset"],
    "analysis": ["analyze series", "diagnostics", "detect anomalies"],
    "forecasting": ["forecast the next", "predict future values"],
    "plotting": ["plot diagnostics", "visualize series"],
    "svg_viz": [
        "create svg",
        "generate svg",
        "draw diagram",
        "dependency graph",
        "architecture diagram",
        "visual representation",
    ],
}
_TOOL_BACKED_DEFAULT_ANTI_TRIGGERS: dict[str, list[str]] = {
    "forecasting": ["plot diagnostics only"],
}
_TOOL_BACKED_DEFAULT_NEXT_SKILLS: dict[str, list[str]] = {
    "data_loading": ["preprocessing", "analysis", "forecasting"],
    "preprocessing": ["analysis", "forecasting"],
    "analysis": ["forecasting"],
}
_TOOL_BACKED_DEFAULT_WHEN_NOT_TO_USE = "another specialized skill is a clearer match"
_FORECASTING_PREFERRED_TOOL_ORDER = (
    "suggest_models",
    "stat_forecast",
    "neural_forecast",
)

_DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_ROUTING_MIN_SCORE = 0.16
_DEFAULT_RECALL_K = 24
_DEFAULT_USAGE_HALF_LIFE_HOURS = 72.0
_DEFAULT_ROUTING_WEIGHTS: dict[str, float] = {
    "bm25": 0.34,
    "dense": 0.22,
    "fuzzy": 0.08,
    "coverage": 0.08,
    "name_or_alias": 0.10,
    "trigger_hit": 0.07,
    "tool_hit": 0.04,
    "example": 0.05,
    "usage_bias": 0.05,
    "recency_bias": 0.04,
    "anti_trigger_penalty": 0.20,
    "negation_penalty": 0.30,
    "guidance_discount": 0.82,
}


@contextlib.contextmanager
def _suppress_fd_output() -> Any:
    """Temporarily silence native library stdout/stderr during model load."""
    old_fd1 = os.dup(1)
    old_fd2 = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(old_fd1, 1)
        os.close(old_fd1)
        os.dup2(old_fd2, 2)
        os.close(old_fd2)


def _skill_token_variants(token: str) -> list[str]:
    """Return lightweight morphological variants for fuzzy skill matching."""
    t = (token or "").strip().lower()
    if len(t) < 3:
        return [t] if t else []

    variants: list[str] = [t]
    seen: set[str] = {t}

    def _add(v: str) -> None:
        v = v.strip().lower()
        if len(v) < 3 or v in seen:
            return
        seen.add(v)
        variants.append(v)

    if t.endswith("ing") and len(t) > 5:
        base = t[:-3]
        _add(base)
        # running -> run, brainstorming -> brainstorm
        if len(base) > 3 and base[-1] == base[-2]:
            _add(base[:-1])
    if t.endswith("ied") and len(t) > 5:
        _add(t[:-3] + "y")
    if t.endswith("ed") and len(t) > 4:
        _add(t[:-2])
    if t.endswith("ies") and len(t) > 5:
        _add(t[:-3] + "y")
    if t.endswith("es") and len(t) > 4:
        _add(t[:-2])
    if t.endswith("s") and len(t) > 4:
        _add(t[:-1])

    return variants


def _skill_tokens(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower()):
        for tok in _skill_token_variants(raw):
            if tok in _SKILL_STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


@dataclass
class SkillScoreBreakdown:
    total: float
    contributions: dict[str, float]


class SkillScoreFeature:
    name = "feature"
    weight = 1.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        return 0.0

    def score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        return float(self.raw_score(query, skill, ctx, catalog)) * float(self.weight)


class NameOrIdSubstringFeature(SkillScoreFeature):
    name = "name_or_id_substring"
    weight = 5.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_l = str(ctx.get("query_l", ""))
        if not query_l:
            return 0.0
        if skill.id.replace("_", " ") in query_l or skill.name.lower() in query_l:
            return 1.0
        return 0.0


class ExactToolHitsFeature(SkillScoreFeature):
    name = "exact_tool_hits"
    weight = 7.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_l = str(ctx.get("query_l", ""))
        if not query_l:
            ctx["exact_tool_hits_count"] = 0
            return 0.0
        hits = sum(1 for tool_name in skill.tool_names if tool_name.lower() in query_l)
        ctx["exact_tool_hits_count"] = hits
        return float(hits)


class AliasHitsFeature(SkillScoreFeature):
    name = "alias_hits"
    weight = 3.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        if not query_tokens:
            ctx["alias_hits_count"] = 0
            return 0.0
        aliases_l = {alias.lower() for alias in skill.aliases}
        hits = query_tokens.intersection(aliases_l)
        ctx["alias_hits_count"] = len(hits)
        return float(len(hits))


class TriggerSubstringFeature(SkillScoreFeature):
    name = "trigger_substring_hits"
    weight = 5.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_l = str(ctx.get("query_l", ""))
        if not query_l:
            ctx["trigger_phrase_hits_count"] = 0
            return 0.0
        hits = sum(1 for trigger in skill.triggers if trigger and trigger.lower() in query_l)
        ctx["trigger_phrase_hits_count"] = hits
        return float(hits)


class TriggerTokenOverlapFeature(SkillScoreFeature):
    name = "trigger_token_overlap"
    weight = 7.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        if not query_tokens:
            return 0.0
        total_overlap = 0.0
        for trigger in skill.triggers:
            trigger_tokens = set(_skill_tokens(trigger))
            if not trigger_tokens:
                continue
            overlap = len(query_tokens.intersection(trigger_tokens))
            if overlap:
                total_overlap += overlap / len(trigger_tokens)
        return total_overlap


class DescriptionCoverageFeature(SkillScoreFeature):
    name = "description_coverage"
    weight = 10.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        if not query_tokens:
            ctx["description_overlap_count"] = 0
            return 0.0
        desc_tokens = set(_skill_tokens(skill.description))
        if not desc_tokens:
            ctx["description_overlap_count"] = 0
            return 0.0
        overlap = len(query_tokens.intersection(desc_tokens))
        ctx["description_overlap_count"] = overlap
        if not overlap:
            return 0.0
        return overlap / len(desc_tokens)


class DescriptionPrecisionFeature(SkillScoreFeature):
    name = "description_precision"
    weight = 6.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        if not query_tokens:
            return 0.0
        overlap = int(ctx.get("description_overlap_count", 0))
        if not overlap:
            return 0.0
        return overlap / len(query_tokens)


class ProfileCoverageFeature(SkillScoreFeature):
    name = "profile_query_coverage"
    weight = 6.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        profile_tokens = set(ctx.get("profile_tokens", set()))
        if not query_tokens or not profile_tokens:
            ctx["query_coverage"] = 0.0
            return 0.0
        overlap = len(query_tokens.intersection(profile_tokens))
        coverage = overlap / max(1, len(query_tokens))
        ctx["query_coverage"] = coverage
        return coverage


class ProfileFuzzySimilarityFeature(SkillScoreFeature):
    name = "profile_fuzzy_similarity"
    weight = 4.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_l = str(ctx.get("query_l", ""))
        profile_text = str(ctx.get("profile_text", ""))
        if not query_l or not profile_text:
            ctx["fuzzy_ratio"] = 0.0
            return 0.0
        ratio = catalog._fuzzy_similarity(query_l, profile_text)
        ctx["fuzzy_ratio"] = ratio
        return ratio


class AntiTriggerPenaltyFeature(SkillScoreFeature):
    name = "anti_trigger_penalty"
    weight = -6.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_l = str(ctx.get("query_l", ""))
        if not query_l:
            return 0.0
        hits = sum(
            1
            for anti_trigger in skill.anti_triggers
            if anti_trigger and anti_trigger.lower() in query_l
        )
        return float(hits)


class KeywordHitsFeature(SkillScoreFeature):
    name = "keyword_hits"
    weight = 1.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        if not query_tokens:
            ctx["keyword_hits_count"] = 0
            return 0.0
        hits = query_tokens.intersection(set(skill.keywords))
        ctx["keyword_hits_count"] = len(hits)
        return float(len(hits))


class ExampleQueryFeature(SkillScoreFeature):
    name = "example_query_score"
    weight = 1.0

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        query_tokens = set(ctx.get("query_tokens", set()))
        query_l = str(ctx.get("query_l", ""))
        return catalog._example_query_score(query_tokens, query_l, skill)


class NegatedChunkPenaltyFeature(SkillScoreFeature):
    name = "negated_chunk_penalty"
    weight = -2.5

    def raw_score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> float:
        negated_chunks = list(ctx.get("negated_chunks", []))
        if not negated_chunks:
            return 0.0
        profile_tokens = set(ctx.get("profile_tokens", set())).union(set(skill.keywords))
        penalty_units = 0.0
        for chunk in negated_chunks:
            chunk_tokens = set(_skill_tokens(chunk))
            if not chunk_tokens:
                continue
            overlap = len(chunk_tokens.intersection(profile_tokens))
            if overlap:
                penalty_units += overlap
        return penalty_units


class CompositeSkillScorer:
    def __init__(self, features: Sequence[SkillScoreFeature]) -> None:
        self.features = list(features)

    def score(
        self, query: str, skill: SkillCard, ctx: dict[str, Any], catalog: "SkillCatalog"
    ) -> SkillScoreBreakdown:
        total = 0.0
        contributions: dict[str, float] = {}
        for feature in self.features:
            delta = feature.score(query, skill, ctx, catalog)
            if not delta:
                continue
            total += delta
            contributions[feature.name] = contributions.get(feature.name, 0.0) + float(delta)
        return SkillScoreBreakdown(total=total, contributions=contributions)


class MarkdownSection(TypedDict):
    heading: str
    body: str


class BootstrapSection(TypedDict):
    name: str
    content: str
    code: str


class ToolSection(TypedDict):
    name: str
    content: str
    code: str
    description: str
    parameters: list[ToolParameter]
    source_path: str
    skill_id: str


class SkillManifest(TypedDict, total=False):
    name: str
    summary: str
    triggers: list[str]
    anti_triggers: list[str]
    aliases: list[str]
    preferred_tools: list[str]
    example_queries: list[str]
    when_not_to_use: list[str]
    next_skills: list[str]
    paths: list[str]


class CodeBlockExtractor:
    def __init__(self) -> None:
        self.in_block = False
        self.code_lines: List[str] = []

    def process_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith("```python"):
            self.in_block = True
        elif stripped == "```" and self.in_block:
            self.in_block = False
        elif self.in_block:
            self.code_lines.append(line)

    def get_code(self) -> str:
        return "\n".join(self.code_lines)


class SkillCatalog:
    # Bump when cache schema changes to force invalidation.
    _CACHE_VERSION = 11

    def __init__(self, *, skills_md_path: Path, skills_dir_path: Path, log: Any) -> None:
        self.skills_md_path = skills_md_path
        self.skills_dir_path = skills_dir_path
        self._log = log
        self._active_lazy_skill_groups: set[str] = set()
        self._skills: dict[str, SkillCard] = {}
        self._tool_docs: dict[str, str] = {}
        self._all_tool_sections: list[ToolSection] = []
        # Startup-built routing index for fast and consistent skill relevance scoring.
        self._routing_profiles: dict[str, str] = {}
        self._routing_tokens: dict[str, set[str]] = {}
        self._routing_term_freqs: dict[str, dict[str, int]] = {}
        self._routing_doc_lengths: dict[str, int] = {}
        self._bm25_doc_freq: dict[str, int] = {}
        self._bm25_avg_doc_len: float = 1.0
        self._skill_scorer = CompositeSkillScorer(
            [
                NameOrIdSubstringFeature(),
                ExactToolHitsFeature(),
                AliasHitsFeature(),
                TriggerSubstringFeature(),
                TriggerTokenOverlapFeature(),
                DescriptionCoverageFeature(),
                DescriptionPrecisionFeature(),
                ProfileCoverageFeature(),
                ProfileFuzzySimilarityFeature(),
                AntiTriggerPenaltyFeature(),
                KeywordHitsFeature(),
                ExampleQueryFeature(),
                NegatedChunkPenaltyFeature(),
            ]
        )
        self._routing_weights = self._resolve_routing_weights()
        self._routing_recall_k = self._resolve_recall_k()
        self._routing_min_score = self._resolve_min_score()
        self._usage_half_life_s = self._resolve_usage_half_life_s()
        self._dense_enabled = self._resolve_dense_enabled()
        self._dense_model_name = str(
            os.getenv("AGENT_SKILL_DENSE_MODEL") or _DEFAULT_DENSE_MODEL
        ).strip()
        self._dense_model: Any | None = None
        self._dense_model_disabled_reason = ""
        self._dense_embedding_cache: dict[str, tuple[float, ...]] = {}
        self._skill_usage_stats: dict[str, dict[str, float]] = {}
        # Last per-skill score breakdown for debugging/tuning.
        self._last_skill_score_breakdown: dict[str, dict[str, Any]] = {}
        self._python_skill_metadata: dict[str, dict[str, Any]] = {}

    @property
    def skills(self) -> dict[str, SkillCard]:
        return self._skills

    @property
    def tool_docs(self) -> dict[str, str]:
        return self._tool_docs

    @property
    def all_tool_sections(self) -> list[ToolSection]:
        """All parsed ToolSections from the last build. Avoids re-parsing on registration."""
        return self._all_tool_sections

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _resolve_dense_enabled(self) -> bool:
        return self._env_flag("AGENT_SKILL_DENSE_ENABLED", True)

    def _resolve_recall_k(self) -> int:
        raw = str(os.getenv("AGENT_SKILL_ROUTING_RECALL_K") or "").strip()
        if not raw:
            return _DEFAULT_RECALL_K
        try:
            return max(8, min(128, int(raw)))
        except Exception:
            return _DEFAULT_RECALL_K

    def _resolve_min_score(self) -> float:
        raw = str(os.getenv("AGENT_SKILL_ROUTING_MIN_SCORE") or "").strip()
        if not raw:
            return _DEFAULT_ROUTING_MIN_SCORE
        try:
            return max(0.0, min(1.0, float(raw)))
        except Exception:
            return _DEFAULT_ROUTING_MIN_SCORE

    def _resolve_usage_half_life_s(self) -> float:
        raw = str(os.getenv("AGENT_SKILL_USAGE_HALF_LIFE_HOURS") or "").strip()
        if not raw:
            return _DEFAULT_USAGE_HALF_LIFE_HOURS * 3600.0
        try:
            hours = max(1.0, float(raw))
            return hours * 3600.0
        except Exception:
            return _DEFAULT_USAGE_HALF_LIFE_HOURS * 3600.0

    def _resolve_routing_weights(self) -> dict[str, float]:
        weights = dict(_DEFAULT_ROUTING_WEIGHTS)
        raw = os.getenv("AGENT_SKILL_ROUTING_WEIGHTS")
        if not raw:
            return weights

        parsed: dict[str, Any] | None = None
        text = str(raw).strip()
        if not text:
            return weights

        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict):
                parsed = maybe_json
        except Exception:
            parsed = None

        if parsed is None:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            parsed = {}
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()

        for key, value in parsed.items():
            k = str(key or "").strip()
            if k not in weights:
                continue
            try:
                weights[k] = float(value)
            except Exception:
                continue
        return weights

    def _cache_path(self) -> Path:
        return self.skills_dir_path / ".skills_catalog_cache.json"

    @staticmethod
    def _normalize_lazy_skill_group_name(value: str) -> str:
        text = str(value or "").strip().lower()
        text = text.strip("/").replace("-", "_").replace(" ", "_")
        if text.startswith("lazy_"):
            text = text[len("lazy_") :]
        text = "".join(ch for ch in text if ch.isalnum() or ch == "_").strip("_")
        return text

    def set_active_lazy_skill_groups(self, groups: Sequence[str] | set[str]) -> None:
        self._active_lazy_skill_groups = {
            group
            for group in (self._normalize_lazy_skill_group_name(item) for item in (groups or []))
            if group
        }

    def _is_lazy_skill_group_dir_name(self, name: str) -> bool:
        return str(name or "").strip().startswith("lazy_")

    def _is_lazy_skill_group_active(self, name: str) -> bool:
        group = self._normalize_lazy_skill_group_name(name)
        return bool(group) and group in self._active_lazy_skill_groups

    def _catalog_fingerprint(self, skill_md_paths: list[Path]) -> str:
        """SHA1 over sorted (path, mtime) pairs + cache version."""
        parts = [f"v{self._CACHE_VERSION}"]
        for p in sorted(skill_md_paths):
            try:
                parts.append(f"{p}:{p.stat().st_mtime_ns}")
            except OSError:
                parts.append(str(p))
        return hashlib.sha1("\n".join(parts).encode()).hexdigest()

    def _load_from_cache(self, fingerprint: str) -> bool:
        """Try to restore catalog from disk cache. Returns True on success."""
        cache_file = self._cache_path()
        if not cache_file.is_file():
            return False
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if raw.get("fingerprint") != fingerprint:
                return False
            cards: dict[str, SkillCard] = {}
            for skill_id, card_dict in raw.get("skills", {}).items():
                cards[skill_id] = SkillCard(**card_dict)
            profiles: dict[str, str] = raw.get("routing_profiles", {})
            tokens: dict[str, set[str]] = {
                k: set(v) for k, v in raw.get("routing_tokens", {}).items()
            }
            self._skills = cards
            # Rebuild indexes to keep retrieval structures compatible with
            # the current scoring implementation even when cache schema grows.
            if profiles and tokens:
                self._routing_profiles = profiles
                self._routing_tokens = tokens
            self._build_routing_index(cards)
            self._log.info(
                "Skill catalog loaded from cache: %d skills (fingerprint=%s…)",
                len(cards),
                fingerprint[:8],
            )
            return True
        except Exception as exc:
            self._log.warning("Skill catalog cache load failed (%s), rebuilding.", exc)
            return False

    def _save_to_cache(self, fingerprint: str) -> None:
        """Persist current catalog to disk for fast subsequent starts."""
        cache_file = self._cache_path()
        try:
            payload = {
                "fingerprint": fingerprint,
                "skills": {sid: asdict(card) for sid, card in self._skills.items()},
                "routing_profiles": self._routing_profiles,
                "routing_tokens": {k: list(v) for k, v in self._routing_tokens.items()},
            }
            cache_file.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            self._log.info(
                "Skill catalog cached to disk: %d skills → %s",
                len(self._skills),
                cache_file.name,
            )
        except Exception as exc:
            self._log.warning("Failed to write skill catalog cache: %s", exc)

    def iter_skills_sources(self) -> list[Path]:
        """Return all skill .md files, scanning subdirectories recursively.

        Search order:
        1. skills_md_path if it is a directory (e.g. the ``skills/`` folder).
        2. A ``skills/`` sibling of the SKILLS.md file.
        3. skills_md_path itself if it is a single file (backward-compat).
        4. skills_dir_path as a final fallback.
        """

        def _collect(directory: Path) -> list[Path]:
            # Use os.walk with followlinks=True to traverse symlinked dirs like
            # 10_superpowers and 20_ralph (rglob follow_symlinks needs Python 3.13+).

            results: list[Path] = []
            for root, dirs, files in os.walk(str(directory), followlinks=True):
                root_path = Path(root)
                try:
                    rel_parts = root_path.relative_to(directory).parts
                except Exception:
                    rel_parts = ()

                if rel_parts:
                    top_level = rel_parts[0]
                    if self._is_lazy_skill_group_dir_name(
                        top_level
                    ) and not self._is_lazy_skill_group_active(top_level):
                        dirs[:] = []
                        continue
                else:
                    dirs[:] = [
                        d
                        for d in dirs
                        if not self._is_lazy_skill_group_dir_name(d)
                        or self._is_lazy_skill_group_active(d)
                    ]
                for fname in files:
                    if fname.lower().endswith(".md"):
                        results.append(root_path / fname)
            return sorted(results)

        if self.skills_md_path.is_dir():
            return _collect(self.skills_md_path)

        sibling_dir = self.skills_md_path.parent / "skills"
        if sibling_dir.is_dir():
            files = _collect(sibling_dir)
            if files:
                return files

        if self.skills_md_path.is_file():
            return [self.skills_md_path]

        if self.skills_dir_path.is_dir():
            return _collect(self.skills_dir_path)

        return []

    def read_skill_source_contents(
        self, sources: Sequence[Path] | None = None
    ) -> list[tuple[Path, str]]:
        out: list[tuple[Path, str]] = []
        failed = 0
        for src in sources or self.iter_skills_sources():
            try:
                out.append((src, src.read_text(encoding="utf-8")))
            except Exception as e:
                failed += 1
                self._log.warning("Skipping unreadable skills source %s: %s", src, e)
                continue
        if failed:
            self._log.warning(
                "Skipped %d unreadable skills source(s) while building catalog.",
                failed,
            )
        return out

    def ensure_skill_catalog(self) -> None:
        if self._skills:
            return
        sources = self.iter_skills_sources()
        if not sources:
            return
        # Fingerprint all markdown sources, because guidance cards may derive
        # routing keywords from sibling markdown references in the same folder.
        fingerprint = self._catalog_fingerprint(sources)
        if self._load_from_cache(fingerprint):
            return
        contents = self.read_skill_source_contents(sources)
        if contents:
            self.build_skill_catalog(contents, _fingerprint=fingerprint)

    def register_python_skill_metadata(
        self,
        *,
        skill_id: str,
        source_path: str,
        skill_meta: dict[str, Any],
    ) -> None:
        sid = str(skill_id or "").strip()
        if not sid or not isinstance(skill_meta, dict) or not skill_meta:
            return
        self._python_skill_metadata[sid] = {
            "source_path": str(source_path or ""),
            "skill_meta": dict(skill_meta),
        }

    def hydrate_metadata_only_python_skills(self, tools: Sequence[Any]) -> bool:
        if not self._python_skill_metadata:
            return False

        changed = False
        cards = self._skills
        available_tool_names = {
            str(getattr(tool, "name", "") or "").strip()
            for tool in tools
            if str(getattr(tool, "name", "") or "").strip()
        }

        for skill_id, payload in self._python_skill_metadata.items():
            meta_raw = payload.get("skill_meta", {})
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            source_path = str(payload.get("source_path") or "")
            card = cards.get(skill_id)
            if card is None:
                card = self._new_tool_backed_skill_card(
                    skill_id=skill_id,
                    source_path=source_path,
                    tool_desc="",
                    skill_meta=meta,
                )
                cards[skill_id] = card
                changed = True

            if self._apply_tool_backed_skill_meta(
                card=card,
                skill_id=skill_id,
                skill_meta=meta,
            ):
                changed = True

            preferred = [name for name in card.preferred_tools if name in available_tool_names]
            if preferred and card.tool_names != preferred:
                card.tool_names = list(preferred)
                changed = True

        if changed:
            self._build_routing_index(cards)
        return changed

    def _parse_superpowers_card(self, source_path: Path, raw_content: str) -> SkillCard | None:
        """Parse a Superpowers-format SKILL.md into a guidance-only SkillCard.

        These files:
        - Wrap everything in a ```skill … ``` fence.
        - Have YAML frontmatter with ``name`` and ``description`` only.
        - Contain no ``## Tool:`` sections — they are purely behavioural guidance.
        """
        manifest, body = self._split_frontmatter(raw_content)
        skill_id = self._skill_id_from_source(source_path)
        raw_name = manifest.get("name") or source_path.parent.name
        display_name = str(raw_name).replace("-", " ").replace("_", " ").title()
        description = str(manifest.get("description") or "").strip()
        summary = description or self._skill_summary(body)
        # Triggers: description sentence is the primary routing signal.
        triggers: list[str] = [description] if description else []
        triggers.extend(self._extract_use_when_clauses(body))
        triggers.extend(self._extract_labeled_items(body, "Triggers"))
        # Aliases: tokens derived from the slug name.
        raw_slug = str(raw_name).strip().lower()
        slug_tokens = re.split(r"[-_\s]+", raw_slug)
        aliases: list[str] = []
        slug_phrase = " ".join(tok for tok in slug_tokens if tok).strip()
        slug_hyphen = "-".join(tok for tok in slug_tokens if tok).strip()
        slug_underscore = "_".join(tok for tok in slug_tokens if tok).strip()
        if raw_slug:
            aliases.append(raw_slug)
        if slug_hyphen:
            aliases.append(slug_hyphen)
            aliases.append(f"superpowers:{slug_hyphen}")
            aliases.append(f"superpower:{slug_hyphen}")
            aliases.append(f"sp:{slug_hyphen}")
        if slug_underscore:
            aliases.append(slug_underscore)
        if slug_phrase:
            aliases.append(slug_phrase)
            aliases.append(f"superpowers {slug_phrase}")
            aliases.append(f"superpower {slug_phrase}")
        aliases.extend(tok for tok in slug_tokens if len(tok) > 2)
        if display_name:
            aliases.append(display_name.lower())
        aliases = list(dict.fromkeys(aliases))
        anti_triggers = self._extract_labeled_items(body, "Avoid when")
        when_not_to_use = anti_triggers[:1]
        # Playbooks: H2 section headings from the body.
        sections = self.parse_markdown_h2_sections(body)
        playbooks = [sec["heading"] for sec in sections if sec["heading"]]
        related_context = self._guidance_related_markdown_content(source_path)
        keywords = self._skill_keywords(
            summary,
            playbooks,
            [],
            triggers=triggers,
            example_queries=[],
            aliases=aliases,
            content=f"{body}\n{related_context}",
            content_max_chars=12000,
        )
        return SkillCard(
            id=skill_id,
            name=display_name,
            summary=summary,
            source_path=str(source_path),
            description=description,
            tool_names=[],  # guidance-only — no executable tools
            playbooks=playbooks,
            keywords=keywords,
            aliases=aliases,
            triggers=triggers,
            anti_triggers=anti_triggers,
            preferred_tools=[],
            example_queries=[],
            when_not_to_use=when_not_to_use,
            next_skills=[],
        )

    def _normalize_skill_slug(self, raw: str) -> str:
        text = re.sub(r"^\d+[_-]*", "", str(raw or ""))
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
        return text or "skill"

    def _skill_root_for_source(self, source_path: Path) -> Path | None:
        start = source_path if source_path.is_dir() else source_path.parent
        for candidate in (start, *start.parents):
            skill_md = candidate / "SKILL.md"
            if skill_md.is_file():
                return candidate
            try:
                if candidate == self.skills_dir_path or candidate == self.skills_md_path:
                    break
            except Exception:
                continue
        return None

    @staticmethod
    def _is_python_skill_module(path: Path) -> bool:
        return (
            path.is_file()
            and path.suffix == ".py"
            and path.name != "__init__.py"
            and not path.name.startswith("_")
        )

    def _skill_root_has_python_tools(self, skill_root: Path) -> bool:
        try:
            for path in skill_root.rglob("*.py"):
                if self._is_python_skill_module(path):
                    return True
        except Exception:
            return False
        return False

    def _parse_tool_backed_skill_card(self, source_path: Path, raw_content: str) -> SkillCard:
        manifest, body = self._split_frontmatter(raw_content)
        skill_id = self._skill_id_from_source(source_path)
        display_name = (
            str(manifest.get("name") or "").strip()
            or source_path.parent.name.replace("-", " ").replace("_", " ").title()
        )
        description = str(manifest.get("description") or "").strip()
        summary = description or self._skill_summary(body)
        playbooks = [
            sec["heading"] for sec in self.parse_markdown_h2_sections(body) if sec["heading"]
        ]
        aliases = self._skill_aliases(skill_id, manifest)
        triggers = self._manifest_list(manifest, "triggers")
        if description:
            triggers = [description, *triggers]
        triggers.extend(self._extract_use_when_clauses(body))
        anti_triggers = self._manifest_list(manifest, "anti_triggers")
        anti_triggers.extend(self._extract_labeled_items(body, "Avoid when"))
        preferred_tools = self._manifest_list(manifest, "preferred_tools")
        example_queries = self._manifest_list(manifest, "example_queries")
        when_not_to_use = self._manifest_list(manifest, "when_not_to_use")
        next_skills = self._manifest_list(manifest, "next_skills")
        paths = self._manifest_list(manifest, "paths")
        related_context = self._guidance_related_markdown_content(source_path)
        keywords = self._skill_keywords(
            summary,
            playbooks,
            [],
            triggers=triggers,
            example_queries=example_queries,
            aliases=aliases,
            content=f"{body}\n{related_context}",
            content_max_chars=12000,
        )
        return SkillCard(
            id=skill_id,
            name=display_name,
            summary=summary,
            source_path=str(source_path),
            description=description or summary,
            tool_names=[],
            playbooks=playbooks,
            keywords=keywords,
            aliases=aliases,
            triggers=list(dict.fromkeys(item for item in triggers if item)),
            anti_triggers=list(dict.fromkeys(item for item in anti_triggers if item)),
            preferred_tools=preferred_tools,
            example_queries=example_queries,
            when_not_to_use=when_not_to_use,
            next_skills=next_skills,
            paths=paths,
        )

    def _guidance_related_markdown_content(self, source_path: Path) -> str:
        """Collect bounded routing text from all markdown files in a guidance skill.

        We index file names, section headings, and compact body snippets so fuzzy
        matching can use the full skill folder (not only SKILL.md description),
        while keeping strict byte limits to avoid over-matching.
        """
        root = source_path.parent
        try:
            md_files = sorted(p for p in root.rglob("*.md") if p.is_file())
        except Exception:
            return ""

        if not md_files:
            return ""

        # Keep SKILL.md first so canonical metadata remains dominant.
        ordered: list[Path] = [source_path]
        ordered.extend(p for p in md_files if p != source_path)

        parts: list[str] = []
        total_chars = 0
        max_files = 32
        max_total_chars = 14000
        max_snippet_chars = 700

        for idx, md_path in enumerate(ordered):
            if idx >= max_files:
                break
            try:
                raw = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            _, body = self._split_frontmatter(raw)
            body = body.strip()
            if not body:
                continue

            try:
                rel = str(md_path.relative_to(root))
            except Exception:
                rel = md_path.name

            headings = [
                sec["heading"].strip()
                for sec in self.parse_markdown_h2_sections(body)
                if sec["heading"].strip()
            ]
            headings_text = " | ".join(headings[:20])
            snippet = self._compact_markdown_excerpt(body, max_chars=max_snippet_chars)
            chunk = f"{rel}\n{headings_text}\n{snippet}\n"

            if total_chars + len(chunk) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining <= 200:
                    break
                chunk = chunk[:remaining].rstrip()
            parts.append(chunk)
            total_chars += len(chunk)
            if total_chars >= max_total_chars:
                break

        return "\n".join(parts)

    def hydrate_tool_backed_skills(self, tools: Sequence[Any]) -> bool:
        """Ensure the catalog has routable skill cards for Python tool groups.

        Many local skills are authored as ``.py`` modules (not markdown files).
        This method synthesizes/updates ``SkillCard`` entries from registered
        tools so routing can target those skills directly.
        """
        changed = False
        cards = self._skills
        for tool in tools:
            skill_id = str(getattr(tool, "skill_id", "") or "").strip()
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not skill_id or not tool_name:
                continue
            if skill_id.startswith("mcp__"):
                continue

            source_path = str(getattr(tool, "source_path", "") or "")
            tool_desc = str(getattr(tool, "description", "") or "").strip()
            skill_meta_raw = getattr(tool, "skill_meta", None)
            skill_meta = skill_meta_raw if isinstance(skill_meta_raw, dict) else {}
            card = cards.get(skill_id)
            if card is None:
                card = self._new_tool_backed_skill_card(
                    skill_id=skill_id,
                    source_path=source_path,
                    tool_desc=tool_desc,
                    skill_meta=skill_meta,
                )
                cards[skill_id] = card
                changed = True

            if self._hydrate_tool_backed_card(
                card=card,
                skill_id=skill_id,
                tool_name=tool_name,
                tool_desc=tool_desc,
                skill_meta=skill_meta,
            ):
                changed = True

        if changed:
            self._build_routing_index(cards)
        return changed

    def _new_tool_backed_skill_card(
        self,
        *,
        skill_id: str,
        source_path: str,
        tool_desc: str,
        skill_meta: dict[str, Any] | None = None,
    ) -> SkillCard:
        meta = skill_meta if isinstance(skill_meta, dict) else {}
        display_name = str(meta.get("name") or "").strip() or skill_id.replace("_", " ").title()
        default_summary = f"Tool-backed skill group '{display_name}' loaded from Python modules."
        description = str(meta.get("description") or "").strip() or tool_desc or default_summary
        aliases = [skill_id.replace("_", " ")]
        aliases.extend(str(item).strip() for item in meta.get("aliases", []) if str(item).strip())
        triggers = list(_TOOL_BACKED_DEFAULT_TRIGGERS.get(skill_id, []))
        triggers.extend(str(item).strip() for item in meta.get("triggers", []) if str(item).strip())
        anti_triggers = list(_TOOL_BACKED_DEFAULT_ANTI_TRIGGERS.get(skill_id, []))
        anti_triggers.extend(
            str(item).strip() for item in meta.get("anti_triggers", []) if str(item).strip()
        )
        preferred_tools = [
            str(item).strip() for item in meta.get("preferred_tools", []) if str(item).strip()
        ]
        example_queries = [
            str(item).strip() for item in meta.get("example_queries", []) if str(item).strip()
        ]
        when_not_to_use = [
            str(item).strip() for item in meta.get("when_not_to_use", []) if str(item).strip()
        ]
        next_skills = [
            str(item).strip() for item in meta.get("next_skills", []) if str(item).strip()
        ]
        playbooks = [str(item).strip() for item in meta.get("workflow", []) if str(item).strip()]
        keywords = self._skill_keywords(
            description,
            playbooks,
            preferred_tools,
            triggers=triggers,
            example_queries=example_queries,
            aliases=aliases,
            content="\n".join([description, *playbooks]),
        )
        return SkillCard(
            id=skill_id,
            name=display_name,
            summary=description,
            source_path=source_path,
            description=description,
            tool_names=[],
            playbooks=playbooks,
            keywords=keywords,
            aliases=list(dict.fromkeys(item for item in aliases if item)),
            triggers=list(dict.fromkeys(item for item in triggers if item)),
            anti_triggers=list(dict.fromkeys(item for item in anti_triggers if item)),
            preferred_tools=preferred_tools,
            example_queries=example_queries,
            when_not_to_use=when_not_to_use or [_TOOL_BACKED_DEFAULT_WHEN_NOT_TO_USE],
            next_skills=list(
                dict.fromkeys(
                    [
                        *next_skills,
                        *list(_TOOL_BACKED_DEFAULT_NEXT_SKILLS.get(skill_id, [])),
                    ]
                )
            ),
        )

    @staticmethod
    def _append_unique_strings(target: list[str], values: Sequence[str]) -> bool:
        changed = False
        for value in values:
            if not value or value in target:
                continue
            target.append(value)
            changed = True
        return changed

    def _rebalance_forecasting_preferred_tools(self, card: SkillCard) -> bool:
        forecast_pref = [
            name for name in _FORECASTING_PREFERRED_TOOL_ORDER if name in card.tool_names
        ]
        if not forecast_pref:
            return False

        for candidate in card.preferred_tools:
            if candidate not in forecast_pref:
                forecast_pref.append(candidate)
        if card.preferred_tools == forecast_pref:
            return False
        card.preferred_tools = forecast_pref
        return True

    def _ensure_tool_backed_card_defaults(
        self,
        *,
        card: SkillCard,
        skill_id: str,
        tool_desc: str,
    ) -> bool:
        changed = False
        if (not card.summary or card.summary == "Skill-defined tool group.") and tool_desc:
            card.summary = tool_desc
            changed = True
        if not card.description and tool_desc:
            card.description = tool_desc
            changed = True

        if skill_id in _TOOL_BACKED_DEFAULT_TRIGGERS and self._append_unique_strings(
            card.triggers,
            _TOOL_BACKED_DEFAULT_TRIGGERS[skill_id],
        ):
            changed = True
        if skill_id in _TOOL_BACKED_DEFAULT_ANTI_TRIGGERS and self._append_unique_strings(
            card.anti_triggers,
            _TOOL_BACKED_DEFAULT_ANTI_TRIGGERS[skill_id],
        ):
            changed = True
        if skill_id in _TOOL_BACKED_DEFAULT_NEXT_SKILLS and self._append_unique_strings(
            card.next_skills,
            _TOOL_BACKED_DEFAULT_NEXT_SKILLS[skill_id],
        ):
            changed = True
        if not card.when_not_to_use:
            card.when_not_to_use = [_TOOL_BACKED_DEFAULT_WHEN_NOT_TO_USE]
            changed = True
        return changed

    def _hydrate_tool_backed_card(
        self,
        *,
        card: SkillCard,
        skill_id: str,
        tool_name: str,
        tool_desc: str,
        skill_meta: dict[str, Any] | None = None,
    ) -> bool:
        changed = False
        if tool_name not in card.tool_names:
            card.tool_names.append(tool_name)
            changed = True

        if tool_name.startswith("suggest_"):
            if tool_name not in card.preferred_tools:
                card.preferred_tools.insert(0, tool_name)
                changed = True
        elif not card.preferred_tools:
            card.preferred_tools = [tool_name]
            changed = True

        if skill_id == "forecasting" and self._rebalance_forecasting_preferred_tools(card):
            changed = True
        if self._ensure_tool_backed_card_defaults(
            card=card,
            skill_id=skill_id,
            tool_desc=tool_desc,
        ):
            changed = True

        desc_triggers = self._extract_labeled_items(tool_desc, "Triggers")
        if self._append_unique_strings(card.triggers, desc_triggers):
            changed = True

        desc_anti = self._extract_labeled_items(tool_desc, "Avoid when")
        if self._append_unique_strings(card.anti_triggers, desc_anti):
            changed = True
        if desc_anti and (
            not card.when_not_to_use
            or card.when_not_to_use == [_TOOL_BACKED_DEFAULT_WHEN_NOT_TO_USE]
        ):
            if card.when_not_to_use != [desc_anti[0]]:
                card.when_not_to_use = [desc_anti[0]]
                changed = True

        keywords = _skill_tokens(f"{tool_name.replace('_', ' ')} {tool_desc}")
        if self._append_unique_strings(card.keywords, keywords):
            changed = True
        if self._apply_tool_backed_skill_meta(card=card, skill_id=skill_id, skill_meta=skill_meta):
            changed = True
        return changed

    def _apply_tool_backed_skill_meta(
        self,
        *,
        card: SkillCard,
        skill_id: str,
        skill_meta: dict[str, Any] | None,
    ) -> bool:
        meta = skill_meta if isinstance(skill_meta, dict) else {}
        if not meta:
            return False

        changed = False
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        aliases = [str(item).strip() for item in meta.get("aliases", []) if str(item).strip()]
        triggers = [str(item).strip() for item in meta.get("triggers", []) if str(item).strip()]
        anti_triggers = [
            str(item).strip() for item in meta.get("anti_triggers", []) if str(item).strip()
        ]
        preferred_tools = [
            str(item).strip() for item in meta.get("preferred_tools", []) if str(item).strip()
        ]
        example_queries = [
            str(item).strip() for item in meta.get("example_queries", []) if str(item).strip()
        ]
        when_not_to_use = [
            str(item).strip() for item in meta.get("when_not_to_use", []) if str(item).strip()
        ]
        next_skills = [
            str(item).strip() for item in meta.get("next_skills", []) if str(item).strip()
        ]
        workflow = [str(item).strip() for item in meta.get("workflow", []) if str(item).strip()]

        if name and card.name != name:
            card.name = name
            changed = True
        if description and card.summary != description:
            card.summary = description
            changed = True
        if description and card.description != description:
            card.description = description
            changed = True
        if self._append_unique_strings(card.aliases, aliases):
            changed = True
        if self._append_unique_strings(card.triggers, triggers):
            changed = True
        if self._append_unique_strings(card.anti_triggers, anti_triggers):
            changed = True
        if self._append_unique_strings(card.preferred_tools, preferred_tools):
            changed = True
        if self._append_unique_strings(card.example_queries, example_queries):
            changed = True
        if self._append_unique_strings(card.next_skills, next_skills):
            changed = True
        if self._append_unique_strings(card.playbooks, workflow):
            changed = True
        if when_not_to_use and card.when_not_to_use != when_not_to_use:
            card.when_not_to_use = when_not_to_use
            changed = True

        content = "\n".join([description, *workflow, *example_queries])
        keywords = self._skill_keywords(
            description or card.summary,
            card.playbooks,
            card.tool_names,
            triggers=card.triggers,
            example_queries=card.example_queries,
            aliases=card.aliases,
            content=content,
        )
        if self._append_unique_strings(card.keywords, keywords):
            changed = True
        return changed

    def build_skill_catalog(
        self,
        skill_contents: list[tuple[Path, str]],
        *,
        _fingerprint: str | None = None,
    ) -> None:
        cards: dict[str, SkillCard] = {}
        all_sections: list[ToolSection] = []
        for source_path, source_content in skill_contents:
            if source_path.name.upper() == "SKILL.MD":
                skill_root = self._skill_root_for_source(source_path)
                if skill_root is not None and self._skill_root_has_python_tools(skill_root):
                    card = self._parse_tool_backed_skill_card(source_path, source_content)
                else:
                    card = self._parse_superpowers_card(source_path, source_content)
                if card:
                    cards[card.id] = card
                continue

            card, tool_sections = self._build_markdown_skill_card(
                source_path=source_path,
                source_content=source_content,
            )
            if card is None:
                continue
            cards[card.id] = card
            all_sections.extend(tool_sections)
            self._cache_tool_docs_from_sections(tool_sections)

        self._finalize_built_skill_catalog(
            cards=cards,
            all_sections=all_sections,
            fingerprint=_fingerprint,
        )

    def _build_markdown_skill_card(
        self, *, source_path: Path, source_content: str
    ) -> tuple[SkillCard | None, list[ToolSection]]:
        skill_id = self._skill_id_from_source(source_path)
        manifest, content = self._split_frontmatter(source_content)
        tool_sections = self.parse_tool_sections(content, source_path)
        if not tool_sections:
            # Do not treat arbitrary markdown references as standalone skills.
            # Guidance skills should be authored via SKILL.md, while regular
            # markdown cards should define at least one ## Tool: section.
            return None, []

        summary = manifest.get("summary") or self._skill_summary(content)
        description = str(manifest.get("description") or summary).strip()
        playbooks = self._skill_playbooks(content)
        tool_names = [sec["name"] for sec in tool_sections]
        aliases = self._skill_aliases(skill_id, manifest)
        triggers = self._manifest_list(manifest, "triggers")
        anti_triggers = self._manifest_list(manifest, "anti_triggers")
        preferred_tools = self._manifest_list(manifest, "preferred_tools")
        example_queries = self._manifest_list(manifest, "example_queries")
        when_not_to_use = self._manifest_list(manifest, "when_not_to_use")
        next_skills = self._manifest_list(manifest, "next_skills")
        keywords = self._skill_keywords(
            summary,
            playbooks,
            tool_names,
            triggers=triggers,
            example_queries=example_queries,
            aliases=aliases,
            content=content,
        )
        return (
            SkillCard(
                id=skill_id,
                name=manifest.get("name") or self._skill_name_from_source(source_path),
                summary=summary,
                source_path=str(source_path),
                description=description,
                tool_names=tool_names,
                playbooks=playbooks,
                keywords=keywords,
                aliases=aliases,
                triggers=triggers,
                anti_triggers=anti_triggers,
                preferred_tools=preferred_tools,
                example_queries=example_queries,
                when_not_to_use=when_not_to_use,
                next_skills=next_skills,
            ),
            tool_sections,
        )

    def _cache_tool_docs_from_sections(self, tool_sections: Sequence[ToolSection]) -> None:
        for sec in tool_sections:
            self._tool_docs[sec["name"]] = self.tool_doc_from_section(sec["content"])

    def _finalize_built_skill_catalog(
        self,
        *,
        cards: dict[str, SkillCard],
        all_sections: list[ToolSection],
        fingerprint: str | None,
    ) -> None:
        self._skills = cards
        self._all_tool_sections = all_sections
        self._build_routing_index(cards)
        self._log.info(
            "Skill routing catalog ready: skills=%d profiles=%d rapidfuzz=%s",
            len(self._skills),
            len(self._routing_profiles),
            "enabled" if HAS_RAPIDFUZZ else "disabled",
        )
        if fingerprint:
            self._save_to_cache(fingerprint)

    def _build_routing_index(self, cards: dict[str, SkillCard]) -> None:
        profiles: dict[str, str] = {}
        token_index: dict[str, set[str]] = {}
        term_freqs: dict[str, dict[str, int]] = {}
        doc_lengths: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        total_doc_len = 0
        for skill_id, card in cards.items():
            profile_parts = [
                card.name,
                card.description,
                card.summary,
                *card.playbooks,
                *card.triggers,
                *card.aliases,
                *card.example_queries,
                " ".join(card.keywords[:220]),
            ]
            profile = " ".join(p for p in profile_parts if p).strip().lower()
            tokens = _skill_tokens(profile)
            profiles[skill_id] = profile
            token_set = set(tokens)
            token_index[skill_id] = token_set
            tf = dict(Counter(tokens))
            term_freqs[skill_id] = tf
            doc_len = max(1, int(sum(tf.values())))
            doc_lengths[skill_id] = doc_len
            total_doc_len += doc_len
            doc_freq.update(token_set)
        self._routing_profiles = profiles
        self._routing_tokens = token_index
        self._routing_term_freqs = term_freqs
        self._routing_doc_lengths = doc_lengths
        self._bm25_doc_freq = dict(doc_freq)
        self._bm25_avg_doc_len = (
            float(total_doc_len) / max(1, len(doc_lengths)) if doc_lengths else 1.0
        )
        self._dense_embedding_cache.clear()

    def _fuzzy_similarity(self, query: str, profile: str) -> float:
        """Return normalized fuzzy similarity [0, 1] using RapidFuzz when available.

        Uses WRatio (weighted combination of multiple algorithms) for the best
        context-aware score, falling back to max(token_set_ratio, partial_ratio)
        if WRatio is unavailable, then to SequenceMatcher.
        """
        query_text = str(query or "")
        profile_text = str(profile or "")
        if not query_text or not profile_text:
            return 0.0
        if HAS_RAPIDFUZZ and _rf_fuzz is not None:
            # WRatio picks the best of token_sort, token_set, partial, and full
            # ratio depending on string length — ideal for skill routing where
            # the query is short and profiles are long, multi-word strings.
            wr = getattr(_rf_fuzz, "WRatio", None)
            if wr is not None:
                return float(wr(query_text, profile_text)) / 100.0
            # Fallback within rapidfuzz if WRatio is somehow absent
            ts = _rf_fuzz.token_set_ratio(query_text, profile_text)
            pr = _rf_fuzz.partial_ratio(query_text, profile_text)
            return max(float(ts), float(pr)) / 100.0
        return SequenceMatcher(None, query_text, profile_text).ratio()

    def _build_query_score_context(self, query: str) -> dict[str, Any]:
        query_l = (query or "").lower()
        query_tokens = set(_skill_tokens(query))
        negated_chunks = [
            m.group(1).strip()
            for m in re.finditer(
                r"(?:do\s+not|don't|avoid|without)\s+([a-z0-9 _-]{2,80})",
                query_l,
            )
            if m.group(1).strip()
        ]
        return {
            "query_l": query_l,
            "query_tokens": query_tokens,
            "negated_chunks": negated_chunks,
        }

    def _build_skill_score_context(
        self, base_ctx: dict[str, Any], skill: SkillCard
    ) -> dict[str, Any]:
        ctx = dict(base_ctx)
        ctx["profile_tokens"] = self._routing_tokens.get(skill.id, set(skill.keywords))
        ctx["profile_text"] = self._routing_profiles.get(skill.id, "")
        return ctx

    def _bm25_raw_score(self, query_tokens: Sequence[str], skill_id: str) -> float:
        if not query_tokens:
            return 0.0
        tf = self._routing_term_freqs.get(skill_id)
        if not tf:
            return 0.0

        n_docs = max(1, len(self._routing_term_freqs))
        doc_len = float(self._routing_doc_lengths.get(skill_id, 1))
        avg_doc_len = max(1e-6, float(self._bm25_avg_doc_len))
        k1 = 1.2
        b = 0.75

        score = 0.0
        qfreq = Counter(query_tokens)
        for term, qcount in qfreq.items():
            f = float(tf.get(term, 0))
            if f <= 0.0:
                continue
            df = float(self._bm25_doc_freq.get(term, 0))
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
            denom = f + (k1 * (1.0 - b + b * (doc_len / avg_doc_len)))
            if denom <= 0.0:
                continue
            tf_norm = (f * (k1 + 1.0)) / denom
            q_boost = 1.0 + math.log1p(max(1, int(qcount)))
            score += idf * tf_norm * q_boost
        return score

    @staticmethod
    def _normalize_score_map(raw_scores: dict[str, float]) -> dict[str, float]:
        if not raw_scores:
            return {}
        positives = [float(v) for v in raw_scores.values() if float(v) > 0.0]
        if not positives:
            return {k: 0.0 for k in raw_scores}
        lo = min(positives)
        hi = max(positives)
        if hi <= lo + 1e-9:
            return {k: (1.0 if float(v) > 0.0 else 0.0) for k, v in raw_scores.items()}
        scale = hi - lo
        out: dict[str, float] = {}
        for key, value in raw_scores.items():
            v = float(value)
            if v <= 0.0:
                out[key] = 0.0
            else:
                out[key] = max(0.0, min(1.0, (v - lo) / scale))
        return out

    def _load_dense_model(self) -> Any | None:
        if not self._dense_enabled:
            return None
        if self._dense_model is not None:
            return self._dense_model
        if self._dense_model_disabled_reason:
            return None
        sentence_transformer = _lazy_import_sentence_transformer()
        if sentence_transformer is None:
            self._dense_model_disabled_reason = "sentence_transformers_missing"
            if _SENTENCE_TRANSFORMER_IMPORT_ERROR:
                self._dense_model_disabled_reason = _SENTENCE_TRANSFORMER_IMPORT_ERROR
            self._log.info("Dense routing disabled: sentence-transformers is not installed.")
            return None
        try:
            # Keep bridge trace clean when the model is materialized lazily
            # during the first routing call.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                from transformers.utils import logging as _tf_logging

                _tf_logging.set_verbosity_error()
                disable_pb = getattr(_tf_logging, "disable_progress_bar", None)
                if callable(disable_pb):
                    disable_pb()
            except Exception:
                pass
            with _suppress_fd_output():
                self._dense_model = sentence_transformer(self._dense_model_name)
            self._log.info("Dense routing model enabled: %s", self._dense_model_name)
            return self._dense_model
        except Exception as exc:
            self._dense_model_disabled_reason = str(exc)
            self._log.warning("Dense routing disabled: failed to load model (%s)", exc)
            return None

    def _encode_dense_vectors(self, texts: Sequence[str]) -> list[tuple[float, ...]] | None:
        if not texts:
            return []
        model = self._load_dense_model()
        if model is None:
            return None
        try:
            vectors = model.encode(list(texts), normalize_embeddings=True)
        except Exception as exc:
            self._dense_model_disabled_reason = str(exc)
            self._dense_model = None
            self._log.warning("Dense routing disabled after encoding failure: %s", exc)
            return None

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if not vectors:
            return []
        if isinstance(vectors[0], (int, float)):
            vectors = [vectors]

        out: list[tuple[float, ...]] = []
        for vec in vectors:
            out.append(tuple(float(x) for x in vec))
        return out

    @staticmethod
    def _dense_dot(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        n = min(len(left), len(right))
        if n <= 0:
            return 0.0
        return float(sum(float(left[i]) * float(right[i]) for i in range(n)))

    def _dense_similarity_map(self, query: str, candidate_ids: Sequence[str]) -> dict[str, float]:
        out = {sid: 0.0 for sid in candidate_ids}
        if not candidate_ids or not str(query or "").strip():
            return out

        q_vectors = self._encode_dense_vectors([query])
        if not q_vectors:
            return out
        query_vec = q_vectors[0]

        missing_ids = [
            sid
            for sid in candidate_ids
            if sid not in self._dense_embedding_cache and self._routing_profiles.get(sid)
        ]
        if missing_ids:
            texts = [self._routing_profiles.get(sid, "") for sid in missing_ids]
            vectors = self._encode_dense_vectors(texts)
            if vectors:
                for sid, vec in zip(missing_ids, vectors):
                    self._dense_embedding_cache[sid] = vec

        for sid in candidate_ids:
            vec = self._dense_embedding_cache.get(sid)
            if not vec:
                continue
            dot = self._dense_dot(query_vec, vec)
            out[sid] = max(0.0, min(1.0, (dot + 1.0) / 2.0))
        return out

    def note_skill_usage(self, skill_id: str, *, timestamp: float | None = None) -> None:
        sid = str(skill_id or "").strip()
        if not sid:
            return
        now = float(timestamp if timestamp is not None else time.time())
        usage = self._skill_usage_stats.setdefault(
            sid,
            {"count": 0.0, "last_used_at": 0.0},
        )
        usage["count"] = float(usage.get("count", 0.0)) + 1.0
        usage["last_used_at"] = now

    def _usage_bias_map(self, candidate_ids: Sequence[str]) -> dict[str, tuple[float, float]]:
        if not candidate_ids:
            return {}
        max_count = 0.0
        for stat in self._skill_usage_stats.values():
            max_count = max(max_count, float(stat.get("count", 0.0)))
        now = time.time()
        out: dict[str, tuple[float, float]] = {}
        for sid in candidate_ids:
            stat = self._skill_usage_stats.get(sid, {})
            count = max(0.0, float(stat.get("count", 0.0)))
            last_used_at = max(0.0, float(stat.get("last_used_at", 0.0)))
            if max_count > 0.0 and count > 0.0:
                usage = math.log1p(count) / math.log1p(max_count)
            else:
                usage = 0.0
            if last_used_at > 0.0:
                age_s = max(0.0, now - last_used_at)
                recency = math.exp(-age_s / max(1.0, self._usage_half_life_s))
            else:
                recency = 0.0
            out[sid] = (max(0.0, min(1.0, usage)), max(0.0, min(1.0, recency)))
        return out

    def _recall_candidate_ids(
        self,
        *,
        query_ctx: dict[str, Any],
        top_k: int,
        forced_skill_ids: Sequence[str],
    ) -> tuple[list[str], dict[str, float], dict[str, float]]:
        query_l = str(query_ctx.get("query_l", ""))
        query_tokens = list(query_ctx.get("query_tokens", set()))
        query_token_set = set(query_tokens)

        bm25_raw: dict[str, float] = {}
        for sid in self._skills:
            bm25_raw[sid] = self._bm25_raw_score(query_tokens, sid)
        bm25_norm = self._normalize_score_map(bm25_raw)

        recall_scores: dict[str, float] = {}
        for sid, skill in self._skills.items():
            exact = (
                1.0
                if (skill.id.replace("_", " ") in query_l or skill.name.lower() in query_l)
                else 0.0
            )
            alias_hit = 0.0
            for alias in skill.aliases:
                if not alias:
                    continue
                alias_l = alias.lower()
                if alias_l in query_l:
                    alias_hit = 1.0
                    break
                alias_tokens = set(_skill_tokens(alias_l))
                if alias_tokens and query_token_set.intersection(alias_tokens):
                    alias_hit = 1.0
                    break
            tool_hit = (
                1.0
                if any(tool_name and tool_name.lower() in query_l for tool_name in skill.tool_names)
                else 0.0
            )
            fuzzy = self._fuzzy_similarity(query_l, self._routing_profiles.get(sid, ""))
            recall_scores[sid] = (
                (bm25_norm.get(sid, 0.0) * 0.70)
                + (exact * 0.14)
                + (alias_hit * 0.06)
                + (tool_hit * 0.05)
                + (fuzzy * 0.05)
            )

        ranked = sorted(
            recall_scores,
            key=lambda sid: (
                recall_scores.get(sid, 0.0),
                len(self._skills.get(sid).tool_names) if sid in self._skills else 0,
                sid,
            ),
            reverse=True,
        )

        forced_ordered: list[str] = []
        seen_forced: set[str] = set()
        for raw_id in forced_skill_ids:
            sid = str(raw_id or "").strip()
            if not sid or sid in seen_forced or sid not in self._skills:
                continue
            seen_forced.add(sid)
            forced_ordered.append(sid)

        limit = max(self._routing_recall_k, max(1, int(top_k)) * 8, len(forced_ordered))
        out: list[str] = list(forced_ordered)
        for sid in ranked:
            if sid in seen_forced:
                continue
            out.append(sid)
            if len(out) >= limit:
                break
        return out, bm25_norm, recall_scores

    def _score_candidate(
        self,
        *,
        query_ctx: dict[str, Any],
        skill: SkillCard,
        bm25_score: float,
        dense_score: float,
        usage_score: float,
        recency_score: float,
        min_score: float,
    ) -> tuple[float, dict[str, float]]:
        query_l = str(query_ctx.get("query_l", ""))
        query_tokens = set(query_ctx.get("query_tokens", set()))
        profile_text = self._routing_profiles.get(skill.id, "")
        profile_tokens = self._routing_tokens.get(skill.id, set(skill.keywords))
        negated_chunks = list(query_ctx.get("negated_chunks", []))
        negated_text = " ".join(negated_chunks)

        fuzzy = self._fuzzy_similarity(query_l, profile_text)
        coverage = 0.0
        if query_tokens and profile_tokens:
            coverage = len(query_tokens.intersection(profile_tokens)) / max(1, len(query_tokens))

        alias_match = False
        for alias in skill.aliases:
            if not alias:
                continue
            alias_l = alias.lower()
            if alias_l in query_l:
                alias_match = True
                break
            alias_tokens = set(_skill_tokens(alias_l))
            if alias_tokens and query_tokens.intersection(alias_tokens):
                alias_match = True
                break

        name_or_alias = (
            1.0
            if (
                skill.id.replace("_", " ") in query_l
                or skill.name.lower() in query_l
                or alias_match
            )
            else 0.0
        )
        trigger_hits = sum(
            1 for trigger in skill.triggers if trigger and trigger.lower() in query_l
        )
        trigger_score = min(1.0, float(trigger_hits))
        if negated_text:
            negated_trigger_hits = sum(
                1 for trigger in skill.triggers if trigger and trigger.lower() in negated_text
            )
            if negated_trigger_hits > 0:
                trigger_score = 0.0
        tool_hits = sum(
            1 for tool_name in skill.tool_names if tool_name and tool_name.lower() in query_l
        )
        tool_score = min(1.0, float(tool_hits))
        example_score = min(
            1.0,
            self._example_query_score(query_tokens, query_l, skill) / 6.0,
        )
        anti_hits = sum(1 for anti in skill.anti_triggers if anti and anti.lower() in query_l)
        anti_penalty = min(1.0, float(anti_hits))

        negated_penalty = 0.0
        for chunk in negated_chunks:
            chunk_tokens = set(_skill_tokens(chunk))
            if not chunk_tokens:
                continue
            overlap = len(chunk_tokens.intersection(profile_tokens)) / max(1, len(chunk_tokens))
            negated_penalty = max(negated_penalty, overlap)

        w = self._routing_weights
        contributions = {
            "bm25": w["bm25"] * bm25_score,
            "dense": w["dense"] * dense_score,
            "fuzzy": w["fuzzy"] * fuzzy,
            "coverage": w["coverage"] * coverage,
            "name_or_alias": w["name_or_alias"] * name_or_alias,
            "trigger_hit": w["trigger_hit"] * trigger_score,
            "tool_hit": w["tool_hit"] * tool_score,
            "example": w["example"] * example_score,
            "usage_bias": w["usage_bias"] * usage_score,
            "recency_bias": w["recency_bias"] * recency_score,
            "anti_trigger_penalty": -w["anti_trigger_penalty"] * anti_penalty,
            "negation_penalty": -w["negation_penalty"] * negated_penalty,
        }
        raw = sum(contributions.values())

        explicit_guidance_hit = bool(name_or_alias > 0.0 or trigger_score > 0.0)
        if not skill.tool_names and not explicit_guidance_hit:
            raw *= float(w.get("guidance_discount", 0.82))
            contributions["guidance_discount"] = raw - sum(contributions.values())

        if skill.id == "think":
            explicit_think = bool(
                "/think" in query_l
                or "think tool" in query_l
                or "use think" in query_l
                or "call think" in query_l
            )
            if not explicit_think:
                return max(0.0, min(1.0, min_score - 0.01)), contributions

        score = max(0.0, min(1.0, raw))
        return score, contributions

    def _apply_guidance_score_adjustments(
        self,
        score: float,
        skill: SkillCard,
        ctx: dict[str, Any],
        *,
        min_score: float,
    ) -> float:
        if skill.tool_names:
            return score

        query_l = str(ctx.get("query_l", ""))
        explicit_guidance_hit = bool(
            skill.id.replace("_", " ") in query_l
            or skill.name.lower() in query_l
            or int(ctx.get("alias_hits_count", 0)) > 0
            or int(ctx.get("trigger_phrase_hits_count", 0)) > 0
        )
        query_coverage = float(ctx.get("query_coverage", 0.0))
        ratio = float(ctx.get("fuzzy_ratio", 0.0))
        keyword_hits = int(ctx.get("keyword_hits_count", 0))
        strong_guidance_match = bool(
            int(ctx.get("trigger_phrase_hits_count", 0)) > 0
            or query_coverage >= 0.40
            or ratio >= 0.48
            or (ratio >= 0.38 and keyword_hits >= 2)
            or score >= (min_score * 3.0)
        )
        if explicit_guidance_hit:
            return score + 1.0
        if strong_guidance_match:
            # Keep guidance cards relevant but less dominant than executable skills.
            return (score * 0.75) - 0.25
        return min(score, min_score - 0.01)

    def _apply_legacy_skill_caps(
        self,
        score: float,
        skill: SkillCard,
        ctx: dict[str, Any],
        *,
        min_score: float,
    ) -> float:
        # Keep the legacy think tool opt-in: don't auto-route unless the
        # user clearly requests it.
        if skill.id == "think":
            query_l = str(ctx.get("query_l", ""))
            explicit_think = bool(
                "/think" in query_l
                or "think tool" in query_l
                or "use think" in query_l
                or "call think" in query_l
            )
            if not explicit_think:
                return min(score, min_score - 0.01)
        return score

    def get_last_skill_score_breakdown(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._last_skill_score_breakdown.items()}

    def _skill_paths_match_cwd(self, card: SkillCard) -> bool:
        """Return True if the skill is active in the current working directory.

        A skill with no ``paths:`` frontmatter key is always active.  When
        ``paths:`` is set each entry is treated as a glob pattern and matched
        against the absolute CWD string (and its parent directory names), so
        entries like ``timeseries/``, ``**/*.csv``, or ``src/`` work as expected.
        """
        if not card.paths:
            return True
        import fnmatch

        cwd = os.getcwd()
        for pattern in card.paths:
            pat = str(pattern or "").strip()
            if not pat:
                continue
            # Match against full cwd, its base name, or any trailing component.
            if (
                fnmatch.fnmatch(cwd, f"*{pat}*")
                or fnmatch.fnmatch(os.path.basename(cwd), pat.rstrip("/"))
                or fnmatch.fnmatch(cwd, pat)
            ):
                return True
        return False

    def route_query_to_skills(
        self,
        query: str,
        available_tool_names: Sequence[str],
        *,
        top_k: int = 3,
        min_score: float | None = None,
        forced_skill_ids: Sequence[str] | None = None,
    ) -> SkillSelection:
        self.ensure_skill_catalog()

        if not self._skills:
            return SkillSelection(
                query=query,
                selected_skills=[],
                selected_tools=[],
                fallback_tools=list(available_tool_names),
            )

        threshold = (
            self._routing_min_score if min_score is None else max(0.0, min(1.0, float(min_score)))
        )
        query_ctx = self._build_query_score_context(query)
        available = set(available_tool_names)

        forced_ids: list[str] = []
        forced_seen: set[str] = set()
        for raw_id in forced_skill_ids or []:
            sid = str(raw_id or "").strip()
            if not sid or sid in forced_seen or sid not in self._skills:
                continue
            forced_ids.append(sid)
            forced_seen.add(sid)

        candidate_ids, bm25_map, recall_map = self._recall_candidate_ids(
            query_ctx=query_ctx,
            top_k=max(1, int(top_k)),
            forced_skill_ids=forced_ids,
        )
        dense_map = self._dense_similarity_map(query, candidate_ids)
        usage_map = self._usage_bias_map(candidate_ids)

        scored: list[tuple[float, SkillCard]] = []
        score_breakdown: dict[str, dict[str, Any]] = {
            sid: {
                "raw_score": 0.0,
                "adjusted_score": 0.0,
                "normalized_score": 0.0,
                "recall_score": recall_map.get(sid, 0.0),
                "contributions": {},
            }
            for sid in self._skills
        }
        for sid in candidate_ids:
            skill = self._skills.get(sid)
            if skill is None:
                continue
            # Skip skills whose paths: filter doesn't match the current CWD.
            if not self._skill_paths_match_cwd(skill):
                continue
            usage_score, recency_score = usage_map.get(sid, (0.0, 0.0))
            score, contributions = self._score_candidate(
                query_ctx=query_ctx,
                skill=skill,
                bm25_score=bm25_map.get(sid, 0.0),
                dense_score=dense_map.get(sid, 0.0),
                usage_score=usage_score,
                recency_score=recency_score,
                min_score=threshold,
            )
            score_breakdown[sid] = {
                "raw_score": score,
                "adjusted_score": score,
                "normalized_score": score,
                "recall_score": recall_map.get(sid, 0.0),
                "contributions": contributions,
            }
            if score >= threshold or sid in forced_seen:
                scored.append((score, skill))

        self._last_skill_score_breakdown = score_breakdown
        scored.sort(
            key=lambda item: (
                item[0],
                len(item[1].tool_names),
                item[1].name,
            ),
            reverse=True,
        )

        forced_skills = [self._skills[sid] for sid in forced_ids if sid in self._skills]
        limit = max(1, int(top_k), len(forced_skills))
        ranked_skills = [skill for _, skill in scored if skill.id not in forced_seen]
        selected_skills = (forced_skills + ranked_skills)[:limit]
        selected_tools: list[str] = []
        seen_tools: set[str] = set()
        for skill in selected_skills:
            for tool_name in self._ordered_skill_tools(skill):
                if tool_name in available and tool_name not in seen_tools:
                    selected_tools.append(tool_name)
                    seen_tools.add(tool_name)

        fallback_tools = [
            tool_name for tool_name in available_tool_names if tool_name not in seen_tools
        ]

        return SkillSelection(
            query=query,
            selected_skills=selected_skills,
            selected_tools=selected_tools,
            fallback_tools=fallback_tools,
        )

    def skill_routing_prompt(
        self,
        query: str,
        available_tool_names: Sequence[str],
        render_tool_prompt: Callable[..., str],
        *,
        use_toon: bool = True,
        mode: Literal["rich", "compact", "json_schema"] = "rich",
        top_k: int = 3,
        include_playbooks: bool = True,
        include_compact_fallback: bool = True,
        include_on_demand_context: bool = True,
        on_demand_context_max_chars: int = 2600,
        on_demand_context_max_skills: int = 2,
        on_demand_context_max_files_per_skill: int = 5,
        forced_skill_ids: Sequence[str] | None = None,
    ) -> tuple[str, SkillSelection]:
        selection = self.route_query_to_skills(
            query,
            available_tool_names,
            top_k=top_k,
            forced_skill_ids=forced_skill_ids,
        )

        # Proceed even when only guidance cards matched (no selected_tools yet).
        has_skills = bool(selection.selected_skills)
        if not has_skills:
            return (
                render_tool_prompt(use_toon=use_toon, mode=mode),
                selection,
            )

        lines = ["\n\nACTIVE SKILLS FOR THIS REQUEST:"]
        for skill in selection.selected_skills:
            summary = skill.summary or "Relevant tools and playbooks for this request."
            guidance_tag = " [guidance]" if not skill.tool_names else ""
            lines.append(f"- {skill.name}{guidance_tag}: {summary}")
            if skill.when_not_to_use:
                lines.append(f"  avoid when: {skill.when_not_to_use[0]}")
            if include_playbooks:
                for playbook in skill.playbooks[:3]:
                    lines.append(f"  playbook: {playbook}")

        if include_on_demand_context:
            on_demand_lines = self._render_on_demand_skill_context(
                selection=selection,
                query=query,
                max_total_chars=max(400, int(on_demand_context_max_chars)),
                max_skills=max(1, int(on_demand_context_max_skills)),
                max_files_per_skill=max(1, int(on_demand_context_max_files_per_skill)),
            )
            if on_demand_lines:
                lines.append("")
                lines.append("ON-DEMAND SKILL CONTEXT:")
                lines.extend(on_demand_lines)

        fallback = selection.fallback_tools[:12] if include_compact_fallback else None
        prompt = render_tool_prompt(
            use_toon=use_toon,
            mode=mode,
            include_tool_names=selection.selected_tools,
            compact_fallback_tool_names=fallback,
        )
        return ("\n".join(lines) + prompt, selection)

    def _render_on_demand_skill_context(
        self,
        *,
        selection: SkillSelection,
        query: str,
        max_total_chars: int,
        max_skills: int,
        max_files_per_skill: int,
    ) -> list[str]:
        lines: list[str] = []
        remaining = max_total_chars
        included_skills = 0

        for skill in selection.selected_skills:
            if included_skills >= max_skills:
                break
            source = Path(skill.source_path)
            if source.name.upper() != "SKILL.MD":
                continue
            excerpts = self._skill_context_excerpts(
                source_path=source,
                query=query,
                max_files=max_files_per_skill,
            )
            if not excerpts:
                continue

            header = f"- {skill.name}:"
            if len(header) + 1 > remaining:
                break
            lines.append(header)
            remaining -= len(header) + 1

            added_for_skill = 0
            for rel_path, heading, excerpt in excerpts:
                entry = f"  [{rel_path} :: {heading}] {excerpt}"
                if len(entry) + 1 > remaining:
                    if added_for_skill == 0:
                        trim_budget = max(80, remaining - len(rel_path) - len(heading) - 12)
                        trimmed = excerpt[:trim_budget].rstrip()
                        if trimmed:
                            entry = f"  [{rel_path} :: {heading}] {trimmed}"
                        else:
                            break
                    else:
                        break
                lines.append(entry)
                remaining -= len(entry) + 1
                added_for_skill += 1
                if remaining <= 120:
                    break

            if added_for_skill > 0:
                included_skills += 1
            if remaining <= 120:
                break

        return lines

    def _skill_context_excerpts(
        self, *, source_path: Path, query: str, max_files: int
    ) -> list[tuple[str, str, str]]:
        if not source_path.is_file():
            return []
        skill_root = source_path.parent
        file_candidates = self._rank_skill_context_files(
            source_path=source_path,
            query=query,
            max_files=max_files,
        )
        query_l = (query or "").lower()
        query_tokens = set(_skill_tokens(query))
        scored: list[tuple[float, str, str, str]] = []

        for md_path in file_candidates:
            try:
                raw = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            _, body = self._split_frontmatter(raw)
            body = body.strip()
            if not body:
                continue

            sections = self.parse_markdown_h2_sections(body)
            if not sections:
                fallback_heading = md_path.stem.replace("_", " ").replace("-", " ")
                sections = [{"heading": fallback_heading, "body": body}]

            for sec in sections:
                heading = (sec.get("heading") or "").strip() or md_path.stem
                snippet = self._compact_markdown_excerpt(sec.get("body", ""), max_chars=520)
                if not snippet:
                    continue
                candidate_text = f"{md_path.name} {heading} {snippet[:220]}".strip().lower()
                ratio = self._fuzzy_similarity(query_l, candidate_text)
                candidate_tokens = set(_skill_tokens(candidate_text))
                overlap = 0.0
                if query_tokens and candidate_tokens:
                    overlap = len(query_tokens.intersection(candidate_tokens)) / max(
                        1, len(query_tokens)
                    )
                score = (ratio * 0.75) + (overlap * 0.25)
                if heading.lower() in query_l:
                    score += 0.2
                if md_path == source_path:
                    score += 0.12
                if score < 0.11 and md_path != source_path:
                    continue
                try:
                    rel = str(md_path.relative_to(skill_root))
                except Exception:
                    rel = md_path.name
                scored.append((score, rel, heading, snippet))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for _score, rel, heading, snippet in scored:
            key = (rel, heading.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((rel, heading, snippet))
            if len(out) >= 3:
                break
        return out

    def _rank_skill_context_files(
        self, *, source_path: Path, query: str, max_files: int
    ) -> list[Path]:
        root = source_path.parent
        query_l = (query or "").lower()
        query_tokens = set(_skill_tokens(query))
        files = [source_path]
        try:
            files.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
        except Exception:
            pass

        scored: list[tuple[float, Path]] = []
        seen: set[str] = set()
        for path in files:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                rel = str(path.relative_to(root))
            except Exception:
                rel = path.name
            rel_text = rel.lower().replace("/", " ").replace("-", " ").replace("_", " ")
            ratio = self._fuzzy_similarity(query_l, rel_text)
            rel_tokens = set(_skill_tokens(rel_text))
            overlap = 0.0
            if query_tokens and rel_tokens:
                overlap = len(query_tokens.intersection(rel_tokens)) / max(1, len(query_tokens))
            score = (ratio * 0.7) + (overlap * 0.3)
            if path == source_path:
                score += 0.4
            if path.name.upper() == "SKILL.MD":
                score += 0.15
            scored.append((score, path))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[Path] = []
        for _, path in scored:
            out.append(path)
            if len(out) >= max(1, max_files):
                break
        if source_path not in out:
            out.insert(0, source_path)
        return out[: max(1, max_files)]

    def _compact_markdown_excerpt(self, text: str, *, max_chars: int = 520) -> str:
        if not text:
            return ""
        without_code = re.sub(r"```[\s\S]*?```", " ", text)
        one_line = re.sub(r"\s+", " ", without_code).strip()
        if len(one_line) <= max_chars:
            return one_line
        return one_line[:max_chars].rstrip() + " ..."

    def parse_markdown_h2_sections(self, content: str) -> list[MarkdownSection]:
        if not HAS_MARKDOWN_IT or MarkdownIt is None:
            return self._parse_markdown_h2_sections_fallback(content)

        md = MarkdownIt()
        tokens = md.parse(content)
        lines = content.splitlines()
        sections: list[MarkdownSection] = []
        n_lines = len(lines)

        h2_entries: list[tuple[str, int]] = []
        for i, tok in enumerate(tokens):
            if tok.type != "heading_open" or tok.tag != "h2":
                continue
            heading = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                heading = tokens[i + 1].content.strip()
            start_line = tok.map[1] if tok.map else 0
            h2_entries.append((heading, start_line))

        for idx, (heading, start_line) in enumerate(h2_entries):
            end_line = n_lines
            if idx + 1 < len(h2_entries):
                end_line = max(start_line, h2_entries[idx + 1][1] - 1)
            body = "\n".join(lines[start_line:end_line]).strip("\n")
            sections.append({"heading": heading, "body": body})
        return sections

    def _parse_markdown_h2_sections_fallback(self, content: str) -> list[MarkdownSection]:
        sections: list[MarkdownSection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines():
            m = re.match(r"^\s*##\s+(.+?)\s*$", line)
            if m:
                if current_heading is not None:
                    sections.append(
                        {
                            "heading": current_heading,
                            "body": "\n".join(current_lines).strip("\n"),
                        }
                    )
                current_heading = m.group(1).strip()
                current_lines = []
                continue
            if current_heading is not None:
                current_lines.append(line)

        if current_heading is not None:
            sections.append(
                {
                    "heading": current_heading,
                    "body": "\n".join(current_lines).strip("\n"),
                }
            )
        return sections

    def parse_bootstrap_sections(self, content: str) -> list[BootstrapSection]:
        sections: list[BootstrapSection] = []
        for sec in self.parse_markdown_h2_sections(content):
            bname = self._extract_bootstrap_name_from_heading(sec["heading"])
            if not bname:
                continue
            sections.append(
                {
                    "name": bname,
                    "content": sec["body"],
                    "code": self._extract_code_from_markdown(sec["body"]),
                }
            )
        return sections

    def parse_tool_sections(self, content: str, source_path: Path) -> list[ToolSection]:
        sections: list[ToolSection] = []
        skill_id = self._skill_id_from_source(source_path)
        for sec in self.parse_markdown_h2_sections(content):
            tool_name = self._extract_tool_name_from_heading(sec["heading"])
            if not tool_name:
                continue
            section: ToolSection = {
                "name": tool_name,
                "content": sec["body"],
                "code": self._extract_code_from_markdown(sec["body"]),
                "description": "",
                "parameters": [],
                "source_path": str(source_path),
                "skill_id": skill_id,
            }
            self._extract_metadata(section)
            sections.append(section)
        return sections

    def tool_doc_from_section(self, content: str) -> str:
        kept_lines: List[str] = []
        in_code_block = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if stripped == "**Implementation:**":
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _skill_id_from_source(self, source_path: Path) -> str:
        skill_root = self._skill_root_for_source(source_path)
        if skill_root is not None and self._skill_root_has_python_tools(skill_root):
            return self._normalize_skill_slug(skill_root.name)

        # SKILL.md files (Superpowers format) use their *parent directory* as the id.
        if source_path.name.upper() == "SKILL.MD":
            raw = source_path.parent.name
            raw = self._normalize_skill_slug(raw)
            return f"sp__{raw}" if raw else "sp__skill"
        stem = source_path.stem
        stem = self._normalize_skill_slug(stem)
        return stem or "skill"

    def _skill_name_from_source(self, source_path: Path) -> str:
        stem = self._skill_id_from_source(source_path)
        return stem.replace("_", " ").title()

    def _skill_summary(self, content: str) -> str:
        lines = [line.strip() for line in content.splitlines()]
        summary_parts: list[str] = []
        for line in lines:
            if not line:
                if summary_parts:
                    break
                continue
            if line.startswith("#"):
                continue
            if line.startswith("```"):
                break
            summary_parts.append(line)
            if len(" ".join(summary_parts)) >= 180:
                break
        summary = " ".join(summary_parts).strip()
        return summary or "Skill-defined tool group."

    def _skill_playbooks(self, content: str) -> list[str]:
        lines = content.splitlines()
        playbooks: list[str] = []
        in_playbooks = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## ") and "playbook" in stripped.lower():
                in_playbooks = True
                continue
            if in_playbooks and stripped.startswith("## "):
                break
            if in_playbooks:
                m = re.match(r"^\s*\d+\.\s+\*\*(.+?)\*\*", line)
                if m:
                    playbooks.append(m.group(1).strip())
        return playbooks

    def _skill_keywords(
        self,
        summary: str,
        playbooks: list[str],
        tool_names: list[str],
        *,
        triggers: list[str],
        example_queries: list[str],
        aliases: list[str],
        content: str = "",
        content_max_chars: int = 3000,
    ) -> list[str]:
        tokens = _skill_tokens(summary)
        for playbook in playbooks:
            tokens.extend(_skill_tokens(playbook))
        for tool_name in tool_names:
            tokens.extend(_skill_tokens(tool_name.replace("_", " ")))
        for trigger in triggers:
            tokens.extend(_skill_tokens(trigger))
        for query in example_queries:
            tokens.extend(_skill_tokens(query))
        for alias in aliases:
            tokens.extend(_skill_tokens(alias))
        if content:
            # Use a bounded slice to avoid over-weighting very large skill docs.
            tokens.extend(_skill_tokens(content[: max(0, int(content_max_chars))]))
        deduped: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                deduped.append(tok)
        return deduped

    def _extract_use_when_clauses(self, content: str) -> list[str]:
        clauses: list[str] = []
        for m in re.finditer(r"(?im)^\s*use when[:\-]?\s*(.+)$", content):
            phrase = m.group(1).strip()
            if phrase:
                clauses.append(phrase)
        return clauses[:8]

    def _extract_labeled_items(self, text: str, label: str) -> list[str]:
        if not text or not label:
            return []
        pattern = re.compile(
            rf"(?is)\b{re.escape(label)}\s*:\s*(.+?)(?:\n[A-Z][A-Za-z ]{{2,}}\s*:|$)"
        )
        m = pattern.search(text)
        if not m:
            return []
        raw = m.group(1).strip()
        if not raw:
            return []
        parts = re.split(r"[,;\n]+", raw)
        out: list[str] = []
        seen: set[str] = set()
        for part in parts:
            item = part.strip().strip(".")
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= 12:
                break
        return out

    def _skill_aliases(self, skill_id: str, manifest: SkillManifest) -> list[str]:
        # Aliases come solely from the skill frontmatter — no hardcoded fallback dict.
        return self._manifest_list(manifest, "aliases")

    def _manifest_list(self, manifest: SkillManifest, key: str) -> list[str]:
        raw = manifest.get(key, [])
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def _ordered_skill_tools(self, skill: SkillCard) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for tool_name in skill.preferred_tools:
            if tool_name in skill.tool_names and tool_name not in seen:
                ordered.append(tool_name)
                seen.add(tool_name)
        for tool_name in skill.tool_names:
            if tool_name not in seen:
                ordered.append(tool_name)
                seen.add(tool_name)
        return ordered

    def _example_query_score(
        self,
        query_tokens: set[str],
        query_l: str,
        skill: SkillCard,
    ) -> float:
        score = 0.0
        for example_query in skill.example_queries:
            example_tokens = set(_skill_tokens(example_query))
            if not example_tokens:
                continue
            overlap = len(query_tokens.intersection(example_tokens))
            if overlap:
                score += min(4.0, overlap * 1.5)
            if example_query.lower() in query_l:
                score += 5.0
        return score

    def _split_frontmatter(self, content: str) -> tuple[SkillManifest, str]:
        manifest, body = split_frontmatter(content)
        return manifest, body
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                current_key = key
                if value:
                    manifest[key] = value
                else:
                    manifest[key] = []
                continue

            if stripped.startswith("- ") and current_key:
                existing = manifest.get(current_key)
                if not isinstance(existing, list):
                    existing = [] if existing in (None, "") else [str(existing)]
                existing.append(stripped[2:].strip())
                manifest[current_key] = existing

        return manifest

    def _extract_bootstrap_name_from_heading(self, heading: str) -> Optional[str]:
        normalized = heading.strip()
        if not normalized.lower().startswith("bootstrap:"):
            return None
        rest = normalized[len("bootstrap:") :].strip()
        return rest if rest else "bootstrap"

    def _extract_tool_name_from_heading(self, heading: str) -> Optional[str]:
        normalized = heading.strip()
        if not normalized.lower().startswith("tool:"):
            return None
        rest = normalized[len("tool:") :].strip()
        return rest.split()[0] if rest else None

    def _extract_code_from_markdown(self, text: str) -> str:
        code_extractor = CodeBlockExtractor()
        for line in text.splitlines():
            code_extractor.process_line(line)
        return code_extractor.get_code()

    def _extract_metadata(self, section: ToolSection) -> None:
        content = section["content"]
        desc_match = re.search(r"\*\*Description:\*\*\s*(.+?)(?:\n\n|\*\*|$)", content, re.DOTALL)
        section["description"] = desc_match.group(1).strip() if desc_match else ""
        section["parameters"] = self._parse_parameters(content)

    def _parse_parameters(self, content: str) -> list[ToolParameter]:
        params = []
        param_section = re.search(
            r"\*\*Parameters:\*\*\s*\n((?:^[-*]\s+.+\n?)+)", content, re.MULTILINE
        )
        if not param_section:
            return params

        param_text = param_section.group(1)
        param_pattern = re.compile(r"[-*]\s+(\w+)\s+\(([^,]+),\s*(required|optional)\):\s*(.+)")
        for match in param_pattern.finditer(param_text):
            params.append(
                ToolParameter(
                    name=match.group(1),
                    type=match.group(2).strip(),
                    description=match.group(4).strip(),
                    required=(match.group(3) == "required"),
                )
            )
        return params


__all__ = ["SkillCatalog", "ToolSection"]
