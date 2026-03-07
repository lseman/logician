from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Sequence, TypedDict

from .runtime import SkillCard, SkillSelection, ToolParameter

try:
    from markdown_it import MarkdownIt

    HAS_MARKDOWN_IT = True
except ImportError:
    MarkdownIt = None  # type: ignore
    HAS_MARKDOWN_IT = False

try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:
    _yaml = None  # type: ignore
    HAS_YAML = False

try:
    from rapidfuzz import fuzz as _rf_fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    _rf_fuzz = None  # type: ignore
    HAS_RAPIDFUZZ = False


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
    _CACHE_VERSION = 6

    def __init__(
        self, *, skills_md_path: Path, skills_dir_path: Path, log: Any
    ) -> None:
        self.skills_md_path = skills_md_path
        self.skills_dir_path = skills_dir_path
        self._log = log
        self._skills: dict[str, SkillCard] = {}
        self._tool_docs: dict[str, str] = {}
        self._all_tool_sections: list[ToolSection] = []
        # Startup-built routing index for fast and consistent skill relevance scoring.
        self._routing_profiles: dict[str, str] = {}
        self._routing_tokens: dict[str, set[str]] = {}

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

    def _cache_path(self) -> Path:
        return self.skills_dir_path / ".skills_catalog_cache.json"

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
            self._routing_profiles = profiles
            self._routing_tokens = tokens
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
            for root, _dirs, files in os.walk(str(directory), followlinks=True):
                for fname in files:
                    if fname.lower().endswith(".md"):
                        results.append(Path(root) / fname)
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

    def read_skill_source_contents(self) -> list[tuple[Path, str]]:
        out: list[tuple[Path, str]] = []
        for src in self.iter_skills_sources():
            try:
                out.append((src, src.read_text(encoding="utf-8")))
            except Exception as e:
                self._log.error("Failed to read skills source %s: %s", src, e)
                return []
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
        contents = []
        for src in sources:
            try:
                contents.append((src, src.read_text(encoding="utf-8")))
            except Exception as e:
                self._log.error("Failed to read skills source %s: %s", src, e)
                return
        if contents:
            self.build_skill_catalog(contents, _fingerprint=fingerprint)

    def _parse_superpowers_card(
        self, source_path: Path, raw_content: str
    ) -> SkillCard | None:
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
        slug_tokens = re.split(r"[-_]+", str(raw_name).lower())
        aliases: list[str] = []
        slug_phrase = " ".join(tok for tok in slug_tokens if tok).strip()
        if slug_phrase:
            aliases.append(slug_phrase)
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
        default_triggers: dict[str, list[str]] = {
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
        default_anti_triggers: dict[str, list[str]] = {
            "forecasting": ["plot diagnostics only"],
        }
        default_next_skills: dict[str, list[str]] = {
            "data_loading": ["preprocessing", "analysis", "forecasting"],
            "preprocessing": ["analysis", "forecasting"],
            "analysis": ["forecasting"],
        }

        for tool in tools:
            skill_id = str(getattr(tool, "skill_id", "") or "").strip()
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not skill_id or not tool_name:
                continue
            if skill_id.startswith("mcp__"):
                continue

            source_path = str(getattr(tool, "source_path", "") or "")
            tool_desc = str(getattr(tool, "description", "") or "").strip()
            display_name = skill_id.replace("_", " ").title()
            default_summary = (
                f"Tool-backed skill group '{display_name}' loaded from Python modules."
            )

            card = cards.get(skill_id)
            if card is None:
                card = SkillCard(
                    id=skill_id,
                    name=display_name,
                    summary=tool_desc or default_summary,
                    source_path=source_path,
                    description=tool_desc or default_summary,
                    tool_names=[],
                    playbooks=[],
                    keywords=[],
                    aliases=[skill_id.replace("_", " ")],
                    triggers=list(default_triggers.get(skill_id, [])),
                    anti_triggers=list(default_anti_triggers.get(skill_id, [])),
                    preferred_tools=[],
                    example_queries=[],
                    when_not_to_use=["another specialized skill is a clearer match"],
                    next_skills=list(default_next_skills.get(skill_id, [])),
                )
                cards[skill_id] = card
                changed = True

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

            if skill_id == "forecasting":
                forecast_pref = [
                    name
                    for name in ("suggest_models", "stat_forecast", "neural_forecast")
                    if name in card.tool_names
                ]
                if forecast_pref:
                    for candidate in card.preferred_tools:
                        if candidate not in forecast_pref:
                            forecast_pref.append(candidate)
                    if card.preferred_tools != forecast_pref:
                        card.preferred_tools = forecast_pref
                        changed = True

            if (not card.summary or card.summary == "Skill-defined tool group.") and (
                tool_desc
            ):
                card.summary = tool_desc
                changed = True

            if not card.description and tool_desc:
                card.description = tool_desc
                changed = True

            if not card.triggers and skill_id in default_triggers:
                card.triggers = list(default_triggers[skill_id])
                changed = True
            if not card.anti_triggers and skill_id in default_anti_triggers:
                card.anti_triggers = list(default_anti_triggers[skill_id])
                changed = True
            if not card.next_skills and skill_id in default_next_skills:
                card.next_skills = list(default_next_skills[skill_id])
                changed = True
            if not card.when_not_to_use:
                card.when_not_to_use = [
                    "another specialized skill is a clearer match"
                ]
                changed = True

            desc_triggers = self._extract_labeled_items(tool_desc, "Triggers")
            for trig in desc_triggers:
                if trig not in card.triggers:
                    card.triggers.append(trig)
                    changed = True

            desc_anti = self._extract_labeled_items(tool_desc, "Avoid when")
            for anti in desc_anti:
                if anti not in card.anti_triggers:
                    card.anti_triggers.append(anti)
                    changed = True
            if desc_anti and (
                not card.when_not_to_use
                or card.when_not_to_use
                == ["another specialized skill is a clearer match"]
            ):
                card.when_not_to_use = [desc_anti[0]]
                changed = True

            for tok in _skill_tokens(f"{tool_name.replace('_', ' ')} {tool_desc}"):
                if tok not in card.keywords:
                    card.keywords.append(tok)
                    changed = True

        if changed:
            self._build_routing_index(cards)
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
            # Superpowers SKILL.md files are guidance-only — handle separately.
            if source_path.name.upper() == "SKILL.MD":
                card = self._parse_superpowers_card(source_path, source_content)
                if card:
                    cards[card.id] = card
                continue

            skill_id = self._skill_id_from_source(source_path)
            manifest, content = self._split_frontmatter(source_content)
            tool_sections = self.parse_tool_sections(content, source_path)
            if not tool_sections:
                # Do not treat arbitrary markdown references as standalone skills.
                # Guidance skills should be authored via SKILL.md, while regular
                # markdown cards should define at least one ## Tool: section.
                continue
            all_sections.extend(tool_sections)
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
            cards[skill_id] = SkillCard(
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
            )
            for sec in tool_sections:
                self._tool_docs[sec["name"]] = self.tool_doc_from_section(
                    sec["content"]
                )
        self._skills = cards
        self._all_tool_sections = all_sections
        self._build_routing_index(cards)
        self._log.info(
            "Skill routing catalog ready: skills=%d profiles=%d rapidfuzz=%s",
            len(self._skills),
            len(self._routing_profiles),
            "enabled" if HAS_RAPIDFUZZ else "disabled",
        )
        if _fingerprint:
            self._save_to_cache(_fingerprint)

    def _build_routing_index(self, cards: dict[str, SkillCard]) -> None:
        profiles: dict[str, str] = {}
        token_index: dict[str, set[str]] = {}
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
            profiles[skill_id] = profile
            token_index[skill_id] = set(_skill_tokens(profile))
        self._routing_profiles = profiles
        self._routing_tokens = token_index

    def _fuzzy_similarity(self, query: str, profile: str) -> float:
        """Return normalized fuzzy similarity [0, 1] using RapidFuzz when available.

        Uses WRatio (weighted combination of multiple algorithms) for the best
        context-aware score, falling back to max(token_set_ratio, partial_ratio)
        if WRatio is unavailable, then to SequenceMatcher.
        """
        if not query or not profile:
            return 0.0
        if HAS_RAPIDFUZZ and _rf_fuzz is not None:
            # WRatio picks the best of token_sort, token_set, partial, and full
            # ratio depending on string length — ideal for skill routing where
            # the query is short and profiles are long, multi-word strings.
            wr = getattr(_rf_fuzz, "WRatio", None)
            if wr is not None:
                return float(wr(query, profile)) / 100.0
            # Fallback within rapidfuzz if WRatio is somehow absent
            ts = _rf_fuzz.token_set_ratio(query, profile)
            pr = _rf_fuzz.partial_ratio(query, profile)
            return max(float(ts), float(pr)) / 100.0
        return SequenceMatcher(None, query, profile).ratio()


    def route_query_to_skills(
        self,
        query: str,
        available_tool_names: Sequence[str],
        *,
        top_k: int = 3,
        min_score: float = 2.0,
        forced_skill_ids: Sequence[str] | None = None,
    ) -> SkillSelection:
        self.ensure_skill_catalog()

        query_l = (query or "").lower()
        query_tokens = set(_skill_tokens(query))
        available = set(available_tool_names)
        scored: list[tuple[float, SkillCard]] = []
        negated_chunks = [
            m.group(1).strip()
            for m in re.finditer(
                r"(?:do\s+not|don't|avoid|without)\s+([a-z0-9 _-]{2,80})",
                query_l,
            )
            if m.group(1).strip()
        ]

        for skill in self._skills.values():
            # Guidance-only cards (Superpowers) have no tool_names but still route.
            # We score them normally; they won't add tools but will surface guidance.

            score = 0.0
            ratio = 0.0
            query_coverage = 0.0
            if skill.id.replace("_", " ") in query_l or skill.name.lower() in query_l:
                score += 5.0

            exact_tool_hits = sum(
                1 for tool_name in skill.tool_names if tool_name.lower() in query_l
            )
            score += exact_tool_hits * 7.0

            alias_hits = query_tokens.intersection(
                {alias.lower() for alias in skill.aliases}
            )
            score += len(alias_hits) * 3.0

            trigger_phrase_hits = sum(
                1
                for trigger in skill.triggers
                if trigger and trigger.lower() in query_l
            )
            score += trigger_phrase_hits * 5.0

            # Similarity against each trigger phrase (not only exact substring).
            for trigger in skill.triggers:
                trigger_tokens = set(_skill_tokens(trigger))
                if not trigger_tokens or not query_tokens:
                    continue
                overlap = len(query_tokens.intersection(trigger_tokens))
                if overlap:
                    score += (overlap / len(trigger_tokens)) * 7.0

            # Explicit similarity against SKILL.md description.
            desc_tokens = set(_skill_tokens(skill.description))
            if desc_tokens and query_tokens:
                overlap = len(query_tokens.intersection(desc_tokens))
                if overlap:
                    score += (overlap / len(desc_tokens)) * 10.0
                    score += (overlap / len(query_tokens)) * 6.0

            profile_tokens = self._routing_tokens.get(skill.id, set(skill.keywords))
            if query_tokens and profile_tokens:
                overlap = len(query_tokens.intersection(profile_tokens))
                # Coverage-style similarity: reward skills that cover more query intent words.
                query_coverage = overlap / max(1, len(query_tokens))
                score += query_coverage * 6.0

            profile_text = self._routing_profiles.get(skill.id, "")
            if profile_text:
                ratio = self._fuzzy_similarity(query_l, profile_text)
                score += ratio * 4.0

            anti_phrase_hits = sum(
                1
                for anti_trigger in skill.anti_triggers
                if anti_trigger and anti_trigger.lower() in query_l
            )
            score -= anti_phrase_hits * 6.0

            keyword_hits = query_tokens.intersection(set(skill.keywords))
            score += len(keyword_hits)

            score += self._example_query_score(query_tokens, query_l, skill)

            if negated_chunks:
                neg_penalty = 0.0
                skill_tokens = profile_tokens.union(set(skill.keywords))
                for chunk in negated_chunks:
                    chunk_tokens = set(_skill_tokens(chunk))
                    if not chunk_tokens:
                        continue
                    overlap = len(chunk_tokens.intersection(skill_tokens))
                    if overlap:
                        neg_penalty += overlap * 2.5
                if neg_penalty:
                    score -= neg_penalty

            # Guidance-only skills should not eclipse executable skills unless the
            # user explicitly asks for that guidance skill by name/alias/trigger.
            if not skill.tool_names:
                explicit_guidance_hit = bool(
                    skill.id.replace("_", " ") in query_l
                    or skill.name.lower() in query_l
                    or alias_hits
                    or trigger_phrase_hits
                )
                strong_guidance_match = bool(
                    trigger_phrase_hits
                    or query_coverage >= 0.40
                    or ratio >= 0.48
                    or (ratio >= 0.38 and len(keyword_hits) >= 2)
                    or score >= (min_score * 3.0)
                )
                if explicit_guidance_hit:
                    score += 1.0
                elif strong_guidance_match:
                    # Keep guidance cards relevant but less dominant than executable skills.
                    score = (score * 0.75) - 0.25
                else:
                    score = min(score, min_score - 0.01)

            # Keep the legacy think tool opt-in: don't auto-route unless the
            # user clearly requests it.
            if skill.id == "think":
                explicit_think = bool(
                    "/think" in query_l
                    or "think tool" in query_l
                    or "use think" in query_l
                    or "call think" in query_l
                )
                if not explicit_think:
                    score = min(score, min_score - 0.01)

            if score >= min_score:
                scored.append((score, skill))

        scored.sort(
            key=lambda item: (
                item[0],
                len(item[1].tool_names),
                item[1].name,
            ),
            reverse=True,
        )

        forced_skills: list[SkillCard] = []
        forced_seen: set[str] = set()
        for raw_id in forced_skill_ids or []:
            sid = str(raw_id or "").strip()
            if not sid or sid in forced_seen:
                continue
            card = self._skills.get(sid)
            if card is None:
                continue
            forced_seen.add(sid)
            forced_skills.append(card)

        limit = max(1, int(top_k))
        if forced_skills:
            limit = max(limit, len(forced_skills))

        ranked_skills = [
            skill for _, skill in scored if skill.id not in forced_seen
        ]
        selected_skills = (forced_skills + ranked_skills)[:limit]
        selected_tools: list[str] = []
        seen_tools: set[str] = set()
        for skill in selected_skills:
            for tool_name in self._ordered_skill_tools(skill):
                if tool_name in available and tool_name not in seen_tools:
                    selected_tools.append(tool_name)
                    seen_tools.add(tool_name)

        fallback_tools = [
            tool_name
            for tool_name in available_tool_names
            if tool_name not in seen_tools
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
                entry = f'  [{rel_path} :: {heading}] {excerpt}'
                if len(entry) + 1 > remaining:
                    if added_for_skill == 0:
                        trim_budget = max(80, remaining - len(rel_path) - len(heading) - 12)
                        trimmed = excerpt[:trim_budget].rstrip()
                        if trimmed:
                            entry = f'  [{rel_path} :: {heading}] {trimmed}'
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
                candidate_text = (
                    f"{md_path.name} {heading} {snippet[:220]}".strip().lower()
                )
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
                overlap = len(query_tokens.intersection(rel_tokens)) / max(
                    1, len(query_tokens)
                )
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

    def _parse_markdown_h2_sections_fallback(
        self, content: str
    ) -> list[MarkdownSection]:
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
        # SKILL.md files (Superpowers format) use their *parent directory* as the id.
        if source_path.name.upper() == "SKILL.MD":
            raw = source_path.parent.name
            raw = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
            return f"sp__{raw}" if raw else "sp__skill"
        stem = source_path.stem
        stem = re.sub(r"^\d+[_-]*", "", stem)
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
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

    def _strip_skill_fence(self, content: str) -> str:
        """Strip the outer ```skill … ``` wrapper used by Superpowers SKILL.md files."""
        lines = content.splitlines()
        if lines and re.match(r"^```skill\s*$", lines[0].strip()):
            # Drop opening fence
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
        return "\n".join(lines)

    def _split_frontmatter(self, content: str) -> tuple[SkillManifest, str]:
        # Superpowers files wrap everything in a ```skill fence — strip it first.
        stripped = self._strip_skill_fence(content)
        if not stripped.startswith("---\n"):
            return SkillManifest(), stripped
        content = stripped

        lines = content.splitlines()
        end_idx: int | None = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            return SkillManifest(), content

        raw_yaml = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
        manifest = self._parse_frontmatter(raw_yaml)
        return manifest, body

    def _parse_frontmatter(self, raw_yaml: str) -> SkillManifest:
        """Parse YAML frontmatter using PyYAML when available, falling back to a simple
        line-by-line parser for environments without the dependency."""
        if HAS_YAML and _yaml is not None:
            try:
                parsed = _yaml.safe_load(raw_yaml)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}  # type: ignore[return-value]
            except Exception as exc:
                self._log.warning(
                    "YAML frontmatter parse failed, using fallback: %s", exc
                )
        return self._parse_frontmatter_lines_fallback(raw_yaml.splitlines())

    def _parse_frontmatter_lines_fallback(self, lines: list[str]) -> SkillManifest:
        """Minimal YAML-subset parser used when PyYAML is not installed."""
        manifest: SkillManifest = {}
        current_key: str | None = None

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if re.match(r"^[A-Za-z0-9_]+:\s*", stripped):
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
        desc_match = re.search(
            r"\*\*Description:\*\*\s*(.+?)(?:\n\n|\*\*|$)", content, re.DOTALL
        )
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
        param_pattern = re.compile(
            r"[-*]\s+(\w+)\s+\(([^,]+),\s*(required|optional)\):\s*(.+)"
        )
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
