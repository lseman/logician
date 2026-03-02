from __future__ import annotations

import re
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


def _skill_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower())
        if tok not in _SKILL_STOPWORDS
    ]


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
    def __init__(
        self, *, skills_md_path: Path, skills_dir_path: Path, log: Any
    ) -> None:
        self.skills_md_path = skills_md_path
        self.skills_dir_path = skills_dir_path
        self._log = log
        self._skills: dict[str, SkillCard] = {}
        self._tool_docs: dict[str, str] = {}
        self._all_tool_sections: list[ToolSection] = []

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

    def iter_skills_sources(self) -> list[Path]:
        """Return all skill .md files, scanning subdirectories recursively.

        Search order:
        1. skills_md_path if it is a directory (e.g. the ``skills/`` folder).
        2. A ``skills/`` sibling of the SKILLS.md file.
        3. skills_md_path itself if it is a single file (backward-compat).
        4. skills_dir_path as a final fallback.
        """

        def _collect(directory: Path) -> list[Path]:
            return sorted(directory.rglob("*.md"))

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
        contents = self.read_skill_source_contents()
        if contents:
            self.build_skill_catalog(contents)

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
        # Aliases: tokens derived from the slug name.
        slug_tokens = re.split(r"[-_]+", str(raw_name).lower())
        aliases = [tok for tok in slug_tokens if len(tok) > 2]
        # Playbooks: H2 section headings from the body.
        sections = self.parse_markdown_h2_sections(body)
        playbooks = [sec["heading"] for sec in sections if sec["heading"]]
        keywords = self._skill_keywords(
            summary,
            playbooks,
            [],
            triggers=triggers,
            example_queries=[],
            aliases=aliases,
        )
        return SkillCard(
            id=skill_id,
            name=display_name,
            summary=summary,
            source_path=str(source_path),
            tool_names=[],  # guidance-only — no executable tools
            playbooks=playbooks,
            keywords=keywords,
            aliases=aliases,
            triggers=triggers,
            anti_triggers=[],
            preferred_tools=[],
            example_queries=[],
            when_not_to_use=[],
            next_skills=[],
        )

    def build_skill_catalog(self, skill_contents: list[tuple[Path, str]]) -> None:
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
            all_sections.extend(tool_sections)
            summary = manifest.get("summary") or self._skill_summary(content)
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
            )
            cards[skill_id] = SkillCard(
                id=skill_id,
                name=manifest.get("name") or self._skill_name_from_source(source_path),
                summary=summary,
                source_path=str(source_path),
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

    def route_query_to_skills(
        self,
        query: str,
        available_tool_names: Sequence[str],
        *,
        top_k: int = 3,
        min_score: float = 2.0,
    ) -> SkillSelection:
        self.ensure_skill_catalog()

        query_l = (query or "").lower()
        query_tokens = set(_skill_tokens(query))
        available = set(available_tool_names)
        scored: list[tuple[float, SkillCard]] = []

        for skill in self._skills.values():
            # Guidance-only cards (Superpowers) have no tool_names but still route.
            # We score them normally; they won't add tools but will surface guidance.

            score = 0.0
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

            anti_phrase_hits = sum(
                1
                for anti_trigger in skill.anti_triggers
                if anti_trigger and anti_trigger.lower() in query_l
            )
            score -= anti_phrase_hits * 6.0

            keyword_hits = query_tokens.intersection(set(skill.keywords))
            score += len(keyword_hits)

            score += self._example_query_score(query_tokens, query_l, skill)

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

        selected_skills = [skill for _, skill in scored[: max(1, int(top_k))]]
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
    ) -> tuple[str, SkillSelection]:
        selection = self.route_query_to_skills(
            query,
            available_tool_names,
            top_k=top_k,
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

        fallback = selection.fallback_tools[:12] if include_compact_fallback else None
        prompt = render_tool_prompt(
            use_toon=use_toon,
            mode=mode,
            include_tool_names=selection.selected_tools,
            compact_fallback_tool_names=fallback,
        )
        return ("\n".join(lines) + prompt, selection)

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
        deduped: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                deduped.append(tok)
        return deduped

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
