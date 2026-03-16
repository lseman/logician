from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Literal

from .types import _SKILL_PREFIX_SEPARATORS
from ..runtime import SkillCard, SkillSelection

_ROUTING_META_TOOL_NAMES = (
    "describe_tool",
    "search_tools",
)


class RegistryRoutingMixin:
    """ToolRegistry mixin."""

    def _invalidate_skill_resolution_cache(self) -> None:
        self._skill_resolution_epoch += 1
        self._resolve_skill_candidate_ids_cached.cache_clear()

    def _sync_catalog_with_registered_tools(self) -> None:
        self._catalog.ensure_skill_catalog()
        changed = self._catalog.hydrate_metadata_only_python_skills(self._tools.values())
        if self._catalog.hydrate_tool_backed_skills(self._tools.values()):
            changed = True
        if changed:
            self._invalidate_skill_resolution_cache()

    def list_skills(self) -> list[SkillCard]:
        self._sync_catalog_with_registered_tools()
        return list(self._catalog.skills.values())

    def route_query_to_skills(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> SkillSelection:
        self._sync_catalog_with_registered_tools()
        forced = self._consume_forced_skill_ids()
        selection = self._catalog.route_query_to_skills(
            query,
            [tool.name for tool in self._iter_tools_for_prompt()],
            top_k=top_k,
            min_score=min_score,
            forced_skill_ids=forced,
        )
        return self._augment_selection_tool_visibility(query, selection)

    def skill_routing_prompt(
        self,
        query: str,
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
    ) -> tuple[str, SkillSelection]:
        self._sync_catalog_with_registered_tools()
        forced = self._consume_forced_skill_ids()
        available_tool_names = [tool.name for tool in self._iter_tools_for_prompt()]

        def _render_augmented_tool_prompt(**kwargs: Any) -> str:
            include_tool_names = kwargs.pop("include_tool_names", None)
            compact_fallback_tool_names = kwargs.pop("compact_fallback_tool_names", None)
            if include_tool_names is None and compact_fallback_tool_names is None:
                return self.tools_schema_prompt(**kwargs)
            (
                include_tool_names,
                compact_fallback_tool_names,
            ) = self._augment_tool_name_lists(
                query=query,
                selected_tool_names=include_tool_names,
                fallback_tool_names=compact_fallback_tool_names,
            )
            return self.tools_schema_prompt(
                **kwargs,
                include_tool_names=include_tool_names,
                compact_fallback_tool_names=compact_fallback_tool_names,
            )

        prompt, selection = self._catalog.skill_routing_prompt(
            query,
            available_tool_names,
            _render_augmented_tool_prompt,
            use_toon=use_toon,
            mode=mode,
            top_k=top_k,
            include_playbooks=include_playbooks,
            include_compact_fallback=include_compact_fallback,
            include_on_demand_context=include_on_demand_context,
            on_demand_context_max_chars=on_demand_context_max_chars,
            on_demand_context_max_skills=on_demand_context_max_skills,
            on_demand_context_max_files_per_skill=on_demand_context_max_files_per_skill,
            # Keep forced skills as explicit context for the next routing pass only.
            forced_skill_ids=forced,
        )
        selection = self._augment_selection_tool_visibility(query, selection)
        return prompt, selection

    def _augment_selection_tool_visibility(
        self,
        query: str,
        selection: SkillSelection,
    ) -> SkillSelection:
        if not selection.selected_skills:
            return selection

        selected_tools, fallback_tools = self._augment_tool_name_lists(
            query=query,
            selected_tool_names=selection.selected_tools,
            fallback_tool_names=selection.fallback_tools,
        )
        return SkillSelection(
            query=selection.query,
            selected_skills=list(selection.selected_skills),
            selected_tools=selected_tools,
            fallback_tools=fallback_tools,
        )

    def _augment_tool_name_lists(
        self,
        *,
        query: str,
        selected_tool_names: list[str] | tuple[str, ...] | None,
        fallback_tool_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[list[str], list[str] | None]:
        selected_tools: list[str] = []
        seen: set[str] = set()

        for tool_name in self._explicit_tool_mentions(query):
            if tool_name in self._tools and tool_name not in seen:
                selected_tools.append(tool_name)
                seen.add(tool_name)

        for tool_name in selected_tool_names or []:
            tname = str(tool_name)
            if tname in self._tools and tname not in seen:
                selected_tools.append(tname)
                seen.add(tname)

        for tool_name in _ROUTING_META_TOOL_NAMES:
            if tool_name in self._tools and tool_name not in seen:
                selected_tools.append(tool_name)
                seen.add(tool_name)

        if fallback_tool_names is None:
            return selected_tools, None
        fallback_tools = [
            str(tool_name)
            for tool_name in fallback_tool_names
            if str(tool_name) in self._tools and str(tool_name) not in seen
        ]
        return selected_tools, fallback_tools

    def _explicit_tool_mentions(self, query: str) -> list[str]:
        text = str(query or "").strip()
        if not text:
            return []

        lower = text.lower()
        mentioned: list[str] = []
        seen: set[str] = set()

        for tool in self._iter_tools_for_prompt():
            name = str(getattr(tool, "name", "") or "").strip()
            if not name or name in seen:
                continue
            lname = name.lower()
            matched = bool(
                re.search(
                    rf"(?<![a-z0-9_]){re.escape(lname)}(?![a-z0-9_])",
                    lower,
                )
            )
            if matched:
                mentioned.append(name)
                seen.add(name)
        return mentioned

    def _consume_forced_skill_ids(self) -> list[str]:
        if not self._forced_skill_ids:
            return []
        out = list(dict.fromkeys(self._forced_skill_ids))
        self._forced_skill_ids.clear()
        self._forced_skill_reason = ""
        return out

    def _score_skill_name_match(self, query: str, card: SkillCard) -> float:
        q = (query or "").strip().lower()
        if not q:
            return 0.0
        fields = [card.id.replace("_", " "), card.name, *card.aliases]
        best = 0.0
        for field in fields:
            f = str(field or "").strip().lower()
            if not f:
                continue
            ratio = self._catalog._fuzzy_similarity(q, f)
            if f == q:
                ratio = max(ratio, 1.0)
            elif f.startswith(q) or q.startswith(f):
                ratio = max(ratio, 0.92)
            best = max(best, ratio)
        return best

    @staticmethod
    def _normalize_skill_lookup(value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        for prefix in ("superpowers", "superpower", "sp"):
            if not text.startswith(prefix):
                continue
            tail = text[len(prefix) :].lstrip()
            if tail[:1] in _SKILL_PREFIX_SEPARATORS:
                text = tail[1:].lstrip()
                break
        while text.startswith("sp__"):
            text = text[4:]
        text = text.replace("_", " ").replace("-", " ").replace("/", " ")
        return " ".join(text.split())

    @lru_cache(maxsize=2048)
    def _resolve_skill_candidate_ids_cached(
        self,
        query: str,
        top_k: int,
        epoch: int,  # included for deterministic invalidation
    ) -> tuple[str, ...]:
        _ = epoch
        q = (query or "").strip()
        if not q:
            return tuple()
        q_l = q.lower()
        q_norm = self._normalize_skill_lookup(q)

        direct_ids: list[str] = []
        seen_direct: set[str] = set()
        for card in self._catalog.skills.values():
            names = [card.id, card.id.replace("_", " "), card.name, *card.aliases]
            norm_names = {
                self._normalize_skill_lookup(str(name or ""))
                for name in names
                if str(name or "").strip()
            }
            if any(str(name or "").strip().lower() == q_l for name in names) or (
                q_norm and q_norm in norm_names
            ):
                if card.id not in seen_direct:
                    seen_direct.add(card.id)
                    direct_ids.append(card.id)
        if direct_ids:
            return tuple(direct_ids[: max(1, top_k)])

        # For explicit skill lookups like "brainstorm", prioritize fuzzy
        # name/alias matching before semantic routing across all skills.
        strong_fuzzy: list[tuple[float, str]] = []
        for card in self._catalog.skills.values():
            score = self._score_skill_name_match(q, card)
            if score >= 0.72:
                strong_fuzzy.append((score, card.id))
        if strong_fuzzy:
            strong_fuzzy.sort(key=lambda item: item[0], reverse=True)
            return tuple(skill_id for _, skill_id in strong_fuzzy[: max(1, top_k)])

        routed = self._catalog.route_query_to_skills(
            q,
            [tool.name for tool in self._iter_tools_for_prompt()],
            top_k=max(3, top_k),
            min_score=0.28,
        ).selected_skills
        if routed:
            return tuple(card.id for card in routed[: max(1, top_k)])

        fuzzy: list[tuple[float, str]] = []
        for card in self._catalog.skills.values():
            score = self._score_skill_name_match(q, card)
            if score >= 0.45:
                fuzzy.append((score, card.id))
        fuzzy.sort(key=lambda item: item[0], reverse=True)
        out_ids: list[str] = []
        for _, skill_id in fuzzy:
            out_ids.append(skill_id)
            if len(out_ids) >= max(1, top_k):
                break
        return tuple(out_ids)

    def _resolve_skill_candidates(self, query: str, *, top_k: int) -> list[SkillCard]:
        self._sync_catalog_with_registered_tools()
        q = (query or "").strip()
        if not q:
            return []
        ids = self._resolve_skill_candidate_ids_cached(
            q,
            max(1, int(top_k)),
            self._skill_resolution_epoch,
        )
        out: list[SkillCard] = []
        for skill_id in ids:
            card = self._catalog.skills.get(skill_id)
            if card is None:
                continue
            out.append(card)
            if len(out) >= max(1, top_k):
                break
        return out

    def _invoke_skill_tool(
        self,
        skill: str,
        reason: str = "",
        top_k: int = 1,
        args: str = "",
    ) -> str:
        q = str(skill or "").strip()
        if not q:
            return json.dumps(
                {"status": "error", "error": "Parameter 'skill' is required."},
                ensure_ascii=False,
            )
        k = max(1, min(3, int(top_k or 1)))
        matches = self._resolve_skill_candidates(q, top_k=k)
        if not matches:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"No skill match found for '{q}'.",
                },
                ensure_ascii=False,
            )

        forced_ids = [card.id for card in matches]
        self._forced_skill_ids = list(dict.fromkeys(forced_ids))
        self._forced_skill_reason = str(reason or "").strip()

        args_str = str(args or "").strip()
        if args_str:
            target_skill = matches[0]
            if not target_skill.tool_names:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Skill '{target_skill.id}' has no executable tools to run with args.",
                        "forced_skill_ids": self._forced_skill_ids,
                    },
                    ensure_ascii=False,
                )
            primary_tool = target_skill.tool_names[0]
            if hasattr(self, "call_tool"):
                try:
                    parsed_args = json.loads(args_str)
                    if not isinstance(parsed_args, dict):
                        raise ValueError("Args must be a JSON dictionary.")
                    execution_result = getattr(self, "call_tool")(primary_tool, **parsed_args)
                    if not isinstance(execution_result, str):
                        execution_result = json.dumps(execution_result, ensure_ascii=False)
                    return execution_result
                except Exception as e:
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Execution of {primary_tool} failed: {e}",
                            "forced_skill_ids": self._forced_skill_ids,
                        },
                        ensure_ascii=False,
                    )

        return json.dumps(
            {
                "status": "ok",
                "forced_skill_ids": self._forced_skill_ids,
                "reason": self._forced_skill_reason,
                "message": (
                    "Forced skill(s) will be injected into the next routing pass."
                ),
                "matches": [
                    {
                        "id": card.id,
                        "name": card.name,
                        "summary": card.summary[:180],
                    }
                    for card in matches
                ],
            },
            ensure_ascii=False,
        )
