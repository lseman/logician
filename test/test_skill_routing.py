import importlib.util
import json
import logging
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

try:
    from src.tools.registry.catalog import SkillCatalog
    from src.tools import Context, ToolRegistry, ToolCall
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(src_dir)]
    sys.modules.setdefault("src", src_pkg)

    logging_spec = importlib.util.spec_from_file_location(
        "src.logging_utils", src_dir / "logging_utils.py"
    )
    assert logging_spec and logging_spec.loader
    logging_module = importlib.util.module_from_spec(logging_spec)
    sys.modules["src.logging_utils"] = logging_module
    logging_spec.loader.exec_module(logging_module)

    tools_spec = importlib.util.spec_from_file_location(
        "src.tools",
        src_dir / "tools" / "__init__.py",
        submodule_search_locations=[str(src_dir / "tools")],
    )
    assert tools_spec and tools_spec.loader
    tools_module = importlib.util.module_from_spec(tools_spec)
    sys.modules["src.tools"] = tools_module
    tools_spec.loader.exec_module(tools_module)

    from src.tools.registry.catalog import SkillCatalog
    from src.tools import Context, ToolRegistry, ToolCall


def _registry(*, load_lazy_skill_groups: tuple[str, ...] = ()) -> ToolRegistry:
    registry = ToolRegistry(auto_load_from_skills=False)
    for group in load_lazy_skill_groups:
        registry.activate_lazy_skill_group(group)
    registry.load_tools_from_skills()
    return registry


class SkillRoutingTests(unittest.TestCase):
    def test_symlinked_superpowers_skill_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skills_root = root / "skills"
            plugin_root = root / "plugins" / "superpowers" / "skills"
            skill_dir = plugin_root / "brainstorming"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                """```skill
---
name: brainstorming
description: Use before implementation to explore alternatives.
---

## Process
Generate options first.
```""",
                encoding="utf-8",
            )

            skills_root.mkdir(parents=True, exist_ok=True)
            link_path = skills_root / "10_superpowers"
            try:
                os.symlink(plugin_root, link_path, target_is_directory=True)
            except (AttributeError, NotImplementedError, OSError):
                self.skipTest("Symlink creation not available on this platform.")

            catalog = SkillCatalog(
                skills_md_path=skills_root,
                skills_dir_path=skills_root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            self.assertIn("sp__brainstorming", catalog.skills)

    def test_unreadable_markdown_source_does_not_abort_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good_skill = root / "brainstorming"
            good_skill.mkdir(parents=True, exist_ok=True)
            (good_skill / "SKILL.md").write_text(
                """```skill
---
name: brainstorming
description: Use before implementation to explore alternatives.
---

## Process
Generate options first.
```""",
                encoding="utf-8",
            )
            # Intentionally invalid UTF-8 markdown source; should be skipped.
            (root / "broken.md").write_bytes(b"\xff\xfe\xfa")

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            self.assertIn("sp__brainstorming", catalog.skills)

    def test_reference_markdown_stays_lazy_until_skill_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "design-discovery"
            refs = skill_dir / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                """```skill
---
name: design-discovery
description: Use before implementation to clarify requirements.
---

## Overview
Ask a few clarifying questions before building.
```""",
                encoding="utf-8",
            )
            (refs / "ideation-glossary.md").write_text(
                """## Blue Sky Ideation
Use blue-sky ideation to explore novel concepts and divergent options.""",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            prompt, selection = catalog.skill_routing_prompt(
                "Clarify requirements before implementation and use blue sky ideation for this product.",
                [],
                lambda **_: "",
                top_k=1,
            )

            self.assertTrue(selection.selected_skills)
            self.assertEqual(selection.selected_skills[0].id, "sp__design_discovery")
            self.assertIn("ideation-glossary.md", prompt)
            self.assertIn("Blue Sky Ideation", prompt)

    def test_brainstorm_token_routes_to_brainstorming_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sp_dir = root / "brainstorming"
            sp_dir.mkdir(parents=True, exist_ok=True)
            (sp_dir / "SKILL.md").write_text(
                """```skill
---
name: brainstorming
description: Use this skill before creative design and implementation.
---

## Process
Ask clarifying questions, then propose design options.
```""",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            selection = catalog.route_query_to_skills(
                "brainstorm",
                available_tool_names=[],
                top_k=1,
            )

            self.assertTrue(selection.selected_skills)
            self.assertEqual(selection.selected_skills[0].id, "sp__brainstorming")

    def test_invoke_skill_forces_next_routing_pass(self) -> None:
        registry = _registry()
        registry.install_context(Context())

        invoke_result = registry.execute(
            ToolCall(
                id="test_invoke",
                name="invoke_skill",
                arguments={"skill": "brainstorm", "reason": "explicit user request"},
            ),
            use_toon=False,
        )
        payload = json.loads(invoke_result)
        self.assertEqual(payload.get("status"), "ok")
        self.assertIn("sp__brainstorming", payload.get("forced_skill_ids", []))

        selection = registry.route_query_to_skills(
            "continue",
            top_k=1,
        )
        self.assertTrue(selection.selected_skills)
        self.assertEqual(selection.selected_skills[0].id, "sp__brainstorming")

    def test_non_skill_reference_markdown_not_promoted_to_skill_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "debugging"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                """```skill
---
name: Debugging
description: Use when debugging flaky tests.
---

## Steps
1. Reproduce.
2. Trace root cause.
```""",
                encoding="utf-8",
            )
            (skill_dir / "root-cause-tracing.md").write_text(
                """## Root Cause Tracing
Walk backwards from the failure site until you find the first bad input.""",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            self.assertEqual(set(catalog.skills.keys()), {"sp__debugging"})

    def test_catalog_startup_scans_entrypoints_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "debugging"
            refs = skill_dir / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                """```skill
---
name: Debugging
description: Use when debugging flaky tests.
---

## Steps
1. Reproduce.
2. Trace root cause.
```""",
                encoding="utf-8",
            )
            (skill_dir / "README.md").write_text("# Debugging helpers", encoding="utf-8")
            (refs / "root-cause-tracing.md").write_text(
                "## Root Cause Tracing\nStart from the symptom and walk backward.",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )

            sources = catalog.iter_skills_sources()
            self.assertEqual(sources, [skill_dir / "SKILL.md"])

    def test_folder_skill_with_scripts_dir_hydrates_tool_backed_card(self) -> None:
        prev_skills_dir = os.environ.get("AGENT_SKILLS_DIR")
        prev_skills_md = os.environ.get("AGENT_SKILLS_MD_PATH")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                skills_root = root / "skills"
                skill_dir = skills_root / "rag"
                scripts_dir = skill_dir / "scripts"
                scripts_dir.mkdir(parents=True, exist_ok=True)
                (skill_dir / "SKILL.md").write_text(
                    """---
name: RAG
description: Use for ingestion and semantic retrieval.
preferred_tools:
  - demo_lookup
---

## Scope

Use when: the user wants semantic retrieval over indexed content.
""",
                    encoding="utf-8",
                )
                (scripts_dir / "lookup.py").write_text(
                    '''def demo_lookup(query: str) -> str:
    """Search indexed content."""
    return query

__tools__ = [demo_lookup]
''',
                    encoding="utf-8",
                )

                os.environ["AGENT_SKILLS_DIR"] = str(skills_root)
                os.environ.pop("AGENT_SKILLS_MD_PATH", None)

                registry = ToolRegistry(auto_load_from_skills=True)
                registry.install_context(Context())
                skills = {skill.id: skill for skill in registry.list_skills()}

                self.assertIn("rag", skills)
                self.assertNotIn("sp__rag", skills)
                self.assertEqual(skills["rag"].name, "RAG")
                self.assertIn("demo_lookup", skills["rag"].tool_names)
                tool = registry.get("demo_lookup")
                self.assertIsNotNone(tool)
                assert tool is not None
                self.assertEqual(tool.skill_id, "rag")
        finally:
            if prev_skills_dir is None:
                os.environ.pop("AGENT_SKILLS_DIR", None)
            else:
                os.environ["AGENT_SKILLS_DIR"] = prev_skills_dir
            if prev_skills_md is None:
                os.environ.pop("AGENT_SKILLS_MD_PATH", None)
            else:
                os.environ["AGENT_SKILLS_MD_PATH"] = prev_skills_md

    def test_on_demand_context_is_loaded_for_selected_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "systematic-debugging"
            refs = skill_dir / "references"
            refs.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                """```skill
---
name: systematic-debugging
description: Use when bugs are flaky or hard to reproduce.
---

## Overview
Find root cause before proposing fixes.
```""",
                encoding="utf-8",
            )
            (refs / "root-cause-tracing.md").write_text(
                """## Root Cause Tracing
Walk backward from the symptom through each caller until the first invalid value appears.""",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            prompt, selection = catalog.skill_routing_prompt(
                "Use systematic debugging to trace root cause through the call stack before any fix.",
                [],
                lambda **_: "",
                top_k=1,
            )

            self.assertTrue(selection.selected_skills)
            self.assertEqual(selection.selected_skills[0].id, "sp__systematic_debugging")
            self.assertIn("ON-DEMAND SKILL CONTEXT:", prompt)
            self.assertIn("root-cause-tracing.md", prompt)
            self.assertIn("Walk backward from the symptom", prompt)

    def test_skill_catalog_reads_explicit_manifest_metadata(self) -> None:
        registry = _registry()
        skills = {skill.id: skill for skill in registry.list_skills()}

        self.assertIn("firecrawl", skills)
        firecrawl = skills["firecrawl"]
        self.assertEqual(firecrawl.name, "Firecrawl")
        self.assertIn("firecrawl_search", firecrawl.tool_names)
        self.assertIn("firecrawl_crawl", firecrawl.tool_names)
        self.assertIn("crawl this site", firecrawl.triggers)
        self.assertEqual(
            firecrawl.preferred_tools[:3],
            ["firecrawl_search", "firecrawl_crawl", "firecrawl_scrape"],
        )

    def test_real_rag_folder_skill_groups_rag_tools(self) -> None:
        registry = _registry()
        skills = {skill.id: skill for skill in registry.list_skills()}

        self.assertIn("rag", skills)
        rag = skills["rag"]
        self.assertEqual(rag.name, "RAG")
        self.assertIn("rag_add_file", rag.tool_names)
        self.assertIn("rag_search", rag.tool_names)
        self.assertIn("rag_tuning_status", rag.tool_names)
        self.assertIn(
            "document ingestion, retrieval, and retrieval-quality tuning",
            rag.description,
        )

    def test_skill_router_prefers_rag_for_retrieval_queries(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Search indexed documents in the rag store and inspect rag coverage before retuning.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertTrue(selected_ids)
        self.assertEqual(selected_ids[0], "rag")
        self.assertEqual(selection.selected_tools[0], "rag_search")
        self.assertIn("rag_add_file", selection.selected_tools)

    def test_skill_router_prefers_firecrawl_for_site_queries(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Crawl this site and scrape all pages under this URL before extracting structured content.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertIn("firecrawl", selected_ids)
        self.assertIn("firecrawl_crawl", selection.selected_tools)

    def test_skill_routing_prompt_includes_active_skill_summary(self) -> None:
        registry = _registry()
        prompt, selection = registry.skill_routing_prompt(
            "Search the docs site before crawling it so we can extract the relevant pages.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertIn("ACTIVE SKILLS FOR THIS REQUEST", prompt)
        self.assertTrue(selected_ids)
        self.assertIn("firecrawl", selected_ids)
        self.assertIn("firecrawl_search", selection.selected_tools)
        self.assertIn("avoid when:", prompt)

    def test_skill_routing_keeps_meta_tools_available_during_routed_turns(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Search indexed documents in the rag store and inspect rag coverage.",
            top_k=2,
        )

        self.assertIn("describe_tool", selection.selected_tools)
        self.assertIn("search_tools", selection.selected_tools)

    def test_skill_routing_prioritizes_explicit_tool_mentions(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Use firecrawl_search to search the docs site before crawling it.",
            top_k=2,
        )

        self.assertTrue(selection.selected_skills)
        self.assertEqual(selection.selected_tools[0], "firecrawl_search")

    def test_skill_routing_prompt_marks_routed_tool_list_as_subset(self) -> None:
        registry = _registry()
        prompt, selection = registry.skill_routing_prompt(
            "Use firecrawl_search to search the docs site before crawling it.",
            top_k=2,
        )

        self.assertTrue(selection.selected_skills)
        self.assertIn("routed subset", prompt)
        self.assertIn("describe_tool", prompt)

    def test_superpowers_guidance_routes_on_fuzzy_intent_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sp_dir = root / "strategic-thinking"
            sp_dir.mkdir(parents=True, exist_ok=True)
            (sp_dir / "SKILL.md").write_text(
                """```skill
---
name: Strategic Thinking
description: Use when a task is ambiguous and needs step-by-step planning.
---

## Playbook
1. Decompose the task before acting.
2. Validate assumptions.

Use when: the request is ambiguous and needs careful decomposition.
Avoid when: the task is already straightforward.
```""",
                encoding="utf-8",
            )

            catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=logging.getLogger("test.skill_routing"),
            )
            catalog.ensure_skill_catalog()

            selection = catalog.route_query_to_skills(
                "This is ambiguous, so do careful step by step planning before running tools.",
                available_tool_names=[],
                top_k=1,
            )

            self.assertTrue(selection.selected_skills)
            self.assertEqual(selection.selected_skills[0].id, "sp__strategic_thinking")

    def test_usage_and_recency_bias_contribute_to_ranking(self) -> None:
        registry = _registry()
        # Ensure the catalog is hydrated before recording usage stats.
        _ = registry.list_skills()
        registry._catalog.note_skill_usage("rag", timestamp=time.time() - 30)
        registry._catalog.note_skill_usage("rag", timestamp=time.time() - 10)
        registry._catalog.note_skill_usage("rag")

        registry.route_query_to_skills(
            "Search indexed documents in the rag store and inspect rag coverage.",
            top_k=2,
        )
        breakdown = registry._catalog.get_last_skill_score_breakdown()
        rag_breakdown = breakdown.get("rag", {})
        contrib = rag_breakdown.get("contributions", {})
        self.assertGreater(float(contrib.get("usage_bias", 0.0)), 0.0)
        self.assertGreater(float(contrib.get("recency_bias", 0.0)), 0.0)

    def test_routing_weights_can_be_overridden_via_env(self) -> None:
        prev = os.environ.get("AGENT_SKILL_ROUTING_WEIGHTS")
        try:
            os.environ["AGENT_SKILL_ROUTING_WEIGHTS"] = json.dumps(
                {"bm25": 0.5, "dense": 0.0}
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                catalog = SkillCatalog(
                    skills_md_path=root,
                    skills_dir_path=root,
                    log=logging.getLogger("test.skill_routing"),
                )
                self.assertAlmostEqual(catalog._routing_weights["bm25"], 0.5)
                self.assertAlmostEqual(catalog._routing_weights["dense"], 0.0)
        finally:
            if prev is None:
                os.environ.pop("AGENT_SKILL_ROUTING_WEIGHTS", None)
            else:
                os.environ["AGENT_SKILL_ROUTING_WEIGHTS"] = prev

    def test_think_skill_routes_on_ambiguous_decompose_query(self) -> None:
        """Think or orchestrator skill should win on deliberate-reasoning / decompose queries."""
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Think through this ambiguous multi-step task and decompose it before taking any action.",
            top_k=3,
        )
        selected_ids = [s.id for s in selection.selected_skills]
        self.assertTrue(selected_ids, "Expected at least one skill to be selected")
        # think or orchestrator must appear — both are valid for deliberate reasoning queries
        self.assertTrue(
            any(sid in selected_ids for sid in ("think", "orchestrator", "sp__think")),
            f"Expected think or orchestrator in {selected_ids}",
        )

    def test_lazy_skill_groups_are_hidden_until_activated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "lazy_demo" / "demo_skill"
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: Demo Skill\ndescription: Lazy-loaded demo skill.\n---\n",
                encoding="utf-8",
            )
            (scripts_dir / "demo.py").write_text(
                "def demo_tool() -> str:\n"
                "    return _safe_json({'status': 'ok'})\n\n"
                "__tools__ = [demo_tool]\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            skills = {skill.id for skill in registry.list_skills()}
            self.assertNotIn("demo_skill", skills)

            changed, group = registry.activate_lazy_skill_group("demo")
            self.assertTrue(changed)
            self.assertEqual(group, "demo")
            registry.reload_skills()

            skills = {skill.id for skill in registry.list_skills()}
            self.assertIn("demo_skill", skills)

    def test_orchestrator_routes_on_fan_out_read_query(self) -> None:
        """orchestrator skill should route when user asks for fan-out then consolidate."""
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Read these five files in parallel then consolidate the findings before writing.",
            top_k=3,
        )
        selected_ids = [s.id for s in selection.selected_skills]
        self.assertTrue(selected_ids, "Expected at least one skill to be selected")
        self.assertTrue(
            any(sid in selected_ids for sid in ("orchestrator", "sp__orchestrator")),
            f"Expected orchestrator in {selected_ids}",
        )

    def test_memory_management_routes_on_context_checkpoint_query(self) -> None:
        """memory_management skill should route on session-checkpoint / context-length queries."""
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Create a session checkpoint before the next phase — context is getting very long.",
            top_k=3,
        )
        selected_ids = [s.id for s in selection.selected_skills]
        self.assertTrue(selected_ids, "Expected at least one skill to be selected")
        self.assertTrue(
            any(sid in selected_ids for sid in ("memory_management", "sp__memory_management")),
            f"Expected memory_management in {selected_ids}",
        )


if __name__ == "__main__":
    unittest.main()
