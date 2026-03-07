import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    from src.tools.catalog import SkillCatalog
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

    from src.tools.catalog import SkillCatalog
    from src.tools import Context, ToolRegistry, ToolCall


def _registry() -> ToolRegistry:
    return ToolRegistry(auto_load_from_skills=True)


class SkillRoutingTests(unittest.TestCase):
    def test_guidance_routing_uses_related_markdown_files(self) -> None:
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

            selection = catalog.route_query_to_skills(
                "let's do blue sky ideation for this product",
                available_tool_names=[],
                top_k=1,
            )

            self.assertTrue(selection.selected_skills)
            self.assertEqual(selection.selected_skills[0].id, "sp__design_discovery")

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

        self.assertIn("forecasting", skills)
        forecasting = skills["forecasting"]
        self.assertEqual(forecasting.name, "Forecasting")
        self.assertIn("stat_forecast", forecasting.tool_names)
        self.assertIn("neural_forecast", forecasting.tool_names)
        self.assertIn("forecast the next", forecasting.triggers)
        self.assertIn("plot diagnostics only", forecasting.anti_triggers)
        self.assertEqual(
            forecasting.preferred_tools[:3],
            ["suggest_models", "stat_forecast", "neural_forecast"],
        )

    def test_skill_router_prefers_forecasting_for_forecast_queries(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Forecast the next 24 steps and compare neural forecast models.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertTrue(selected_ids)
        self.assertEqual(selected_ids[0], "forecasting")
        self.assertEqual(selection.selected_tools[0], "suggest_models")
        self.assertIn("stat_forecast", selection.selected_tools)
        self.assertIn("neural_forecast", selection.selected_tools)

    def test_skill_router_prefers_data_loading_for_csv_queries(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Load a CSV file and inspect the dataset columns before analysis.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertIn("data_loading", selected_ids)
        self.assertIn("load_csv_data", selection.selected_tools)

    def test_skill_routing_prompt_includes_active_skill_summary(self) -> None:
        registry = _registry()
        prompt, selection = registry.skill_routing_prompt(
            "Plot diagnostics and inspect anomalies in the series.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertIn("ACTIVE SKILLS FOR THIS REQUEST", prompt)
        self.assertTrue(selected_ids)
        self.assertTrue(
            any(skill_id in {"analysis", "plotting"} for skill_id in selected_ids)
        )
        self.assertTrue(
            any(
                tool_name in selection.selected_tools
                for tool_name in ("plot_diagnostics", "detect_anomalies")
            )
        )
        self.assertIn("avoid when:", prompt)

    def test_anti_triggers_reduce_incorrect_skill_matches(self) -> None:
        registry = _registry()
        selection = registry.route_query_to_skills(
            "Load a CSV file, inspect dataset columns, and do not plot diagnostics yet.",
            top_k=2,
        )

        selected_ids = [skill.id for skill in selection.selected_skills]
        self.assertTrue(selected_ids)
        self.assertEqual(selected_ids[0], "data_loading")
        self.assertNotEqual(selected_ids[0], "plotting")

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


if __name__ == "__main__":
    unittest.main()
