import importlib.util
import sys
import types
import unittest
from pathlib import Path

try:
    from src.tools import ToolRegistry
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

    from src.tools import ToolRegistry


def _registry() -> ToolRegistry:
    return ToolRegistry(auto_load_from_skills=True)


class SkillRoutingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
