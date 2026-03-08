import json
import sys
import unittest
from pathlib import Path
from typing import Literal

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.tools import Context, Tool, ToolCall, ToolParameter, ToolRegistry


class ToolRegistryEnhancementsTests(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(auto_load_from_skills=False)
        registry.install_context(Context())
        return registry

    def test_argument_type_coercion_for_integer_parameters(self) -> None:
        registry = self._registry()
        registry.register(
            name="add_numbers",
            description="Add two numbers.",
            parameters=[
                ToolParameter(name="a", type="integer", description="", required=True),
                ToolParameter(name="b", type="int", description="", required=True),
            ],
            function=lambda a, b: {"status": "ok", "sum": a + b},
        )

        raw = registry.execute(
            ToolCall(
                id="call_1",
                name="add_numbers",
                arguments={"a": "2", "b": "3"},
            ),
            use_toon=False,
        )
        payload = json.loads(raw)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("sum"), 5)

    def test_argument_schema_validation_for_missing_and_unknown_fields(self) -> None:
        registry = self._registry()
        registry.register(
            name="echo_pair",
            description="Echo two fields.",
            parameters=[
                ToolParameter(name="left", type="string", description="", required=True),
                ToolParameter(name="right", type="string", description="", required=True),
            ],
            function=lambda left, right: {"status": "ok", "pair": [left, right]},
        )

        missing_raw = registry.execute(
            ToolCall(
                id="call_missing",
                name="echo_pair",
                arguments={"left": "A"},
            ),
            use_toon=False,
        )
        missing_payload = json.loads(missing_raw)
        self.assertEqual(missing_payload.get("status"), "error")
        self.assertEqual(missing_payload.get("error_type"), "schema_validation_failed")
        self.assertIn("right", missing_payload.get("missing_required", []))

        unknown_raw = registry.execute(
            ToolCall(
                id="call_unknown",
                name="echo_pair",
                arguments={"left": "A", "right": "B", "extra": "C"},
            ),
            use_toon=False,
        )
        unknown_payload = json.loads(unknown_raw)
        self.assertEqual(unknown_payload.get("status"), "error")
        self.assertEqual(unknown_payload.get("error_type"), "schema_validation_failed")
        self.assertIn("extra", unknown_payload.get("unknown_arguments", []))

    def test_describe_and_search_builtin_tools(self) -> None:
        registry = self._registry()
        registry._register_builtin_tools()
        registry.register(
            name="run_tests",
            description="Run project tests.",
            parameters=[],
            function=lambda: {"status": "ok"},
        )

        describe_raw = registry.execute(
            ToolCall(
                id="call_describe",
                name="describe_tool",
                arguments={"name": "run_tests"},
            ),
            use_toon=False,
        )
        describe_payload = json.loads(describe_raw)
        self.assertEqual(describe_payload.get("status"), "ok")
        self.assertEqual(describe_payload.get("tool", {}).get("name"), "run_tests")

        search_raw = registry.execute(
            ToolCall(
                id="call_search",
                name="search_tools",
                arguments={"query": "tests", "top_k": "5"},
            ),
            use_toon=False,
        )
        search_payload = json.loads(search_raw)
        self.assertEqual(search_payload.get("status"), "ok")
        names = [item.get("name") for item in search_payload.get("matches", [])]
        self.assertIn("run_tests", names)

    def test_skills_health_builtin_reports_catalog_diagnostics(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        raw = registry.execute(
            ToolCall(
                id="call_skills_health",
                name="skills_health",
                arguments={"include_sources": True, "max_items": 8},
            ),
            use_toon=False,
        )
        payload = json.loads(raw)
        self.assertEqual(payload.get("status"), "ok")

        discovery = payload.get("discovery", {})
        catalog = payload.get("catalog", {})
        coding = payload.get("coding", {})
        organization = payload.get("organization", {})
        checks = payload.get("checks", {})
        self.assertGreater(discovery.get("source_count", 0), 0)
        self.assertGreaterEqual(
            discovery.get("source_count", 0), discovery.get("readable_count", 0)
        )
        self.assertGreater(catalog.get("skills_total", 0), 0)
        self.assertTrue(checks.get("brainstorming_present"))
        self.assertIn(coding.get("maturity"), {"state_of_the_art", "strong", "good", "baseline"})
        self.assertGreaterEqual(coding.get("required_coverage_pct", 0.0), 0.0)
        self.assertIsInstance(coding.get("groups", {}), dict)
        self.assertIn(
            organization.get("status"),
            {"ok", "needs_attention", "missing"},
        )
        self.assertGreaterEqual(organization.get("issues_count", 0), 0)
        self.assertIn("bootstrap_only_modules", organization)
        self.assertIn(
            "00_bootstrap.py",
            organization.get("bootstrap_only_modules", []),
        )
        self.assertIn("coding_required_coverage_pct", checks)
        self.assertIn("organization_issues_count", checks)

    def test_tool_execution_stats_capture_errors_and_successes(self) -> None:
        registry = self._registry()
        registry.register(
            name="flaky",
            description="Sometimes fails.",
            parameters=[],
            function=lambda: {"status": "error", "error": "boom"},
        )
        registry.register(
            name="stable",
            description="Always succeeds.",
            parameters=[],
            function=lambda: {"status": "ok"},
        )

        registry.execute(
            ToolCall(id="call_flaky", name="flaky", arguments={}),
            use_toon=False,
        )
        registry.execute(
            ToolCall(id="call_stable", name="stable", arguments={}),
            use_toon=False,
        )

        stats = registry.tool_execution_stats()
        self.assertIn("flaky", stats)
        self.assertIn("stable", stats)
        self.assertEqual(stats["flaky"].get("errors"), 1)
        self.assertEqual(stats["stable"].get("successes"), 1)

    def test_create_sample_data_signature_infers_numeric_types(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        tool = registry.get("create_sample_data")
        self.assertIsNotNone(tool)
        assert tool is not None

        params = {p.name: p.type for p in tool.parameters}
        self.assertEqual(params.get("pattern"), "string")
        self.assertEqual(params.get("n_points"), "int")
        self.assertEqual(params.get("noise_level"), "float")

        out = registry.execute(
            ToolCall(
                id="sample_numeric_strings",
                name="create_sample_data",
                arguments={"pattern": "trend", "n_points": "120", "noise_level": "0.7"},
            ),
            use_toon=False,
        )
        payload = json.loads(out)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("n"), 120)

    def test_argument_aliases_reduce_schema_loops(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        out = registry.execute(
            ToolCall(
                id="sample_aliases",
                name="create_sample_data",
                arguments={"type": "seasonal", "points": 90, "noise": "1.0"},
            ),
            use_toon=False,
        )
        payload = json.loads(out)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("pattern"), "seasonal")
        self.assertEqual(payload.get("n"), 90)

    def test_stringified_annotations_are_typed_correctly(self) -> None:
        registry = self._registry()

        def synthetic(max_results=10, strict=False, ratio=0.5):  # noqa: ANN001
            return {"status": "ok"}

        synthetic.__annotations__ = {
            "max_results": "int",
            "strict": "bool | None",
            "ratio": "float",
        }

        params = registry._parameters_from_signature(synthetic)  # type: ignore[attr-defined]
        pmap = {p.name: p.type for p in params}
        self.assertEqual(pmap.get("max_results"), "int")
        self.assertEqual(pmap.get("strict"), "bool")
        self.assertEqual(pmap.get("ratio"), "float")

    def test_literal_annotations_emit_enum_and_enforce_values(self) -> None:
        registry = self._registry()

        def choose_mode(mode: Literal["fast", "safe"] = "fast"):  # noqa: ANN001
            return {"status": "ok", "mode": mode}

        params = registry._parameters_from_signature(choose_mode)  # type: ignore[attr-defined]
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].type, "string")
        self.assertEqual(params[0].enum, ["fast", "safe"])

        registry.register(
            name="choose_mode",
            description="Choose execution mode.",
            parameters=params,
            function=choose_mode,
        )

        schema = registry.openai_tool_schemas(["choose_mode"])[0]
        mode_schema = schema["function"]["parameters"]["properties"]["mode"]
        self.assertEqual(mode_schema.get("enum"), ["fast", "safe"])

        ok_payload = json.loads(
            registry.execute(
                ToolCall(
                    id="literal_ok",
                    name="choose_mode",
                    arguments={"mode": "safe"},
                ),
                use_toon=False,
            )
        )
        self.assertEqual(ok_payload.get("status"), "ok")
        self.assertEqual(ok_payload.get("mode"), "safe")

        bad_payload = json.loads(
            registry.execute(
                ToolCall(
                    id="literal_bad",
                    name="choose_mode",
                    arguments={"mode": "turbo"},
                ),
                use_toon=False,
            )
        )
        self.assertEqual(bad_payload.get("status"), "error")
        self.assertEqual(
            bad_payload.get("error_type"),
            "schema_type_validation_failed",
        )
        first_error = (bad_payload.get("type_errors") or [{}])[0]
        self.assertEqual(first_error.get("allowed_values"), ["fast", "safe"])
        self.assertIn("expected one of", str(first_error.get("error", "")))

    def test_real_skill_enums_are_exported_and_enforced(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        expected = {
            ("create_sample_data", "pattern"): [
                "trend",
                "seasonal",
                "random",
                "anomaly",
                "stationary",
                "cyclic_trend",
            ],
            ("write_file", "mode"): ["w", "a"],
            ("smart_quality_gate", "mode"): ["fast", "balanced", "full"],
            ("fd_find", "file_type"): ["f", "d", ""],
            ("detect_anomalies", "method"): [
                "zscore",
                "iqr",
                "hampel",
                "stl_resid",
                "iforest",
            ],
        }

        for (tool_name, param_name), enum_values in expected.items():
            tool = registry.get(tool_name)
            self.assertIsNotNone(tool, msg=f"{tool_name} should be registered")
            assert tool is not None
            param = next((p for p in tool.parameters if p.name == param_name), None)
            self.assertIsNotNone(
                param,
                msg=f"{tool_name}.{param_name} should exist",
            )
            assert param is not None
            self.assertEqual(param.enum, enum_values)

            schema = registry.openai_tool_schemas([tool_name])[0]
            prop_schema = schema["function"]["parameters"]["properties"][param_name]
            self.assertEqual(prop_schema.get("enum"), enum_values)

        bad_sample = json.loads(
            registry.execute(
                ToolCall(
                    id="enum_bad_sample",
                    name="create_sample_data",
                    arguments={"pattern": "foo"},
                ),
                use_toon=False,
            )
        )
        self.assertEqual(bad_sample.get("status"), "error")
        self.assertEqual(
            bad_sample.get("error_type"),
            "schema_type_validation_failed",
        )

    def test_coding_and_timeseries_schema_audit(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        allowed_types = {"string", "int", "float", "bool", "list", "dict"}
        suspicious_allowlist = {("svg_pipeline", "steps")}
        suspicious: list[tuple[str, str, str]] = []

        tool_names = []
        for tool in registry.list_tools():
            source_path = str(tool.source_path or "")
            if (
                "/skills/01_coding/" not in source_path
                and "/skills/02_timeseries/" not in source_path
                and "/skills/04_svg/" not in source_path
                and "/skills/05_rag/" not in source_path
            ):
                continue
            tool_names.append(tool.name)

            seen_param_names: set[str] = set()
            for param in tool.parameters:
                self.assertTrue(
                    str(param.description or "").strip(),
                    msg=f"{tool.name}.{param.name} has empty description",
                )
                self.assertIn(
                    param.type,
                    allowed_types,
                    msg=f"{tool.name}.{param.name} has unsupported type: {param.type}",
                )
                self.assertNotIn(
                    param.name,
                    seen_param_names,
                    msg=f"{tool.name} has duplicate parameter name: {param.name}",
                )
                seen_param_names.add(param.name)

                n = param.name.lower()
                looks_numeric = n.startswith(("n_", "num_", "max_", "min_", "top_")) or n.endswith(
                    ("_count", "_idx", "_index", "_points", "_steps", "_window", "_k")
                ) or n in {
                    "k",
                    "n",
                    "count",
                    "limit",
                    "window",
                    "horizon",
                    "steps",
                    "epochs",
                    "batch_size",
                    "seed",
                    "period",
                }
                looks_bool = n.startswith(("is_", "has_", "use_", "with_", "enable_")) or n in {
                    "recursive",
                    "overwrite",
                    "dry_run",
                    "strict",
                    "verbose",
                }
                if param.type == "string" and (looks_numeric or looks_bool):
                    suspicious.append((tool.name, param.name, param.type))

        schemas = registry.openai_tool_schemas(tool_names)
        for schema in schemas:
            fn = schema["function"]
            props = fn["parameters"]["properties"]
            required = fn["parameters"]["required"]
            for req_name in required:
                self.assertIn(
                    req_name,
                    props,
                    msg=f"{fn['name']} marks '{req_name}' required but it is missing from properties",
                )

        suspicious = [
            item
            for item in suspicious
            if (item[0], item[1]) not in suspicious_allowlist
        ]
        self.assertEqual(
            suspicious,
            [],
            msg=f"Unexpected suspicious string-typed params: {suspicious}",
        )

    def test_tool_parameter_model_validation(self) -> None:
        with self.assertRaises(Exception):
            ToolParameter(name="", type="string", description="bad")

        p = ToolParameter("limit", "integer", "max items")
        self.assertEqual(p.name, "limit")
        self.assertEqual(p.type, "int")

    def test_tool_model_rejects_non_callable_function(self) -> None:
        with self.assertRaises(Exception):
            Tool(
                name="bad_tool",
                description="not callable",
                parameters=[],
                function="oops",
            )


if __name__ == "__main__":
    unittest.main()
