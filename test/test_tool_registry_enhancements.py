import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.tools import Context, Tool, ToolCall, ToolParameter, ToolRegistry
from src.tools.registry import SkillCatalog


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

    def test_prepare_call_normalizes_arguments_without_executing(self) -> None:
        registry = self._registry()
        registry.register(
            name="add_numbers",
            description="Add two numbers.",
            parameters=[
                ToolParameter(name="a", type="integer", description="", required=True),
                ToolParameter(name="b", type="integer", description="", required=True),
            ],
            function=lambda a, b: {"status": "ok", "sum": a + b},
        )

        prepared, error = registry.prepare_call(
            ToolCall(
                id="prepare_ok",
                name="add_numbers",
                arguments={"a": "2", "b": "3"},
            )
        )

        self.assertIsNone(error)
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(prepared.arguments, {"a": 2, "b": 3})
        self.assertEqual(registry.tool_execution_stats(), {})

    def test_prepare_call_returns_structured_schema_error(self) -> None:
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

        prepared, error = registry.prepare_call(
            ToolCall(
                id="prepare_bad",
                name="echo_pair",
                arguments={"left": "A", "extra": "C"},
            )
        )

        self.assertIsNone(prepared)
        self.assertIsNotNone(error)
        payload = json.loads(error or "{}")
        self.assertEqual(payload.get("error_type"), "schema_validation_failed")
        self.assertIn("right", payload.get("missing_required", []))
        self.assertIn("extra", payload.get("unknown_arguments", []))

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

    def test_default_prompt_tool_names_prefers_core_subset(self) -> None:
        registry = self._registry()
        registry.register(
            name="read_file",
            description="Read a file.",
            parameters=[],
            function=lambda: {"status": "ok"},
        )
        registry.register(
            name="run_python",
            description="Run python.",
            parameters=[],
            function=lambda: {"status": "ok"},
        )
        registry.register(
            name="svg_render",
            description="Render SVG output.",
            parameters=[],
            function=lambda: {"status": "ok"},
        )

        selected = registry.default_prompt_tool_names()

        self.assertIn("read_file", selected)
        self.assertIn("run_python", selected)
        self.assertNotIn("svg_render", selected)

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
            "bootstrap.py",
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
        registry.activate_lazy_skill_group("lazy_timeseries")
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

    def test_skill_catalog_fuzzy_similarity_coerces_non_string_profiles(self) -> None:
        registry = self._registry()
        catalog = SkillCatalog(
            skills_md_path=registry.skills_md_path,
            skills_dir_path=registry.skills_dir_path,
            log=registry._log,
        )
        score_list = catalog._fuzzy_similarity("review rust cli", ["rust", "cli"])
        score_int = catalog._fuzzy_similarity("review rust cli", 12345)
        self.assertIsInstance(score_list, float)
        self.assertIsInstance(score_int, float)
        self.assertGreaterEqual(score_list, 0.0)
        self.assertLessEqual(score_list, 1.0)
        self.assertGreaterEqual(score_int, 0.0)
        self.assertLessEqual(score_int, 1.0)

    def test_argument_aliases_reduce_schema_loops(self) -> None:
        registry = self._registry()
        registry.activate_lazy_skill_group("lazy_timeseries")
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

    def test_python_skill_module_can_register_tools_via_explicit_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "coding" / "toy" / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "toy.py"
            module_path.write_text(
                """
def hello(name: str = "world") -> str:
    \"\"\"Use when: Return a greeting string.\"\"\"
    return _safe_json({"status": "ok", "message": f"hello {name}"})


__tools__ = [hello]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = self._registry()
            registry.skills_dir_path = root
            registry.skills_md_path = root
            registry._catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=registry._log,
            )

            registry.load_tools_from_skills()

            tool = registry.get("hello")
            self.assertIsNotNone(tool)
            assert tool is not None
            self.assertEqual(tool.description, "Return a greeting string.")

            raw = registry.execute(
                ToolCall(id="call_hello", name="hello", arguments={"name": "codex"}),
                use_toon=False,
            )
            payload = json.loads(raw)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("message"), "hello codex")

    def test_python_skill_module_can_register_tools_via_tool_decorator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "coding" / "toy" / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "toy.py"
            module_path.write_text(
                """
from skills.coding.bootstrap.runtime_access import tool


@tool
def hello(name: str = "world") -> str:
    \"\"\"Return a greeting string.\"\"\"
    return _safe_json({"status": "ok", "message": f"hello {name}"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = self._registry()
            registry.skills_dir_path = root
            registry.skills_md_path = root
            registry._catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=registry._log,
            )

            registry.load_tools_from_skills()

            tool = registry.get("hello")
            self.assertIsNotNone(tool)
            assert tool is not None
            self.assertEqual(tool.description, "Return a greeting string.")

    def test_tool_decorator_injects_module_skill_context_into_docstring(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        tool = registry.get("get_file_outline")
        self.assertIsNotNone(tool)
        assert tool is not None
        doc = str(getattr(tool.function, "__doc__", "") or "")
        self.assertIn("Skill context:", doc)
        self.assertIn("Skill: Explore", doc)
        self.assertIn("Preferred tools in this skill: get_file_outline, rg_search, fd_find", doc)
        self.assertIn("Skill failure recovery:", doc)

    def test_core_tools_remain_canonical_when_coding_skills_overlap(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        from src.tools.core import apply_edit_block, read_file, write_file

        registry._register_collected_python_tools(
            tool_entries=[
                (read_file, getattr(read_file, "__llm_tool_meta__", {})),
                (write_file, getattr(write_file, "__llm_tool_meta__", {})),
                (apply_edit_block, getattr(apply_edit_block, "__llm_tool_meta__", {})),
            ],
            module_path=Path("src/tools/core/files.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        read_tool = registry.get("read_file")
        write_tool = registry.get("write_file")
        edit_block_tool = registry.get("apply_edit_block")
        explore_tool = registry.get("get_file_outline")
        multi_edit_tool = registry.get("multi_edit")

        self.assertIsNotNone(read_tool)
        self.assertIsNotNone(write_tool)
        self.assertIsNotNone(edit_block_tool)
        self.assertIsNotNone(explore_tool)
        self.assertIsNotNone(multi_edit_tool)
        assert read_tool is not None
        assert write_tool is not None
        assert edit_block_tool is not None
        assert explore_tool is not None
        assert multi_edit_tool is not None

        self.assertEqual(read_tool.skill_id, "core")
        self.assertEqual(write_tool.skill_id, "core")
        self.assertEqual(edit_block_tool.skill_id, "core")
        self.assertEqual(explore_tool.skill_id, "explore")
        self.assertEqual(multi_edit_tool.skill_id, "multi_edit")

    def test_capability_audit_normalizes_legacy_required_tool_names(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        from src.tools.core import apply_edit_block, edit_file, list_dir, read_file, write_file

        registry._register_collected_python_tools(
            tool_entries=[
                (read_file, getattr(read_file, "__llm_tool_meta__", {})),
                (write_file, getattr(write_file, "__llm_tool_meta__", {})),
                (edit_file, getattr(edit_file, "__llm_tool_meta__", {})),
                (list_dir, getattr(list_dir, "__llm_tool_meta__", {})),
                (apply_edit_block, getattr(apply_edit_block, "__llm_tool_meta__", {})),
            ],
            module_path=Path("src/tools/core/files.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        audit = registry._coding_capability_audit(registry.list_skills())
        missing = set(audit.get("missing_required_tools", []))
        self.assertNotIn("list_directory", missing)
        self.assertNotIn("edit_file_replace", missing)

    def test_python_skill_explicit_exports_are_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "coding" / "toy" / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "toy.py"
            module_path.write_text(
                """
from skills.coding.bootstrap.runtime_access import tool

@tool
def exported() -> str:
    return _safe_json({"status": "ok", "name": "exported"})

@tool
def hidden() -> str:
    return _safe_json({"status": "ok", "name": "hidden"})

__tools__ = [exported]
""".strip()
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root.parent / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            self.assertIsNotNone(registry.get("exported"))
            self.assertIsNone(registry.get("hidden"))

    def test_metadata_only_python_skill_still_routes_after_dedup(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        from src.tools.core import apply_edit_block

        registry._register_collected_python_tools(
            tool_entries=[
                (apply_edit_block, getattr(apply_edit_block, "__llm_tool_meta__", {})),
            ],
            module_path=Path("src/tools/core/files.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        skills = {skill.id: skill for skill in registry.list_skills()}
        self.assertIn("edit_block", skills)
        card = skills["edit_block"]
        self.assertEqual(card.name, "Edit Block")
        self.assertIn("apply_edit_block", card.preferred_tools)
        self.assertIn("apply_edit_block", card.tool_names)

    def test_python_skill_modules_can_use_lazy_numpy_and_pandas_globals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "coding" / "array_ops" / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "array_ops.py"
            module_path.write_text(
                """
def summarize() -> str:
    arr = np.asarray([1, 2, 3], dtype=float)
    frame = pd.DataFrame({"value": arr.tolist()})
    return _safe_json(
        {
            "status": "ok",
            "sum": float(arr.sum()),
            "rows": int(len(frame)),
        }
    )


__tools__ = [summarize]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = self._registry()
            registry.skills_dir_path = root
            registry.skills_md_path = root
            registry._catalog = SkillCatalog(
                skills_md_path=root,
                skills_dir_path=root,
                log=registry._log,
            )

            registry.load_tools_from_skills()
            raw = registry.execute(
                ToolCall(id="lazy_np_pd", name="summarize", arguments={}),
                use_toon=False,
            )
            payload = json.loads(raw)
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("sum"), 6.0)
            self.assertEqual(payload.get("rows"), 3)

    def test_import_src_tools_does_not_eagerly_import_heavy_optional_deps(self) -> None:
        script = f"""
import json
import sys
sys.path.insert(0, {str(AGENT_ROOT)!r})
import src.tools
print(json.dumps({{
    "numpy": "numpy" in sys.modules,
    "pandas": "pandas" in sys.modules,
    "sentence_transformers": "sentence_transformers" in sys.modules,
}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=AGENT_ROOT,
            check=True,
        )
        payload = json.loads(proc.stdout.strip())
        self.assertFalse(payload["numpy"])
        self.assertFalse(payload["pandas"])
        self.assertFalse(payload["sentence_transformers"])

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
        registry.activate_lazy_skill_group("lazy_timeseries")
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
        registry.activate_lazy_skill_group("timeseries")
        registry.load_tools_from_skills()

        allowed_types = {"string", "int", "float", "bool", "list", "dict"}
        suspicious_allowlist = {("svg_pipeline", "steps")}
        suspicious: list[tuple[str, str, str]] = []

        tool_names = []
        for tool in registry.list_tools():
            source_path = str(tool.source_path or "")
            if (
                "/skills/coding/" not in source_path
                and "/skills/lazy_timeseries/" not in source_path
                and "/skills/svg/" not in source_path
                and "/skills/rag/" not in source_path
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

    def test_editing_tools_register_grammars(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        for tool_name in (
            "write_file",
            "edit_file",
            "multi_edit",
            "apply_edit_block",
        ):
            grammar = registry.get_grammar(tool_name)
            self.assertIsNotNone(grammar, msg=f"missing grammar for {tool_name}")
            assert grammar is not None
            self.assertGreater(len(grammar), 50)
            self.assertIn(tool_name, grammar)

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
