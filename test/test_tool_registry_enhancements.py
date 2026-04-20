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

    def test_manual_register_can_declare_runtime_metadata(self) -> None:
        registry = self._registry()
        registry.register(
            name="custom_reader",
            description="Read a file.",
            parameters=[
                ToolParameter(name="path", type="string", description="File path.", required=True)
            ],
            function=lambda path: {"status": "ok", "path": path},
            runtime={"read_only": True, "cacheable": True, "content_reader": True},
        )

        tool = registry.get("custom_reader")
        assert tool is not None
        self.assertTrue(tool.runtime.read_only)
        self.assertTrue(tool.runtime.cacheable)
        self.assertTrue(tool.runtime.content_reader)

    def test_builtin_tool_registration_uses_materialized_runtime_defaults(self) -> None:
        registry = self._registry()

        registered = registry._register_builtin_tool(
            {
                "name": "builtin_echo",
                "description": "Echo text.",
                "parameters": [
                    ToolParameter(name="text", type="string", description="Text.", required=True)
                ],
                "function": lambda text: {"status": "ok", "text": text},
                "doc": "Echo text.",
            }
        )

        self.assertTrue(registered)
        tool = registry.get("builtin_echo")
        assert tool is not None
        self.assertEqual(tool.description, "Echo text.")
        self.assertEqual(tool.source_path, "<builtin>")
        self.assertEqual(tool.skill_id, "meta_skills")

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

    def test_additional_skill_dirs_load_external_plugin_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_skill = root / "claude-mem" / "skills" / "memory"
            plugin_skill.mkdir(parents=True, exist_ok=True)
            (plugin_skill / "MEMORY.md").write_text(
                """
---
name: Memory
description: External plugin memory skill
---
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = self._registry()
            registry.add_skills_dir_paths([root / "claude-mem" / "skills"])
            registry.load_tools_from_skills()

            self.assertTrue(
                any("memory" in skill_id for skill_id in registry._catalog.skills),
                "External plugin skill should be discovered from additional skill directories",
            )

    def test_additional_command_dirs_load_external_plugin_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_command = root / "claude-mem" / "commands" / "memory"
            plugin_command.mkdir(parents=True, exist_ok=True)
            (plugin_command / "MEMORY.md").write_text(
                """
---
name: Memory
summary: External plugin memory command
---
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = self._registry()
            registry.add_skills_dir_paths([root / "claude-mem" / "commands"])
            registry.load_tools_from_skills()

            self.assertTrue(
                any("memory" in skill_id for skill_id in registry._catalog.skills),
                "External plugin command directory should be discovered from additional skill directories",
            )

    def test_legacy_tool_decorator_no_longer_registers_tools_without_explicit_exports(
        self,
    ) -> None:
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
            self.assertIsNone(tool)

    def test_tool_decorator_injects_module_skill_context_into_docstring(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        tool = registry.get("rg_search")
        self.assertIsNotNone(tool)
        assert tool is not None
        doc = str(getattr(tool.function, "__doc__", "") or "")
        self.assertIn("Skill context:", doc)
        self.assertIn("Skill: Explore", doc)
        self.assertIn(
            "Preferred tools in this skill: get_project_map, find_symbol, get_file_outline, rg_search, fd_find",
            doc,
        )
        self.assertIn("Skill failure recovery:", doc)

    def test_core_tools_remain_canonical_when_coding_skills_overlap(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        from src.tools import core as _core_tools

        registry._register_collected_python_tools(
            tool_entries=[
                (fn, getattr(fn, "__llm_tool_meta__", {})) for fn in _core_tools.CORE_TOOL_FUNCTIONS
            ],
            module_path=Path("src/tools/core/__init__.py"),
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
        self.assertEqual(explore_tool.skill_id, "core")
        self.assertEqual(multi_edit_tool.skill_id, "multi_edit")

    def test_fetch_url_comes_from_core_web_tools(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        from src.tools import core as _core_tools

        registry._register_collected_python_tools(
            tool_entries=[
                (fn, getattr(fn, "__llm_tool_meta__", {})) for fn in _core_tools.CORE_TOOL_FUNCTIONS
            ],
            module_path=Path("src/tools/core/__init__.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        fetch_tool = registry.get("fetch_url")
        web_search_tool = registry.get("web_search")

        self.assertIsNotNone(fetch_tool)
        self.assertIsNotNone(web_search_tool)
        assert fetch_tool is not None
        assert web_search_tool is not None

        self.assertEqual(fetch_tool.skill_id, "core")
        self.assertEqual(web_search_tool.skill_id, "core")
        self.assertTrue(str(fetch_tool.source_path or "").endswith("src/tools/core/__init__.py"))
        self.assertTrue(
            str(web_search_tool.source_path or "").endswith("src/tools/core/__init__.py")
        )

    def test_docs_context_audit_recognizes_core_web_provider(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        from src.tools import core as _core_tools

        registry._register_collected_python_tools(
            tool_entries=[
                (fn, getattr(fn, "__llm_tool_meta__", {})) for fn in _core_tools.CORE_TOOL_FUNCTIONS
            ],
            module_path=Path("src/tools/core/__init__.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        audit = registry._coding_capability_audit(registry.list_skills())
        docs_group = audit.get("groups", {}).get("docs_context", {})
        providers = docs_group.get("provider_details", {})

        self.assertEqual(docs_group.get("canonical_provider"), "core")
        self.assertEqual(providers.get("fetch_url", {}).get("skill_id"), "core")
        self.assertEqual(providers.get("web_search", {}).get("skill_id"), "core")

    def test_coding_organization_audit_allows_metadata_only_modules(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        organization = registry._coding_organization_audit()

        self.assertIn("web.py", organization.get("metadata_only_modules", []))
        self.assertIn("file_ops.py", organization.get("metadata_only_modules", []))
        self.assertNotIn("web.py", organization.get("modules_without_registered_tools", []))

    def test_shell_and_git_overlap_review_shows_no_exact_core_conflicts(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        from src.tools import core as _core_tools

        registry._register_collected_python_tools(
            tool_entries=[
                (fn, getattr(fn, "__llm_tool_meta__", {})) for fn in _core_tools.CORE_TOOL_FUNCTIONS
            ],
            module_path=Path("src/tools/core/__init__.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        audit = registry._coding_capability_audit(registry.list_skills())
        overlap_review = audit.get("overlap_review", {})
        shell_review = overlap_review.get("shell", {})
        git_review = overlap_review.get("git", {})

        self.assertEqual(shell_review.get("exact_core_name_conflicts"), [])
        self.assertEqual(git_review.get("exact_core_name_conflicts"), [])
        self.assertEqual(
            shell_review.get("promoted_to_core"),
            [
                "set_venv",
                "set_working_directory",
                "install_packages",
                "show_coding_config",
                "start_background_process",
                "send_input_to_process",
                "get_process_output",
                "kill_process",
                "list_processes",
                "run_python",
                "check_imports",
                "list_installed_packages",
            ],
        )
        self.assertEqual(shell_review.get("semantic_alias_overlaps", {}).get("run_shell"), ["bash"])
        self.assertEqual(shell_review.get("requires_shared_core_state"), [])
        self.assertEqual(
            git_review.get("semantic_alias_overlaps", {}).get("git_status"),
            ["get_git_status"],
        )
        self.assertEqual(
            git_review.get("semantic_alias_overlaps", {}).get("git_diff"),
            ["get_git_diff"],
        )
        self.assertFalse(shell_review.get("should_promote_to_core"))
        self.assertFalse(git_review.get("should_promote_to_core"))

    def test_run_python_helpers_are_registered_from_core(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()
        from src.tools import core as _core_tools

        registry._register_collected_python_tools(
            tool_entries=[
                (fn, getattr(fn, "__llm_tool_meta__", {})) for fn in _core_tools.CORE_TOOL_FUNCTIONS
            ],
            module_path=Path("src/tools/core/__init__.py"),
            skill_id="core",
            skill_meta={"always_on": True},
        )

        run_python_tool = registry.get("run_python")
        check_imports_tool = registry.get("check_imports")
        list_packages_tool = registry.get("list_installed_packages")

        self.assertIsNotNone(run_python_tool)
        self.assertIsNotNone(check_imports_tool)
        self.assertIsNotNone(list_packages_tool)
        assert run_python_tool is not None
        assert check_imports_tool is not None
        assert list_packages_tool is not None

        self.assertEqual(run_python_tool.skill_id, "core")
        self.assertEqual(check_imports_tool.skill_id, "core")
        self.assertEqual(list_packages_tool.skill_id, "core")

        for tool_name in (
            "set_venv",
            "set_working_directory",
            "install_packages",
            "show_coding_config",
            "start_background_process",
            "send_input_to_process",
            "get_process_output",
            "kill_process",
            "list_processes",
        ):
            tool = registry.get(tool_name)
            self.assertIsNotNone(tool)
            assert tool is not None
            self.assertEqual(tool.skill_id, "core")

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
            module_path=Path("src/tools/core/__init__.py"),
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
            module_path=Path("src/tools/core/__init__.py"),
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

    def test_auto_discovery_of_scripts_tools_without_explicit_exports(self) -> None:
        """Test that tools in scripts/ folders are auto-discovered without __tools__ export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create skill structure with scripts/ subfolder
            skill_dir = root / "wiki" / "wiki_ops"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Wiki Ops Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            # Create tool module WITHOUT __tools__ export
            module_path = scripts_dir / "wiki_ops.py"
            module_path.write_text(
                """
def wiki_list(path: str | None = None) -> dict:
    '''List wiki source documents available for compilation.'''
    return _safe_json({
        "status": "ok",
        "path": path or "default",
    })

def wiki_list_raw(raw_dir: str | None = None) -> dict:
    '''List artifacts collected in the raw directory.'''
    return _safe_json({
        "status": "ok",
        "raw_dir": raw_dir or "default",
    })

# Note: No __tools__ export; skill tools are lazily registered as proxies by load_tools_from_skills
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            # Skill tools under skills/ should not be eagerly auto-loaded.
            self.assertIsNone(registry.get("wiki_list"), "wiki_list should not be auto-discovered")
            self.assertIsNone(
                registry.get("wiki_list_raw"), "wiki_list_raw should not be auto-discovered"
            )

            result = registry.call_tool(
                "invoke_skill",
                skill="wiki_ops",
                args='{"path": null}',
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["path"], "default")

    def test_invoke_skill_loads_skill_tools_without_executing_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_ops"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wiki ops\ndescription: Wiki operations skill.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "wiki_ops.py"
            module_path.write_text(
                "def wiki_list():\n    return {'status': 'ok', 'value': 'done'}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="wiki_ops",
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["loaded_skill_id"], "wiki_ops")
            self.assertEqual(payload["tool"], "wiki_list")
            self.assertEqual(payload["available_tools"], ["wiki_list"])
            self.assertEqual(payload["newly_available_tools"], ["wiki_list"])
            self.assertIsNotNone(registry.get("wiki_list"))

    def test_invoke_skill_renders_skill_prompt_with_embedded_shell_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_ops"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wiki ops\ndescription: Wiki operations skill.\n---\n"
                "Run this skill for path `${path}`.\n"
                "!`echo hello ${path}`\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="wiki_ops",
                args='{"path": "test-path"}',
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertIn("Run this skill for path test-path", payload["prompt"])
            self.assertIn("hello test-path", payload["prompt"])

    def test_invoke_skill_accepts_direct_tool_name_for_skill_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wiki skills\ndescription: Wiki skill group.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "wiki_ops.py"
            module_path.write_text(
                "def wiki_list():\n    return {'status': 'ok', 'value': 'done'}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="wiki_list",
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["loaded_skill_id"], "wiki_skills")
            self.assertEqual(payload["tool"], "wiki_list")
            self.assertEqual(payload["available_tools"], ["wiki_list"])
            self.assertIsNotNone(registry.get("wiki_list"))

    def test_invoke_skill_accepts_skill_path_for_lazy_skill_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "academic" / "arxiv"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: arXiv\ndescription: arXiv search skill.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "arxiv_search.py"
            module_path.write_text(
                "def arxiv_search():\n    return {'status': 'ok', 'source': 'arxiv'}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="academic/arxiv",
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["loaded_skill_id"], "arxiv")
            self.assertEqual(payload["tool"], "arxiv_search")
            self.assertEqual(payload["available_tools"], ["arxiv_search"])
            self.assertIsNotNone(registry.get("arxiv_search"))

    def test_invoke_skill_loads_tool_without_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "academic" / "arxiv"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: arXiv\ndescription: arXiv search skill.\npreferred_tools:\n  - arxiv_search\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "arxiv.py"
            module_path.write_text(
                "from typing import Any\n\n"
                "def arxiv_search(query: str) -> dict[str, Any]:\n"
                "    return {'status': 'ok', 'source': 'arxiv', 'query': query}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool("invoke_skill", skill="academic/arxiv")
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["tool"], "arxiv_search")
            self.assertEqual(payload["available_tools"], ["arxiv_search"])
            self.assertEqual(payload["newly_available_tools"], ["arxiv_search"])
            self.assertIsNotNone(registry.get("arxiv_search"))

            result2 = registry.call_tool("arxiv_search", query="time series")
            payload2 = json.loads(result2)
            self.assertEqual(payload2["status"], "ok")
            self.assertEqual(payload2["source"], "arxiv")
            self.assertEqual(payload2["query"], "time series")

    def test_invoke_skill_path_with_args_executes_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "academic" / "arxiv"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: arXiv\ndescription: arXiv search skill.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "arxiv_search.py"
            module_path.write_text(
                "def arxiv_search(query: str):\n    return {'status': 'ok', 'source': 'arxiv', 'query': query}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="academic/arxiv",
                args='{"query": "time series forecasting"}',
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source"], "arxiv")
            self.assertEqual(payload["query"], "time series forecasting")
            self.assertEqual(payload["loaded_skill_id"], "arxiv")
            self.assertEqual(payload["available_tools"], ["arxiv_search"])

    def test_invoke_skill_loads_class_based_provider_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "academic" / "arxiv"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: arXiv\ndescription: arXiv provider skill.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "arxiv.py"
            module_path.write_text(
                "class ArxivSource:\n"
                "    name = 'arxiv'\n"
                "\n"
                "    def search(self, query: str):\n"
                "        return {'status': 'ok', 'source': 'arxiv', 'query': query}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool(
                "invoke_skill",
                skill="academic/arxiv",
                args='{"query": "timeseries forecasting"}',
            )
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["source"], "arxiv")
            self.assertEqual(payload["query"], "timeseries forecasting")
            self.assertEqual(payload["loaded_skill_id"], "arxiv")
            self.assertEqual(payload["available_tools"], ["arxiv_search"])

    def test_invoke_skill_loads_all_tools_from_scripts_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_ops"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wiki ops\ndescription: Wiki operations skill.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "wiki_ops.py").write_text(
                "def wiki_list():\n    return {'status': 'ok', 'tool': 'wiki_list'}\n",
                encoding="utf-8",
            )
            (scripts_dir / "wiki_admin.py").write_text(
                "def wiki_reindex():\n    return {'status': 'ok', 'tool': 'wiki_reindex'}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            result = registry.call_tool("invoke_skill", skill="wiki_ops")
            payload = json.loads(result)

            self.assertEqual(payload["status"], "ok")
            self.assertCountEqual(payload["available_tools"], ["wiki_list", "wiki_reindex"])
            self.assertCountEqual(
                payload["newly_available_tools"],
                ["wiki_list", "wiki_reindex"],
            )
            self.assertIsNotNone(registry.get("wiki_list"))
            self.assertIsNotNone(registry.get("wiki_reindex"))

    def test_direct_tool_name_lazy_loads_skill_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: wiki skills\ndescription: Wiki skill group.\n---\n",
                encoding="utf-8",
            )
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            module_path = scripts_dir / "wiki_ops.py"
            module_path.write_text(
                "def wiki_list():\n    return {'status': 'ok', 'value': 'done'}\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())

            self.assertIsNone(registry.get("wiki_list"))
            result = registry.call_tool("wiki_list")
            payload = json.loads(result)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["value"], "done")

    def test_root_level_python_modules_ignored_when_scripts_subdir_exists(self) -> None:
        """Tools under root skill directories should be loaded only from scripts/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "wiki" / "wiki_ops"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Wiki Ops Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            root_module = skill_dir / "legacy_tool.py"
            root_module.write_text(
                """
def legacy_tool() -> dict:
    '''Legacy tool should be ignored.'''
    return _safe_json({"status": "ignored"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            script_module = scripts_dir / "wiki_ops.py"
            script_module.write_text(
                """
def wiki_list() -> dict:
    '''List wiki source documents available for compilation.'''
    return _safe_json({"status": "ok"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            self.assertIsNone(registry.get("wiki_list"))
            self.assertIsNone(registry.get("legacy_tool"))

    def test_auto_discovery_excludes_private_and_internal_names(self) -> None:
        """Test that auto-discovery excludes private (starting with _) and internal names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "test_skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Test Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            module_path = scripts_dir / "test.py"
            module_path.write_text(
                """
def public_tool() -> dict:
    '''A public tool.'''
    return _safe_json({"status": "ok"})

def _private_tool() -> dict:
    '''A private tool.'''
    return _safe_json({"status": "private"})

def __dunder_tool() -> dict:
    '''A dunder tool.'''
    return _safe_json({"status": "dunder"})

# Note: No __tools__ export
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            # Only public_tool should be discovered
            self.assertIsNotNone(registry.get("public_tool"))
            self.assertIsNone(registry.get("_private_tool"))
            self.assertIsNone(registry.get("__dunder_tool"))

    def test_auto_discovery_with_explicit_metadata(self) -> None:
        """Test that auto-discovery respects explicit __llm_tool_meta__ metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "test_skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Test Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            module_path = scripts_dir / "test.py"
            module_path.write_text(
                """
def public_tool() -> dict:
    '''A public tool.'''
    # Add custom metadata
    __llm_tool_meta__ = {"description": "Custom description for public tool"}
    return _safe_json({"status": "ok"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            tool = registry.get("public_tool")
            self.assertIsNotNone(tool)
            self.assertEqual(tool.description, "Custom description for public tool")

    def test_auto_discovery_excludes_common_non_tool_objects(self) -> None:
        """Test that auto-discovery excludes common non-tool globals like json, np, pd, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "test_skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Test Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            module_path = scripts_dir / "test.py"
            module_path.write_text(
                """
import json
import numpy as np
import pandas as pd

def my_tool() -> dict:
    '''A real tool.'''
    return _safe_json({"status": "ok"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            # Only my_tool should be discovered
            self.assertIsNotNone(registry.get("my_tool"))
            self.assertIsNone(registry.get("json"))
            self.assertIsNone(registry.get("np"))
            self.assertIsNone(registry.get("pd"))
            self.assertIsNone(registry.get("__name__"))
            self.assertIsNone(registry.get("__file__"))

    def test_auto_discovery_excludes_imported_builtin_classes(self) -> None:
        """Test that imported builtin callable classes are not auto-discovered as tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "test_skill"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text("# Test Skill\n", encoding="utf-8")

            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            module_path = scripts_dir / "test.py"
            module_path.write_text(
                """
from datetime import datetime

def my_tool() -> dict:
    '''A real tool.'''
    return _safe_json({"status": "ok"})
""".strip()
                + "\n",
                encoding="utf-8",
            )

            registry = ToolRegistry(auto_load_from_skills=False)
            registry.skills_dir_path = root
            registry.skills_md_path = root / "SKILLS.md"
            registry.install_context(Context())
            registry.load_tools_from_skills()

            self.assertIsNotNone(registry.get("my_tool"))
            self.assertIsNone(registry.get("datetime"))

    def test_real_skill_enums_are_exported_and_enforced(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        expected = {
            ("smart_quality_gate", "mode"): ["fast", "balanced", "full"],
            ("fd_find", "file_type"): ["f", "d", ""],
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

        bad_quality_gate = json.loads(
            registry.execute(
                ToolCall(
                    id="enum_bad_quality_gate",
                    name="smart_quality_gate",
                    arguments={"path": ".", "mode": "turbo"},
                ),
                use_toon=False,
            )
        )
        self.assertEqual(bad_quality_gate.get("status"), "error")
        self.assertEqual(
            bad_quality_gate.get("error_type"),
            "schema_type_validation_failed",
        )

    def test_coding_svg_and_rag_schema_audit(self) -> None:
        registry = self._registry()
        registry.load_tools_from_skills()

        allowed_types = {"string", "int", "float", "bool", "list", "dict"}
        suspicious_allowlist = {("svg_pipeline", "steps")}
        suspicious: list[tuple[str, str, str]] = []

        tool_names = []
        for tool in registry.list_tools():
            source_path = str(tool.source_path or "")
            if (
                "/skills/coding/" not in source_path
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
                looks_numeric = (
                    n.startswith(("n_", "num_", "max_", "min_", "top_"))
                    or n.endswith(
                        ("_count", "_idx", "_index", "_points", "_steps", "_window", "_k")
                    )
                    or n
                    in {
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
                )
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

        suspicious = [item for item in suspicious if (item[0], item[1]) not in suspicious_allowlist]
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
