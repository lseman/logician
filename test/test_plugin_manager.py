import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from plugin_manager import PluginManager
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src_dir))
    from plugin_manager import PluginManager


class PluginManagerTests(unittest.TestCase):
    def test_local_install_creates_versioned_cache_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            plugins_dir = tmp_root / "claude_plugins"
            manager = PluginManager(plugins_dir=plugins_dir)

            plugin_src = tmp_root / "local_plugin"
            plugin_src.mkdir(parents=True)
            plugin_manifest = plugin_src / ".claude-plugin"
            plugin_manifest.mkdir(parents=True)
            (plugin_manifest / "plugin.json").write_text(
                json.dumps({"name": "test-plugin", "version": "1.0.0"}),
                encoding="utf-8",
            )
            skills_dir = plugin_src / "skills" / "example"
            skills_dir.mkdir(parents=True)
            memory_dir = plugin_src / "memory"
            memory_dir.mkdir(parents=True)
            (memory_dir / "MEMORY.md").write_text(
                "---\nname: Project memory\ndescription: Plugin memory index\n---\n",
                encoding="utf-8",
            )
            (skills_dir / "SKILL.md").write_text(
                "```skill\n---\nname: example\ndescription: simple\n---\n```",
                encoding="utf-8",
            )
            commands_dir = plugin_src / "commands" / "memory"
            commands_dir.mkdir(parents=True)
            (commands_dir / "MEMORY.md").write_text(
                "---\nname: Memory\ndescription: Command-based plugin memory skill\n---\n",
                encoding="utf-8",
            )

            result = manager.install(str(plugin_src))
            self.assertEqual(result["status"], "installed")
            self.assertEqual(result["name"], "test-plugin")
            self.assertEqual(result["version"], "1.0.0")

            cache_root = plugins_dir / "cache" / "local" / "test-plugin" / "1-0-0"
            self.assertTrue(cache_root.is_dir())
            self.assertTrue((cache_root / "skills" / "example" / "SKILL.md").exists())
            self.assertTrue((cache_root / "skill_index.json").is_file())

            info = manager.info("test-plugin")
            self.assertEqual(info["status"], "ok")
            self.assertTrue(info["enabled"])
            self.assertEqual(info["version"], "1.0.0")

            listing = manager.list_plugins()
            self.assertEqual(listing["status"], "ok")
            self.assertEqual(len(listing["plugins"]), 1)
            self.assertTrue(listing["plugins"][0]["enabled"])
            self.assertEqual(listing["plugins"][0]["plugin_id"], "test-plugin@local")

            self.assertEqual(manager.all_plugin_ids(), ["test-plugin@local"])
            self.assertEqual(
                manager.skills_paths(),
                [cache_root / "skills", cache_root / "commands"],
            )
            self.assertEqual(
                manager.memory_paths(),
                [
                    cache_root / "memory",
                    cache_root / "commands" / "memory",
                    cache_root / "commands",
                    cache_root,
                ],
            )

            disable_result = manager.disable("test-plugin")
            self.assertEqual(disable_result["status"], "disabled")
            self.assertFalse(manager.registry.get("test-plugin@local").enabled)
            self.assertEqual(manager.skills_paths(), [])

            enable_result = manager.enable("test-plugin")
            self.assertEqual(enable_result["status"], "enabled")
            self.assertTrue(manager.registry.get("test-plugin@local").enabled)
            self.assertEqual(
                manager.skills_paths(),
                [cache_root / "skills", cache_root / "commands"],
            )

    def test_local_install_without_plugin_manifest_still_indexes_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            plugins_dir = tmp_root / "claude_plugins"
            manager = PluginManager(plugins_dir=plugins_dir)

            repo_src = tmp_root / "hermes_repo"
            repo_src.mkdir(parents=True)
            skills_dir = repo_src / "skills" / "sample"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text(
                "```skill\n---\nname: sample\ndescription: plain skills repo\n---\n```",
                encoding="utf-8",
            )

            result = manager.install(str(repo_src))
            self.assertEqual(result["status"], "installed")
            self.assertEqual(result["name"], "hermes_repo")
            self.assertTrue(
                result["install_path"].endswith("/cache/local/hermes_repo/" + result["version"])
            )

            cache_root = Path(result["install_path"])
            self.assertTrue((cache_root / "skills" / "sample" / "SKILL.md").exists())
            self.assertTrue((cache_root / "skill_index.json").is_file())
            self.assertEqual(manager.list_plugins()["plugins"][0]["plugin_id"], "hermes_repo@local")
            self.assertEqual(manager.skills_paths(), [cache_root / "skills"])

    def test_component_skill_roots_are_indexed_at_install_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            plugins_dir = tmp_root / "claude_plugins"
            manager = PluginManager(plugins_dir=plugins_dir)

            plugin_src = tmp_root / "component_plugin"
            plugin_src.mkdir(parents=True)
            plugin_manifest = plugin_src / ".claude-plugin"
            plugin_manifest.mkdir(parents=True)
            (plugin_manifest / "plugin.json").write_text(
                json.dumps(
                    {
                        "name": "component-plugin",
                        "version": "2.0.0",
                        "components": {"skills": ["addons/custom-skills"]},
                    }
                ),
                encoding="utf-8",
            )
            skill_dir = plugin_src / "addons" / "custom-skills" / "triage"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "```skill\n---\nname: triage\ndescription: classify issues\n---\n```",
                encoding="utf-8",
            )

            result = manager.install(str(plugin_src))
            self.assertEqual(result["status"], "installed")

            cache_root = plugins_dir / "cache" / "local" / "component-plugin" / "2-0-0"
            roots = manager.skills_source_roots()
            self.assertIn(cache_root / "addons" / "custom-skills", [root.path for root in roots])
            self.assertTrue((cache_root / "skill_index.json").is_file())
