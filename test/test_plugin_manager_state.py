from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.plugin_manager.state import InstalledPluginsRegistry, PluginInstall, _now_iso
from src.plugin_manager.state import iter_enabled_plugin_install_paths


class PluginManagerStateTests(unittest.TestCase):
    def test_iter_enabled_plugin_install_paths_returns_only_enabled_existing_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = Path(tmpdir)
            live_plugin = plugins_dir / "cache" / "demo" / "live" / "1.0.0"
            live_plugin.mkdir(parents=True, exist_ok=True)
            disabled_plugin = plugins_dir / "cache" / "demo" / "disabled" / "1.0.0"
            disabled_plugin.mkdir(parents=True, exist_ok=True)
            missing_plugin = plugins_dir / "cache" / "demo" / "missing" / "1.0.0"

            registry = InstalledPluginsRegistry(plugins_dir)
            timestamp = _now_iso()
            registry.upsert(
                "live@demo",
                PluginInstall(
                    scope="user",
                    install_path=str(live_plugin),
                    version="1.0.0",
                    installed_at=timestamp,
                    last_updated=timestamp,
                    enabled=True,
                ),
            )
            registry.upsert(
                "disabled@demo",
                PluginInstall(
                    scope="user",
                    install_path=str(disabled_plugin),
                    version="1.0.0",
                    installed_at=timestamp,
                    last_updated=timestamp,
                    enabled=False,
                ),
            )
            registry.upsert(
                "missing@demo",
                PluginInstall(
                    scope="user",
                    install_path=str(missing_plugin),
                    version="1.0.0",
                    installed_at=timestamp,
                    last_updated=timestamp,
                    enabled=True,
                ),
            )

            installs = iter_enabled_plugin_install_paths(plugins_dir)

        self.assertEqual(installs, [("live@demo", live_plugin.resolve())])


if __name__ == "__main__":
    unittest.main()
