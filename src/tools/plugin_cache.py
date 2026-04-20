"""
Versioned plugin cache system (OpenClaude-style).

This module provides versioned plugin storage with scope tracking (user/project/local),
similar to OpenClaude's installed_plugins.json V2 format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Maximum number of plugin versions to keep per plugin
MAX_VERSIONS_PER_PLUGIN = 5

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "logician" / "plugins"


@dataclass
class PluginInstallationEntry:
    """
    Represents a plugin installation at a specific scope.

    This mirrors OpenClaude's PluginInstallationEntry with support for
    multiple scopes (user/project/local) and version tracking.
    """

    scope: str  # "user", "project", or "local"
    install_path: str
    version: str = "unknown"
    installed_at: str = ""
    last_updated: str = ""
    git_commit_sha: str | None = None
    project_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope,
            "install_path": self.install_path,
            "version": self.version,
            "installed_at": self.installed_at,
            "last_updated": self.last_updated,
            "git_commit_sha": self.git_commit_sha,
            "project_path": self.project_path,
        }


@dataclass
class PluginInfo:
    """
    Metadata about a plugin from its manifest.
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    repository: str = ""
    license: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "repository": self.repository,
            "license": self.license,
            "enabled": self.enabled,
        }


class PluginCacheManager:
    """
    Manages versioned plugin cache with scope tracking.

    This mirrors OpenClaude's installed_plugins.json V2 format, which:
    - Tracks plugins in a single file (not dual V1/V2 files)
    - Uses versioned cache paths: ~/.cache/logician/plugins/{marketplace}/{plugin}/{version}
    - Supports multiple scopes per plugin (user/project/local)
    - Tracks git commit SHA for git-based plugins
    - Provides migration from legacy formats
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._installed_plugins: dict[str, list[PluginInstallationEntry]] = {}
        self._plugin_manifests: dict[str, PluginInfo] = {}
        self._dirty = False

    def _get_installed_plugins_path(self) -> Path:
        """Get path to installed_plugins.json."""
        return self.cache_dir / "installed_plugins.json"

    def _load_installed_plugins(self) -> None:
        """Load installed plugins from disk."""
        path = self._get_installed_plugins_path()
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
                self._installed_plugins = {
                    plugin_id: [
                        PluginInstallationEntry(**entry)
                        for entry in entries
                        if isinstance(entry, dict)
                    ]
                    for plugin_id, entries in data.get("plugins", {}).items()
                }
        except (json.JSONDecodeError, OSError):
            self._installed_plugins = {}

    def _save_installed_plugins(self) -> None:
        """Save installed plugins to disk."""
        path = self._get_installed_plugins_path()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 2,
            "plugins": {
                plugin_id: [entry.to_dict() for entry in entries]
                for plugin_id, entries in self._installed_plugins.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._dirty = False

    def register_plugin(
        self,
        plugin_id: str,
        scope: str,
        install_path: str,
        version: str = "unknown",
        installed_at: str | None = None,
        last_updated: str | None = None,
        git_commit_sha: str | None = None,
        project_path: str | None = None,
    ) -> None:
        """
        Register a plugin installation.

        Creates the versioned cache directory structure:
        ~/.cache/logician/plugins/{marketplace}/{plugin}/{version}
        """
        # Parse plugin_id to get marketplace name
        # Format: "plugin-name@marketplace-name"
        if "@" in plugin_id:
            _, marketplace = plugin_id.rsplit("@", 1)
        else:
            marketplace = "local"

        # Create versioned cache path
        cache_path = self.cache_dir / marketplace / plugin_id.replace("@", "/") / version
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy plugin files to versioned location
        src_path = Path(install_path)
        if src_path.exists():
            import shutil

            shutil.copytree(src_path, cache_path, dirs_exist_ok=True)

        # Create or update installation entry
        entry = PluginInstallationEntry(
            scope=scope,
            install_path=str(cache_path),
            version=version,
            installed_at=installed_at or datetime.now().isoformat(),
            last_updated=last_updated or datetime.now().isoformat(),
            git_commit_sha=git_commit_sha,
            project_path=project_path,
        )

        # Get or create list for this plugin
        if plugin_id not in self._installed_plugins:
            self._installed_plugins[plugin_id] = []

        # Find existing entry for this scope+project_path
        existing_index = next(
            (
                i
                for i, e in enumerate(self._installed_plugins[plugin_id])
                if e.scope == scope and e.project_path == project_path
            ),
            None,
        )

        if existing_index is not None:
            self._installed_plugins[plugin_id][existing_index] = entry
        else:
            self._installed_plugins[plugin_id].append(entry)

        self._dirty = True

    def unregister_plugin(self, plugin_id: str, scope: str, project_path: str | None = None) -> bool:
        """
        Unregister a plugin installation.

        Returns True if successfully unregistered, False otherwise.
        """
        if plugin_id not in self._installed_plugins:
            return False

        entries = self._installed_plugins[plugin_id]
        remaining = [
            e
            for e in entries
            if not (e.scope == scope and e.project_path == project_path)
        ]

        if len(remaining) == len(entries):
            return False

        self._installed_plugins[plugin_id] = remaining

        # Clean up cache directory if no more installations
        if not remaining:
            self._cleanup_plugin_cache(plugin_id)

        self._dirty = True
        return True

    def _cleanup_plugin_cache(self, plugin_id: str) -> None:
        """Remove plugin's cache directory if no installations remain."""
        # Parse plugin_id
        if "@" in plugin_id:
            _, marketplace = plugin_id.rsplit("@", 1)
        else:
            marketplace = "local"

        # Try both formats: plugin@marketplace or just plugin
        base_paths = [
            self.cache_dir / marketplace / plugin_id,
            self.cache_dir / plugin_id,
        ]

        for base_path in base_paths:
            if base_path.exists() and not any(base_path.iterdir()):
                import shutil

                shutil.rmtree(base_path)

    def get_installed_plugins(self) -> dict[str, list[PluginInstallationEntry]]:
        """Get all installed plugins."""
        if not self._dirty:
            self._load_installed_plugins()
        return self._installed_plugins.copy()

    def is_plugin_installed(self, plugin_id: str) -> bool:
        """Check if a plugin is installed in any scope."""
        if not self._dirty:
            self._load_installed_plugins()

        if plugin_id not in self._installed_plugins:
            return False

        return any(e.scope in ("user", "project", "local") for e in self._installed_plugins[plugin_id])

    def get_plugin_installations(
        self, plugin_id: str, scope: str | None = None
    ) -> list[PluginInstallationEntry]:
        """Get installations for a plugin, optionally filtered by scope."""
        if not self._dirty:
            self._load_installed_plugins()

        if plugin_id not in self._installed_plugins:
            return []

        entries = self._installed_plugins[plugin_id]
        if scope is None:
            return entries

        return [e for e in entries if e.scope == scope]

    def is_plugin_globally_installed(self, plugin_id: str) -> bool:
        """Check if a plugin is installed at user or managed scope."""
        if not self._dirty:
            self._load_installed_plugins()

        if plugin_id not in self._installed_plugins:
            return False

        return any(
            e.scope in ("user", "managed") for e in self._installed_plugins[plugin_id]
        )

    def get_versioned_cache_path(
        self, plugin_id: str, version: str
    ) -> Path | None:
        """Get the versioned cache path for a plugin."""
        if "@" in plugin_id:
            _, marketplace = plugin_id.rsplit("@", 1)
        else:
            marketplace = "local"

        return self.cache_dir / marketplace / plugin_id.replace("@", "/") / version

    def register_plugin_manifest(self, plugin_id: str, info: PluginInfo) -> None:
        """Register plugin manifest information."""
        self._plugin_manifests[plugin_id] = info
        self._dirty = True

    def get_plugin_manifest(self, plugin_id: str) -> PluginInfo | None:
        """Get manifest information for a plugin."""
        return self._plugin_manifests.get(plugin_id)

    def get_versioned_cache_path_for_plugin(
        self, plugin_id: str, version: str
    ) -> Path | None:
        """Get the versioned cache path for a plugin version."""
        if "@" in plugin_id:
            _, marketplace = plugin_id.rsplit("@", 1)
        else:
            marketplace = "local"

        return self.cache_dir / marketplace / plugin_id.replace("@", "/") / version

    def is_installation_relevant_to_current_project(
        self, entry: PluginInstallationEntry, current_project_path: str | None = None
    ) -> bool:
        """Check if an installation is relevant to the current project."""
        return (
            entry.scope in ("user", "managed")
            or (entry.project_path and entry.project_path == current_project_path)
        )

    def get_relevant_installations(
        self, plugin_id: str, current_project_path: str | None = None
    ) -> list[PluginInstallationEntry]:
        """Get installations relevant to the current project."""
        if not self._dirty:
            self._load_installed_plugins()

        if plugin_id not in self._installed_plugins:
            return []

        entries = self._installed_plugins[plugin_id]
        relevant = [
            e
            for e in entries
            if self.is_installation_relevant_to_current_project(
                e, current_project_path
            )
        ]
        return relevant

    def cleanup_old_versions(
        self, plugin_id: str, keep_count: int = MAX_VERSIONS_PER_PLUGIN
    ) -> int:
        """
        Clean up old versions of a plugin.

        Returns the number of versions removed.
        """
        if plugin_id not in self._installed_plugins:
            return 0

        cache_path = self.get_versioned_cache_path_for_plugin(plugin_id, "unknown")
        if not cache_path or not cache_path.exists():
            return 0

        # List all version directories
        versions = [
            d.name
            for d in cache_path.parent.iterdir()
            if d.is_dir() and d.name != plugin_id
        ]

        # Keep the newest ones
        versions.sort(reverse=True)
        versions_to_remove = versions[keep_count:]

        removed_count = 0
        for version in versions_to_remove:
            version_path = cache_path / version
            try:
                import shutil

                shutil.rmtree(version_path)
                removed_count += 1
            except OSError:
                pass

        return removed_count

    def save(self) -> None:
        """Force save to disk."""
        if self._dirty:
            self._save_installed_plugins()


def create_plugin_id(name: str, marketplace: str) -> str:
    """Create a plugin ID in 'name@marketplace' format."""
    return f"{name}@{marketplace}"


def parse_plugin_id(plugin_id: str) -> tuple[str, str]:
    """Parse plugin ID into (name, marketplace) tuple."""
    if "@" in plugin_id:
        return plugin_id.rsplit("@", 1)
    return plugin_id, "local"


# =============================================================================
# Migration helpers
# =============================================================================


def migrate_legacy_cache(legacy_cache_dir: Path) -> int:
    """
    Migrate legacy flat cache to versioned structure.

    Legacy: ~/.cache/logician/plugins/{plugin-name}/
    Versioned: ~/.cache/logician/plugins/{marketplace}/{plugin}/{version}/

    Returns the number of plugins migrated.
    """
    migrated = 0

    if not legacy_cache_dir.exists():
        return 0

    for plugin_dir in legacy_cache_dir.iterdir():
        if not plugin_dir.is_dir():
            continue

        # Try to get version from directory name or manifest
        version = "latest"
        manifest_path = plugin_dir / ".claude-plugin" / "plugin.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                    version = data.get("version", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        # Create versioned path
        marketplace = "local"
        new_path = DEFAULT_CACHE_DIR / marketplace / plugin_dir.name / version
        new_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copytree(plugin_dir, new_path, dirs_exist_ok=True)

        migrated += 1

    # Clean up legacy directory
    if migrated > 0:
        import shutil

        shutil.rmtree(legacy_cache_dir)

    return migrated
