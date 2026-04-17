# -*- coding: utf-8 -*-
"""
Plugin state — reads/writes ~/.claude/plugins/installed_plugins.json (V2 format).

Compatible with Claude Code's own installed_plugins.json so plugins installed
via `claude /plugin` and via logician's plugin CLI share the same registry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Scope = Literal["user", "project", "local", "managed"]


@dataclass
class PluginInstall:
    """One scope-installation record inside installed_plugins.json."""

    scope: Scope
    install_path: str
    version: str
    installed_at: str
    last_updated: str
    git_commit_sha: str = ""
    enabled: bool = True
    project_path: str = ""  # only for project/local scopes
    dependencies: list[str] = field(default_factory=list)  # plugin_id references

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scope": self.scope,
            "installPath": self.install_path,
            "version": self.version,
            "installedAt": self.installed_at,
            "lastUpdated": self.last_updated,
            "enabled": self.enabled,
        }
        if self.git_commit_sha:
            d["gitCommitSha"] = self.git_commit_sha
        if self.project_path and self.scope in ("project", "local"):
            d["projectPath"] = self.project_path
        if self.dependencies:
            d["dependencies"] = self.dependencies
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PluginInstall":
        return cls(
            scope=d.get("scope", "user"),
            install_path=d.get("installPath", ""),
            version=d.get("version", "unknown"),
            installed_at=d.get("installedAt", ""),
            last_updated=d.get("lastUpdated", ""),
            git_commit_sha=d.get("gitCommitSha", ""),
            enabled=d.get("enabled", True),
            project_path=d.get("projectPath", ""),
            dependencies=d.get("dependencies", []),
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _openclaude_plugins_dir() -> Path | None:
    """Find openclaude's plugins directory if it exists."""
    claude_config = Path.home() / ".claude"
    openclaude_plugins = claude_config / "openclaude" / "plugins"
    if openclaude_plugins.is_dir():
        return openclaude_plugins
    return None


def _openclaude_repo_plugins_dir() -> Path | None:
    """Find plugins in the openclaude repo checked out under logician."""
    root = Path(__file__).resolve().parent.parent.parent
    openclaude_repo = root / "repos" / "openclaude"
    if openclaude_repo.is_dir():
        plugins_dir = openclaude_repo / "plugins"
        if plugins_dir.is_dir():
            return plugins_dir
    return None


def _plugins_base_dir() -> Path:
    """Resolve ~/.claude/plugins (respects CLAUDE_CODE_PLUGIN_CACHE_DIR env var).

    Priority:
    1. CLAUDE_CODE_PLUGIN_CACHE_DIR env var (explicit override)
    2. openclaude's plugins dir (~/.claude/openclaude/plugins) if it exists
    3. Default: ~/.claude/plugins
    """
    override = os.environ.get("CLAUDE_CODE_PLUGIN_CACHE_DIR", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if p.is_dir():
            return p
    openclaude = _openclaude_plugins_dir()
    if openclaude is not None:
        return openclaude
    return Path.home() / ".claude" / "plugins"


class InstalledPluginsRegistry:
    """
    Read/write ~/.claude/plugins/installed_plugins.json.

    The V2 format is:
        {
          "version": 2,
          "plugins": {
            "<name>@<marketplace>": [<PluginInstall>, ...]
          }
        }

    We use the same format so Claude Code and logician share state.
    """

    VERSION = 2

    def __init__(self, plugins_dir: Path | None = None) -> None:
        base = plugins_dir or _plugins_base_dir()
        base.mkdir(parents=True, exist_ok=True)
        self._path = base / "installed_plugins.json"
        self._data = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, plugin_id: str, scope: Scope = "user") -> PluginInstall | None:
        """Return the install record for plugin_id@scope, or None."""
        records = self._data["plugins"].get(plugin_id, [])
        for r in records:
            inst = PluginInstall.from_dict(r)
            if inst.scope == scope:
                return inst
        return None

    def upsert(self, plugin_id: str, install: PluginInstall) -> None:
        """Insert or replace the record for plugin_id at install.scope."""
        records: list[dict] = self._data["plugins"].setdefault(plugin_id, [])
        for i, r in enumerate(records):
            if r.get("scope") == install.scope:
                records[i] = install.to_dict()
                self._save()
                return
        records.append(install.to_dict())
        self._save()

    def remove(self, plugin_id: str, scope: Scope = "user") -> bool:
        """Remove one scope-record. Returns True if something was removed."""
        records = self._data["plugins"].get(plugin_id, [])
        before = len(records)
        self._data["plugins"][plugin_id] = [r for r in records if r.get("scope") != scope]
        if not self._data["plugins"][plugin_id]:
            del self._data["plugins"][plugin_id]
        changed = len(self._data["plugins"].get(plugin_id, [])) != before - (1 if before > 0 else 0)
        if before != len(self._data["plugins"].get(plugin_id, [])) + (
            1 if plugin_id not in self._data["plugins"] else 0
        ):
            self._save()
            return True
        self._save()
        return before > len(self._data["plugins"].get(plugin_id, []))

    def all_plugin_ids(self) -> list[str]:
        return sorted(self._data["plugins"].keys())

    def all_installs(self) -> list[tuple[str, PluginInstall]]:
        """Return (plugin_id, install) for every scope-record."""
        out: list[tuple[str, PluginInstall]] = []
        for pid, records in self._data["plugins"].items():
            for r in records:
                out.append((pid, PluginInstall.from_dict(r)))
        return out

    # ------------------------------------------------------------------
    # Known marketplaces  (known_marketplaces.json)
    # ------------------------------------------------------------------

    @property
    def _marketplaces_path(self) -> Path:
        return self._path.parent / "known_marketplaces.json"

    def known_marketplaces(self) -> dict[str, Any]:
        if self._marketplaces_path.exists():
            try:
                return json.loads(self._marketplaces_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def register_marketplace(
        self, name: str, source: dict[str, Any], install_location: str
    ) -> None:
        """Add or update an entry in known_marketplaces.json."""
        data = self.known_marketplaces()
        data[name] = {
            "source": source,
            "installLocation": install_location,
            "lastUpdated": _now_iso(),
        }
        self._marketplaces_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                # Migrate V1 → V2
                if raw.get("version", 1) < self.VERSION:
                    return self._migrate_v1(raw)
                return raw
            except Exception:
                pass
        return {"version": self.VERSION, "plugins": {}}

    @staticmethod
    def _migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
        """Best-effort migration from old logician state format."""
        plugins: dict[str, list[dict]] = {}
        for name, info in raw.get("plugins", {}).items():
            if isinstance(info, list):
                # Already array format
                plugins[name] = info
            elif isinstance(info, dict):
                # Old logician format: {name, owner, version, checkout_path, ...}
                owner = info.get("owner", "unknown")
                pid = f"{name}@{owner}"
                plugins[pid] = [
                    {
                        "scope": "user",
                        "installPath": info.get("checkout_path", ""),
                        "version": info.get("version", "unknown"),
                        "installedAt": info.get("installed_at", _now_iso()),
                        "lastUpdated": info.get("installed_at", _now_iso()),
                        "gitCommitSha": "",
                    }
                ]
        return {"version": 2, "plugins": plugins}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
