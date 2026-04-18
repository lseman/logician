# -*- coding: utf-8 -*-
"""
Plugin hook loader - loads SessionStart (and other) hooks from installed plugins.

This module integrates with the PluginManager registry to discover and load
hooks from enabled plugins, following the OpenClaude hook loading pattern.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    HookCommand,
    HookCommandType,
    HookDefinition,
    HookEventType,
    HookMatcher,
    load_plugin_hooks,
)

if TYPE_CHECKING:
    from src.plugin_manager.manager import PluginManager


@dataclass
class LoadedHook:
    """A hook loaded from a plugin, ready for execution."""

    plugin_id: str
    plugin_name: str
    plugin_dir: Path  # Root directory for setting CLAUDE_PLUGIN_ROOT
    event_type: HookEventType
    definition: HookDefinition


class HookLoader:
    """Loads hooks from the plugin registry."""

    _cache_lock = threading.Lock()
    _event_cache: dict[
        tuple[HookEventType, str],
        tuple[tuple[Any, ...], tuple[LoadedHook, ...]],
    ] = {}

    def __init__(self, plugin_manager: PluginManager | None = None) -> None:
        from src.plugin_manager.manager import PluginManager as PM

        self.plugin_manager = plugin_manager or PM()

    def get_session_start_hooks(self) -> list[LoadedHook]:
        """Get all SessionStart hooks from enabled plugins."""
        return self._get_hooks_for_event(HookEventType.SESSION_START)

    def _get_hooks_for_event(
        self, event_type: HookEventType, source: str = "startup"
    ) -> list[LoadedHook]:
        """Get all hooks for a specific event type from enabled plugins."""
        cache_key = (event_type, str(source or "").strip().lower() or "startup")
        signature = self._cache_signature()
        with self._cache_lock:
            cached = self._event_cache.get(cache_key)
            if cached is not None and cached[0] == signature:
                return list(cached[1])

        loaded: list[LoadedHook] = []

        for plugin_id, inst in self.plugin_manager.registry.all_installs():
            if not inst.enabled:
                continue

            plugin_dir = Path(inst.install_path)
            if not plugin_dir.is_dir():
                continue

            plugin_hooks = load_plugin_hooks(plugin_dir)
            definitions = plugin_hooks.get(event_type, [])

            # Get plugin name from plugin.json
            plugin_name = self._get_plugin_name(plugin_dir, plugin_id)

            for definition in definitions:
                if self._matcher_matches(definition.matcher, cache_key[1]):
                    loaded.append(
                        LoadedHook(
                            plugin_id=plugin_id,
                            plugin_name=plugin_name,
                            plugin_dir=plugin_dir,
                            event_type=event_type,
                            definition=definition,
                        )
                    )

        with self._cache_lock:
            self._event_cache[cache_key] = (signature, tuple(loaded))
        return loaded

    def _cache_signature(self) -> tuple[Any, ...]:
        registry_path = getattr(self.plugin_manager.registry, "_path", None)
        registry_sig = self._path_signature(Path(registry_path)) if registry_path else None
        plugin_sigs: list[tuple[Any, ...]] = []

        for plugin_id, inst in self.plugin_manager.registry.all_installs():
            plugin_dir = Path(inst.install_path)
            plugin_sigs.append(
                (
                    str(plugin_id or "").strip(),
                    bool(inst.enabled),
                    str(plugin_dir),
                    self._path_signature(plugin_dir / ".claude-plugin" / "plugin.json"),
                    self._path_signature(plugin_dir / "hooks" / "hooks.json"),
                )
            )

        return (registry_sig, tuple(plugin_sigs))

    @staticmethod
    def _path_signature(path: Path) -> tuple[bool, int | None, int | None]:
        try:
            stat = path.stat()
        except Exception:
            return (False, None, None)
        return (True, int(getattr(stat, "st_mtime_ns", 0)), int(stat.st_size))

    def _get_plugin_name(self, plugin_dir: Path, plugin_id: str) -> str:
        """Get plugin name from plugin.json or fallback to plugin_id."""
        try:
            plugin_json = plugin_dir / ".claude-plugin" / "plugin.json"
            if plugin_json.exists():
                data = json.loads(plugin_json.read_text(encoding="utf-8"))
                name = data.get("name")
                if name and isinstance(name, str):
                    return name
        except Exception:
            pass

        # Fallback to parsing plugin_id (format: "name@marketplace")
        return plugin_id.split("@")[0] if "@" in plugin_id else plugin_id

    def _matcher_matches(self, matcher: HookMatcher | None, source: str) -> bool:
        """Check if a hook matcher matches the given source.

        Supports pipe-separated patterns like 'startup|clear|compact'.
        Returns True if any of the pattern alternatives match the source.
        """
        if matcher is None:
            return True  # No matcher = matches all

        pattern = matcher.matcher
        if not pattern:
            return True

        # Handle pipe-separated alternatives (e.g., "startup|clear|compact")
        sources = [s.strip().lower() for s in source.split("|")]
        patterns = [p.strip().lower() for p in pattern.split("|")]

        # True if any source matches any pattern alternative
        for s in sources:
            for p in patterns:
                if s == p:
                    return True

        # Also support simple contains for backwards compatibility
        return pattern.lower() in source.lower()
