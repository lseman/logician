#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugin — Logician plugin marketplace CLI.

Usage:
    plugin marketplace add <owner/name[@version]>
    plugin install <owner/name | name>
    plugin list
    plugin remove <name> [--keep-checkout]
    plugin update <name | --all>
    plugin enable <name>
    plugin disable <name>
    plugin deps [name]

Examples:
    plugin marketplace add thedotmack/claude-mem
    plugin install claude-mem
    plugin install thedotmack/claude-mem      # clone + install in one step
    plugin list
    plugin update claude-mem
    plugin remove claude-mem
    plugin deps                               # check all plugins
    plugin deps claude-mem                    # check single plugin
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when run directly
_HERE = Path(__file__).resolve().parent
if str(_HERE / "src") not in sys.path:
    sys.path.insert(0, str(_HERE / "src"))

from plugin_manager import PluginManager  # noqa: E402


def _print_result(result: dict) -> None:
    msg = result.get("message", "")
    status = result.get("status", "")
    if msg:
        print(msg)
    elif status:
        print(f"[{status}]")


def cmd_marketplace(args: list[str]) -> int:
    if not args:
        print("Usage: plugin marketplace <subcommand> [args]")
        print("  add <owner/name[@version]>")
        return 1

    sub = args[0]
    if sub == "add":
        if len(args) < 2:
            print("Usage: plugin marketplace add <owner/name[@version]>")
            return 1
        ref = args[1]
        mgr = PluginManager()
        result = mgr.marketplace_add(ref)
        _print_result(result)
        return 0 if result["status"] in ("cloned", "already_cloned") else 1
    else:
        print(f"Unknown marketplace subcommand: {sub!r}")
        return 1


def cmd_install(args: list[str]) -> int:
    if not args:
        print("Usage: plugin install <owner/name | name>")
        return 1
    name = args[0]
    mgr = PluginManager()
    result = mgr.install(name)
    _print_result(result)
    return 0 if result["status"] in ("installed", "already_installed") else 1


def cmd_list(args: list[str]) -> int:
    mgr = PluginManager()
    result = mgr.list_plugins()
    plugins = result.get("plugins", [])
    if not plugins:
        print("No plugins installed.")
        return 0
    print(f"{'NAME':<24} {'VERSION':<12} {'ENABLED':<8} {'INSTALLED AT':<26}")
    print("-" * 74)
    for p in plugins:
        enabled = "yes" if p.get("enabled", False) else "no"
        installed_at = p.get("installed_at", "")[:19].replace("T", " ")
        print(f"{p['name']:<24} {p['version']:<12} {enabled:<8} {installed_at:<26}")
    return 0


def cmd_remove(args: list[str]) -> int:
    if not args:
        print("Usage: plugin remove <name> [--keep-checkout]")
        return 1
    name = args[0]
    keep = "--keep-checkout" in args
    mgr = PluginManager()
    result = mgr.remove(name, keep_cache=keep)
    _print_result(result)
    return 0 if result["status"] == "removed" else 1


def cmd_update(args: list[str]) -> int:
    if not args:
        print("Usage: plugin update <name | --all>")
        return 1
    mgr = PluginManager()
    if args[0] == "--all":
        names = mgr.all_plugin_ids()
        if not names:
            print("No plugins installed.")
            return 0
        rc = 0
        for name in names:
            result = mgr.update(name)
            _print_result(result)
            if result["status"] == "error":
                rc = 1
        return rc
    else:
        result = mgr.update(args[0])
        _print_result(result)
        return 0 if result["status"] in ("updated", "up_to_date") else 1


def cmd_enable(args: list[str]) -> int:
    if not args:
        print("Usage: plugin enable <name>")
        return 1
    mgr = PluginManager()
    result = mgr.enable(args[0])
    _print_result(result)
    return 0 if result["status"] in ("enabled", "already_enabled") else 1


def cmd_disable(args: list[str]) -> int:
    if not args:
        print("Usage: plugin disable <name>")
        return 1
    mgr = PluginManager()
    result = mgr.disable(args[0])
    _print_result(result)
    return 0 if result["status"] in ("disabled", "already_disabled") else 1


def cmd_deps(args: list[str]) -> int:
    """Check plugin dependencies."""
    mgr = PluginManager()
    if not args:
        # Validate all plugins
        result = mgr.validate_dependencies()
        if result["issues"]:
            print("Dependency issues found:")
            for issue in result["issues"]:
                print(f"  {issue['plugin_id']}: {issue['status']}")
                if issue.get("missing"):
                    print(f"    Missing: {', '.join(issue['missing'])}")
            return 1
        else:
            print("All plugin dependencies OK.")
            return 0

    name = args[0]
    plugin_id = mgr._resolve_plugin_id(name)
    if not plugin_id:
        print(f"Plugin '{name}' not found.")
        return 1

    inst = mgr.registry.get(plugin_id)
    if not inst:
        print(f"Plugin '{name}' not installed.")
        return 1

    manifest = mgr._load_plugin_manifest(Path(inst.install_path))
    deps = manifest.get("dependencies", [])

    if deps:
        try:
            required, missing = mgr._resolve_dependencies(plugin_id)
            print(f"Plugin: {name}")
            print(f"Dependencies: {', '.join(deps)}")
            print(f"Resolved: {'yes' if required else 'none'}")
            if missing:
                print(f"Missing: {', '.join(missing)}")
                return 1
            else:
                print("All dependencies satisfied.")
        except ValueError as e:
            print(f"Circular dependency detected: {e}")
            return 1
    else:
        print(f"Plugin: {name}")
        print("No declared dependencies.")
    return 0


_COMMANDS = {
    "marketplace": cmd_marketplace,
    "install": cmd_install,
    "list": cmd_list,
    "remove": cmd_remove,
    "update": cmd_update,
    "enable": cmd_enable,
    "disable": cmd_disable,
    "deps": cmd_deps,
}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    cmd = args[0]
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd!r}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        return 1

    return _COMMANDS[cmd](args[1:])


if __name__ == "__main__":
    sys.exit(main())
