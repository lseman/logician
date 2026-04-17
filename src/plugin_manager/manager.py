# -*- coding: utf-8 -*-
"""
PluginManager — openclaude-compatible plugin lifecycle.

Cache layout (mirrors ~/.claude/plugins/):
    ~/.claude/plugins/
    ├── cache/
    │   └── <marketplace>/
    │       └── <plugin-name>/
    │           └── <version>/      ← extracted plugin content
    ├── marketplaces/
    │   └── <marketplace>/
    │       └── .claude-plugin/
    │           └── marketplace.json
    ├── installed_plugins.json      ← V2 shared with Claude Code
    └── known_marketplaces.json
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from src.skills.path_resolution import (
        SkillSourceRoot,
        resolve_plugin_skill_roots,
        write_plugin_skill_index,
    )
except ModuleNotFoundError:  # pragma: no cover - top-level package compatibility for tests
    from skills.path_resolution import (
        SkillSourceRoot,
        resolve_plugin_skill_roots,
        write_plugin_skill_index,
    )
from .state import InstalledPluginsRegistry, PluginInstall, Scope, _now_iso, _plugins_base_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_BASE = "https://github.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Sanitize a path segment (matches openclaude rules)."""
    return re.sub(r"[^a-zA-Z0-9\-_]", "-", name).strip("-")


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _parse_ref(ref: str) -> tuple[str, str, str]:
    """Parse 'owner/name[@version]' → (owner, name, version_or_empty)."""
    version = ""
    if "@" in ref:
        ref, version = ref.rsplit("@", 1)
    if "/" not in ref:
        raise ValueError(f"Expected 'owner/name', got {ref!r}")
    owner, name = ref.split("/", 1)
    return owner.strip(), name.strip(), version.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_plugin_json(root: Path) -> Path | None:
    """Find .claude-plugin/plugin.json starting from root."""
    direct = root / ".claude-plugin" / "plugin.json"
    if direct.exists():
        return direct
    # Some repos have it one level deeper (e.g. repo/plugin/.claude-plugin/plugin.json)
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            nested = sub / ".claude-plugin" / "plugin.json"
            if nested.exists():
                return nested
    return None


def _find_marketplace_json(root: Path) -> Path | None:
    """Find .claude-plugin/marketplace.json at root."""
    candidate = root / ".claude-plugin" / "marketplace.json"
    return candidate if candidate.exists() else None


def _git_head_sha(repo_dir: Path) -> str:
    """Return the HEAD commit SHA of a git repo, or empty string."""
    rc, out, _ = _run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir))
    return out.strip() if rc == 0 and len(out.strip()) == 40 else ""


def _git_remote_owner(repo_dir: Path) -> str:
    """Extract GitHub owner from git remote URL."""
    rc, url, _ = _run(["git", "remote", "get-url", "origin"], cwd=str(repo_dir))
    if rc != 0 or not url:
        return ""
    # https://github.com/owner/repo.git  or  git@github.com:owner/repo.git
    url = url.strip().rstrip("/").removesuffix(".git")
    if "github.com" in url:
        parts = re.split(r"[:/]", url)
        if len(parts) >= 2:
            return parts[-2]
    return ""


def _copy_dir(src: Path, dst: Path) -> None:
    """Copy directory tree, preserving relative symlinks (security: no traversal)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_symlink():
            link_target = os.readlink(item)
            # Only preserve relative symlinks to prevent path traversal
            if not os.path.isabs(link_target):
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(link_target)
        elif item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(item, target)


def _version_from_path(install_path: str) -> str:
    """Extract version from  .../cache/<mkt>/<plugin>/<version>  path."""
    return Path(install_path).name if install_path else "unknown"


def _refresh_plugin_skill_index(cache_path: Path) -> None:
    """Write or refresh the install-time skill index for a cached plugin."""
    try:
        write_plugin_skill_index(cache_path)
    except Exception:
        # Best effort only; discovery falls back to filesystem scanning.
        pass


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------


class PluginManager:
    """
    Plugin lifecycle manager compatible with ~/.claude/plugins/.

    All cached plugins live at:
        ~/.claude/plugins/cache/<marketplace>/<plugin-name>/<version>/

    The registry at installed_plugins.json is shared with Claude Code.
    """

    def __init__(self, plugins_dir: Path | None = None) -> None:
        self.plugins_dir = (plugins_dir or _plugins_base_dir()).resolve()
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.plugins_dir / "cache"
        self.marketplaces_dir = self.plugins_dir / "marketplaces"
        self.registry = InstalledPluginsRegistry(self.plugins_dir)

    # ------------------------------------------------------------------
    # marketplace add  — clone marketplace repo + cache all its plugins
    # ------------------------------------------------------------------

    def marketplace_add(self, ref: str) -> dict[str, Any]:
        """
        Register a marketplace from GitHub and cache its plugins.

        ref can be:
          - 'owner/name'                 e.g. 'thedotmack/claude-mem'
          - 'owner/name@version'
          - full HTTPS URL

        The marketplace.json is cloned to:
            ~/.claude/plugins/marketplaces/<owner>/
        """
        if ref.startswith("https://") or ref.startswith("git@"):
            url = ref.rstrip("/").removesuffix(".git")
            owner = url.split("/")[-2]
            repo_name = url.split("/")[-1]
            version = ""
            url = ref if ref.endswith(".git") else ref + ".git"
        else:
            owner, repo_name, version = _parse_ref(ref)
            url = f"{_GITHUB_BASE}/{owner}/{repo_name}.git"

        mkt_name = _sanitize(owner)
        mkt_dir = self.marketplaces_dir / mkt_name

        if mkt_dir.exists():
            # Pull latest
            rc, _, err = _run(["git", "pull", "--ff-only"], cwd=str(mkt_dir))
            action = "updated" if rc == 0 else "exists"
        else:
            # Clone
            cmd = ["git", "clone", "--depth=1"]
            if version:
                cmd += ["--branch", version]
            cmd += [url, str(mkt_dir)]
            rc, _, err = _run(cmd)
            if rc != 0:
                return {"status": "error", "message": f"git clone failed: {err}"}
            action = "cloned"

        sha = _git_head_sha(mkt_dir)

        # Read marketplace manifest
        mkt_json_path = _find_marketplace_json(mkt_dir)
        mkt_json = _read_json(mkt_json_path) if mkt_json_path else {}

        # Register in known_marketplaces.json
        self.registry.register_marketplace(
            name=mkt_name,
            source={"source": "github", "repo": f"{owner}/{repo_name}"},
            install_location=str(mkt_dir),
        )

        plugins_listed = [p.get("name", "?") for p in mkt_json.get("plugins", [])]
        return {
            "status": action,
            "marketplace": mkt_name,
            "sha": sha,
            "plugins": plugins_listed,
            "message": (
                f"Marketplace '{mkt_name}' {action}. "
                f"Available plugins: {', '.join(plugins_listed) or 'none listed'}.\n"
                f"Run `plugin install <name>` to install any of them."
            ),
        }

    # ------------------------------------------------------------------
    # install — clone plugin repo, copy to versioned cache, record state
    # ------------------------------------------------------------------

    def install(self, ref: str, scope: Scope = "user", project_path: str = "") -> dict[str, Any]:
        """
        Install a plugin from GitHub into the versioned cache.

        ref: 'owner/name[@version]' or bare 'name' if already in a known marketplace.
        """
        # Resolve owner + name, including local directory installs.
        local_source = None
        if ref.startswith("file://"):
            local_source = Path(ref[len("file://") :]).expanduser()
        elif Path(ref).exists():
            local_source = Path(ref).expanduser()

        version = ""
        if local_source is not None:
            if not local_source.is_dir():
                return {
                    "status": "error",
                    "message": "Local plugin source must be a directory.",
                }
            clone_dir = local_source.resolve()
            sha = ""
            plugin_json_path = _find_plugin_json(clone_dir)
            plugin_json = _read_json(plugin_json_path) if plugin_json_path else {}
            resolved_name = plugin_json.get("name", local_source.name)
            owner = plugin_json.get("marketplace") or plugin_json.get("owner") or "local"
            mkt_name = _sanitize(owner)
            plugin_id = f"{resolved_name}@{mkt_name}"

            existing = self.registry.get(plugin_id, scope)
            if existing and Path(existing.install_path).is_dir():
                return {
                    "status": "already_installed",
                    "message": f"Plugin '{resolved_name}' v{existing.version} is already installed at {existing.install_path}.",
                    "install_path": existing.install_path,
                }

            resolved_version = plugin_json.get("version")
            if not resolved_version:
                mtime = int(clone_dir.stat().st_mtime)
                resolved_version = f"local-{mtime}"
        else:
            if "/" in ref.split("@")[0]:
                owner, name, version = _parse_ref(ref)
            else:
                # Bare name — search known marketplaces for it
                name = ref.split("@")[0]
                version = ref.split("@")[1] if "@" in ref else ""
                owner = self._find_owner_for(name)
                if not owner:
                    return {
                        "status": "error",
                        "message": (
                            f"Plugin '{name}' not found in any known marketplace. "
                            f"Run `plugin marketplace add <owner>/{name}` first, "
                            f"or pass the full 'owner/{name}' reference."
                        ),
                    }

            mkt_name = _sanitize(owner)
            plugin_id = f"{name}@{mkt_name}"

            existing = self.registry.get(plugin_id, scope)
            if existing and Path(existing.install_path).is_dir():
                return {
                    "status": "already_installed",
                    "message": f"Plugin '{name}' v{existing.version} is already installed at {existing.install_path}.",
                    "install_path": existing.install_path,
                }

            with tempfile.TemporaryDirectory(prefix="logician_plugin_") as tmp_str:
                tmp = Path(tmp_str)
                clone_dir = tmp / name
                url = f"{_GITHUB_BASE}/{owner}/{name}.git"
                cmd = ["git", "clone", "--depth=1"]
                if version:
                    cmd += ["--branch", version]
                cmd += [url, str(clone_dir)]
                rc, _, err = _run(cmd, timeout=180)
                if rc != 0:
                    return {"status": "error", "message": f"git clone failed: {err}"}

                sha = _git_head_sha(clone_dir)
                plugin_json_path = _find_plugin_json(clone_dir)
                plugin_json = _read_json(plugin_json_path) if plugin_json_path else {}
                resolved_version = (
                    plugin_json.get("version") or version or (sha[:12] if sha else "unknown")
                )
                resolved_name = plugin_json.get("name", name)

                cache_path = (
                    self.cache_dir
                    / _sanitize(mkt_name)
                    / _sanitize(resolved_name)
                    / _sanitize(resolved_version)
                )

                if cache_path.exists():
                    shutil.rmtree(cache_path)
                _copy_dir(clone_dir, cache_path)

                install_script = cache_path / "install.sh"
                install_log = ""
                if install_script.exists():
                    irc, iout, ierr = _run(["bash", str(install_script)], cwd=str(cache_path))
                    install_log = iout or ierr
                    if irc != 0:
                        shutil.rmtree(cache_path, ignore_errors=True)
                        return {"status": "error", "message": f"install.sh failed: {ierr}"}
                _refresh_plugin_skill_index(cache_path)

            # Extract and record dependencies from manifest
            manifest = self._load_plugin_manifest(cache_path)
            deps = manifest.get("dependencies", [])

            install = PluginInstall(
                scope=scope,
                install_path=str(cache_path),
                version=resolved_version,
                installed_at=_now_iso(),
                last_updated=_now_iso(),
                git_commit_sha=sha,
                project_path=project_path if scope in ("project", "local") else "",
                dependencies=deps,
            )
            self.registry.upsert(plugin_id, install)

            parts = [f"Plugin '{resolved_name}' v{resolved_version} installed to {cache_path}."]
            if install_log:
                parts.append(f"Install output: {install_log[:300]}")

            if deps:
                required, missing = self._resolve_dependencies(plugin_id)
                if missing:
                    return {
                        "status": "installed_with_missing_deps",
                        "message": "\n".join(parts) + f"\nMissing dependencies: {', '.join(missing)}.",
                        "plugin_id": plugin_id,
                        "name": resolved_name,
                        "version": resolved_version,
                        "install_path": str(cache_path),
                        "sha": sha,
                        "dependencies": deps,
                        "missing": missing,
                    }

            return {
                "status": "installed",
                "message": "\n".join(parts),
                "plugin_id": plugin_id,
                "name": resolved_name,
                "version": resolved_version,
                "install_path": str(cache_path),
                "sha": sha,
            }

        cache_path = (
            self.cache_dir
            / _sanitize(mkt_name)
            / _sanitize(resolved_name)
            / _sanitize(resolved_version)
        )

        if cache_path.exists():
            shutil.rmtree(cache_path)
        _copy_dir(clone_dir, cache_path)

        install_script = cache_path / "install.sh"
        install_log = ""
        if install_script.exists():
            irc, iout, ierr = _run(["bash", str(install_script)], cwd=str(cache_path))
            install_log = iout or ierr
            if irc != 0:
                shutil.rmtree(cache_path, ignore_errors=True)
                return {"status": "error", "message": f"install.sh failed: {ierr}"}
        _refresh_plugin_skill_index(cache_path)

        # Extract and record dependencies from manifest
        manifest = self._load_plugin_manifest(cache_path)
        deps = manifest.get("dependencies", [])

        # Record in registry
        install = PluginInstall(
            scope=scope,
            install_path=str(cache_path),
            version=resolved_version,
            installed_at=_now_iso(),
            last_updated=_now_iso(),
            git_commit_sha=sha,
            project_path=project_path if scope in ("project", "local") else "",
            dependencies=deps,
        )
        self.registry.upsert(plugin_id, install)

        parts = [f"Plugin '{resolved_name}' v{resolved_version} installed to {cache_path}."]
        if install_log:
            parts.append(f"Install output: {install_log[:300]}")

        if deps:
            required, missing = self._resolve_dependencies(plugin_id)
            if missing:
                return {
                    "status": "installed_with_missing_deps",
                    "message": "\n".join(parts) + f"\nMissing dependencies: {', '.join(missing)}.",
                    "plugin_id": plugin_id,
                    "name": resolved_name,
                    "version": resolved_version,
                    "install_path": str(cache_path),
                    "sha": sha,
                    "dependencies": deps,
                    "missing": missing,
                }

        return {
            "status": "installed",
            "message": "\n".join(parts),
            "plugin_id": plugin_id,
            "name": resolved_name,
            "version": resolved_version,
            "install_path": str(cache_path),
            "sha": sha,
        }

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list_plugins(self) -> dict[str, Any]:
        installs = self.registry.all_installs()
        if not installs:
            return {"status": "ok", "message": "No plugins installed.", "plugins": []}

        rows = []
        for plugin_id, inst in installs:
            name, mkt = plugin_id.split("@", 1) if "@" in plugin_id else (plugin_id, "")
            rows.append(
                {
                    "plugin_id": plugin_id,
                    "name": name,
                    "marketplace": mkt,
                    "version": inst.version,
                    "scope": inst.scope,
                    "enabled": inst.enabled,
                    "install_path": inst.install_path,
                    "sha": inst.git_commit_sha[:12] if inst.git_commit_sha else "",
                    "installed_at": inst.installed_at,
                    "last_updated": inst.last_updated,
                    "on_disk": Path(inst.install_path).is_dir(),
                }
            )
        return {"status": "ok", "plugins": rows}

    def all_plugin_ids(self) -> list[str]:
        return self.registry.all_plugin_ids()

    def set_enabled(self, name: str, enabled: bool, scope: Scope = "user") -> dict[str, Any]:
        plugin_id = self._resolve_plugin_id(name)
        if not plugin_id:
            return {"status": "error", "message": f"Plugin '{name}' not found in registry."}

        inst = self.registry.get(plugin_id, scope)
        if not inst:
            return {
                "status": "error",
                "message": f"Plugin '{name}' not installed at scope '{scope}'.",
            }

        if inst.enabled == enabled:
            return {
                "status": "already_enabled" if enabled else "already_disabled",
                "message": f"Plugin '{name}' is already {'enabled' if enabled else 'disabled'}.",
                "plugin_id": plugin_id,
                "enabled": inst.enabled,
            }

        updated = PluginInstall(
            scope=inst.scope,
            install_path=inst.install_path,
            version=inst.version,
            installed_at=inst.installed_at,
            last_updated=_now_iso(),
            git_commit_sha=inst.git_commit_sha,
            enabled=enabled,
            project_path=inst.project_path,
        )
        self.registry.upsert(plugin_id, updated)
        return {
            "status": "enabled" if enabled else "disabled",
            "message": f"Plugin '{name}' has been {'enabled' if enabled else 'disabled'}.",
            "plugin_id": plugin_id,
            "enabled": enabled,
        }

    def enable(self, name: str, scope: Scope = "user") -> dict[str, Any]:
        return self.set_enabled(name, True, scope)

    def disable(self, name: str, scope: Scope = "user") -> dict[str, Any]:
        return self.set_enabled(name, False, scope)

    # ------------------------------------------------------------------
    # update — pull latest, copy to new versioned cache dir
    # ------------------------------------------------------------------

    def update(self, name: str, scope: Scope = "user") -> dict[str, Any]:
        plugin_id = self._resolve_plugin_id(name)
        if not plugin_id:
            return {"status": "error", "message": f"Plugin '{name}' not found in registry."}

        inst = self.registry.get(plugin_id, scope)
        if not inst:
            return {
                "status": "error",
                "message": f"Plugin '{name}' not installed at scope '{scope}'.",
            }

        # Re-derive owner + name from cache path structure or plugin_id
        pname, mkt = plugin_id.split("@", 1) if "@" in plugin_id else (plugin_id, "")

        # marketplace name == owner in most cases
        owner = mkt

        with tempfile.TemporaryDirectory(prefix="logician_update_") as tmp_str:
            tmp = Path(tmp_str)
            clone_dir = tmp / pname
            url = f"{_GITHUB_BASE}/{owner}/{pname}.git"
            rc, _, err = _run(["git", "clone", "--depth=1", url, str(clone_dir)], timeout=180)
            if rc != 0:
                return {"status": "error", "message": f"git clone failed: {err}"}

            sha = _git_head_sha(clone_dir)
            if sha and sha == inst.git_commit_sha:
                return {
                    "status": "up_to_date",
                    "message": f"Plugin '{pname}' is already at latest commit ({sha[:12]}).",
                }

            plugin_json_path = _find_plugin_json(clone_dir)
            plugin_json = _read_json(plugin_json_path) if plugin_json_path else {}
            new_version = plugin_json.get("version") or (sha[:12] if sha else "unknown")

            cache_path = self.cache_dir / _sanitize(mkt) / _sanitize(pname) / _sanitize(new_version)
            if cache_path.exists():
                shutil.rmtree(cache_path)
            _copy_dir(clone_dir, cache_path)
            _refresh_plugin_skill_index(cache_path)

        # Extract and record dependencies from updated manifest
        manifest = self._load_plugin_manifest(cache_path)
        deps = manifest.get("dependencies", [])

        updated = PluginInstall(
            scope=scope,
            install_path=str(cache_path),
            version=new_version,
            installed_at=inst.installed_at,
            last_updated=_now_iso(),
            git_commit_sha=sha,
            enabled=inst.enabled,
            project_path=inst.project_path,
            dependencies=deps,
        )
        self.registry.upsert(plugin_id, updated)

        # Check for new broken deps after update
        if deps:
            _, missing = self._resolve_dependencies(plugin_id)
            if missing:
                return {
                    "status": "updated_with_broken_deps",
                    "message": f"Plugin '{pname}' updated to v{new_version} ({sha[:12]}), but dependencies broken: {', '.join(missing)}.",
                    "version": new_version,
                    "install_path": str(cache_path),
                    "sha": sha,
                    "dependencies": deps,
                    "missing": missing,
                }

        return {
            "status": "updated",
            "message": f"Plugin '{pname}' updated to v{new_version} ({sha[:12]}).",
            "version": new_version,
            "install_path": str(cache_path),
            "sha": sha,
        }

    # ------------------------------------------------------------------
    # remove — unregister and optionally delete cached files
    # ------------------------------------------------------------------

    def remove(
        self, name: str, scope: Scope = "user", *, keep_cache: bool = False
    ) -> dict[str, Any]:
        plugin_id = self._resolve_plugin_id(name)
        if not plugin_id:
            return {"status": "error", "message": f"Plugin '{name}' not found in registry."}

        inst = self.registry.get(plugin_id, scope)
        if not inst:
            return {
                "status": "error",
                "message": f"Plugin '{name}' not installed at scope '{scope}'.",
            }

        removed = []
        if not keep_cache and inst.install_path:
            cache = Path(inst.install_path)
            if cache.is_dir():
                shutil.rmtree(cache, ignore_errors=True)
                removed.append(str(cache))

        self.registry.remove(plugin_id, scope)
        return {
            "status": "removed",
            "message": (
                f"Plugin '{name}' removed from registry."
                + (f" Deleted: {', '.join(removed)}" if removed else "")
            ),
        }

    # ------------------------------------------------------------------
    # info — inspect one plugin's cached content
    # ------------------------------------------------------------------

    def info(self, name: str, scope: Scope = "user") -> dict[str, Any]:
        plugin_id = self._resolve_plugin_id(name)
        if not plugin_id:
            return {"status": "error", "message": f"Plugin '{name}' not found."}

        inst = self.registry.get(plugin_id, scope)
        if not inst:
            return {
                "status": "error",
                "message": f"Plugin '{name}' not installed at scope '{scope}'.",
            }

        cache = Path(inst.install_path)
        plugin_json_path = _find_plugin_json(cache)
        plugin_json = _read_json(plugin_json_path) if plugin_json_path else {}

        skills: list[str] = []
        skills_dir = cache / "skills"
        if skills_dir.is_dir():
            skills = [d.name for d in sorted(skills_dir.iterdir()) if d.is_dir()]

        commands: list[str] = []
        commands_dir = cache / "commands"
        if commands_dir.is_dir():
            commands = [f.name for f in sorted(commands_dir.rglob("*.md"))]

        return {
            "status": "ok",
            "plugin_id": plugin_id,
            "version": inst.version,
            "sha": inst.git_commit_sha,
            "enabled": inst.enabled,
            "install_path": inst.install_path,
            "on_disk": cache.is_dir(),
            "manifest": plugin_json,
            "skills": skills,
            "commands": commands,
        }

    # ------------------------------------------------------------------
    # skills_paths — return paths usable by ToolRegistry
    # ------------------------------------------------------------------

    def skills_source_roots(self, scope: Scope | None = None) -> list[SkillSourceRoot]:
        """Return typed skill/command roots for enabled plugin installs."""
        roots: list[SkillSourceRoot] = []
        seen: set[tuple[str, str]] = set()
        for plugin_id, inst in self.registry.all_installs():
            if scope and inst.scope != scope:
                continue
            if not inst.enabled:
                continue
            cache = Path(inst.install_path)
            if not cache.is_dir():
                continue
            for root in resolve_plugin_skill_roots(cache):
                try:
                    key = (str(root.path.resolve()), root.kind)
                except Exception:
                    key = (str(root.path), root.kind)
                if key in seen:
                    continue
                seen.add(key)
                roots.append(root)
        return roots

    def skills_paths(self, scope: Scope | None = None) -> list[Path]:
        """
        Return install paths for plugin-provided skills or command sources.

        This is the integration point with logician's ToolRegistry:
        pass these paths to wherever skill loading happens so the agent
        can use plugin-provided skills and markdown command docs without
        relying on symlinks.

        Includes both traditional `skills/` subdirectories and component-based
        `components.skills` paths from plugin.json manifests.
        """
        return [root.path for root in self.skills_source_roots(scope=scope)]

    def memory_paths(self, scope: Scope | None = None) -> list[Path]:
        """
        Return plugin memory directories or files for enabled plugin installs.

        This supports OpenClaude-style plugin memory summaries and observation
        index loading when a plugin exposes a memory index under memory/ or
        commands/memory/ in its cached install tree.
        """
        paths: list[Path] = []
        seen: set[str] = set()
        for plugin_id, inst in self.registry.all_installs():
            if scope and inst.scope != scope:
                continue
            if not inst.enabled:
                continue
            cache = Path(inst.install_path)
            if not cache.is_dir():
                continue
            candidates = [
                cache / "memory",
                cache / "commands" / "memory",
                cache / "commands",
                cache,
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                key = str(candidate.resolve())
                if key in seen:
                    continue
                seen.add(key)
                paths.append(candidate)
        return paths

    def get_plugin_skill_dirs(self) -> list[tuple[Path, str]]:
        """Get SKILL.md directories from plugin components.

        Reads the `components.skills` field from each enabled plugin's
        plugin.json manifest and returns directories containing SKILL.md
        files. Also includes the traditional `skills/` subdirectory.

        Returns list of (directory_path, plugin_id) tuples.
        """
        results: list[tuple[Path, str]] = []
        for plugin_id, inst in self.registry.all_installs():
            if not inst.enabled:
                continue
            cache = Path(inst.install_path)
            if not cache.is_dir():
                continue
            for root in resolve_plugin_skill_roots(cache):
                if root.kind != "skills":
                    continue
                if (root.path / "SKILL.md").is_file():
                    results.append((root.path, plugin_id))
                    continue
                for md_file in sorted(root.path.rglob("SKILL.md")):
                    results.append((md_file.parent, plugin_id))

        return results

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _load_plugin_manifest(self, cache_path: Path) -> dict[str, Any]:
        """Read plugin.json and extract dependencies."""
        path = _find_plugin_json(cache_path)
        if not path:
            return {}
        return _read_json(path)

    def _resolve_dependencies(
        self,
        plugin_id: str,
        visited: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Resolve dependencies for a plugin.

        Returns ``(required_deps, missing_deps)``.
        Raises ``ValueError`` on circular dependencies.
        """
        if visited is None:
            visited = set()
        if plugin_id in visited:
            raise ValueError(f"Circular dependency detected involving '{plugin_id}'")
        visited.add(plugin_id)

        # Normalize: resolve bare names to plugin_id
        resolved_id = self._resolve_plugin_id(plugin_id) or plugin_id
        inst = self.registry.get(resolved_id)
        if not inst:
            # Check if it's a full plugin_id directly
            inst = self.registry.get(plugin_id)

        if not inst or not inst.install_path:
            return [], [plugin_id]

        manifest = self._load_plugin_manifest(Path(inst.install_path))
        deps = manifest.get("dependencies", [])
        missing: list[str] = []
        required: list[str] = []

        for dep_id in deps:
            # Normalize dep_id: may be bare name or name@marketplace
            resolved_dep = self._resolve_plugin_id(dep_id) or dep_id
            dep_inst = self.registry.get(resolved_dep)
            if not dep_inst or not dep_inst.install_path or not Path(dep_inst.install_path).is_dir():
                missing.append(dep_id)
            else:
                required.append(dep_id)
                # Transitive resolution
                sub_req, sub_missing = self._resolve_dependencies(resolved_dep, visited)
                for r in sub_req:
                    if r not in required:
                        required.append(r)
                for m in sub_missing:
                    if m not in missing:
                        missing.append(m)

        return required, missing

    def validate_dependencies(
        self, scope: Scope | None = None
    ) -> dict[str, Any]:
        """Check all installed plugins for dependency issues."""
        issues: list[dict] = []
        checked = 0

        for plugin_id, inst in self.registry.all_installs():
            if scope and inst.scope != scope:
                continue
            if not inst.enabled:
                continue
            checked += 1
            try:
                _, missing = self._resolve_dependencies(plugin_id)
                if missing:
                    issues.append({
                        "plugin_id": plugin_id,
                        "status": "missing_dependencies",
                        "missing": missing,
                    })
            except ValueError as e:
                issues.append({
                    "plugin_id": plugin_id,
                    "status": "circular_dependency",
                    "error": str(e),
                })

        return {
            "status": "ok" if not issues else "issues_found",
            "issues": issues,
            "total_checked": checked,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_plugin_id(self, name: str) -> str | None:
        """Find plugin_id (name@marketplace) for a bare plugin name."""
        # Already a full plugin_id
        if "@" in name:
            if self.registry.get(name):
                return name

        # Search all registered plugins for a name match
        for pid in self.registry.all_plugin_ids():
            pname = pid.split("@")[0] if "@" in pid else pid
            if pname.lower() == name.lower():
                return pid
        return None

    def _find_owner_for(self, name: str) -> str:
        """
        Search known marketplaces for a plugin named `name`.
        Returns the owner/marketplace name if found, else empty string.
        """
        for mkt_name, mkt_info in self.registry.known_marketplaces().items():
            mkt_dir = Path(mkt_info.get("installLocation", ""))
            if not mkt_dir.is_dir():
                continue
            mkt_json_path = _find_marketplace_json(mkt_dir)
            if not mkt_json_path:
                continue
            mkt_data = _read_json(mkt_json_path)
            for plugin_entry in mkt_data.get("plugins", []):
                if plugin_entry.get("name", "").lower() == name.lower():
                    return mkt_name
        return ""
