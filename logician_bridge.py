from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import shlex
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

BRIDGE_DIR = Path(__file__).resolve().parent


def _resolve_root_dir() -> Path:
    # In repo root / packaged layout, source is directly under BRIDGE_DIR/src/agent.
    if (BRIDGE_DIR / "src" / "agent").exists():
        return BRIDGE_DIR

    # In dev layout (bridge under cli/), source lives one level up.
    parent = BRIDGE_DIR.parent
    if (parent / "src" / "agent").exists():
        return parent

    # Best-effort fallback: keep prior behavior of using bridge directory.
    return BRIDGE_DIR


ROOT_DIR = _resolve_root_dir()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.runtime_paths import state_path


DB_PATH = state_path("agent_sessions.db")
VECTOR_PATH = state_path("message_history.vector")
RAG_VECTOR_PATH = state_path("rag_docs.vector")


@lru_cache(maxsize=1)
def _agent_factory():
    from src.agent import create_agent

    return create_agent


@lru_cache(maxsize=1)
def _has_rapidfuzz() -> bool:
    from src.tools.registry.catalog import HAS_RAPIDFUZZ

    return bool(HAS_RAPIDFUZZ)


@lru_cache(maxsize=1)
def _has_tiktoken() -> bool:
    from src.backends.common import HAS_TIKTOKEN

    return bool(HAS_TIKTOKEN)


def _new_session_id() -> str:
    return f"cli_{uuid.uuid4().hex[:8]}"


_BRIDGE_COMMAND_SPECS: list[tuple[str, str]] = [
    ("/help", "Show command list"),
    ("/version", "Show version/runtime info"),
    ("/status", "Runtime state snapshot"),
    ("/skills-health", "Skill loader diagnostics"),
    ("/agents", "List loaded agents"),
    ("/agent", "Switch active agent (`/agent <name>`)"),
    ("/pipeline", "Set/stop inter-agent pipeline (`/pipeline <a> <b> [rounds]`)"),
    ("/context", "Show session/data context"),
    ("/compact [keep_last]", "Summarize old history"),
    ("/reset", "Reset runtime tool state"),
    ("/sessions", "List stored sessions"),
    ("/load", "Load prior session (`/load <id_prefix>`)"),
    ("/export", "Export transcript (`/export [path]`)"),
    ("/mount", "Mount codebase (`/mount <dir> [glob] [max] [depth]`)"),
    ("/mount-code", "Alias for /mount"),
    ("/upload", "Ingest one doc (`/upload <file> [label]`)"),
    ("/upload-dir", "Bulk ingest docs (`/upload-dir <dir> [glob] [max]`)"),
    ("/docs", "Fetch Context7 docs (`/docs <library> [query]`)"),
    ("/changes", "Show git status + diff preview (`/changes [path] [--staged]`)"),
    ("/new", "Start new session"),
    ("/reload", "Reload config + agents"),
    ("/quit", "Exit"),
    ("/exit", "Alias for /quit"),
    ("/q", "Alias for /quit"),
]


def _bridge_commands_manifest() -> list[dict[str, str]]:
    return [
        {"command": command, "description": description}
        for command, description in _BRIDGE_COMMAND_SPECS
    ]


def _extract_context7_library_id(raw: Any) -> str | None:
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

    # Plain-text response format from Context7.
    match = re.search(r"Context7-compatible library ID:\s*(\S+)", text)
    if match:
        return match.group(1).strip()

    # JSON response variants.
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("results", "libraries", "items"):
            items = payload.get(key)
            if not isinstance(items, list) or not items:
                continue
            first = items[0]
            if not isinstance(first, dict):
                continue
            for id_key in ("id", "libraryId", "library_id"):
                val = first.get(id_key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # Sometimes the tool returns the ID directly.
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first_line.startswith("/"):
        return first_line
    return None


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".svg")
_IMAGE_KEY_HINTS = ("image", "plot", "chart", "figure", "output", "path", "file")
_IMAGE_PATH_RE = re.compile(
    r"(?i)(?:https?://|file://|[a-z]:)?[a-z0-9_./\\-]+\.(?:png|jpe?g|webp|bmp|gif|svg)"
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_HTML_IMAGE_SRC_RE = re.compile(
    r"""<img[^>]+src=["']([^"']+\.(?:png|jpe?g|webp|bmp|gif|svg))["']""",
    re.IGNORECASE,
)


def _normalize_image_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    path = value.strip().strip("\"'<>")
    if not path:
        return None
    if path.startswith("file://"):
        path = path[7:]
    # Keep local path stable for terminal renderer cache keys.
    path = path.split("?", 1)[0].split("#", 1)[0].strip()
    if not path:
        return None
    if any(ch.isspace() for ch in path):
        return None

    lower = path.lower()
    if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
        return path
    return None


def _extract_image_paths(payload: Any, *, limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    stack: list[Any] = [payload]
    scanned = 0

    def _add(raw: Any) -> None:
        if len(out) >= limit:
            return
        path = _normalize_image_path(raw)
        if not path or path in seen:
            return
        seen.add(path)
        out.append(path)

    while stack and len(out) < limit and scanned < 500:
        scanned += 1
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                key_l = str(key or "").lower()
                if isinstance(value, str) and any(h in key_l for h in _IMAGE_KEY_HINTS):
                    _add(value)
                stack.append(value)
            continue
        if isinstance(cur, (list, tuple, set)):
            stack.extend(list(cur))
            continue
        if not isinstance(cur, str):
            continue

        _add(cur)
        text = cur.strip()
        if not text:
            continue
        for match in _MARKDOWN_IMAGE_RE.finditer(text):
            _add(match.group(1))
        for match in _HTML_IMAGE_SRC_RE.finditer(text):
            _add(match.group(1))
        for match in _IMAGE_PATH_RE.finditer(text):
            _add(match.group(0))

    return out


def _resolve_config_path(raw_path: str | None) -> Path:
    """Resolve config path using practical lookup order for CLI bridge calls."""
    if raw_path:
        candidate = Path(str(raw_path)).expanduser()
    else:
        candidate = Path("agent_config.json")

    if candidate.is_absolute():
        return candidate

    # Prefer the caller's current working directory first.
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Then try relative to the bridge folder itself (agent/cli).
    bridge_candidate = (BRIDGE_DIR / candidate).resolve()
    if bridge_candidate.exists():
        return bridge_candidate

    # Finally fall back to agent root where config commonly lives.
    root_candidate = (ROOT_DIR / candidate).resolve()
    if root_candidate.exists():
        return root_candidate

    # Return best-effort bridge-relative path for clear error reporting.
    return bridge_candidate


@lru_cache(maxsize=1)
def _detected_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }

    pyproject = ROOT_DIR / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
            match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', text)
            if match:
                versions["agent"] = match.group(1).strip()
        except Exception:
            pass

    cargo = ROOT_DIR / "rust-cli" / "Cargo.toml"
    if cargo.is_file():
        try:
            text = cargo.read_text(encoding="utf-8")
            match = re.search(
                r'(?sm)^\[package\].*?^\s*version\s*=\s*"([^"]+)"\s*$',
                text,
            )
            if match:
                versions["cli"] = match.group(1).strip()
        except Exception:
            pass

    return versions


@contextlib.contextmanager
def _silent_load():
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    old_fd1 = os.dup(1)
    old_fd2 = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(old_fd1, 1)
        os.close(old_fd1)
        os.dup2(old_fd2, 2)
        os.close(old_fd2)


class BridgeServer:
    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {}
        self.agents: dict[str, Any] = {}
        self.active: str | None = None
        self.sessions: dict[str, str] = {}
        self.pipeline: dict[str, Any] | None = None

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        msg = {"event": event, **payload}
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _response(
        self, rid: Any, ok: bool, result: Any = None, error: str = ""
    ) -> None:
        msg = {"id": rid, "ok": ok}
        if ok:
            msg["result"] = result
        else:
            msg["error"] = error
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _load_agents(self, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        def build_overrides(block: dict[str, Any]) -> dict[str, Any]:
            from src.config import Config as AgentConfig

            merged = {**cfg, **block}
            allowed = set(getattr(AgentConfig, "__dataclass_fields__", {}).keys())
            ov = {
                key: value
                for key, value in merged.items()
                if key in allowed and value is not None
            }
            # Keep vector store local to this bridge workspace.
            ov["vector_path"] = str(VECTOR_PATH)
            ov["rag_vector_path"] = str(RAG_VECTOR_PATH)
            # Backward-compat alias used by agent_config.json.
            if "mcp_servers" not in ov and (mcp := merged.get("mcp")):
                ov["mcp_servers"] = mcp
            return ov

        def make_agent(block: dict[str, Any]):
            create_agent = _agent_factory()
            merged = {**cfg, **block}
            # Apply config-provided environment variables before agent creation.
            # Supports top-level env plus per-agent env overrides in multi-agent mode.
            env_map: dict[str, Any] = {}
            if isinstance(cfg.get("env"), dict):
                env_map.update(dict(cfg.get("env") or {}))
            if isinstance(block.get("env"), dict):
                env_map.update(dict(block.get("env") or {}))
            for key, value in env_map.items():
                k = str(key or "").strip()
                if not k or value is None:
                    continue
                os.environ[k] = str(value)
            for ek, env_key in [
                ("firecrawl_url", "FIRECRAWL_URL"),
                ("firecrawl_api_key", "FIRECRAWL_API_KEY"),
            ]:
                if value := merged.get(ek):
                    os.environ[env_key] = value
            return create_agent(
                llm_url=merged.get("endpoint", "http://localhost:8080"),
                system_prompt=merged.get("system_prompt"),
                use_chat_api=merged.get("use_chat_api", True),
                chat_template=merged.get("chat_template", "chatml"),
                db_path=str(DB_PATH),
                embedding_model=merged.get("embedding_model"),
                config_overrides=build_overrides(merged),
            )

        raw_map = cfg.get("agents", {})
        agents: dict[str, Any] = {}
        with _silent_load():
            if raw_map:
                for name, block in raw_map.items():
                    agents[name] = make_agent(block)
            else:
                agents[cfg.get("agent_name", "main")] = make_agent(cfg)
        return cfg, agents

    def _active_agent(self):
        if self.active is None:
            raise RuntimeError("No active agent")
        return self.agents[self.active]

    def _active_mcp_names(self) -> list[str]:
        active = self.active or ""
        if not active or active not in self.agents:
            return []
        try:
            clients = getattr(self.agents[active], "_mcp_clients", []) or []
            names = [
                str(getattr(client, "name", "")).strip()
                for client in clients
                if str(getattr(client, "name", "")).strip()
            ]
            return sorted(dict.fromkeys(names))
        except Exception:
            return []

    def _state_snapshot(self) -> dict[str, Any]:
        active = self.active or ""
        sid = self.sessions.get(active, "")
        msgs = 0
        if active and sid and active in self.agents:
            try:
                msgs = (
                    self.agents[active]
                    .describe_runtime_context(sid)
                    .get("persisted_messages", 0)
                )
            except Exception:
                msgs = 0
        tool_count = 0
        skill_count = 0
        if active and active in self.agents:
            try:
                ag = self.agents[active]
                tool_registry = getattr(ag, "tools", None)
                if tool_registry is not None:
                    # Keep counts stable by fully syncing the catalog before
                    # reading counters. Without this, lazy-hydrated skills can
                    # make the displayed number jump across turns.
                    sync_fn = getattr(
                        tool_registry, "_sync_catalog_with_registered_tools", None
                    )
                    if callable(sync_fn):
                        sync_fn()
                    # ToolRegistry stores tools in ._tools dict
                    tool_dict = getattr(tool_registry, "_tools", None) or {}
                    tool_count = len(tool_dict)
                    # Skills are in ._catalog._skills dict
                    catalog = getattr(tool_registry, "_catalog", None)
                    if catalog is not None:
                        # Fallback path when sync helper is unavailable.
                        ensure_fn = getattr(catalog, "ensure_skill_catalog", None)
                        if callable(ensure_fn) and not callable(sync_fn):
                            ensure_fn()
                        skill_dict = getattr(catalog, "_skills", None) or {}
                        skill_count = len(skill_dict)
            except Exception:
                pass
        return {
            "active": active,
            "session": sid,
            "msg_count": msgs,
            "agents": sorted(self.agents.keys()),
            "mcp_servers": self._active_mcp_names(),
            "pipeline": self.pipeline,
            "rapidfuzz": _has_rapidfuzz(),
            "tiktoken": _has_tiktoken(),
            "tool_count": tool_count,
            "skill_count": skill_count,
        }

    def init(self, params: dict[str, Any]) -> dict[str, Any]:
        cp = params.get("config_path")
        config_path = _resolve_config_path(cp)
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")

        self.cfg, self.agents = self._load_agents(config_path)
        first = next(iter(self.agents.keys()))
        self.active = first
        self.sessions = {first: _new_session_id()}
        self.pipeline = None

        return {
            "config_path": str(config_path),
            "state": self._state_snapshot(),
            "commands": _bridge_commands_manifest(),
        }

    def slash(self, params: dict[str, Any]) -> dict[str, Any]:
        raw = str(params.get("raw", "")).strip()
        if not raw:
            return {"messages": [], "state": self._state_snapshot(), "exit": False}

        parts = shlex.split(raw)
        cmd = parts[0].lower()
        args = parts[1:]
        out: list[str] = []
        should_exit = False

        def _parse_tool_payload(payload: Any) -> dict[str, Any]:
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                return {"status": "error", "error": payload}
            return {"status": "error", "error": str(payload)}

        def _parse_json_dict(payload: Any) -> dict[str, Any] | None:
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
            return None

        def _context7_tools(agent: Any) -> list[Any]:
            tool_registry = getattr(agent, "tools", None)
            if tool_registry is None:
                return []
            try:
                tools = list(tool_registry.list_tools())
            except Exception:
                return []
            out: list[Any] = []
            for tool in tools:
                name = str(getattr(tool, "name", "") or "").strip()
                if not name:
                    continue
                lname = name.lower()
                skill_id = str(getattr(tool, "skill_id", "") or "").lower()
                if (
                    skill_id == "mcp__context7"
                    or lname.startswith("context7__")
                    or "context7" in lname
                    or lname in {"resolve_library_id", "get_library_docs", "query_docs"}
                ):
                    out.append(tool)
            return out

        if cmd in ("/quit", "/exit", "/q"):
            should_exit = True
            out.append("bye")

        elif cmd == "/help":
            lines = ["# Command Palette", ""]
            for item in _bridge_commands_manifest():
                command = str(item.get("command", "")).strip()
                description = str(item.get("description", "")).strip()
                if not command:
                    continue
                lines.append(f"- `{command}`  {description}")
            out.append("\n".join(lines))

        elif cmd == "/version":
            versions = _detected_versions()
            out.append(
                "\n".join(
                    [
                        "# Version",
                        "",
                        f"- agent: `{versions.get('agent', 'unknown')}`",
                        f"- cli: `{versions.get('cli', 'unknown')}`",
                        f"- bridge: `{Path(__file__).name}`",
                        f"- python: `{versions.get('python', 'unknown')}`",
                    ]
                )
            )

        elif cmd == "/status":
            snap = self._state_snapshot()
            agents = ", ".join(snap.get("agents", []) or []) or "-"
            mcps = ", ".join(snap.get("mcp_servers", []) or []) or "-"
            pipeline = snap.get("pipeline")
            if pipeline:
                ptxt = (
                    f"{pipeline.get('a', '?')} -> {pipeline.get('b', '?')} "
                    f"x{pipeline.get('rounds', '?')}"
                )
            else:
                ptxt = "off"
            out.extend(
                [
                    f"active: {snap.get('active', '-')}",
                    f"session: {snap.get('session', '-')}",
                    f"messages: {snap.get('msg_count', 0)}",
                    f"agents: {agents}",
                    f"mcp: {mcps}",
                    f"pipeline: {ptxt}",
                    f"rapidfuzz: {'enabled' if snap.get('rapidfuzz') else 'disabled'}",
                    f"tiktoken: {'enabled' if snap.get('tiktoken') else 'disabled'}",
                ]
            )

        elif cmd == "/skills-health":
            ag = self._active_agent()
            include_sources = False
            max_items = 25
            for arg in args:
                if arg in {"--sources", "-s"}:
                    include_sources = True
                    continue
                try:
                    max_items = max(1, min(200, int(arg)))
                except Exception:
                    out.append(f"ignored invalid argument: {arg}")

            raw = ag.tools.call_tool(
                "skills_health",
                include_sources=include_sources,
                max_items=max_items,
            )
            payload = _parse_tool_payload(raw)
            if str(payload.get("status", "")).lower() != "ok":
                out.append(
                    f"skills_health failed: {payload.get('error', 'unknown error')}"
                )
            else:
                paths = payload.get("paths", {}) or {}
                discovery = payload.get("discovery", {}) or {}
                catalog = payload.get("catalog", {}) or {}
                coding = payload.get("coding", {}) or {}
                organization = payload.get("organization", {}) or {}
                registry = payload.get("registry", {}) or {}
                checks = payload.get("checks", {}) or {}

                out.extend(
                    [
                        f"skills_dir: {paths.get('skills_dir_path', '-')}",
                        f"skills_md: {paths.get('skills_md_path', '-')}",
                        "discovery: "
                        f"sources={discovery.get('source_count', 0)} "
                        f"readable={discovery.get('readable_count', 0)} "
                        f"unreadable={discovery.get('unreadable_count', 0)} "
                        f"superpowers={discovery.get('superpowers_skill_md_count', 0)}",
                        "catalog: "
                        f"total={catalog.get('skills_total', 0)} "
                        f"guidance={catalog.get('guidance_only_skills', 0)} "
                        f"tool_backed={catalog.get('tool_backed_skills', 0)} "
                        f"superpowers={catalog.get('superpowers_skills', 0)}",
                        "coding: "
                        f"maturity={coding.get('maturity', 'unknown')} "
                        f"coverage={coding.get('required_coverage_pct', 0.0)}% "
                        f"missing_required={coding.get('missing_required_count', 0)} "
                        f"coding_tools={coding.get('coding_tools_total', 0)}",
                        "organization: "
                        f"status={organization.get('status', 'unknown')} "
                        f"modules={organization.get('coding_modules_count', 0)} "
                        f"issues={organization.get('issues_count', 0)}",
                        "registry: "
                        f"tools={registry.get('tools_total', 0)} "
                        f"python={registry.get('python_skill_tools', 0)} "
                        f"builtin={registry.get('builtin_tools', 0)}",
                        "checks: "
                        f"brainstorming_present={checks.get('brainstorming_present', False)} "
                        f"missing_tool_links={checks.get('missing_tools_by_skill_count', 0)}",
                    ]
                )

                missing = checks.get("missing_tools_by_skill", []) or []
                if missing:
                    out.append("missing tool links:")
                    for item in missing[:10]:
                        skill_id = str(item.get("skill_id", "?"))
                        miss = item.get("missing_tools", []) or []
                        preview = ", ".join(str(v) for v in miss[:6])
                        if len(miss) > 6:
                            preview += f" … (+{len(miss) - 6})"
                        out.append(f"  {skill_id}: {preview}")

                missing_required = coding.get("missing_required_tools", []) or []
                if missing_required:
                    preview = ", ".join(str(v) for v in missing_required[:10])
                    if len(missing_required) > 10:
                        preview += f" … (+{len(missing_required) - 10})"
                    out.append(f"missing required capability tools: {preview}")

                org_issues = organization.get("issues", []) or []
                if org_issues:
                    out.append("organization issues:")
                    for issue in org_issues[:10]:
                        out.append(f"  - {issue}")

                if include_sources:
                    sources = discovery.get("sources", []) or []
                    if sources:
                        out.append("sample sources:")
                        for src in sources[:10]:
                            out.append(f"  {src}")

        elif cmd == "/agents":
            if self.active is None:
                out.append("No active agent")
            else:
                out.append("agents:")
                for name in self.agents:
                    mark = " ◀" if name == self.active else ""
                    mcp = [
                        c.name for c in getattr(self.agents[name], "_mcp_clients", [])
                    ]
                    mcp_tag = f"  mcp: {', '.join(mcp)}" if mcp else ""
                    out.append(f"  {name}{mark}{mcp_tag}")

        elif cmd == "/agent":
            if self.active is None:
                out.append("No active agent")
            elif not args:
                out.append(f"active: {self.active}")
            elif args[0] not in self.agents:
                out.append(f"unknown '{args[0]}'")
            else:
                self.active = args[0]
                self.sessions.setdefault(self.active, _new_session_id())
                out.append(
                    f"switched to '{self.active}' · {self.sessions[self.active]}"
                )

        elif cmd == "/pipeline":
            if args and args[0].lower() == "stop":
                self.pipeline = None
                out.append("pipeline cancelled")
            elif len(args) < 2:
                out.append("usage: /pipeline <a> <b> [rounds]")
            elif args[0] not in self.agents or args[1] not in self.agents:
                out.append("unknown agent")
            else:
                rounds = 3
                if len(args) >= 3:
                    rounds = max(1, int(args[2]))
                self.pipeline = {"a": args[0], "b": args[1], "rounds": rounds}
                out.append(
                    f"pipeline: {args[0]} -> {args[1]} x{rounds} (send message to run)"
                )

        elif cmd == "/context":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            info = ag.describe_runtime_context(sid)
            out.append(f"session: {sid}")
            out.append(
                f"messages: {info.get('persisted_messages', 0)} / limit {info.get('history_limit', 0)}"
            )
            rt = info.get("runtime", {})
            if rt.get("loaded"):
                cols = ", ".join(rt.get("value_columns", [])[:6]) or "none"
                out.append(
                    f"dataset: {rt.get('data_name', 'unnamed')} rows={rt.get('row_count', 0)} cols={cols}"
                )
            else:
                out.append("dataset: none")

        elif cmd == "/compact":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            keep = int(args[0]) if args else None
            result = ag.compact_session(sid, keep_last_messages=keep)
            out.append(result.get("message", "done"))

        elif cmd == "/reset":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            ag.reset_runtime_state(sid)
            out.append("runtime state cleared")

        elif cmd == "/new":
            if self.active is None:
                out.append("No active agent")
            else:
                self.sessions[self.active] = _new_session_id()
                ag = self._active_agent()
                try:
                    ag.detach_runtime_state()
                except Exception:
                    pass
                self.pipeline = None
                out.append(f"new session · {self.sessions[self.active]}")

        elif cmd == "/reload":
            cp = _resolve_config_path(params.get("config_path"))
            self.cfg, self.agents = self._load_agents(cp)
            self.active = next(iter(self.agents.keys()))
            self.sessions = {self.active: _new_session_id()}
            self.pipeline = None
            out.append(f"reloaded · {len(self.agents)} agent(s)")

        elif cmd == "/sessions":
            ag = self._active_agent()
            sess_list = ag.list_sessions()
            if not sess_list:
                out.append("no sessions found")
            else:
                out.append(f"sessions ({len(sess_list)}):")
                for sid, ts in sess_list:
                    mark = (
                        " ◀" if sid == self.sessions.get(self.active or "", "") else ""
                    )
                    short = sid[:24] + ("…" if len(sid) > 24 else "")
                    out.append(f"  {short}  {ts}{mark}")

        elif cmd == "/load":
            if not args:
                out.append("usage: /load <session_id>")
            else:
                ag = self._active_agent()
                sess_list = ag.list_sessions()
                matched = [
                    s for s, _ in sess_list if s == args[0] or s.startswith(args[0])
                ]
                if not matched:
                    out.append(f"no match for '{args[0]}'")
                elif len(matched) > 1:
                    out.append(f"ambiguous — {len(matched)} matches")
                else:
                    sid = matched[0]
                    assert self.active is not None
                    self.sessions[self.active] = sid
                    ag._activate_session_runtime(sid)
                    n = ag.memory.count_session_messages(sid)
                    out.append(f"loaded · {sid}")
                    out.append(f"messages: {n}")

        elif cmd == "/export":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = args[0] if args else f"export_{(sid or 'nosid')[:8]}_{ts}.md"
            msgs = ag.memory.get_session_messages(sid) if sid else []
            lines = [
                "# Session Export",
                "",
                f"**session:** `{sid}`",
                f"**exported:** {datetime.datetime.now().isoformat()}",
                f"**messages:** {len(msgs)}",
                "",
                "---",
                "",
            ]
            for m in msgs:
                role = m.role.value.upper()
                ntag = f" `{m.name}`" if m.name else ""
                lines.extend([f"### {role}{ntag}", "", m.content or "", "", "---", ""])
            Path(out_path).write_text("\n".join(lines), encoding="utf-8")
            out.append(f"exported {len(msgs)} messages -> {out_path}")

        elif cmd in {"/mount", "/mount-code"}:
            if not args:
                out.append("usage: /mount <directory> [glob] [max_files] [map_depth]")
            else:
                ag = self._active_agent()
                sid = self.sessions.get(self.active or "", "") or None
                sid_text = str(sid or self.sessions.get(self.active or "", "") or "-")
                directory = args[0]
                glob_pattern = (
                    args[1]
                    if len(args) > 1
                    else "**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}"
                )

                max_files = 120
                if len(args) > 2:
                    try:
                        max_files = max(1, min(400, int(args[2])))
                    except Exception:
                        out.append(f"invalid max_files '{args[2]}', using 120")

                map_depth = 3
                if len(args) > 3:
                    try:
                        map_depth = max(1, min(10, int(args[3])))
                    except Exception:
                        out.append(f"invalid map_depth '{args[3]}', using 3")

                project_map_payload: dict[str, Any]
                try:
                    project_map_raw = ag.run_tool_direct(
                        "get_project_map",
                        {"directory": directory, "max_depth": map_depth},
                        session_id=sid,
                        persist_to_history=True,
                    )
                    project_map_payload = _parse_tool_payload(project_map_raw)
                except Exception as exc:
                    project_map_payload = {"status": "error", "error": str(exc)}

                promote_payload: dict[str, Any]
                used_fallback = False
                try:
                    promote_raw = ag.run_tool_direct(
                        "rag_promote_paths",
                        {
                            "paths": json.dumps([directory], ensure_ascii=False),
                            "recursive": True,
                            "glob": glob_pattern,
                            "chunk_size": 400,
                            "overlap": 0.2,
                            "max_files": max_files,
                        },
                        session_id=sid,
                        persist_to_history=True,
                    )
                    promote_payload = _parse_tool_payload(promote_raw)
                except Exception:
                    used_fallback = True
                    try:
                        fallback_raw = ag.run_tool_direct(
                            "docling_add_dir",
                            {
                                "directory": directory,
                                "glob": glob_pattern,
                                "max_files": max_files,
                            },
                            session_id=sid,
                            persist_to_history=True,
                        )
                        fallback_payload = _parse_tool_payload(fallback_raw)
                        promote_payload = {
                            "status": fallback_payload.get("status", "error"),
                            "files_processed": fallback_payload.get("files_processed", 0),
                            "total_chunks_added": fallback_payload.get(
                                "total_chunks_added",
                                fallback_payload.get("chunks_added", 0),
                            ),
                            "error": fallback_payload.get("error"),
                        }
                    except Exception as exc:
                        promote_payload = {"status": "error", "error": str(exc)}

                out.append(f"mounted path: {directory}")
                out.append(f"session: {sid_text}")

                if str(project_map_payload.get("status", "")).lower() == "ok":
                    out.append(
                        "context persisted: yes "
                        f"(project map: {project_map_payload.get('file_count', 0)} files, depth={map_depth})"
                    )
                else:
                    out.append(
                        "context persisted: partial "
                        "(project map failed: "
                        f"{project_map_payload.get('error', 'unknown error')}"
                        ")"
                    )

                if str(promote_payload.get("status", "")).lower() == "ok":
                    files_processed = int(promote_payload.get("files_processed", 0) or 0)
                    chunks_added = int(promote_payload.get("total_chunks_added", 0) or 0)
                    fallback_note = " (fallback=docling_add_dir)" if used_fallback else ""
                    out.append(
                        "rag persisted: yes "
                        f"({files_processed} files, {chunks_added} chunks{fallback_note})"
                    )
                else:
                    out.append(
                        "rag persisted: no "
                        "(mount failed: "
                        f"{promote_payload.get('error', 'unknown error')}"
                        ")"
                    )

                try:
                    ctx_info = ag.describe_runtime_context(sid_text)
                    out.append(
                        "context window readiness: "
                        f"session_messages={ctx_info.get('persisted_messages', 0)} "
                        f"loaded_per_turn={ctx_info.get('loaded_message_budget', 0)}/"
                        f"{ctx_info.get('history_limit', 0)}"
                    )
                except Exception:
                    pass

                out.append(
                    "ready: ask with repo-relative paths; retrieval will use mounted RAG + live tools."
                )

        elif cmd == "/upload":
            if not args:
                out.append("usage: /upload <file_path> [source_label]")
            else:
                ag = self._active_agent()
                file_path = args[0]
                source_label = " ".join(args[1:]).strip() if len(args) > 1 else ""
                raw_result = ag.tools.call_tool(
                    "docling_add_file", path=file_path, source_label=source_label
                )
                parsed = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )
                if parsed.get("status") == "ok":
                    out.append(
                        f"uploaded {parsed.get('path', file_path)} ({parsed.get('chunks_added', 0)} chunks)"
                    )
                else:
                    out.append(f"upload failed: {parsed.get('error', 'unknown error')}")

        elif cmd == "/upload-dir":
            if not args:
                out.append("usage: /upload-dir <directory> [glob] [max_files]")
            else:
                ag = self._active_agent()
                directory = args[0]
                glob_pattern = args[1] if len(args) > 1 else "**/*"
                max_files = max(1, int(args[2])) if len(args) > 2 else 25
                raw_result = ag.tools.call_tool(
                    "docling_add_dir",
                    directory=directory,
                    glob=glob_pattern,
                    max_files=max_files,
                )
                parsed = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )
                if parsed.get("status") == "ok":
                    out.append(
                        "upload-dir done "
                        f"({parsed.get('files_processed', 0)} files, {parsed.get('total_chunks_added', 0)} chunks)"
                    )
                else:
                    out.append(
                        f"upload-dir failed: {parsed.get('error', 'unknown error')}"
                    )

        elif cmd == "/docs":
            if not args:
                out.append("usage: /docs <library> [query]")
            else:
                ag = self._active_agent()
                library = args[0].strip()
                query = " ".join(args[1:]).strip() or "quickstart usage examples"
                ctx7_tools = _context7_tools(ag)
                if not ctx7_tools:
                    out.append(
                        "context7 unavailable: no Context7 MCP tools are loaded for this agent"
                    )
                else:
                    resolve_tool = next(
                        (
                            tool
                            for tool in ctx7_tools
                            if "resolve" in tool.name.lower()
                            and "library" in tool.name.lower()
                        ),
                        None,
                    )
                    docs_tool = next(
                        (
                            tool
                            for tool in ctx7_tools
                            if (
                                "query_docs" in tool.name.lower()
                                or "get_library_docs" in tool.name.lower()
                                or (
                                    "doc" in tool.name.lower()
                                    and "resolve" not in tool.name.lower()
                                )
                            )
                        ),
                        None,
                    )
                    if resolve_tool is None:
                        out.append("context7 unavailable: resolve-library tool not found")
                    elif docs_tool is None:
                        out.append("context7 unavailable: docs query tool not found")
                    else:
                        resolve_params = [
                            str(getattr(p, "name", "") or "").strip()
                            for p in getattr(resolve_tool, "parameters", []) or []
                        ]
                        resolve_params = [p for p in resolve_params if p]
                        resolve_args: dict[str, Any] = {}
                        for pname in resolve_params:
                            lname = pname.lower()
                            if any(
                                token in lname
                                for token in ("query", "library", "name", "id")
                            ):
                                resolve_args[pname] = library
                        if not resolve_args:
                            if resolve_params:
                                resolve_args[resolve_params[0]] = library
                            else:
                                resolve_args["query"] = library

                        resolve_raw = ag.tools.call_tool(resolve_tool.name, **resolve_args)
                        resolve_payload = _parse_json_dict(resolve_raw)
                        if (
                            isinstance(resolve_payload, dict)
                            and str(resolve_payload.get("status", "")).lower() == "error"
                        ):
                            out.append(
                                "context7 resolve failed: "
                                f"{resolve_payload.get('error', 'unknown error')}"
                            )
                        else:
                            library_id = _extract_context7_library_id(resolve_raw)
                            if not library_id:
                                out.append(
                                    "context7 resolve failed: could not extract a library ID"
                                )
                                out.append(str(resolve_raw)[:280])
                            else:
                                docs_params = [
                                    str(getattr(p, "name", "") or "").strip()
                                    for p in getattr(docs_tool, "parameters", []) or []
                                ]
                                docs_params = [p for p in docs_params if p]
                                id_param: str | None = None
                                query_param: str | None = None
                                for pname in docs_params:
                                    lname = pname.lower()
                                    if id_param is None and (
                                        "context7" in lname
                                        or "libraryid" in lname
                                        or lname in {"library_id", "library", "id"}
                                    ):
                                        id_param = pname
                                    if query_param is None and (
                                        "query" in lname
                                        or "topic" in lname
                                        or "question" in lname
                                        or "search" in lname
                                    ):
                                        query_param = pname
                                if id_param is None and docs_params:
                                    id_param = docs_params[0]
                                if not id_param:
                                    out.append(
                                        "context7 docs failed: could not infer docs tool parameters"
                                    )
                                else:
                                    docs_args: dict[str, Any] = {id_param: library_id}
                                    if query:
                                        if query_param:
                                            docs_args[query_param] = query
                                        else:
                                            docs_args["query"] = query
                                    docs_raw = ag.tools.call_tool(docs_tool.name, **docs_args)
                                    docs_payload = _parse_json_dict(docs_raw)
                                    if (
                                        isinstance(docs_payload, dict)
                                        and str(docs_payload.get("status", "")).lower()
                                        == "error"
                                    ):
                                        out.append(
                                            "context7 docs failed: "
                                            f"{docs_payload.get('error', 'unknown error')}"
                                        )
                                    else:
                                        docs_text = (
                                            docs_raw
                                            if isinstance(docs_raw, str)
                                            else json.dumps(
                                                docs_raw,
                                                ensure_ascii=False,
                                                indent=2,
                                            )
                                        )
                                        if len(docs_text) > 12000:
                                            docs_text = (
                                                docs_text[:12000].rstrip()
                                                + "\n...[truncated]"
                                            )
                                        out.append(f"context7 library: {library_id}")
                                        out.append(
                                            f"context7 query: {query}  (tool: {docs_tool.name})"
                                        )
                                        out.append("--- docs ---")
                                        out.append(docs_text)

        elif cmd == "/changes":
            ag = self._active_agent()
            staged = False
            target_path = ""
            for arg in args:
                if arg in ("--staged", "-s"):
                    staged = True
                elif not target_path:
                    target_path = arg

            status_raw = ag.tools.call_tool("git_status", repo_path=target_path)
            status = _parse_tool_payload(status_raw)
            if status.get("status") == "ok":
                repo = status.get("repo", "-")
                branch = status.get("branch", "-")
                staged_items = status.get("staged", []) or []
                unstaged_items = status.get("unstaged", []) or []
                untracked_items = status.get("untracked", []) or []
                out.append(f"repo: {repo}")
                out.append(f"branch: {branch}")
                out.append(
                    "changes: "
                    f"staged={len(staged_items)} "
                    f"unstaged={len(unstaged_items)} "
                    f"untracked={len(untracked_items)}"
                )
                files: list[str] = []
                files.extend(
                    str(item.get("file", "")).strip()
                    for item in staged_items
                    if isinstance(item, dict)
                )
                files.extend(
                    str(item.get("file", "")).strip()
                    for item in unstaged_items
                    if isinstance(item, dict)
                )
                files.extend(
                    str(item).strip() for item in untracked_items if str(item).strip()
                )
                uniq_files = [f for f in dict.fromkeys(files) if f]
                if uniq_files:
                    preview = ", ".join(uniq_files[:20])
                    if len(uniq_files) > 20:
                        preview += f" … (+{len(uniq_files) - 20} more)"
                    out.append(f"files: {preview}")
            else:
                out.append(
                    f"git_status failed: {status.get('error', 'unknown error')}"
                )

            diff_raw = ag.tools.call_tool(
                "git_diff",
                path=target_path,
                staged=staged,
            )
            diff = _parse_tool_payload(diff_raw)
            if diff.get("status") != "ok":
                out.append(f"git_diff failed: {diff.get('error', 'unknown error')}")
            else:
                diff_text = str(diff.get("diff", "") or "").strip()
                if not diff_text:
                    out.append("diff: clean working tree")
                else:
                    out.append("--- diff preview ---")
                    out.append(diff_text)
                    if diff.get("truncated"):
                        out.append("[diff truncated]")

        else:
            out.append(f"unknown: {cmd}")

        return {
            "messages": out,
            "state": self._state_snapshot(),
            "exit": should_exit,
        }

    def chat(self, params: dict[str, Any]) -> dict[str, Any]:
        raw = str(params.get("message", "")).strip()
        if not raw:
            return {"messages": [], "state": self._state_snapshot()}

        active = self.active
        if active is None:
            raise RuntimeError("No active agent")

        sid = self.sessions.setdefault(active, _new_session_id())
        token_seen = False
        emitted_image_events: set[tuple[str, str]] = set()

        def _token(tok: str):
            nonlocal token_seen
            if not token_seen:
                token_seen = True
                self._emit(
                    "phase",
                    {"state": "streaming", "note": "model output"},
                )
            self._emit("token", {"token": tok})

        def _tool(
            name: str,
            tool_args: dict[str, Any],
            meta: dict[str, Any] | None = None,
        ):
            info = meta if isinstance(meta, dict) else {}
            stage = str(info.get("stage", "start") or "start").strip().lower()
            sequence = int(info.get("sequence", 0) or 0)
            if stage == "end":
                payload: dict[str, Any] = {
                    "name": name,
                    "sequence": sequence,
                    "status": str(info.get("status", "ok") or "ok"),
                    "duration_ms": int(info.get("duration_ms", 0) or 0),
                    "cache_hit": bool(info.get("cache_hit", False)),
                }
                error = str(info.get("error", "") or "").strip()
                if error:
                    payload["error"] = error
                self._emit("tool_end", payload)
                return

            self._emit(
                "tool_start",
                {"name": name, "args": tool_args or {}, "sequence": sequence},
            )
            for path in _extract_image_paths(tool_args or {}):
                key = (str(name or "").strip(), path)
                if key in emitted_image_events:
                    continue
                emitted_image_events.add(key)
                self._emit(
                    "image",
                    {"tool": key[0] or "unknown_tool", "path": path, "source": "args"},
                )

        def _skill(skill_ids: list[str], selected_tools: list[str]):
            self._emit(
                "skill",
                {"skill_ids": skill_ids or [], "selected_tools": selected_tools or []},
            )

        if self.pipeline:
            pm = self.pipeline
            assert pm is not None
            turns: list[dict[str, str]] = []
            message = raw
            for i in range(int(pm["rounds"])):
                for ag_name in (pm["a"], pm["b"]):
                    cur = self.agents.get(ag_name)
                    if cur is None:
                        continue
                    sid_cur = self.sessions.setdefault(ag_name, _new_session_id())
                    self._emit(
                        "phase",
                        {
                            "state": "thinking",
                            "note": f"pipeline {i + 1}/{pm['rounds']} {ag_name}",
                        },
                    )
                    try:
                        resp = cur.chat(
                            message,
                            session_id=sid_cur,
                            verbose=False,
                            use_semantic_retrieval=True,
                            retrieval_mode="hybrid",
                        )
                    except Exception as exc:
                        resp = f"[error] {exc}"
                    turns.append({"agent": str(ag_name), "text": str(resp)})
                    message = str(resp)
            state = self._state_snapshot()
            return {"pipeline": True, "turns": turns, "state": state}

        ag = self._active_agent()
        self._emit("phase", {"state": "thinking", "note": "running agent"})
        run_resp = ag.run(
            raw,
            session_id=sid,
            verbose=False,
            use_semantic_retrieval=True,
            retrieval_mode="hybrid",
            stream_callback=_token,
            tool_callback=_tool,
            skill_callback=_skill,
        )
        response_text = str(getattr(run_resp, "final_response", "") or "")
        tool_calls_payload: list[dict[str, Any]] = []
        for call in list(getattr(run_resp, "tool_calls", []) or []):
            call_name = str(getattr(call, "name", "") or "")
            call_args = dict(getattr(call, "arguments", {}) or {})
            tool_calls_payload.append(
                {
                    "name": call_name,
                    "arguments": call_args,
                }
            )
            for path in _extract_image_paths(call_args):
                key = (call_name, path)
                if key in emitted_image_events:
                    continue
                emitted_image_events.add(key)
                self._emit(
                    "image",
                    {"tool": call_name or "unknown_tool", "path": path, "source": "call"},
                )

        for msg in list(getattr(run_resp, "messages", []) or []):
            role = str(getattr(msg, "role", "") or "").lower()
            if not role.endswith("tool"):
                continue
            tool_name = str(getattr(msg, "name", "") or "tool_result")
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            for path in _extract_image_paths(content):
                key = (tool_name, path)
                if key in emitted_image_events:
                    continue
                emitted_image_events.add(key)
                self._emit(
                    "image",
                    {
                        "tool": tool_name,
                        "path": path,
                        "source": "tool_result",
                    },
                )
        # Some backends/modes return a full answer without incremental token callbacks
        # (for example constrained decoding or non-streaming server paths). Emit a
        # synthetic token stream so the TUI still shows streaming output.
        if not token_seen and response_text:
            self._emit(
                "phase",
                {"state": "streaming", "note": "rendering response"},
            )
            text = response_text
            chunk_size = 96
            for i in range(0, len(text), chunk_size):
                self._emit("token", {"token": text[i : i + chunk_size]})
        return {
            "pipeline": False,
            "assistant": response_text,
            "iterations": int(getattr(run_resp, "iterations", 0) or 0),
            "tool_call_count": len(tool_calls_payload),
            "tool_calls": tool_calls_payload,
            "state": self._state_snapshot(),
        }


def main() -> None:
    server = BridgeServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}

            if method == "init":
                result = server.init(params)
            elif method == "slash":
                result = server.slash(params)
            elif method == "chat":
                result = server.chat(params)
            elif method == "state":
                result = server._state_snapshot()
            else:
                raise ValueError(f"Unknown method: {method}")

            server._response(rid, True, result=result)
        except Exception as exc:
            try:
                rid = req.get("id") if isinstance(req, dict) else None
            except Exception:
                rid = None
            server._response(rid, False, error=str(exc))


if __name__ == "__main__":
    main()
