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

DB_PATH = ROOT_DIR / "agent_sessions.db"
VECTOR_PATH = ROOT_DIR / "message_history.vector"


@lru_cache(maxsize=1)
def _agent_factory():
    from src.agent import create_agent

    return create_agent


@lru_cache(maxsize=1)
def _has_rapidfuzz() -> bool:
    from src.tools.catalog import HAS_RAPIDFUZZ

    return bool(HAS_RAPIDFUZZ)


def _new_session_id() -> str:
    return f"cli_{uuid.uuid4().hex[:8]}"


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


@contextlib.contextmanager
def _silent_load():
    null = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = null
    sys.stderr = null
    old_fd1 = os.dup(1)
    old_fd2 = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(old_fd1, 1)
        os.close(old_fd1)
        os.dup2(old_fd2, 2)
        os.close(old_fd2)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null.close()


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
            # Backward-compat alias used by agent_config.json.
            if "mcp_servers" not in ov and (mcp := merged.get("mcp")):
                ov["mcp_servers"] = mcp
            return ov

        def make_agent(block: dict[str, Any]):
            create_agent = _agent_factory()
            merged = {**cfg, **block}
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
            "pipeline": self.pipeline,
            "rapidfuzz": _has_rapidfuzz(),
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
            out.extend(
                [
                    "Command palette:",
                    "  /status                          runtime state snapshot",
                    "  /agents                          list loaded agents",
                    "  /agent <name>                    switch active agent",
                    "  /pipeline <a> <b> [rounds]       set inter-agent pipeline",
                    "  /pipeline stop                   disable pipeline",
                    "  /context                         session/data context",
                    "  /compact [keep_last]             summarize old history",
                    "  /reset                           reset runtime tool state",
                    "  /sessions                        list stored sessions",
                    "  /load <id_prefix>                load prior session",
                    "  /export [path]                   export transcript",
                    "  /upload <file> [label]           ingest one doc to RAG",
                    "  /upload-dir <dir> [glob] [max]   bulk ingest docs to RAG",
                    "  /docs <library> [query]          fetch Context7 documentation",
                    "  /changes [path] [--staged]        show git status + diff preview",
                    "  /trace [on|off]                  client-side trace toggle",
                    "  /clear                           client-side transcript clear",
                    "  /doctor                          client-side health checks",
                    "  /bug [note]                      client-side bug report file",
                    "  /new                             start new session",
                    "  /reload                          reload config + agents",
                    "  /quit                            exit",
                ]
            )

        elif cmd == "/status":
            snap = self._state_snapshot()
            agents = ", ".join(snap.get("agents", []) or []) or "-"
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
                    f"pipeline: {ptxt}",
                    f"rapidfuzz: {'enabled' if snap.get('rapidfuzz') else 'disabled'}",
                ]
            )

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

        def _token(tok: str):
            nonlocal token_seen
            if not token_seen:
                token_seen = True
                self._emit(
                    "phase",
                    {"state": "streaming", "note": "model output"},
                )
            self._emit("token", {"token": tok})

        def _tool(name: str, tool_args: dict[str, Any]):
            self._emit("tool", {"name": name, "args": tool_args or {}})

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
        response = ag.chat(
            raw,
            session_id=sid,
            verbose=False,
            use_semantic_retrieval=True,
            retrieval_mode="hybrid",
            stream=_token,
            tool_callback=_tool,
            skill_callback=_skill,
        )
        # Some backends/modes return a full answer without incremental token callbacks
        # (for example constrained decoding or non-streaming server paths). Emit a
        # synthetic token stream so the TUI still shows streaming output.
        if not token_seen and response:
            self._emit(
                "phase",
                {"state": "streaming", "note": "rendering response"},
            )
            text = str(response)
            chunk_size = 96
            for i in range(0, len(text), chunk_size):
                self._emit("token", {"token": text[i : i + chunk_size]})
        return {
            "pipeline": False,
            "assistant": response,
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
