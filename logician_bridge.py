from __future__ import annotations

import contextlib
import datetime
import importlib
import json
import os
import re
import shlex
import sys
import threading
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

# Disable bytecode cache during bridge execution to avoid stale .pyc imports
# and reduce interpreter issues when loading dynamic skill modules.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
importlib.invalidate_caches()

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message=r"invalid escape sequence '.*'",
)

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

from src.runtime_paths import message_history_vector_path, rag_vector_path, session_db_path


def _db_path() -> Path:
    return session_db_path()


def _vector_path() -> Path:
    return message_history_vector_path()


def _rag_vector_path() -> Path:
    return rag_vector_path()


# Project-scoped cross-session memory directory (relative to CWD at launch).
MEMORY_DIR = Path.cwd() / ".logician" / "memory"


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
    ("/context", "Show runtime context or prompt preview (`/context preview [query]`)"),
    ("/compact [keep_last]", "Summarize old history"),
    ("/reset", "Reset runtime tool state"),
    ("/sessions", "List stored sessions"),
    ("/load", "Load prior session (`/load <id_prefix>`)"),
    ("/export", "Export transcript (`/export [path]`)"),
    (
        "/mount",
        "Mount codebase (`/mount <dir> [glob] [max] [depth] [-exclude subdir]`)",
    ),
    ("/mount-code", "Alias for /mount"),
    ("/upload", "Ingest one doc (`/upload <file> [label]`)"),
    ("/upload-dir", "Bulk ingest docs (`/upload-dir <dir> [glob] [max]`)"),
    ("/plugins", "List installed plugins or enable/disable plugin state"),
    (
        "/repo",
        "Manage repo memory (`/repo add|list|use|ingest|remove ...`)",
    ),
    (
        "/rag",
        "Inspect RAG store (`/rag list [repo]`, `/rag search <query> [top_k]`, `/rag clear`)",
    ),
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
_STREAM_EVENT_CHAR_BATCH = 192


def _parse_exclude_args(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    seen: set[str] = set()
    excludes: list[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            item = part.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            excludes.append(item)
    return excludes


def _parse_mount_args(args: list[str]) -> dict[str, Any]:
    if not args:
        return {
            "status": "error",
            "error": "usage: /mount <directory> [glob] [max_files] [map_depth] [-exclude subdir]",
        }

    directory = args[0]
    positional: list[str] = []
    excludes: list[str] = []
    idx = 1
    while idx < len(args):
        token = args[idx]
        if token in {"-exclude", "--exclude"}:
            if idx + 1 >= len(args):
                return {"status": "error", "error": f"missing value for {token}"}
            excludes.extend(_parse_exclude_args(args[idx + 1]))
            idx += 2
            continue
        positional.append(token)
        idx += 1

    if len(positional) > 3:
        return {
            "status": "error",
            "error": "usage: /mount <directory> [glob] [max_files] [map_depth] [-exclude subdir]",
        }

    return {
        "status": "ok",
        "directory": directory,
        "glob": positional[0]
        if positional
        else "**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}",
        "max_files": positional[1] if len(positional) > 1 else None,
        "map_depth": positional[2] if len(positional) > 2 else None,
        "exclude": ",".join(excludes),
        "exclude_display": excludes,
    }


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _estimate_token_count(payload: Any) -> int:
    if isinstance(payload, dict):
        for key in (
            "token_count",
            "tokens",
            "estimated_tokens",
            "context_tokens",
            "prompt_tokens",
        ):
            if key in payload:
                count = _safe_int(payload.get(key))
                if count > 0:
                    return count

        for key in ("project_map", "tree", "map", "content", "preview"):
            text = payload.get(key)
            if isinstance(text, str) and text.strip():
                return max(1, len(text) // 4)

    try:
        rendered = json.dumps(payload, ensure_ascii=False)
    except Exception:
        rendered = str(payload or "")
    return max(0, len(rendered) // 4)


def _upsert_runtime_record(
    records: list[dict[str, Any]], item: dict[str, Any], *, key: str
) -> None:
    identity = str(item.get(key) or "").strip()
    if not identity:
        return
    for idx, existing in enumerate(records):
        if str(existing.get(key) or "").strip() == identity:
            records[idx] = item
            return
    records.append(item)


def _remember_mount_context(
    agent: Any,
    *,
    path: str,
    glob: str,
    file_count: int,
    token_count: int,
    map_depth: int,
) -> None:
    ctx = getattr(agent, "ctx", None)
    if ctx is None:
        return
    mounts = getattr(ctx, "mounted_paths", None)
    if not isinstance(mounts, list):
        mounts = []
        setattr(ctx, "mounted_paths", mounts)
    _upsert_runtime_record(
        mounts,
        {
            "path": path,
            "glob": glob,
            "file_count": max(0, file_count),
            "token_count": max(0, token_count),
            "map_depth": max(0, map_depth),
        },
        key="path",
    )


def _remember_rag_doc(
    agent: Any,
    *,
    path: str,
    source_label: str,
    chunks: int,
    token_count: int,
    kind: str,
) -> None:
    ctx = getattr(agent, "ctx", None)
    if ctx is None:
        return
    docs = getattr(ctx, "rag_docs", None)
    if not isinstance(docs, list):
        docs = []
        setattr(ctx, "rag_docs", docs)
    _upsert_runtime_record(
        docs,
        {
            "path": path,
            "label": source_label,
            "chunks": max(0, chunks),
            "token_count": max(0, token_count),
            "kind": kind,
        },
        key="path",
    )


def _repo_runtime_record(repo: dict[str, Any]) -> dict[str, Any]:
    git = dict(repo.get("git") or {})
    return {
        "id": str(repo.get("id") or "").strip(),
        "name": str(repo.get("name") or "").strip(),
        "path": str(repo.get("path") or "").strip(),
        "last_ingested_at": str(repo.get("last_ingested_at") or "").strip(),
        "files_processed": _safe_int(repo.get("files_processed")),
        "chunks_added": _safe_int(repo.get("chunks_added")),
        "graph_nodes": _safe_int(repo.get("graph_nodes")),
        "graph_edges": _safe_int(repo.get("graph_edges")),
        "graph_symbols": _safe_int(repo.get("graph_symbols")),
        "branch": str(git.get("branch") or "").strip(),
        "commit": str(git.get("commit") or "").strip(),
        "last_graph_built_at": str(repo.get("last_graph_built_at") or "").strip(),
    }


def _active_repo_records(agent: Any) -> list[dict[str, Any]]:
    ctx = getattr(agent, "ctx", None)
    if ctx is None:
        return []
    active = getattr(ctx, "active_repos", None)
    if not isinstance(active, list):
        active = []
        setattr(ctx, "active_repos", active)
    return [dict(item) for item in active if isinstance(item, dict)]


def _set_active_repo_records(agent: Any, repos: list[dict[str, Any]]) -> None:
    ctx = getattr(agent, "ctx", None)
    if ctx is None:
        return
    setattr(
        ctx,
        "active_repos",
        [_repo_runtime_record(repo) for repo in repos if isinstance(repo, dict)],
    )


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
        self._recent_rag_queries: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._session_start_hook_plugin_ids_cache: set[str] | None = None
        self._startup_hook_lock = threading.Lock()
        self._startup_hook_thread: threading.Thread | None = None
        self._startup_hook_generation = 0
        self._startup_hook_started = False
        self._startup_hook_complete = True
        self._startup_hook_contexts: list[str] = []
        self._startup_hook_count = 0
        self._startup_hook_errors: list[str] = []

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        msg = {"event": event, **payload}
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _response(self, rid: Any, ok: bool, result: Any = None, error: str = "") -> None:
        msg = {"id": rid, "ok": ok}
        if ok:
            msg["result"] = result
        else:
            msg["error"] = error
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _load_agents(
        self,
        config_path: Path,
        *,
        fast_init: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        def build_overrides(block: dict[str, Any]) -> dict[str, Any]:
            from src.config import Config as AgentConfig

            merged = {**cfg, **block}
            allowed = set(getattr(AgentConfig, "__dataclass_fields__", {}).keys())
            ov = {
                key: value for key, value in merged.items() if key in allowed and value is not None
            }
            # Keep vector store local to this bridge workspace.
            ov["vector_path"] = str(_vector_path())
            ov["rag_vector_path"] = str(_rag_vector_path())
            # Do not force a default RAG vector backend here – leave it
            # unset so the agent/workspace can opt into a backend explicitly
            # (empty string == raw/no backend by default).
            if "rag_rerank_enabled" not in ov:
                ov["rag_rerank_enabled"] = False
            # Backward-compat alias used by agent_config.json.
            if "mcp_servers" not in ov and (mcp := merged.get("mcp")):
                ov["mcp_servers"] = mcp
            if fast_init:
                ov["lazy_mcp_init"] = True
            ov["runtime_hooks_enabled"] = True
            return ov

        def make_agent(block: dict[str, Any]):
            create_agent = _agent_factory()
            merged = {**cfg, **block}
            os.environ["LOGICIAN_PROJECT_MEMORY_ENABLED"] = (
                "1" if bool(merged.get("project_memory_enabled", True)) else "0"
            )
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
                db_path=str(_db_path()),
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

    def _invalidate_startup_hook_cache(self) -> None:
        self._session_start_hook_plugin_ids_cache = None
        with self._startup_hook_lock:
            self._startup_hook_generation += 1
            self._startup_hook_thread = None
            self._startup_hook_started = False
            self._startup_hook_complete = True
            self._startup_hook_contexts = []
            self._startup_hook_count = 0
            self._startup_hook_errors = []

    def _session_start_hook_plugin_ids(self) -> set[str]:
        if self._session_start_hook_plugin_ids_cache is not None:
            return set(self._session_start_hook_plugin_ids_cache)

        plugin_ids: set[str] = set()
        try:
            from src.hooks.loader import HookLoader

            plugin_ids = {
                str(hook.plugin_id or "").strip()
                for hook in HookLoader().get_session_start_hooks()
                if str(hook.plugin_id or "").strip()
            }
        except Exception:
            plugin_ids = set()

        self._session_start_hook_plugin_ids_cache = set(plugin_ids)
        return plugin_ids

    def _startup_hook_snapshot(self) -> tuple[list[str], int, bool, list[str]]:
        with self._startup_hook_lock:
            return (
                list(self._startup_hook_contexts),
                int(self._startup_hook_count or 0),
                bool(self._startup_hook_complete),
                list(self._startup_hook_errors),
            )

    def _warm_mcp_servers_async(self) -> None:
        def _worker() -> None:
            for agent in list(self.agents.values()):
                try:
                    ensure_loaded = getattr(agent, "_ensure_mcp_servers_loaded", None)
                    if callable(ensure_loaded):
                        ensure_loaded()
                except Exception:
                    continue

        thread = threading.Thread(target=_worker, name="bridge-mcp-warmup", daemon=True)
        thread.start()

    def _start_startup_hooks_async(self, source: str = "startup") -> None:
        with self._startup_hook_lock:
            if self._startup_hook_started:
                thread = self._startup_hook_thread
                if thread is not None and thread.is_alive():
                    return
                if self._startup_hook_complete:
                    return
            generation = self._startup_hook_generation
            self._startup_hook_started = True
            self._startup_hook_complete = False
            self._startup_hook_contexts = []
            self._startup_hook_count = 0
            self._startup_hook_errors = []

        def _worker() -> None:
            def _progress(kind: str, payload: dict[str, Any]) -> None:
                self._handle_startup_hook_progress(generation, kind, payload)

            try:
                from src.hooks import HookEngine

                engine = HookEngine(progress_callback=_progress)
                active_name = self._hook_agent_name()
                active_sid = self.sessions.get(active_name, "") if active_name else ""
                result = engine.execute_session_start_hooks(
                    source,
                    session_id=active_sid,
                    transcript_path=self._hook_transcript_path(active_name),
                )
            except Exception as exc:
                with self._startup_hook_lock:
                    if generation != self._startup_hook_generation:
                        return
                    self._startup_hook_complete = True
                    self._startup_hook_errors.append(str(exc))
                self._emit(
                    "lifecycle",
                    {
                        "subsystem": "startup_hook",
                        "payload": {"state": "failed", "error": str(exc)},
                    },
                )
                return

            with self._startup_hook_lock:
                if generation != self._startup_hook_generation:
                    return
                self._startup_hook_contexts = list(result.additional_contexts)
                self._startup_hook_count = int(result.hook_count or 0)
                self._startup_hook_errors = list(result.errors or [])
                self._startup_hook_complete = True

            self._inject_startup_hook_contexts()

        thread = threading.Thread(
            target=_worker,
            name=f"bridge-startup-hooks-{generation}",
            daemon=True,
        )
        with self._startup_hook_lock:
            self._startup_hook_thread = thread
        thread.start()

    def _wait_for_startup_hooks(self) -> None:
        thread: threading.Thread | None = None
        with self._startup_hook_lock:
            thread = self._startup_hook_thread
        if thread is not None and thread.is_alive():
            thread.join()

    def _hook_agent_name(self, agent_name: str | None = None) -> str:
        target = str(agent_name or self.active or "").strip()
        if target and target in self.agents:
            return target
        return next(iter(self.agents.keys()), "") if self.agents else ""

    def _hook_transcript_path(self, agent_name: str | None = None) -> str:
        target = self._hook_agent_name(agent_name)
        if not target:
            return ""
        agent = self.agents.get(target)
        if agent is None:
            return ""
        memory = getattr(agent, "memory", None)
        path = str(getattr(memory, "db_path", "") or "").strip()
        if not path:
            return ""
        try:
            return str(Path(path).expanduser())
        except Exception:
            return path

    def _fire_session_end_hooks(self, reason: str) -> None:
        seen_sessions: set[tuple[str, str]] = set()
        for agent_name, sid in list(self.sessions.items()):
            session_id = str(sid or "").strip()
            if not session_id:
                continue
            key = (agent_name, session_id)
            if key in seen_sessions:
                continue
            seen_sessions.add(key)
            agent = self.agents.get(agent_name)
            if agent is None:
                continue
            try:
                agent.end_session(reason=reason, session_id=session_id)
            except Exception:
                continue

    def _handle_startup_hook_progress(
        self,
        generation: int,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        if generation != self._startup_hook_generation:
            return

        if kind == "discovered":
            hook_count = int(payload.get("hook_count") or 0)
            with self._startup_hook_lock:
                if generation != self._startup_hook_generation:
                    return
                self._startup_hook_count = hook_count
            self._emit(
                "lifecycle",
                {
                    "subsystem": "startup_hook",
                    "payload": {"state": "running", "hook_count": hook_count},
                },
            )
            return

        if kind == "context":
            context = str(payload.get("context") or "").strip()
            if not context:
                return
            with self._startup_hook_lock:
                if generation != self._startup_hook_generation:
                    return
                self._startup_hook_contexts.append(context)
            self._emit(
                "lifecycle",
                {
                    "subsystem": "startup_hook",
                    "payload": {
                        "state": "context",
                        "context": context,
                        "plugin_id": str(payload.get("plugin_id") or ""),
                        "plugin_name": str(payload.get("plugin_name") or ""),
                    },
                },
            )
            return

        if kind in {"hook_started", "hook_finished"}:
            self._emit(
                "lifecycle",
                {
                    "subsystem": "startup_hook",
                    "payload": {
                        "state": "running" if kind == "hook_started" else "hook_finished",
                        "plugin_id": str(payload.get("plugin_id") or ""),
                        "plugin_name": str(payload.get("plugin_name") or ""),
                        "ordinal": int(payload.get("ordinal") or 0),
                    },
                },
            )
            return

        if kind == "error":
            error = str(payload.get("error") or "").strip()
            if not error:
                return
            with self._startup_hook_lock:
                if generation != self._startup_hook_generation:
                    return
                self._startup_hook_errors.append(error)
            self._emit(
                "lifecycle",
                {
                    "subsystem": "startup_hook",
                    "payload": {"state": "error", "error": error},
                },
            )
            return

        if kind == "completed":
            self._emit(
                "lifecycle",
                {
                    "subsystem": "startup_hook",
                    "payload": {
                        "state": "complete",
                        "hook_count": int(payload.get("hook_count") or 0),
                        "context_count": int(payload.get("context_count") or 0),
                        "errors": list(payload.get("errors") or []),
                    },
                },
            )

    def _plugin_memory_sources(
        self, *, exclude_plugin_ids: set[str] | None = None
    ) -> list[tuple[str, Path]]:
        try:
            from src.plugin_manager.state import iter_enabled_plugin_install_paths

            excluded = {str(pid or "").strip() for pid in (exclude_plugin_ids or set())}
            sources: list[tuple[str, Path]] = []
            seen: set[str] = set()

            for plugin_id, cache in iter_enabled_plugin_install_paths():
                plugin_id = str(plugin_id or "").strip()
                if not plugin_id or plugin_id in excluded:
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
                    sources.append((plugin_id, candidate))
            return sources
        except Exception:
            return []

    def _plugin_memory_paths(self) -> list[Path]:
        return [path for _, path in self._plugin_memory_sources()]

    def _find_memory_index(self, root: Path) -> Path | None:
        if root.is_file() and root.name.upper() in ("MEMORY.MD", "CLAUDE.MD"):
            return root
        if not root.is_dir():
            return None

        direct = root / "MEMORY.md"
        if direct.exists():
            return direct

        # Prefer CLAUDE.md within a .claude-plugin directory anywhere under the plugin root.
        plugin_claude_md = root / ".claude-plugin" / "CLAUDE.md"
        if plugin_claude_md.exists():
            return plugin_claude_md

        for candidate in sorted(root.rglob("CLAUDE.md")):
            if not candidate.is_file():
                continue
            if candidate.parent.name == ".claude-plugin":
                return candidate

        # Also check for direct CLAUDE.md in the plugin root.
        claude_md = root / "CLAUDE.md"
        if claude_md.exists():
            return claude_md

        for candidate in sorted(root.rglob("MEMORY.md")):
            if candidate.is_file():
                return candidate

        # Also recursively search for CLAUDE.md in nested plugin directories
        first_candidate: Path | None = None
        for candidate in sorted(root.rglob("CLAUDE.md")):
            if not candidate.is_file():
                continue
            if candidate.parent.name == ".claude-plugin":
                return candidate
            if first_candidate is None:
                first_candidate = candidate
        return first_candidate

    def _parse_memory_dir_summary(self, root: Path) -> dict[str, Any]:
        memory_index = self._find_memory_index(root)
        if memory_index is None or not memory_index.exists():
            return {
                "has_memories": False,
                "total_entries": 0,
                "obs_count": 0,
                "facts_count": 0,
                "sections": [],
            }

        content = memory_index.read_text(encoding="utf-8", errors="replace")
        obs_count = 0
        obs_index = memory_index.parent / "obs" / "index.json"
        if not obs_index.exists():
            for parent in memory_index.parents:
                candidate = parent / "obs" / "index.json"
                if candidate.exists():
                    obs_index = candidate
                    break
        if obs_index.exists():
            try:
                import json as _json

                obs_count = len(_json.loads(obs_index.read_text(encoding="utf-8")))
            except Exception:
                pass

        sections: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_entries: list[str] = []
        facts_count = 0

        # Check if this looks like claude-mem format (has ### date headers and table rows)
        is_claude_mem_format = "### " in content and "| ID |" in content

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("<!--") or stripped.startswith("# "):
                continue
            if stripped.startswith("**#S") and stripped.endswith(")"):
                if current_heading is not None and current_entries:
                    sections.append({"heading": current_heading, "entries": current_entries})
                current_heading = stripped.strip("*")
                current_entries = []
            elif stripped.startswith("### "):
                # claude-mem date header like "### Oct 25, 2026"
                if current_heading is not None and current_entries:
                    sections.append({"heading": current_heading, "entries": current_entries})
                current_heading = stripped[4:].strip()
                current_entries = []
            elif stripped.startswith("## "):
                if current_heading is not None and current_entries:
                    sections.append({"heading": current_heading, "entries": current_entries})
                current_heading = stripped[3:].strip()
                current_entries = []
            elif stripped.startswith("|") and current_heading is not None:
                parts = [p.strip() for p in stripped.split("|") if p.strip()]
                if is_claude_mem_format:
                    # claude-mem format: preserve table header, separator, and data rows.
                    current_entries.append(stripped)
                elif stripped.startswith("| #") and len(parts) >= 3:
                    current_entries.append(" ".join(parts[-2:]))
            elif stripped.startswith("- ") and current_heading is not None:
                entry = stripped[2:].strip()
                current_entries.append(entry)
                if current_heading == "Facts":
                    facts_count += 1

        if current_heading is not None and current_entries:
            sections.append({"heading": current_heading, "entries": current_entries})

        total = obs_count + facts_count
        if total == 0 and sections:
            total = sum(len(section.get("entries", [])) for section in sections)
        return {
            "enabled": True,
            "has_memories": total > 0,
            "total_entries": total,
            "obs_count": obs_count,
            "facts_count": facts_count,
            "sections": sections,
        }

    def _parse_memory_summary(
        self, *, exclude_session_start_hook_plugins: bool = True
    ) -> dict[str, Any]:
        if not self.cfg.get("project_memory_enabled", True):
            return {
                "enabled": False,
                "has_memories": False,
                "total_entries": 0,
                "obs_count": 0,
                "facts_count": 0,
                "sections": [],
            }
        summary = self._parse_memory_dir_summary(MEMORY_DIR)
        excluded = (
            self._session_start_hook_plugin_ids() if exclude_session_start_hook_plugins else set()
        )
        for _plugin_id, plugin_path in self._plugin_memory_sources(exclude_plugin_ids=excluded):
            if plugin_path == MEMORY_DIR or plugin_path in MEMORY_DIR.parents:
                continue
            plugin_summary = self._parse_memory_dir_summary(plugin_path)
            if not plugin_summary["has_memories"]:
                continue
            summary["obs_count"] += plugin_summary["obs_count"]
            summary["facts_count"] += plugin_summary["facts_count"]
            summary["total_entries"] += plugin_summary["total_entries"]
            summary["sections"].extend(plugin_summary["sections"])
            summary["has_memories"] = summary["total_entries"] > 0
        return summary

    def _inject_project_memory(self, *, exclude_session_start_hook_plugins: bool = True) -> None:
        if not self.cfg.get("project_memory_enabled", True):
            return
        memory_blocks: list[str] = []
        memory_index = MEMORY_DIR / "MEMORY.md"
        if memory_index.exists():
            content = memory_index.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                memory_blocks.append(content)

        excluded = (
            self._session_start_hook_plugin_ids() if exclude_session_start_hook_plugins else set()
        )
        for _plugin_id, plugin_path in self._plugin_memory_sources(exclude_plugin_ids=excluded):
            if plugin_path == MEMORY_DIR or plugin_path in MEMORY_DIR.parents:
                continue
            plugin_index = self._find_memory_index(plugin_path)
            if plugin_index is None or not plugin_index.exists():
                continue
            content = plugin_index.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                memory_blocks.append(content)

        if not memory_blocks:
            return

        block = "\n\n".join(
            f"<session-memory>\n{content}\n</session-memory>" for content in memory_blocks
        )
        for agent in self.agents.values():
            sp = getattr(agent, "system_prompt", None) or ""
            agent.system_prompt = f"{sp}\n\n{block}" if sp else block

    def _normalize_system_prompt(self, text: str) -> str:
        if not text:
            return ""

        pattern = r"<startup-hook-context>.*?</startup-hook-context>"
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            normalized_contents: list[str] = []
            for block in matches:
                inner = re.sub(
                    r"</?startup-hook-context>",
                    "",
                    block,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
                if inner:
                    normalized_contents.append(inner)

            merged_block = "<startup-hook-context>\n"
            merged_block += "\n\n".join(normalized_contents)
            merged_block += "\n</startup-hook-context>"

            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            text = f"{text}\n\n{merged_block}" if text else merged_block

        sections = [s.strip() for s in re.split(r"\n{2,}", text.strip()) if s.strip()]
        seen: set[str] = set()
        deduped: list[str] = []
        for sec in sections:
            if len(sec) > 140 or sec.count("\n") > 3:
                if sec in seen:
                    continue
                seen.add(sec)
            deduped.append(sec)
        return "\n\n".join(deduped)

    def _format_context_preview(
        self,
        preview: dict[str, Any],
        sid: str,
        runtime_info: dict[str, Any] | None = None,
    ) -> str:
        system_prompt = self._normalize_system_prompt(str(preview.get("system_prompt", "")))
        out: list[str] = []
        out.append(f"session: {preview.get('session_id', sid)}")
        out.append(f"classification: {preview.get('classified_as', 'execution')}")
        domains = preview.get("domain_groups") or []
        if domains:
            out.append(f"domain tools: {', '.join(str(item) for item in domains)}")
        out.append(
            "messages in prompt: "
            f"{preview.get('trimmed_message_count', 0)}"
            f" (loaded={preview.get('history_loaded_count', 0)}, "
            f"pre-trim={preview.get('untrimmed_message_count', 0)}, "
            f"limit={preview.get('history_limit', 0)})"
        )
        out.append(f"context token budget: {preview.get('context_token_budget', 0)}")
        out.append("")
        out.append("## System Prompt")
        out.append("")
        out.append("```md")
        out.append(system_prompt.rstrip())
        out.append("```")
        out.append("")
        out.append("## Message Window")
        out.append("")
        try:
            hook_matches = re.findall(
                r"<startup-hook-context>(.*?)</startup-hook-context>",
                system_prompt,
                flags=re.DOTALL | re.IGNORECASE,
            )
            suppress_texts: set[str] = set(h.strip() for h in hook_matches if h and h.strip())
            for sec in re.split(r"\n{2,}", system_prompt or ""):
                s = sec.strip()
                if not s:
                    continue
                if len(s) > 140 or s.count("\n") > 3:
                    suppress_texts.add(s)
        except Exception:
            suppress_texts = set()

        filtered_messages = []
        for msg in preview.get("messages", []) or []:
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            skip = False
            for st in suppress_texts:
                if not st:
                    continue
                if content == st or content in st or st in content:
                    skip = True
                    break
            if skip:
                continue
            filtered_messages.append(msg)

        for idx, msg in enumerate(filtered_messages, start=1):
            role = str(msg.get("role", "unknown"))
            name = str(msg.get("name", "") or "").strip()
            label = f"{idx}. {role}"
            if name:
                label += f" ({name})"
            out.append(f"### {label}")
            out.append("")
            out.append("```text")
            out.append(str(msg.get("content", "")).rstrip())
            out.append("```")
            out.append("")

        if runtime_info is not None:
            out.append("## Runtime Context")
            out.append("")
            out.append(
                f"messages: {runtime_info.get('persisted_messages', 0)} / limit {runtime_info.get('history_limit', 0)}"
            )
            rt = runtime_info.get("runtime", {})
            if rt.get("loaded"):
                cols = ", ".join(rt.get("value_columns", [])[:6]) or "none"
                out.append(
                    f"dataset: {rt.get('data_name', 'unnamed')} rows={rt.get('row_count', 0)} cols={cols}"
                )
            else:
                out.append("dataset: none")
            active_repos = list(rt.get("active_repos") or [])
            if active_repos:
                repo_preview = ", ".join(
                    str(item.get("id") or item.get("name") or "").strip()
                    for item in active_repos[:8]
                    if str(item.get("id") or item.get("name") or "").strip()
                )
                if repo_preview:
                    out.append(f"active_repos: {repo_preview}")

        return "\n".join(out)

    def _normalize_startup_hook_blocks(self, text: str) -> str:
        """Normalize startup hook context blocks in system prompt."""
        if not text:
            return ""

        pattern = r"<startup-hook-context>.*?</startup-hook-context>"
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            normalized_contents: list[str] = []
            for block in matches:
                inner = re.sub(
                    r"</?startup-hook-context>",
                    "",
                    block,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
                if inner:
                    normalized_contents.append(inner)

            merged_block = "<startup-hook-context>\n"
            merged_block += "\n\n".join(normalized_contents)
            merged_block += "\n</startup-hook-context>"

            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            text = f"{text}\n\n{merged_block}" if text else merged_block

        sections = [s.strip() for s in re.split(r"\n{2,}", text.strip()) if s.strip()]
        seen: set[str] = set()
        deduped: list[str] = []
        for sec in sections:
            if len(sec) > 140 or sec.count("\n") > 3:
                if sec in seen:
                    continue
                seen.add(sec)
            deduped.append(sec)
        return "\n\n".join(deduped)

    def _inject_startup_hook_contexts(self) -> None:
        if not self._startup_hook_contexts:
            return

        block = (
            "<startup-hook-context>\n"
            + "\n\n".join(self._startup_hook_contexts)
            + "\n</startup-hook-context>"
        )
        for agent in self.agents.values():
            sp = getattr(agent, "system_prompt", None) or ""
            cleaned = self._normalize_startup_hook_blocks(sp)
            cleaned = re.sub(
                r"<startup-hook-context>.*?</startup-hook-context>\s*",
                "",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            agent.system_prompt = f"{cleaned}\n\n{block}" if cleaned else block

    def _active_agent(self):
        if self.active is None:
            raise RuntimeError("No active agent")
        return self.agents[self.active]

    def _active_mcp_names(self) -> list[str]:
        # If no active agent is explicitly selected, fall back to the
        # first registered agent so /status can show a meaningful context
        # preview in unattended runs.
        active = self.active or (next(iter(self.agents.keys()), "") if self.agents else "")
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

    def _session_tool_call_count(self, active: str, sid: str) -> int:
        if not active or active not in self.agents:
            return 0
        try:
            ctx = getattr(self.agents[active], "ctx", None)
            return int(getattr(ctx, "tool_call_count", 0))
        except Exception:
            return 0

    def _session_plan_mode(self, active: str, sid: str) -> bool:
        if not active or active not in self.agents:
            return False
        try:
            ctx = getattr(self.agents[active], "ctx", None)
            return bool(getattr(ctx, "plan_mode", False))
        except Exception:
            return False

    def _persist_active_runtime_state(self) -> None:
        active = self.active or ""
        if not active or active not in self.agents:
            return
        sid = self.sessions.get(active, "")
        if not sid:
            return
        persist = getattr(self.agents[active], "_persist_runtime_state", None)
        if callable(persist):
            try:
                persist(sid)
            except Exception:
                pass

    def _session_key(self, active: str, sid: str) -> tuple[str, str] | None:
        if not active or not sid:
            return None
        return (active, sid)

    def _recent_rag_query_items(self, active: str, sid: str) -> list[dict[str, Any]]:
        key = self._session_key(active, sid)
        if key is None:
            return []
        return [dict(item) for item in self._recent_rag_queries.get(key, [])]

    def _remember_rag_query(self, active: str, sid: str, item: dict[str, Any]) -> None:
        key = self._session_key(active, sid)
        if key is None:
            return
        query = str(item.get("query") or "").strip()
        if not query:
            return
        payload = dict(item)
        hits = payload.get("hits")
        if isinstance(hits, list):
            payload["hits"] = [dict(entry) for entry in hits[:8] if isinstance(entry, dict)]
        history = self._recent_rag_queries.setdefault(key, [])
        history.insert(0, payload)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for existing in history:
            dedupe_key = (
                str(existing.get("query") or "").strip().lower(),
                json.dumps(existing.get("repo_filter") or [], ensure_ascii=False),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(existing)
            if len(deduped) >= 6:
                break
        self._recent_rag_queries[key] = deduped

    def _clear_recent_rag_queries(self, active: str, sid: str) -> None:
        key = self._session_key(active, sid)
        if key is None:
            return
        self._recent_rag_queries.pop(key, None)

    def _repo_library(self) -> list[dict[str, Any]]:
        try:
            from src.repo.registry import load_repo_index

            repos = load_repo_index()
        except Exception:
            return []
        return [
            dict(item)
            for item in repos
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]

    def _resolve_repo_ref(self, raw: str) -> dict[str, Any] | None:
        target = str(raw or "").strip()
        if not target:
            return None
        target_lower = target.lower()
        for repo in self._repo_library():
            repo_id = str(repo.get("id") or "").strip()
            repo_name = str(repo.get("name") or "").strip()
            repo_path = str(repo.get("path") or "").strip()
            if (
                target == repo_id
                or target == repo_name
                or target == repo_path
                or target_lower == repo_id.lower()
                or target_lower == repo_name.lower()
                or target_lower == repo_path.lower()
            ):
                return repo
        return None

    def _repo_chunk_count(self, agent: Any, repo_id: str) -> int | None:
        clean_repo_id = str(repo_id or "").strip()
        if not clean_repo_id:
            return None
        try:
            memory = getattr(agent, "memory", None)
            if memory is None:
                return None
            ensure_doc_db = getattr(memory, "_ensure_doc_db", None)
            if callable(ensure_doc_db):
                ensure_doc_db()
            doc_db = getattr(memory, "_doc_db", None)
            if doc_db is None:
                return None
            return int(doc_db.count(where={"repo_id": clean_repo_id}) or 0)
        except Exception:
            return None

    def _repo_ingest_settings(self, agent: Any) -> dict[str, Any]:
        config = getattr(agent, "config", None)
        memory = getattr(agent, "memory", None)
        embedding_model_name = str(
            getattr(memory, "_embedding_model_name", "")
            or getattr(config, "embedding_model", "")
            or ""
        ).strip()
        vector_backend = str(getattr(config, "rag_vector_backend", "") or "").strip()
        return {
            "vector_path": str(_rag_vector_path()),
            "embedding_model_name": embedding_model_name or None,
            "vector_backend": vector_backend or None,
        }

    def _rag_inventory_snapshot(
        self,
        agent: Any,
        active_repos: list[dict[str, Any]],
        repo_library: list[dict[str, Any]],
        retrieval_insights: list[dict[str, Any]],
        rag_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        settings = self._repo_ingest_settings(agent)
        repo_entries = sorted(
            [
                {
                    "repo_id": str(item.get("id") or "").strip(),
                    "repo_name": str(item.get("name") or item.get("id") or "").strip(),
                    "chunks": _safe_int(item.get("chunks_added")),
                    "files": _safe_int(item.get("files_processed")),
                    "last_ingested_at": str(item.get("last_ingested_at") or "").strip(),
                }
                for item in repo_library
                if isinstance(item, dict)
                and (
                    str(item.get("id") or "").strip()
                    or str(item.get("name") or "").strip()
                    or _safe_int(item.get("chunks_added")) > 0
                )
            ],
            key=lambda item: (-int(item.get("chunks") or 0), str(item.get("repo_name") or "")),
        )
        active_repo_ids = {
            str(item.get("id") or "").strip()
            for item in active_repos
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        return {
            "vector_path": settings.get("vector_path") or str(_rag_vector_path()),
            "vector_backend": settings.get("vector_backend") or "unknown",
            "active_doc_count": len(rag_docs),
            "active_doc_chunks": sum(_safe_int(item.get("chunks")) for item in rag_docs),
            "repo_count": len(repo_entries),
            "active_repo_count": len(active_repo_ids),
            "repo_chunks": sum(int(item.get("chunks") or 0) for item in repo_entries),
            "retrieval_count": len(retrieval_insights),
            "legacy_paths": self._legacy_rag_paths(),
            "top_repos": repo_entries[:6],
        }

    def _repo_needs_ingest(self, agent: Any, repo: dict[str, Any]) -> bool:
        actual_chunk_count = self._repo_chunk_count(agent, str(repo.get("id") or ""))
        if actual_chunk_count is not None:
            return actual_chunk_count <= 0
        return int(repo.get("chunks_added", 0) or 0) <= 0

    def _legacy_rag_paths(self) -> list[str]:
        try:
            from src.rag_runtime import legacy_rag_vector_paths

            return [str(path) for path in legacy_rag_vector_paths(ROOT_DIR) if path.exists()]
        except Exception:
            return []

    def _ingest_repo_record(
        self,
        agent: Any,
        repo: dict[str, Any],
        *,
        glob_pattern: str,
        max_files: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from src.repo.ingest import ingest_repo

        payload = ingest_repo(
            str(repo.get("path") or ""),
            name=str(repo.get("name") or ""),
            base_dir=ROOT_DIR,
            glob_pattern=glob_pattern,
            max_files=max_files,
            vector_path=self._repo_ingest_settings(agent)["vector_path"],
            embedding_model_name=self._repo_ingest_settings(agent)["embedding_model_name"],
            vector_backend=self._repo_ingest_settings(agent)["vector_backend"],
        )
        updated_repo = dict(payload.get("repo") or repo)
        if str(payload.get("status") or "").lower() in {"ok", "partial"}:
            _remember_rag_doc(
                agent,
                path=str(updated_repo.get("path") or ""),
                source_label=f"repo {updated_repo.get('name', updated_repo.get('id', ''))}",
                chunks=_safe_int((payload.get("ingest") or {}).get("total_chunks_added")),
                token_count=_estimate_token_count(payload.get("ingest") or {}),
                kind="repo",
            )
        return payload, updated_repo

    def _auto_ingest_repos(
        self,
        agent: Any,
        repos: list[dict[str, Any]],
        *,
        glob_pattern: str,
        max_files: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        updated_repos: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        for repo in repos:
            if not isinstance(repo, dict):
                continue
            if not self._repo_needs_ingest(agent, repo):
                updated_repos.append(repo)
                continue
            payload, updated_repo = self._ingest_repo_record(
                agent,
                repo,
                glob_pattern=glob_pattern,
                max_files=max_files,
            )
            updated_repos.append(updated_repo)
            reports.append({"repo": updated_repo, "payload": payload})
        return updated_repos, reports

    def _preview_skill_selection(
        self,
        agent: Any,
        query: str,
    ) -> tuple[list[str], list[str]]:
        tool_registry = getattr(agent, "tools", None)
        if tool_registry is None:
            return [], []
        try:
            sync_fn = getattr(tool_registry, "_sync_catalog_with_registered_tools", None)
            if callable(sync_fn):
                sync_fn()
            catalog = getattr(tool_registry, "_catalog", None)
            iter_tools = getattr(tool_registry, "_iter_tools_for_prompt", None)
            augment = getattr(tool_registry, "_augment_selection_tool_visibility", None)
            if catalog is None or not callable(iter_tools):
                return [], []
            available_tool_names = [
                str(getattr(tool, "name", "") or "").strip()
                for tool in iter_tools()
                if str(getattr(tool, "name", "") or "").strip()
            ]
            forced_ids = list(dict.fromkeys(getattr(tool_registry, "_forced_skill_ids", []) or []))
            selection = catalog.route_query_to_skills(
                query,
                available_tool_names,
                top_k=3,
                forced_skill_ids=forced_ids,
            )
            if callable(augment):
                selection = augment(query, selection)
            skill_ids = [
                str(getattr(card, "id", "") or "").strip()
                for card in list(getattr(selection, "selected_skills", []) or [])
                if str(getattr(card, "id", "") or "").strip()
            ]
            selected_tools = [
                str(name).strip()
                for name in list(getattr(selection, "selected_tools", []) or [])
                if str(name).strip()
            ]
            return skill_ids, selected_tools
        except Exception:
            return [], []

    def _infer_observation_type(
        self,
        query: str,
        response: str,
        wrote_files: bool,
    ) -> str:
        text = f"{query}\n{response}".lower()
        if any(
            token in text
            for token in ("bug", "fix", "error", "failing", "broken", "regression", "wrong")
        ):
            return "bugfix"
        if any(
            token in text for token in ("refactor", "cleanup", "restructure", "rename", "simplif")
        ):
            return "refactor"
        if any(
            token in text
            for token in ("feature", "implement", "add support", "new capability", "introduc")
        ):
            return "feature"
        if any(
            token in text for token in ("decision", "trade-off", "tradeoff", "rationale", "why we")
        ):
            return "decision"
        if not wrote_files:
            return "discovery"
        return "change"

    def _derive_observation_title(
        self,
        query: str,
        response: str,
        paths: list[str],
    ) -> str:
        candidates: list[str] = []
        for raw_line in str(response or "").splitlines():
            line = re.sub(r"`+", "", raw_line).strip(" #-*>\t")
            if len(line) < 12:
                continue
            line = re.sub(r"^(implemented|updated|fixed|added|changed)\s+", "", line, flags=re.I)
            if line:
                candidates.append(line.rstrip("."))
        if candidates:
            return candidates[0][:100].rstrip()

        cleaned_query = re.sub(
            r"^(please|can you|could you|let'?s|lets|now[, ]+|we need to)\s+",
            "",
            str(query or "").strip(),
            flags=re.I,
        ).rstrip("?.! ")
        if cleaned_query:
            return cleaned_query[:100]

        if paths:
            short = ", ".join(Path(path).name for path in paths[:2])
            if len(paths) > 2:
                short += " and more"
            return f"{short} updated"
        return "Project memory update"

    def _record_turn_observation(
        self,
        *,
        query: str,
        response: str,
        session_id: str,
        tool_calls: list[dict[str, Any]],
        written_paths: list[str],
    ) -> None:
        if not self.cfg.get("project_memory_enabled", True):
            return
        if not written_paths:
            return
        try:
            from src.memory.project import compact_text, record_observation

            tool_names = [
                str(call.get("name", "") or "").strip()
                for call in tool_calls
                if str(call.get("name", "") or "").strip()
            ]
            unique_paths = list(dict.fromkeys(path for path in written_paths if path))
            obs_type = self._infer_observation_type(query, response, wrote_files=bool(unique_paths))
            title = self._derive_observation_title(query, response, unique_paths)
            body_lines = [
                f"**Request:** {compact_text(query, 500)}",
                f"**Outcome:** {compact_text(response, 1200)}",
                f"**Files:** {', '.join(unique_paths)}",
            ]
            if tool_names:
                body_lines.append(f"**Tools:** {', '.join(tool_names)}")
            record_observation(
                obs_type=obs_type,
                title=title,
                content="\n".join(body_lines),
                files=unique_paths,
                session_key=session_id,
            )
        except Exception:
            pass

    def _state_snapshot(
        self,
        *,
        include_repo_library: bool = True,
        include_rag_inventory: bool = True,
        include_context_preview: bool = False,
    ) -> dict[str, Any]:
        active = self.active or (next(iter(self.agents.keys()), "") if self.agents else "")
        sid = self.sessions.get(active, "")
        msgs = 0
        todo = []
        mounted_paths = []
        rag_docs = []
        active_repos = []
        retrieval_insights = []
        loaded_tools: list[str] = []
        loaded_skills: list[str] = []
        context_size = 0
        context_limit = 0
        context_token_budget = 0
        if active and sid and active in self.agents:
            try:
                ctx = self.agents[active].describe_runtime_context(sid)
                msgs = ctx.get("persisted_messages", 0)
                todo = ctx.get("todo", [])
                runtime = ctx.get("runtime", {}) if isinstance(ctx, dict) else {}
                mounted_paths = list(runtime.get("mounted_paths") or [])
                rag_docs = list(runtime.get("rag_docs") or [])
                active_repos = list(runtime.get("active_repos") or [])
                retrieval_insights = list(runtime.get("retrieval_insights") or [])
                context_size = int(
                    ctx.get("loaded_message_budget", ctx.get("persisted_messages", 0))
                )
                context_limit = int(ctx.get("history_limit", 0))
                if include_context_preview:
                    preview = self.agents[active].preview_prompt_context("", sid)
                    info = self.agents[active].describe_runtime_context(sid)
                    rendered = self._format_context_preview(preview, sid, info)
                    context_size = len(rendered)
                    context_limit = 0
                    context_token_budget = int(preview.get("context_token_budget", 0))
                else:
                    context_token_budget = int(
                        getattr(
                            getattr(self.agents[active], "config", None),
                            "context_token_budget",
                            0,
                        )
                        or 0
                    )
            except Exception:
                msgs = 0
                todo = []
                mounted_paths = []
                rag_docs = []
                active_repos = []
                retrieval_insights = []
                context_size = 0
                context_limit = 0
                context_token_budget = 0

        tool_count = 0
        skill_count = 0
        if active and active in self.agents:
            try:
                ag = self.agents[active]
                tool_registry = getattr(ag, "tools", None)
                if tool_registry is not None:
                    # ToolRegistry stores tools in ._tools dict
                    tool_dict = getattr(tool_registry, "_tools", None) or {}
                    tool_count = len(tool_dict)
                    loaded_tools = sorted(
                        dict.fromkeys(
                            str(getattr(tool, "name", "") or "").strip()
                            for tool in tool_dict.values()
                            if str(getattr(tool, "name", "") or "").strip()
                        )
                    )
                    # Avoid forcing skill-catalog hydration during startup/state
                    # snapshots. If the catalog has already been built elsewhere,
                    # report it; otherwise leave skills empty and keep init cheap.
                    catalog = getattr(tool_registry, "_catalog", None)
                    if catalog is not None:
                        skill_dict = getattr(catalog, "_skills", None) or {}
                        skill_count = len(skill_dict)
                        loaded_skills = sorted(
                            dict.fromkeys(
                                str(
                                    getattr(card, "name", None) or getattr(card, "id", "") or ""
                                ).strip()
                                for card in skill_dict.values()
                                if str(
                                    getattr(card, "name", None) or getattr(card, "id", "") or ""
                                ).strip()
                            )
                        )
            except Exception:
                pass
        repo_library = self._repo_library() if include_repo_library else []
        rag_inventory = self._rag_inventory_snapshot(
            self.agents.get(active),
            active_repos,
            repo_library if include_rag_inventory else [],
            retrieval_insights,
            rag_docs,
        )
        return {
            "active": active,
            "session": sid,
            "msg_count": msgs,
            "context_size": context_size,
            "context_limit": context_limit,
            "context_token_budget": context_token_budget,
            "agents": sorted(self.agents.keys()),
            "mcp_servers": self._active_mcp_names(),
            "pipeline": self.pipeline,
            "rapidfuzz": _has_rapidfuzz(),
            "tiktoken": _has_tiktoken(),
            "tool_count": tool_count,
            "skill_count": skill_count,
            "loaded_tools": loaded_tools,
            "loaded_skills": loaded_skills,
            "todo": todo,
            "active_repos": active_repos,
            "retrieval_insights": retrieval_insights,
            "repo_library": repo_library,
            "mounted_paths": mounted_paths,
            "rag_docs": rag_docs,
            "rag_inventory": rag_inventory,
            "recent_rag_queries": self._recent_rag_query_items(active, sid),
            "tool_call_count": self._session_tool_call_count(active, sid),
            "plan_mode": self._session_plan_mode(active, sid),
        }

    def _startup_context(
        self,
        *,
        include_memory_summary: bool = True,
        include_hooks: bool = True,
        fast_hooks: bool = False,
    ) -> dict[str, Any]:
        if include_hooks:
            if fast_hooks:
                self._start_startup_hooks_async("startup")
            else:
                self._start_startup_hooks_async("startup")
                self._wait_for_startup_hooks()

        hook_context, hook_count, hook_context_complete, _hook_errors = (
            self._startup_hook_snapshot()
        )

        memory_summary = (
            self._parse_memory_summary(exclude_session_start_hook_plugins=False)
            if include_memory_summary
            else {
                "enabled": bool(self.cfg.get("project_memory_enabled", True)),
                "has_memories": False,
                "total_entries": 0,
                "obs_count": 0,
                "facts_count": 0,
                "sections": [],
            }
        )
        return {
            "memory_summary": memory_summary,
            "hook_context": hook_context,
            "hook_count": hook_count,
            "hook_context_complete": hook_context_complete,
            "project_memory_enabled": bool(self.cfg.get("project_memory_enabled", True)),
        }

    def init(self, params: dict[str, Any]) -> dict[str, Any]:
        cp = params.get("config_path")
        config_path = _resolve_config_path(cp)
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")

        fast_init = bool(params.get("fast", True))
        self.cfg, self.agents = self._load_agents(config_path, fast_init=fast_init)
        first = next(iter(self.agents.keys()))
        self.active = first
        self.sessions = {name: _new_session_id() for name in self.agents.keys()}
        self.pipeline = None
        self._invalidate_startup_hook_cache()
        if self.cfg.get("project_memory_enabled", True):
            self._inject_project_memory(exclude_session_start_hook_plugins=False)
        if fast_init:
            self._start_startup_hooks_async("startup")
            self._warm_mcp_servers_async()

        return {
            "config_path": str(config_path),
            "state": self._state_snapshot(
                include_repo_library=not fast_init,
                include_rag_inventory=not fast_init,
            ),
            "commands": _bridge_commands_manifest(),
            **self._startup_context(
                include_memory_summary=True,
                include_hooks=True,
                fast_hooks=fast_init,
            ),
            "fast_init": fast_init,
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
            self._fire_session_end_hooks("exit")
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
            snap = self._state_snapshot(include_context_preview=True)
            agents = ", ".join(snap.get("agents", []) or []) or "-"
            mcps = ", ".join(snap.get("mcp_servers", []) or []) or "-"
            rag = dict(snap.get("rag_inventory") or {})
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
                    f"ctx: {snap.get('context_size', 0)}",
                    f"agents: {agents}",
                    f"mcp: {mcps}",
                    f"pipeline: {ptxt}",
                    f"rapidfuzz: {'enabled' if snap.get('rapidfuzz') else 'disabled'}",
                    f"tiktoken: {'enabled' if snap.get('tiktoken') else 'disabled'}",
                    f"repos: {len(snap.get('repo_library', []) or [])}",
                    f"active_repos: {len(snap.get('active_repos', []) or [])}",
                    f"rag_backend: {rag.get('vector_backend', 'unknown')}",
                    f"rag_chunks: {rag.get('repo_chunks', 0)}",
                ]
            )
        elif cmd == "/context":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            mode = (args[0].lower() if args else "").strip()
            query = ""
            if mode in {"preview", "prompt"}:
                query = " ".join(args[1:]).strip()
            preview = ag.preview_prompt_context(query, sid)
            info = ag.describe_runtime_context(sid)
            rendered = self._format_context_preview(preview, sid, info)
            out.extend(rendered.split("\n"))

        elif cmd == "/compact":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            keep = int(args[0]) if args else None
            result = ag.compact_session(sid, keep_last_messages=keep)
            out.append(result.get("message", "done"))

        elif cmd == "/reset":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "")
            ag.reset_runtime_state(sid, reason="clear")
            out.append("runtime state cleared")

        elif cmd == "/new":
            if self.active is None:
                out.append("No active agent")
            else:
                old_sid = self.sessions.get(self.active, "")
                self.sessions[self.active] = _new_session_id()
                ag = self._active_agent()
                try:
                    ag.detach_runtime_state(session_id=old_sid, reason="clear")
                except Exception:
                    pass
                self.pipeline = None
                out.append(f"new session · {self.sessions[self.active]}")

        elif cmd == "/reload":
            self._fire_session_end_hooks("reload")
            cp = _resolve_config_path(params.get("config_path"))
            self.cfg, self.agents = self._load_agents(cp)
            self.active = next(iter(self.agents.keys()))
            self.sessions = {name: _new_session_id() for name in self.agents.keys()}
            self.pipeline = None
            self._invalidate_startup_hook_cache()
            if self.cfg.get("project_memory_enabled", True):
                self._inject_project_memory(exclude_session_start_hook_plugins=False)
            out.append(f"reloaded · {len(self.agents)} agent(s)")

        elif cmd == "/sessions":
            ag = self._active_agent()
            sess_list = ag.list_sessions()
            if not sess_list:
                out.append("no sessions found")
            else:
                out.append(f"sessions ({len(sess_list)}):")
                for sid, ts in sess_list:
                    mark = " ◀" if sid == self.sessions.get(self.active or "", "") else ""
                    short = sid[:24] + ("…" if len(sid) > 24 else "")
                    out.append(f"  {short}  {ts}{mark}")

        elif cmd == "/load":
            if not args:
                out.append("usage: /load <session_id>")
            else:
                ag = self._active_agent()
                sess_list = ag.list_sessions()
                matched = [s for s, _ in sess_list if s == args[0] or s.startswith(args[0])]
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
            parsed_mount = _parse_mount_args(args)
            if parsed_mount.get("status") != "ok":
                out.append(str(parsed_mount.get("error", "invalid /mount arguments")))
            else:
                ag = self._active_agent()
                sid = self.sessions.get(self.active or "", "") or None
                sid_text = str(sid or self.sessions.get(self.active or "", "") or "-")
                directory = str(parsed_mount["directory"])
                glob_pattern = str(parsed_mount["glob"])
                exclude = str(parsed_mount.get("exclude", ""))
                exclude_display = list(parsed_mount.get("exclude_display", []))

                max_files = 120
                raw_max_files = parsed_mount.get("max_files")
                if raw_max_files is not None:
                    try:
                        max_files = max(1, min(400, int(raw_max_files)))
                    except Exception:
                        out.append(f"invalid max_files '{raw_max_files}', using 120")

                map_depth = 3
                raw_map_depth = parsed_mount.get("map_depth")
                if raw_map_depth is not None:
                    try:
                        map_depth = max(1, min(10, int(raw_map_depth)))
                    except Exception:
                        out.append(f"invalid map_depth '{raw_map_depth}', using 3")

                project_map_payload: dict[str, Any]
                try:
                    project_map_raw = ag.run_tool_direct(
                        "get_project_map",
                        {
                            "directory": directory,
                            "max_depth": map_depth,
                            "exclude": exclude,
                        },
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
                            "exclude": exclude,
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
                                "exclude": exclude,
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
                if exclude_display:
                    out.append(f"excluded: {', '.join(exclude_display)}")
                out.append(f"session: {sid_text}")

                if str(project_map_payload.get("status", "")).lower() == "ok":
                    file_count = _safe_int(project_map_payload.get("file_count"))
                    token_count = _estimate_token_count(project_map_payload)
                    _remember_mount_context(
                        ag,
                        path=directory,
                        glob=glob_pattern,
                        file_count=file_count,
                        token_count=token_count,
                        map_depth=map_depth,
                    )
                    out.append(
                        "context persisted: yes "
                        f"(project map: {file_count} files, depth={map_depth})"
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
                    _remember_rag_doc(
                        ag,
                        path=directory,
                        source_label="mounted codebase",
                        chunks=chunks_added,
                        token_count=_estimate_token_count(promote_payload),
                        kind="mount",
                    )
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
                parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                if parsed.get("status") == "ok":
                    _remember_rag_doc(
                        ag,
                        path=str(parsed.get("path", file_path)),
                        source_label=source_label,
                        chunks=_safe_int(parsed.get("chunks_added")),
                        token_count=_estimate_token_count(parsed),
                        kind="file",
                    )
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
                parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                if parsed.get("status") == "ok":
                    _remember_rag_doc(
                        ag,
                        path=directory,
                        source_label="bulk upload",
                        chunks=_safe_int(parsed.get("total_chunks_added")),
                        token_count=_estimate_token_count(parsed),
                        kind="directory",
                    )
                    out.append(
                        "upload-dir done "
                        f"({parsed.get('files_processed', 0)} files, {parsed.get('total_chunks_added', 0)} chunks)"
                    )
                else:
                    out.append(f"upload-dir failed: {parsed.get('error', 'unknown error')}")

        elif cmd == "/repo":
            ag = self._active_agent()
            sid = self.sessions.get(self.active or "", "") or None
            subcmd = args[0].lower() if args else "list"
            repo_library = self._repo_library()
            active_runtime = _active_repo_records(ag)
            active_ids = {
                str(item.get("id") or "").strip()
                for item in active_runtime
                if str(item.get("id") or "").strip()
            }

            if subcmd == "list":
                if not repo_library:
                    out.append("no repos registered yet")
                    out.append("usage: /repo add <path> [name]")
                else:
                    out.append(f"repos ({len(repo_library)}):")
                    for repo in repo_library:
                        repo_id = str(repo.get("id") or "").strip()
                        name = str(repo.get("name") or repo_id or "repo").strip()
                        path = str(repo.get("path") or "").strip()
                        files_processed = _safe_int(repo.get("files_processed"))
                        chunks_added = _safe_int(repo.get("chunks_added"))
                        mark = " *active" if repo_id in active_ids else ""
                        out.append(f"  {repo_id} · {name}{mark}")
                        out.append(f"    path: {path}")
                        details = []
                        if files_processed > 0:
                            details.append(f"files={files_processed}")
                        if chunks_added > 0:
                            details.append(f"chunks={chunks_added}")
                        graph_nodes = _safe_int(repo.get("graph_nodes"))
                        graph_edges = _safe_int(repo.get("graph_edges"))
                        if graph_nodes > 0:
                            details.append(f"graph_nodes={graph_nodes}")
                        if graph_edges > 0:
                            details.append(f"graph_edges={graph_edges}")
                        git = dict(repo.get("git") or {})
                        branch = str(git.get("branch") or "").strip()
                        commit = str(git.get("commit") or "").strip()
                        if branch:
                            details.append(f"branch={branch}")
                        if commit:
                            details.append(f"commit={commit}")
                        if details:
                            out.append(f"    {'  '.join(details)}")
                if active_runtime:
                    preview = ", ".join(
                        str(item.get("id") or item.get("name") or "").strip()
                        for item in active_runtime[:8]
                        if str(item.get("id") or item.get("name") or "").strip()
                    )
                    if preview:
                        out.append(f"active now: {preview}")

            elif subcmd == "add":
                if len(args) < 2:
                    out.append("usage: /repo add <path> [name]")
                else:
                    repo_path = args[1]
                    repo_name = " ".join(args[2:]).strip()
                    try:
                        from src.repo.registry import register_repo, update_repo

                        repo = register_repo(repo_path, name=repo_name)
                        repo = (
                            update_repo(
                                str(repo.get("id") or ""),
                                last_used_at=datetime.datetime.now(datetime.timezone.utc)
                                .replace(microsecond=0)
                                .isoformat()
                                .replace("+00:00", "Z"),
                            )
                            or repo
                        )
                        merged = [repo]
                        merged.extend(
                            item
                            for item in active_runtime
                            if str(item.get("id") or "").strip()
                            != str(repo.get("id") or "").strip()
                        )
                        _set_active_repo_records(ag, merged)
                        merged, auto_reports = self._auto_ingest_repos(
                            ag,
                            merged,
                            glob_pattern="**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}",
                            max_files=120,
                        )
                        _set_active_repo_records(ag, merged)
                        self._persist_active_runtime_state()
                        out.append(f"repo added: {repo.get('id', '?')} · {repo.get('name', '?')}")
                        out.append(f"path: {repo.get('path', '-')}")
                        out.append("active for this session: yes")
                        if auto_reports:
                            report = auto_reports[0]
                            payload = dict(report.get("payload") or {})
                            ingest = dict(payload.get("ingest") or {})
                            out.append(
                                "auto-ingested: "
                                f"{_safe_int(ingest.get('files_processed'))} files · "
                                f"{_safe_int(ingest.get('total_chunks_added'))} chunks"
                            )
                        else:
                            out.append("next: /repo ingest <repo_id>")
                    except Exception as exc:
                        out.append(f"repo add failed: {exc}")

            elif subcmd == "use":
                if len(args) < 2:
                    if not active_runtime:
                        out.append("no active repos in this session")
                    else:
                        out.append("active repos:")
                        for repo in active_runtime:
                            out.append(
                                f"  {repo.get('id', '?')} · {repo.get('name', '?')} · {repo.get('path', '-')}"
                            )
                else:
                    refs = args[1:]
                    if len(refs) == 1 and refs[0].lower() in {"none", "clear", "off"}:
                        _set_active_repo_records(ag, [])
                        self._persist_active_runtime_state()
                        out.append("active repos cleared for this session")
                    else:
                        resolved: list[dict[str, Any]] = []
                        missing: list[str] = []
                        for ref in refs:
                            repo = self._resolve_repo_ref(ref)
                            if repo is None:
                                missing.append(ref)
                                continue
                            if not any(
                                str(item.get("id") or "").strip()
                                == str(repo.get("id") or "").strip()
                                for item in resolved
                            ):
                                resolved.append(repo)
                                from src.repo.registry import update_repo

                                update_repo(
                                    str(repo.get("id") or ""),
                                    last_used_at=datetime.datetime.now(datetime.timezone.utc)
                                    .replace(microsecond=0)
                                    .isoformat()
                                    .replace("+00:00", "Z"),
                                )
                        if missing:
                            out.append("missing repos: " + ", ".join(missing))
                        if resolved:
                            resolved, auto_reports = self._auto_ingest_repos(
                                ag,
                                resolved,
                                glob_pattern="**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}",
                                max_files=120,
                            )
                            _set_active_repo_records(ag, resolved)
                            self._persist_active_runtime_state()
                            out.append(
                                "active repos set: "
                                + ", ".join(
                                    str(item.get("id") or item.get("name") or "?")
                                    for item in resolved
                                )
                            )
                            for report in auto_reports:
                                payload = dict(report.get("payload") or {})
                                ingest = dict(payload.get("ingest") or {})
                                repo_report = dict(report.get("repo") or {})
                                status = str(payload.get("status") or "").lower()
                                if status in {"ok", "partial"}:
                                    out.append(
                                        "auto-ingested: "
                                        f"{repo_report.get('id', '?')} · "
                                        f"{_safe_int(ingest.get('files_processed'))} files · "
                                        f"{_safe_int(ingest.get('total_chunks_added'))} chunks"
                                    )
                                else:
                                    out.append(
                                        "auto-ingest failed: "
                                        f"{repo_report.get('id', '?')} · "
                                        f"{payload.get('errors', [payload.get('error', 'unknown error')])[0]}"
                                    )
                        elif not missing:
                            out.append("no repos resolved")

            elif subcmd == "ingest":
                if len(args) < 2:
                    out.append("usage: /repo ingest <repo_id> [glob] [max_files]")
                else:
                    repo = self._resolve_repo_ref(args[1])
                    if repo is None:
                        out.append(f"unknown repo: {args[1]}")
                    else:
                        glob_pattern = (
                            args[2]
                            if len(args) > 2
                            else "**/*.{py,rs,ts,tsx,js,jsx,java,go,rb,php,c,cc,cpp,h,hpp,cs,kt,swift,md,toml,yaml,yml,json,sql,sh}"
                        )
                        try:
                            max_files = max(1, min(400, int(args[3]))) if len(args) > 3 else 120
                        except Exception:
                            max_files = 120
                            out.append(f"invalid max_files '{args[3]}', using 120")

                        try:
                            parsed, updated = self._ingest_repo_record(
                                ag,
                                repo,
                                glob_pattern=glob_pattern,
                                max_files=max_files,
                            )
                        except Exception as exc:
                            parsed = {"status": "error", "error": str(exc)}
                            updated = repo

                        if str(parsed.get("status", "")).lower() in {"ok", "partial"}:
                            ingest_payload = dict(parsed.get("ingest") or {})
                            graph_payload = dict(parsed.get("graph") or {})
                            files_processed = _safe_int(ingest_payload.get("files_processed"))
                            chunks_added = _safe_int(ingest_payload.get("total_chunks_added"))
                            merged = [updated]
                            merged.extend(
                                item
                                for item in active_runtime
                                if str(item.get("id") or "").strip()
                                != str(updated.get("id") or "").strip()
                            )
                            _set_active_repo_records(ag, merged)
                            _remember_rag_doc(
                                ag,
                                path=str(updated.get("path") or ""),
                                source_label=f"repo {updated.get('name', updated.get('id', ''))}",
                                chunks=chunks_added,
                                token_count=_estimate_token_count(parsed),
                                kind="repo",
                            )
                            self._persist_active_runtime_state()
                            out.append(
                                f"repo ingested: {updated.get('id', '?')} · {files_processed} files · {chunks_added} chunks"
                            )
                            if str(graph_payload.get("status", "")).lower() == "ok":
                                out.append(
                                    "graph built: "
                                    f"{_safe_int(graph_payload.get('nodes'))} nodes · "
                                    f"{_safe_int(graph_payload.get('edges'))} edges · "
                                    f"{_safe_int(graph_payload.get('symbols'))} symbols"
                                )
                            else:
                                out.append(
                                    "graph build failed: "
                                    f"{graph_payload.get('error', 'unknown error')}"
                                )
                            out.append("active for this session: yes")
                        else:
                            out.append(
                                f"repo ingest failed: {parsed.get('error', 'unknown error')}"
                            )

            elif subcmd == "remove":
                if len(args) < 2:
                    out.append("usage: /repo remove <repo_id>")
                else:
                    repo = self._resolve_repo_ref(args[1])
                    if repo is None:
                        out.append(f"unknown repo: {args[1]}")
                    else:
                        from src.repo.registry import remove_repo

                        removed = remove_repo(str(repo.get("id") or ""))
                        if removed is None:
                            out.append(f"repo not found: {args[1]}")
                        else:
                            kept_active = [
                                item
                                for item in active_runtime
                                if str(item.get("id") or "").strip()
                                != str(removed.get("id") or "").strip()
                            ]
                            _set_active_repo_records(ag, kept_active)
                            self._persist_active_runtime_state()
                            out.append(f"repo removed from library: {removed.get('id', '?')}")
                            out.append("note: existing indexed RAG chunks were left intact")

            else:
                out.append("usage: /repo add|list|use|ingest|remove ...")

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
                            if "resolve" in tool.name.lower() and "library" in tool.name.lower()
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
                            if any(token in lname for token in ("query", "library", "name", "id")):
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
                                        and str(docs_payload.get("status", "")).lower() == "error"
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
                                                docs_text[:12000].rstrip() + "\n...[truncated]"
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
                files.extend(str(item).strip() for item in untracked_items if str(item).strip())
                uniq_files = [f for f in dict.fromkeys(files) if f]
                if uniq_files:
                    preview = ", ".join(uniq_files[:20])
                    if len(uniq_files) > 20:
                        preview += f" … (+{len(uniq_files) - 20} more)"
                    out.append(f"files: {preview}")
            else:
                out.append(f"git_status failed: {status.get('error', 'unknown error')}")

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
            "state": self._state_snapshot(include_context_preview=True),
            "exit": should_exit,
        }

    def chat(self, params: dict[str, Any]) -> dict[str, Any]:
        raw = str(params.get("message", "")).strip()
        if not raw:
            return {"messages": [], "state": self._state_snapshot(include_context_preview=True)}

        active = self.active
        if active is None:
            raise RuntimeError("No active agent")

        sid = self.sessions.setdefault(active, _new_session_id())
        token_seen = False
        thinking_token_seen = False
        emitted_image_events: set[tuple[str, str]] = set()
        turn_tool_errors: list[int] = [0]  # mutable counter accessible in nested fn
        written_paths: list[str] = []
        token_buffer: list[str] = []
        thinking_buffer: list[str] = []

        def _parse_result_payload(payload: Any) -> dict[str, Any] | None:
            if isinstance(payload, dict):
                return payload
            if not isinstance(payload, str):
                return None
            try:
                parsed = json.loads(payload)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        def _flush_token_buffer() -> None:
            if not token_buffer:
                return
            chunk = "".join(token_buffer)
            token_buffer.clear()
            if chunk:
                self._emit("token", {"token": chunk})

        def _flush_thinking_buffer() -> None:
            if not thinking_buffer:
                return
            chunk = "".join(thinking_buffer)
            thinking_buffer.clear()
            if chunk:
                self._emit("thinking_token", {"token": chunk})

        def _flush_stream_buffers() -> None:
            _flush_thinking_buffer()
            _flush_token_buffer()

        def _token(tok: str):
            nonlocal token_seen
            if not token_seen:
                token_seen = True
                _phase("streaming", "model output")
                self._emit("token", {"token": tok})
                return
            token_buffer.append(tok)
            if "\n" in tok or sum(len(part) for part in token_buffer) >= _STREAM_EVENT_CHAR_BATCH:
                _flush_token_buffer()

        def _phase(state: str, note: str):
            nonlocal thinking_token_seen
            _flush_stream_buffers()
            if state != "thinking":
                thinking_token_seen = False
            self._emit("phase", {"state": state, "note": note})

        def _thinking_token(tok: str):
            nonlocal thinking_token_seen
            if not thinking_token_seen:
                thinking_token_seen = True
                _phase("thinking", "pre-turn plan")
                self._emit("thinking_token", {"token": tok})
                return
            thinking_buffer.append(tok)
            if (
                "\n" in tok
                or sum(len(part) for part in thinking_buffer) >= _STREAM_EVENT_CHAR_BATCH
            ):
                _flush_thinking_buffer()

        def _emit_file_diff(name: str, tool_args: dict[str, Any], info: dict[str, Any]):
            """Emit an exact file_diff event based on the tool's real result payload."""
            status_val = str(info.get("status", "ok") or "ok").strip().lower()
            if status_val != "ok":
                return

            payload = _parse_result_payload(info.get("result_output"))
            if not isinstance(payload, dict):
                return

            diff_text = str(payload.get("diff", "") or "").strip()
            if not diff_text:
                return

            path = str(
                payload.get("path") or tool_args.get("path") or tool_args.get("file_path") or ""
            ).strip()
            if not path:
                return

            if path not in written_paths:
                written_paths.append(path)

            self._emit(
                "file_diff",
                {
                    "tool": name,
                    "path": path,
                    "diff": diff_text,
                },
            )

        def _tool(
            name: str,
            tool_args: dict[str, Any],
            meta: dict[str, Any] | None = None,
        ):
            info = meta if isinstance(meta, dict) else {}
            stage = str(info.get("stage", "start") or "start").strip().lower()
            sequence = int(info.get("sequence", 0) or 0)
            _flush_stream_buffers()
            if stage == "end":
                status_val = str(info.get("status", "ok") or "ok")
                if status_val.lower() in ("error", "failed"):
                    turn_tool_errors[0] += 1
                payload: dict[str, Any] = {
                    "name": name,
                    "sequence": sequence,
                    "status": status_val,
                    "duration_ms": int(info.get("duration_ms", 0) or 0),
                    "cache_hit": bool(info.get("cache_hit", False)),
                    "args": tool_args,  # Include args for consistency with tool_start
                }
                error = str(info.get("error", "") or "").strip()
                if error:
                    payload["error"] = error
                result_preview = str(info.get("result_preview", "") or "").strip()
                if result_preview:
                    payload["result_preview"] = result_preview[:160]

                # Emit exact file diffs for write tools when their payload includes one.
                _emit_file_diff(name, tool_args, info)
                if (
                    not written_paths
                    and name
                    in {
                        "write_file",
                        "edit_file",
                        "apply_edit_block",
                        "smart_edit",
                        "edit_file_libcst",
                        "replace_function_body",
                        "replace_docstring",
                        "replace_decorators",
                        "replace_argument",
                        "insert_after_function",
                        "delete_function",
                    }
                    and str(info.get("status", "ok") or "").lower() == "ok"
                ):
                    fallback_path = str(
                        tool_args.get("path") or tool_args.get("file_path") or ""
                    ).strip()
                    if fallback_path:
                        written_paths.append(fallback_path)

                self._emit("tool_end", payload)
                return

            # Update phase note so the status bar shows the active tool name
            _phase(
                "jambering",
                f"tool#{sequence} {name}",
            )
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

        def _repair(meta: dict[str, Any] | None = None):
            info = meta if isinstance(meta, dict) else {}
            stage = str(info.get("stage", "") or "").strip().lower()
            if not stage:
                return
            _flush_stream_buffers()
            payload = {
                "stage": stage,
                "attempt": int(info.get("attempt", 0) or 0),
                "tool": str(info.get("tool", "") or "unknown"),
                "error_type": str(info.get("error_type", "") or ""),
                "message": str(info.get("message", "") or ""),
            }
            if stage == "attempt":
                _phase(
                    "thinking",
                    f"repairing {payload['tool']}",
                )
            self._emit("tool_repair", payload)

        def _decision(meta: dict[str, Any] | None = None):
            info = meta if isinstance(meta, dict) else {}
            stage = str(info.get("stage", "") or "").strip().lower()
            if not stage:
                return
            _flush_stream_buffers()
            payload = {
                "mode": str(info.get("mode", "") or "hybrid_decision"),
                "stage": stage,
                "message": str(info.get("message", "") or ""),
            }
            if stage == "stream_answer":
                _phase(
                    "streaming",
                    "streamed answer mode",
                )
            self._emit("decision", payload)

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
                        )
                    except Exception as exc:
                        resp = f"[error] {exc}"
                    turns.append({"agent": str(ag_name), "text": str(resp)})
                    message = str(resp)
            state = self._state_snapshot(include_context_preview=True)
            return {"pipeline": True, "turns": turns, "state": state}

        ag = self._active_agent()
        from src.agent.classify import classify_turn

        classification = classify_turn(raw)
        if classification.intent not in {"social", "informational"}:
            skill_ids, selected_tools = self._preview_skill_selection(ag, raw)
            if skill_ids or selected_tools:
                self._emit(
                    "skill",
                    {
                        "skill_ids": skill_ids,
                        "selected_tools": selected_tools,
                    },
                )
        _phase("thinking", "running agent")
        run_resp = ag.run(
            raw,
            session_id=sid,
            stream_callback=_token,
            thinking_callback=_thinking_token,
            tool_callback=_tool,
            repair_callback=_repair,
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
        _flush_stream_buffers()
        # Some backends/modes return a full answer without incremental token callbacks
        # (for example constrained decoding or non-streaming server paths). Emit a
        # synthetic token stream so the TUI still shows streaming output.
        if not token_seen and response_text:
            _phase(
                "streaming",
                "rendering response",
            )
            text = response_text
            chunk_size = 96
            for i in range(0, len(text), chunk_size):
                self._emit("token", {"token": text[i : i + chunk_size]})
        self._record_turn_observation(
            query=raw,
            response=response_text,
            session_id=sid,
            tool_calls=tool_calls_payload,
            written_paths=written_paths,
        )
        return {
            "pipeline": False,
            "assistant": response_text,
            "final_response": response_text,
            "iterations": int(getattr(run_resp, "iterations", 0) or 0),
            "tool_calls": tool_calls_payload,
            "tool_errors": turn_tool_errors[0],
            "session_id": sid,
            "message_count": len(list(getattr(run_resp, "messages", []) or [])),
            "state": self._state_snapshot(include_context_preview=True),
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
                # When asked for state over RPC (e.g. from rust-cli), include a
                # context preview so callers receive system prompt and message
                # window information without requiring additional params.
                result = server._state_snapshot(
                    include_repo_library=bool(params.get("include_repo_library", True)),
                    include_rag_inventory=bool(params.get("include_rag_inventory", True)),
                    include_context_preview=True,
                )
            elif method == "startup_context":
                result = server._startup_context(
                    include_memory_summary=bool(params.get("include_memory_summary", True)),
                    include_hooks=bool(params.get("include_hooks", True)),
                    fast_hooks=bool(params.get("fast_hooks", False)),
                )
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
