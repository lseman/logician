from __future__ import annotations

import json
import queue
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.factory import create_agent
from ..tools.core.tasks import todo
from .data import (
    extract_repo_focus_paths,
    get_current_todos,
    get_memory_overview,
    get_repo_focus_graph,
    get_repo_graph,
    get_session_detail,
    list_memory_facts,
    list_memory_observations,
    list_repos,
    list_sessions,
    rename_session,
    search_rag,
    truncate_session_at,
)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

try:
    import multipart  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    multipart = None  # type: ignore


_AGENT_LOCK = Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    fresh_session: bool = False
    repo_id: str | None = None


class SessionPatchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class SessionTruncateRequest(BaseModel):
    at_id: int = Field(ge=0)


class TodoRequest(BaseModel):
    command: str = "view"
    items: list[dict[str, Any]] | None = None
    id: int | None = None
    status: str | None = None
    title: str | None = None
    note: str | None = None


class RagRequest(BaseModel):
    query: str = Field(min_length=1)
    repo_id: str | None = None
    n_results: int = 8


@lru_cache(maxsize=1)
def _build_agent():
    return create_agent()


def create_app() -> FastAPI:
    app = FastAPI(title="Logician Web UI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    build_dir = Path(__file__).resolve().parents[2] / "webui" / "dist"
    assets_dir = build_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/backend/status")
    def backend_status() -> dict[str, Any]:
        agent = _build_agent()
        llm_url = str(agent.config.llama_cpp_url or "").rstrip("/")
        payload: dict[str, Any] = {
            "web": "ok",
            "llm_url": llm_url,
            "llm_reachable": False,
            "detail": "",
        }
        if not llm_url:
            payload["detail"] = "No llama.cpp URL configured."
            return payload
        if httpx is None:
            payload["detail"] = "httpx is not installed in the backend environment."
            return payload

        candidate_urls = [
            f"{llm_url}/health",
            f"{llm_url}/v1/models",
            llm_url,
        ]
        for candidate in candidate_urls:
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(candidate)
                if response.status_code < 500:
                    payload["llm_reachable"] = True
                    payload["detail"] = f"Connected via {candidate}"
                    return payload
            except Exception as exc:
                payload["detail"] = str(exc)
        return payload

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return {
            "sessions": list_sessions(limit=12),
            "todos": get_current_todos(),
            "memory": get_memory_overview(),
            "repos": list_repos(),
        }

    @app.get("/api/sessions")
    def sessions() -> dict[str, Any]:
        return {"sessions": list_sessions()}

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str) -> dict[str, Any]:
        return get_session_detail(session_id)

    @app.patch("/api/sessions/{session_id}")
    def patch_session(session_id: str, request: SessionPatchRequest) -> dict[str, Any]:
        rename_session(session_id, request.title)
        return {"id": session_id, "title": request.title.strip()}

    @app.post("/api/sessions/{session_id}/truncate")
    def truncate_session(session_id: str, request: SessionTruncateRequest) -> dict[str, Any]:
        truncate_session_at(session_id, request.at_id)
        return {"ok": True}

    if multipart is not None:

        @app.post("/api/upload")
        async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
            content = await file.read()
            filename = str(file.filename or "file")

            if filename.lower().endswith(".pdf"):
                try:
                    import io

                    import pypdf  # type: ignore[import]
                    reader = pypdf.PdfReader(io.BytesIO(content))
                    pages = [page.extract_text() or "" for page in reader.pages]
                    text = "\n\n".join(p.strip() for p in pages if p.strip())
                    if not text:
                        text = "[PDF contained no extractable text]"
                except ImportError:
                    text = "[PDF extraction requires pypdf: pip install pypdf]"
                except Exception as exc:
                    text = f"[PDF extraction failed: {exc}]"
                return {"text": text, "name": filename, "kind": "pdf", "pages": len(reader.pages) if "reader" in dir() else 0}

            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = "[Binary file — cannot display as text]"
            return {"text": text, "name": filename, "kind": "text"}

    else:

        @app.post("/api/upload")
        async def upload_file_unavailable() -> dict[str, Any]:
            raise HTTPException(
                status_code=503,
                detail="File uploads require python-multipart in the backend environment.",
            )

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            with _AGENT_LOCK:
                response = _build_agent().run(
                    request.message,
                    session_id=request.session_id,
                    fresh_session=request.fresh_session,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        session_id = str(response.debug.get("session_id") or request.session_id or "")
        detail = get_session_detail(session_id)
        return {
            "session_id": session_id,
            "assistant_message": response.final_response,
            "thinking_log": list(response.thinking_log or []),
            "messages": detail.get("messages") or [],
            "todos": detail.get("todos") or get_current_todos(),
            "runtime": detail.get("runtime") or {},
        }

    def _chat_stream_response(
        *,
        message: str,
        session_id: str | None = None,
        fresh_session: bool = False,
        repo_id: str | None = None,
    ) -> StreamingResponse:
        event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        selected_repo = str(repo_id or "").strip()
        resolved_session_id = str(session_id or uuid.uuid4())

        def _push(event: str, data: dict[str, Any]) -> None:
            event_queue.put({"event": event, "data": data})

        def _maybe_push_graph(
            *,
            query_text: str = "",
            focus_paths: list[str] | None = None,
        ) -> None:
            if not selected_repo:
                return
            try:
                graph_payload = get_repo_focus_graph(
                    selected_repo,
                    focus_paths=focus_paths,
                    query=query_text,
                )
            except Exception:
                return
            _push("graph", graph_payload)

        def _run_chat() -> None:
            try:
                _push("session", {"session_id": resolved_session_id})
                if selected_repo:
                    _maybe_push_graph(query_text=message)

                token_buffer = {"text": ""}

                def _on_token(token: str) -> None:
                    token_buffer["text"] += token
                    _push("token", {"token": token, "text": token_buffer["text"]})

                def _on_thinking(content: str) -> None:
                    _push("thinking", {"content": content})

                def _on_tool(
                    tool_name: str,
                    arguments: dict[str, Any],
                    meta: dict[str, Any],
                ) -> None:
                    stage = str((meta or {}).get("stage") or "")
                    if stage == "start":
                        # Reset the token buffer so the next LLM iteration streams
                        # clean text without the previous tool-call markup.
                        token_buffer["text"] = ""
                        _push("token_reset", {})
                    payload = {
                        "tool": tool_name,
                        "arguments": dict(arguments or {}),
                        "meta": dict(meta or {}),
                    }
                    _push("tool", payload)
                    focus_paths = extract_repo_focus_paths(
                        selected_repo,
                        arguments=arguments,
                        result_output=str(meta.get("result_output") or ""),
                    )
                    if focus_paths:
                        _maybe_push_graph(
                            query_text=message,
                            focus_paths=focus_paths,
                        )

                with _AGENT_LOCK:
                    response = _build_agent().run(
                        message,
                        session_id=resolved_session_id,
                        fresh_session=fresh_session,
                        stream_callback=_on_token,
                        thinking_callback=_on_thinking,
                        tool_callback=_on_tool,
                    )
                resolved_session_id_final = str(
                    response.debug.get("session_id") or resolved_session_id
                )
                detail = get_session_detail(resolved_session_id_final)
                _push(
                    "done",
                    {
                        "session_id": resolved_session_id_final,
                        "assistant_message": response.final_response,
                        "thinking_log": list(response.thinking_log or []),
                        "messages": detail.get("messages") or [],
                        "todos": detail.get("todos") or get_current_todos(),
                        "runtime": detail.get("runtime") or {},
                    },
                )
            except Exception as exc:
                _push("agent_error", {"message": str(exc)})
            finally:
                event_queue.put(None)

        def _iter_events():
            worker = Thread(target=_run_chat, daemon=True)
            worker.start()
            while True:
                item = event_queue.get()
                if item is None:
                    break
                yield (
                    f"event: {item['event']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            _iter_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/chat/stream")
    def chat_stream(
        message: str = Query(min_length=1),
        session_id: str | None = None,
        fresh_session: bool = False,
        repo_id: str | None = None,
    ) -> StreamingResponse:
        return _chat_stream_response(
            message=message,
            session_id=session_id,
            fresh_session=fresh_session,
            repo_id=repo_id,
        )

    @app.post("/api/chat/stream")
    def chat_stream_post(request: ChatRequest) -> StreamingResponse:
        return _chat_stream_response(
            message=request.message,
            session_id=request.session_id,
            fresh_session=request.fresh_session,
            repo_id=request.repo_id,
        )

    @app.get("/api/todos")
    def get_todos() -> dict[str, Any]:
        return {"todos": get_current_todos()}

    @app.post("/api/todos")
    def post_todos(request: TodoRequest) -> dict[str, Any]:
        payload = todo(
            command=request.command,
            items=request.items,
            id=request.id,
            status=request.status,
            title=request.title,
            note=request.note,
        )
        return payload

    @app.get("/api/memory")
    def memory(query: str = "", obs_type: str = "", limit: int = 120) -> dict[str, Any]:
        return {
            "overview": get_memory_overview(),
            "facts": list_memory_facts(),
            "observations": list_memory_observations(
                query=query,
                obs_type=obs_type,
                limit=limit,
            ),
        }

    @app.get("/api/repos")
    def repos() -> dict[str, Any]:
        return {"repos": list_repos()}

    @app.get("/api/repos/{repo_id}/graph")
    def repo_graph(repo_id: str, query: str = "", max_nodes: int = 120) -> dict[str, Any]:
        try:
            return get_repo_graph(repo_id, query=query, max_nodes=max_nodes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown repo '{repo_id}'") from exc

    @app.post("/api/rag/search")
    def rag_search(request: RagRequest) -> dict[str, Any]:
        try:
            return search_rag(
                request.query,
                repo_id=str(request.repo_id or ""),
                n_results=request.n_results,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def root() -> Any:
        if build_dir.exists():
            return FileResponse(build_dir / "index.html")
        return JSONResponse(
            {
                "status": "ok",
                "message": (
                    "Frontend not built yet. Run `npm install && npm run build` in `webui/`, "
                    "then start the server with `logician-web`."
                ),
            }
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        if build_dir.exists():
            candidate = build_dir / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(build_dir / "index.html")
        raise HTTPException(status_code=404, detail="Frontend build not found")

    return app
