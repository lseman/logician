from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime_paths import repo_root, state_path

DEFAULT_RAG_EMBEDDING_MODEL = (
    "BAAI/bge-m3|Snowflake/snowflake-arctic-embed-l-v2.0|"
    "Qwen/Qwen3-Embedding-0.6B|nomic-ai/nomic-embed-text-v1.5|"
    "intfloat/e5-mistral-7b-instruct|BAAI/bge-small-en-v1.5"
)


def workspace_agent_config_path(workspace_root: str | Path | None = None) -> Path:
    root = Path(workspace_root).expanduser().resolve() if workspace_root else repo_root()
    return root / "agent_config.json"


def load_workspace_agent_config(workspace_root: str | Path | None = None) -> dict[str, Any]:
    config_path = workspace_agent_config_path(workspace_root)
    try:
        if not config_path.is_file():
            return {}
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def managed_rag_vector_path() -> Path:
    return state_path("rag_docs.vector")


def rag_runtime_settings(workspace_root: str | Path | None = None) -> dict[str, str]:
    payload = load_workspace_agent_config(workspace_root)
    vector_backend = (
        str(payload.get("rag_vector_backend") or payload.get("vector_backend") or "usearch")
        .strip()
        .lower()
    )
    embedding_model_name = str(
        payload.get("embedding_model") or DEFAULT_RAG_EMBEDDING_MODEL
    ).strip()
    return {
        "vector_path": str(managed_rag_vector_path()),
        "vector_backend": vector_backend or "usearch",
        "embedding_model_name": embedding_model_name or DEFAULT_RAG_EMBEDDING_MODEL,
    }


def legacy_rag_vector_paths(workspace_root: str | Path | None = None) -> list[Path]:
    root = Path(workspace_root).expanduser().resolve() if workspace_root else repo_root()
    active = managed_rag_vector_path().resolve()
    candidates = [(root / "rag_docs.vector").resolve()]
    return [candidate for candidate in candidates if candidate != active]


__all__ = [
    "DEFAULT_RAG_EMBEDDING_MODEL",
    "legacy_rag_vector_paths",
    "load_workspace_agent_config",
    "managed_rag_vector_path",
    "rag_runtime_settings",
    "workspace_agent_config_path",
]
