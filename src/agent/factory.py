from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import Config

if TYPE_CHECKING:
    from .core import Agent


def create_agent(
    llm_url: str = "http://localhost:8080",
    system_prompt: str | None = None,
    use_chat_api: bool = True,
    chat_template: str = "chatml",
    db_path: str = "agent_sessions.db",
    embedding_model: str | None = "BAAI/bge-base-en-v1.5",
    *,
    config_overrides: dict[str, Any] | None = None,
) -> Agent:
    from .core import Agent

    cfg = Config(
        llama_cpp_url=llm_url,
        use_chat_api=use_chat_api,
        chat_template=chat_template,
        use_toon_for_tools=True,
    )

    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return Agent(
        llm_url=cfg.llama_cpp_url,
        system_prompt=system_prompt,
        config=cfg,
        use_chat_api=cfg.use_chat_api,
        chat_template=cfg.chat_template,
        db_path=db_path,
        embedding_model=embedding_model,
        lazy_rag=True,
    )


__all__ = ["create_agent"]
