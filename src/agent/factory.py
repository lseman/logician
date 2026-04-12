from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..config import Config, ThinkingConfig

if TYPE_CHECKING:
    from .core import Agent


def create_agent(
    llm_url: str | None = None,
    system_prompt: str | None = None,
    use_chat_api: bool = True,
    chat_template: str = "chatml",
    db_path: str = "agent_sessions.db",
    embedding_model: str | None = None,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> Agent:
    from .core import Agent

    resolved_llm_url = (
        str(llm_url or "").strip()
        or str(os.getenv("LOGICIAN_LLM_URL") or "").strip()
        or str(os.getenv("LLAMA_CPP_URL") or "").strip()
        or "http://localhost:8080"
    )

    cfg = Config(
        llama_cpp_url=resolved_llm_url,
        use_chat_api=use_chat_api,
        chat_template=chat_template,
        use_toon_for_tools=False,
    )

    if config_overrides:
        for k, v in config_overrides.items():
            if k == "thinking":
                if isinstance(v, dict):
                    v = ThinkingConfig(
                        **{
                            kk: vv
                            for kk, vv in v.items()
                            if hasattr(ThinkingConfig, kk)
                            or kk in ThinkingConfig.__dataclass_fields__
                        }
                    )
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            elif hasattr(cfg, k):
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
