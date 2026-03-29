from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency guidance
        raise SystemExit(
            "uvicorn is required for the web UI. Install it with `pip install -e .[web]`."
        ) from exc

    host = str(os.getenv("LOGICIAN_WEB_HOST", "127.0.0.1"))
    port = int(os.getenv("LOGICIAN_WEB_PORT", "8000"))
    llm_url = (
        str(os.getenv("LOGICIAN_LLM_URL") or "").strip()
        or str(os.getenv("LLAMA_CPP_URL") or "").strip()
        or "http://localhost:8080"
    )
    reload_enabled = str(os.getenv("LOGICIAN_WEB_RELOAD", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    print(f"Logician web UI listening on http://{host}:{port}")
    print(f"Using LLM backend: {llm_url}")

    uvicorn.run(
        "src.webui.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
