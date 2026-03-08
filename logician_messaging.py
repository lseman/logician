#!/usr/bin/env python3
"""
logician_messaging.py — Telegram / WhatsApp (Twilio) front-end for the Logician agent.

Usage
-----
    # Telegram
    PLATFORM=telegram TELEGRAM_TOKEN=<token> python logician_messaging.py

    # WhatsApp via Twilio (runs an HTTP webhook server)
    PLATFORM=whatsapp TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... \
        TWILIO_WHATSAPP_FROM=whatsapp:+14155238886 \
        python logician_messaging.py

    # Override LLM endpoint / system prompt via env:
    LLM_URL=http://localhost:8080 AGENT_SYSTEM_PROMPT="You are a helpful assistant." ...

Dependencies
------------
    pip install python-telegram-bot>=21 twilio flask

Only the library for the chosen platform needs to be installed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("logician_messaging")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.runtime_paths import state_path  # noqa: E402


def _resolve_config_path() -> Path:
    raw = str(os.getenv("LOGICIAN_CONFIG_PATH", "") or "").strip()
    if not raw:
        # Backward-compatible fallback.
        raw = str(os.getenv("FOREBLOCKS_CONFIG_PATH", "") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return ROOT / "agent_config.json"


CONFIG_PATH = _resolve_config_path()

# ---------------------------------------------------------------------------
# Agent bootstrap
# ---------------------------------------------------------------------------

from src.agent.factory import create_agent  # noqa: E402


def _load_agent():
    """Instantiate the agent from agent_config.json + env overrides."""
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = json.load(f)

    # Honor generic env vars from config so runtime feature flags can be
    # toggled without shell export steps.
    if isinstance(cfg.get("env"), dict):
        for key, value in dict(cfg.get("env") or {}).items():
            k = str(key or "").strip()
            if not k or value is None:
                continue
            os.environ[k] = str(value)

    llm_url = os.getenv("LLM_URL", cfg.get("endpoint", "http://localhost:8080"))
    system_prompt = os.getenv("AGENT_SYSTEM_PROMPT", cfg.get("system_prompt"))
    chat_template = os.getenv("CHAT_TEMPLATE", cfg.get("chat_template", "chatml"))
    use_chat_api = (
        os.getenv("USE_CHAT_API", str(cfg.get("use_chat_api", True))).lower() == "true"
    )

    overrides: dict = {
        "max_iterations": int(
            os.getenv("MAX_ITERATIONS", str(cfg.get("max_iterations", 5)))
        ),
        "temperature": float(
            os.getenv("TEMPERATURE", str(cfg.get("temperature", 0.7)))
        ),
        "max_tokens": int(os.getenv("MAX_TOKENS", str(cfg.get("max_tokens", 2048)))),
        "history_limit": int(
            os.getenv("HISTORY_LIMIT", str(cfg.get("history_limit", 18)))
        ),
        "vector_path": str(state_path("message_history.vector")),
        "rag_vector_path": str(state_path("rag_docs.vector")),
    }

    mcp = cfg.get("mcp")
    if mcp:
        overrides["mcp_servers"] = mcp

    agent = create_agent(
        llm_url=llm_url,
        system_prompt=system_prompt,
        use_chat_api=use_chat_api,
        chat_template=chat_template,
        db_path=str(state_path("messaging_sessions.db")),
        embedding_model=str(cfg.get("embedding_model"))
        if cfg.get("embedding_model")
        else None,
        config_overrides=overrides,
    )
    log.info("Agent loaded  endpoint=%s  template=%s", llm_url, chat_template)
    return agent


# ---------------------------------------------------------------------------
# Shared session store  (user_key → session_id)
# ---------------------------------------------------------------------------


class SessionStore:
    """Thread-safe map from user identity → agent session id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, str] = {}

    def get_or_create(self, user_key: str) -> str:
        with self._lock:
            if user_key not in self._sessions:
                self._sessions[user_key] = f"msg_{uuid.uuid4().hex[:10]}"
                log.info(
                    "New session  user=%s  sid=%s", user_key, self._sessions[user_key]
                )
            return self._sessions[user_key]

    def reset(self, user_key: str) -> str:
        with self._lock:
            old = self._sessions.pop(user_key, None)
            self._sessions[user_key] = f"msg_{uuid.uuid4().hex[:10]}"
            log.info(
                "Reset session  user=%s  old=%s  new=%s",
                user_key,
                old,
                self._sessions[user_key],
            )
            return self._sessions[user_key]

    def get(self, user_key: str) -> str | None:
        with self._lock:
            return self._sessions.get(user_key)


# ---------------------------------------------------------------------------
# Agent runner (sync, called from a thread pool)
# ---------------------------------------------------------------------------


class AgentRunner:
    """Thin wrapper that serialises calls through an executor."""

    _MAX_MSG_LEN = 4000  # safe limit for both Telegram and WhatsApp

    def __init__(self, agent, sessions: SessionStore, max_workers: int = 4) -> None:
        self._agent = agent
        self._sessions = sessions
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="agent"
        )

    def ask(self, user_key: str, text: str) -> str:
        sid = self._sessions.get_or_create(user_key)
        try:
            response: str = self._agent.chat(
                text,
                session_id=sid,
                use_semantic_retrieval=True,
                retrieval_mode="hybrid",
            )
        except Exception as exc:
            log.exception("Agent error  user=%s", user_key)
            response = f"Internal error: {exc}"
        return self._truncate(response)

    def reset(self, user_key: str) -> None:
        old_sid = self._sessions.get(user_key)
        new_sid = self._sessions.reset(user_key)
        if old_sid:
            try:
                self._agent.reset(session_id=old_sid)
            except Exception:
                pass

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= AgentRunner._MAX_MSG_LEN:
            return text
        return text[: AgentRunner._MAX_MSG_LEN - 40] + "\n\n… _(response truncated)_"

    def submit(self, fn: Callable, *args) -> "Future":
        return self._pool.submit(fn, *args)


# ---------------------------------------------------------------------------
# Telegram adapter
# ---------------------------------------------------------------------------


def run_telegram(runner: AgentRunner) -> None:
    """Start the Telegram bot using python-telegram-bot v21+ (asyncio)."""
    try:
        from telegram import Update
        from telegram.constants import ChatAction
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
    except ImportError:
        log.error(
            "python-telegram-bot not installed.  Run: pip install python-telegram-bot>=21"
        )
        sys.exit(1)

    token = os.environ.get("TELEGRAM_TOKEN", "")
    if not token:
        log.error("TELEGRAM_TOKEN env var is required for Telegram mode.")
        sys.exit(1)

    HELP_TEXT = textwrap.dedent("""
        *Logician Agent*

        Send any message and I'll respond.

        Commands:
          /start — greet and start a new session
          /reset — wipe conversation history and start fresh
          /help  — show this message
    """).strip()

    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_key = str(update.effective_user.id)
        runner.reset(user_key)
        await update.message.reply_text(
            "👋 Hello! I'm the Logician agent. Ask me anything.\n"
            "Use /reset to start a fresh conversation, /help for more.",
        )

    async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_key = str(update.effective_user.id)
        runner.reset(user_key)
        await update.message.reply_text("🔄 Session reset. Fresh start!")

    async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")

    async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        user_key = str(update.effective_user.id)
        text = (update.message.text or "").strip()
        if not text:
            return

        await update.message.chat.send_action(ChatAction.TYPING)
        log.info("Telegram  user=%s  msg_len=%d", user_key, len(text))

        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(runner._pool, runner.ask, user_key, text)

        # Telegram hard limit is 4096 chars; split if needed
        chunks = [response[i : i + 4096] for i in range(0, len(response), 4096)]
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="Markdown")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Telegram bot starting (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ---------------------------------------------------------------------------
# WhatsApp / Twilio adapter  (Flask webhook)
# ---------------------------------------------------------------------------


def run_whatsapp(runner: AgentRunner) -> None:
    """
    Expose a Flask webhook that Twilio calls when a WhatsApp message arrives.

    Twilio sandbox setup:
      1. https://www.twilio.com/console/sms/whatsapp/sandbox
      2. Set the "When a message comes in" webhook URL to:
           http://<your-host>:<PORT>/whatsapp
      3. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM env vars.

    For local dev, expose via ngrok:  ngrok http 5000
    """
    try:
        from flask import Flask, Response, request
        from twilio.request_validator import RequestValidator
        from twilio.twiml.messaging_response import MessagingResponse
    except ImportError:
        log.error("twilio and flask not installed.  Run: pip install twilio flask")
        sys.exit(1)

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    port = int(os.environ.get("PORT", "5000"))
    validate_sig = os.environ.get("VALIDATE_TWILIO_SIG", "true").lower() == "true"

    if not account_sid or not auth_token:
        log.error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN env vars are required.")
        sys.exit(1)

    validator = RequestValidator(auth_token)
    flask_app = Flask("logician_messaging")

    @flask_app.post("/whatsapp")
    def whatsapp_webhook():
        # Twilio signature validation (skip when VALIDATE_TWILIO_SIG=false for local dev)
        if validate_sig:
            url = request.url
            body = request.form.to_dict()
            sig = request.headers.get("X-Twilio-Signature", "")
            if not validator.validate(url, body, sig):
                log.warning("Invalid Twilio signature — rejected")
                return Response("Forbidden", status=403)

        from_number_raw = request.form.get("From", "")  # e.g. whatsapp:+5541...
        body = (request.form.get("Body") or "").strip()

        if not from_number_raw or not body:
            return Response("", status=200)

        user_key = from_number_raw  # use the sender's WhatsApp number as key
        log.info("WhatsApp  user=%s  msg_len=%d", user_key, len(body))

        # Handle /reset command
        if body.lower() in ("/reset", "reset"):
            runner.reset(user_key)
            reply_text = "🔄 Session reset. Fresh start!"
        elif body.lower() in ("/help", "help"):
            reply_text = (
                "Logician Agent\n\n"
                "Send any message to chat.\n"
                "Commands:\n"
                "  reset — start a fresh conversation\n"
                "  help  — show this message"
            )
        else:
            reply_text = runner.ask(user_key, body)

        resp = MessagingResponse()
        # WhatsApp via Twilio: 1600 char limit per message segment
        chunk_size = 1500
        chunks = [
            reply_text[i : i + chunk_size]
            for i in range(0, len(reply_text), chunk_size)
        ]
        for chunk in chunks:
            resp.message(chunk)

        return Response(str(resp), status=200, mimetype="application/xml")

    @flask_app.get("/health")
    def health():
        return {"status": "ok", "platform": "whatsapp"}

    log.info("WhatsApp webhook starting on port %d", port)
    flask_app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    platform = os.environ.get("PLATFORM", "telegram").lower().strip()
    if platform not in ("telegram", "whatsapp"):
        log.error("PLATFORM must be 'telegram' or 'whatsapp', got: %r", platform)
        sys.exit(1)

    log.info("Platform: %s", platform)

    agent = _load_agent()
    sessions = SessionStore()
    runner = AgentRunner(
        agent, sessions, max_workers=int(os.environ.get("AGENT_WORKERS", "4"))
    )

    if platform == "telegram":
        run_telegram(runner)
    else:
        run_whatsapp(runner)


if __name__ == "__main__":
    main()
