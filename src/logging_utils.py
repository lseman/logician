# agent_core/logging_utils.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any


def _resolve_log_level(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().upper()
        return getattr(logging, value, logging.ERROR)
    return logging.ERROR


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extras (avoid default record attrs)
        for k, v in record.__dict__.items():
            if k.startswith("_"):
                continue
            if k in (
                "args",
                "name",
                "msg",
                "levelno",
                "levelname",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                continue
            try:
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = str(v)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(
    name: str,
    level: str | int | None = None,
    json_logs: bool | None = None,
) -> logging.Logger:
    """
    Create/reuse a module logger with a single ConsoleHandler.
    Defaults pulled from env or sensible DEBUG defaults.

    This configures the *root* logger once (idempotent) so that
    all sub-loggers share the same handler/format.
    """
    logger = logging.getLogger(name)

    # Configure root once
    root = logging.getLogger()
    if not getattr(root, "_configured_by_agent_logger", False):
        env_level = os.getenv("AGENT_LOG_LEVEL")
        env_json = os.getenv("AGENT_LOG_JSON", "0").lower()
        root.setLevel(_resolve_log_level(env_level or logging.ERROR))

        if not root.handlers:
            ch = logging.StreamHandler()
            if env_json in ("1", "true", "yes"):
                ch.setFormatter(_JsonFormatter())
            else:
                fmt = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
                ch.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
            root.addHandler(ch)

        root._configured_by_agent_logger = True  # type: ignore[attr-defined]

    # Optional per-logger overrides
    if level is not None:
        logger.setLevel(_resolve_log_level(level))
    if json_logs is not None:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler):
                h.setFormatter(
                    _JsonFormatter()
                    if json_logs
                    else logging.Formatter(
                        fmt="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                    )
                )
    return logger
