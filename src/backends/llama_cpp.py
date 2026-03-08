from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from ..logging_utils import get_logger
from ..messages import Message
from .common import (
    count_tokens_local,
    format_messages_for_template,
    normalize_chat_api_messages,
)


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        use_chat_api: bool,
        chat_template: str,
        stop: Iterable[str],
        retry_attempts: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.use_chat = use_chat_api
        self.template = chat_template
        self.stop = list(stop)
        self.retry_attempts = max(0, retry_attempts)
        self._log = get_logger("agent.llama")

    def _request(
        self,
        client: Any,
        method: str,
        url: str,
        **kw: Any,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_attempts + 2):
            try:
                self._log.debug("HTTP %s %s (attempt %d)", method, url, attempt)
                resp = client.request(method, url, **kw)
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                self._log.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    self.retry_attempts + 1,
                    e,
                )
                if attempt <= self.retry_attempts:
                    backoff = 0.6 * attempt
                    self._log.debug("Backoff %.2fs", backoff)
                    time.sleep(backoff)
                else:
                    self._log.error("Giving up after %d attempts", attempt)
                    raise
        assert False, last_exc  # pragma: no cover

    @staticmethod
    def _decode_stream_line(line: str | bytes) -> str:
        if isinstance(line, bytes):
            return line.decode("utf-8", errors="ignore").strip()
        return line.strip()

    def _parse_stream_json_line(self, line: str | bytes) -> dict[str, Any] | None:
        raw = self._decode_stream_line(line)
        if not raw:
            return None
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        if raw == "[DONE]":
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        if httpx is None:
            raise ImportError(
                "httpx is required for the HTTP llama.cpp backend. "
                "Install it with: pip install httpx"
            )
        with httpx.Client(timeout=self.timeout) as client:
            if self.use_chat:
                strict_jinja = self.template.strip().lower() == "jinja"
                chat_messages = normalize_chat_api_messages(
                    messages,
                    collapse_system_to_first=strict_jinja,
                )
                if strict_jinja and len(chat_messages) != len(messages):
                    self._log.debug(
                        "Normalized chat messages for strict Jinja template: %d -> %d",
                        len(messages),
                        len(chat_messages),
                    )
                # Constrained decoding via native function-calling disables streaming
                # (llama.cpp tool_calls delta handling differs; we keep it simple).
                _use_stream = bool(stream) and not bool(tools)
                payload = {
                    "messages": [
                        {"role": m.role.value, "content": m.content}
                        for m in chat_messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": self.stop,
                    "stream": _use_stream,
                }
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"
                url = f"{self.base_url}/v1/chat/completions"
                self._log.info(
                    "POST /v1/chat/completions temp=%s max_tokens=%s stream=%s tools=%d",
                    temperature,
                    max_tokens,
                    _use_stream,
                    len(tools) if tools else 0,
                )
                if _use_stream:
                    with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        full: list[str] = []
                        _stream_finish_reason: str | None = None
                        for line in r.iter_lines():
                            if not line:
                                continue
                            data = self._parse_stream_json_line(line)
                            if not data:
                                continue
                            _choice = data.get("choices", [{}])[0]
                            delta = _choice.get("delta", {}).get("content")
                            if delta:
                                full.append(delta)
                                if on_token:
                                    on_token(delta)
                            if _choice.get("finish_reason"):
                                _stream_finish_reason = _choice["finish_reason"]
                    if _stream_finish_reason == "length":
                        self._log.warning(
                            "Response truncated (finish_reason=length): "
                            "max_tokens limit reached mid-generation"
                        )
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    if data.get("choices", [{}])[0].get("finish_reason") == "length":
                        self._log.warning(
                            "Response truncated (finish_reason=length): "
                            "max_tokens limit reached mid-generation"
                        )
                    msg = data["choices"][0]["message"]
                    # Native function-call response -> reformat to existing tool_call JSON
                    # so the existing parser handles it without modification.
                    if msg.get("tool_calls"):
                        tc = msg["tool_calls"][0]
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            args = {}
                        return json.dumps(
                            {"tool_call": {"name": name, "arguments": args}}
                        )
                    return (msg.get("content") or "").strip()
            else:
                prompt = format_messages_for_template(messages, self.template)
                payload = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "n_predict": max_tokens,
                    "stop": self.stop,
                    "stream": bool(stream),
                }
                url = f"{self.base_url}/completion"
                self._log.info(
                    "POST /completion temp=%s n_predict=%s stream=%s",
                    temperature,
                    max_tokens,
                    stream,
                )
                if stream:
                    with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        full: list[str] = []
                        _stopped_limit = False
                        for line in r.iter_lines():
                            if not line:
                                continue
                            data = self._parse_stream_json_line(line)
                            if not data:
                                continue
                            tok = data.get("content")
                            if tok:
                                full.append(tok)
                                if on_token:
                                    on_token(tok)
                            if data.get("stopped_limit"):
                                _stopped_limit = True
                    if _stopped_limit:
                        self._log.warning(
                            "Response truncated (stopped_limit=true): "
                            "n_predict limit reached mid-generation"
                        )
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    if data.get("stopped_limit"):
                        self._log.warning(
                            "Response truncated (stopped_limit=true): "
                            "n_predict limit reached mid-generation"
                        )
                    return data.get("content", "").strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens for *text*.

        Priority:
        1. tiktoken (local, zero-latency) via count_tokens_local()
        2. llama.cpp ``/tokenize`` endpoint (server-exact, adds HTTP round-trip)
        3. char//4 heuristic (last resort)
        """
        # Fast local path — tiktoken gives ≈95% accuracy vs the server tokenizer
        # at zero latency, which is ideal for budget trimming decisions.
        from .common import HAS_TIKTOKEN
        if HAS_TIKTOKEN:
            return count_tokens_local(text)

        # Slower-but-exact server path when tiktoken is not installed.
        if httpx is None:
            return count_tokens_local(text)
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.post(
                    f"{self.base_url}/tokenize",
                    json={"content": text},
                )
                r.raise_for_status()
                data = r.json()
                return max(1, len(data.get("tokens", [])))
        except Exception:
            return count_tokens_local(text)
