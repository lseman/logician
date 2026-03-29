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

    @staticmethod
    def _tool_call_response_text(tool_calls: list[dict[str, Any]]) -> str:
        normalized: list[dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            raw_args = fn.get("arguments")
            if not name:
                continue
            try:
                arguments = json.loads(raw_args or "{}")
            except Exception:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            normalized.append({"name": name, "arguments": arguments})

        if not normalized:
            return ""
        if len(normalized) == 1:
            return json.dumps({"tool_call": normalized[0]})
        return json.dumps({"tool_calls": normalized})

    def _handle_chat_stream(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        full: list[str] = []
        streamed_tool_calls: list[dict[str, Any]] = []

        with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = self._parse_stream_json_line(line)
                if not data:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if isinstance(delta, dict):
                    content_delta = delta.get("content")
                    if content_delta:
                        full.append(content_delta)
                        if on_token:
                            on_token(content_delta)
                    self._merge_tool_call_delta(
                        streamed_tool_calls,
                        delta.get("tool_calls"),
                    )

                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                    if finish == "length":
                        self._log.warning(
                            "Response truncated (finish_reason=length)"
                        )

        if streamed_tool_calls:
            return self._tool_call_response_text(streamed_tool_calls)
        return "".join(full).strip()

    @staticmethod
    def _merge_tool_call_delta(
        tool_calls: list[dict[str, Any]],
        delta_calls: Any,
    ) -> None:
        if not isinstance(delta_calls, list):
            return
        for raw_item in delta_calls:
            if not isinstance(raw_item, dict):
                continue
            try:
                idx = int(raw_item.get("index", len(tool_calls)))
            except Exception:
                idx = len(tool_calls)
            while len(tool_calls) <= idx:
                tool_calls.append({"id": "", "type": "function", "function": {}})

            current = tool_calls[idx]
            if not isinstance(current, dict):
                current = {"id": "", "type": "function", "function": {}}
                tool_calls[idx] = current

            if raw_item.get("id"):
                current["id"] = str(raw_item["id"])
            if raw_item.get("type"):
                current["type"] = str(raw_item["type"])

            fn_delta = raw_item.get("function")
            if not isinstance(fn_delta, dict):
                continue
            fn = current.setdefault("function", {})
            if not isinstance(fn, dict):
                fn = {}
                current["function"] = fn

            if fn_delta.get("name"):
                fn["name"] = str(fn_delta["name"])
            if fn_delta.get("arguments") is not None:
                fn["arguments"] = str(fn.get("arguments") or "") + str(
                    fn_delta.get("arguments") or ""
                )

    @staticmethod
    def _chat_payload_messages(messages: list[Message]) -> list[dict[str, str]]:
        payload_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "tool" or getattr(msg.role, "value", None) == "tool":
                label = str(msg.name or "tool").strip() or "tool"
                content = str(msg.content or "").strip()
                payload_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool result: {label}]\n{content or '<empty>'}",
                    }
                )
                continue
            payload_messages.append(
                {
                    "role": msg.role.value,
                    "content": str(msg.content or ""),
                }
            )
        return payload_messages

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        grammar: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
                _use_stream = bool(stream)
                payload = {
                    "messages": self._chat_payload_messages(chat_messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": self.stop,
                    "stream": _use_stream,
                }
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = (
                        tool_choice if tool_choice is not None else "auto"
                    )
                    self._log.info(
                        "POST /v1/chat/completions temp=%s max_tokens=%s stream=%s tools=%d tool_choice=%s grammar=%s mode=native_constrained",
                        temperature,
                        max_tokens,
                        _use_stream,
                        len(tools),
                        payload["tool_choice"],
                        False,
                    )
                elif grammar:
                    payload["grammar"] = grammar
                    self._log.info(
                        "POST /v1/chat/completions temp=%s max_tokens=%s stream=%s tools=%d tool_choice=%s grammar=%s mode=manual_grammar",
                        temperature,
                        max_tokens,
                        _use_stream,
                        0,
                        None,
                        True,
                    )
                else:
                    self._log.info(
                        "POST /v1/chat/completions temp=%s max_tokens=%s stream=%s tools=%d tool_choice=%s grammar=%s mode=plain_chat",
                        temperature,
                        max_tokens,
                        _use_stream,
                        0,
                        None,
                        False,
                    )
                url = f"{self.base_url}/v1/chat/completions"
                if _use_stream:
                    return self._handle_chat_stream(
                        client,
                        url,
                        payload,
                        on_token=on_token,
                    )
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
                        return self._tool_call_response_text(msg.get("tool_calls") or [])
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
                if grammar:
                    payload["grammar"] = grammar
                url = f"{self.base_url}/completion"
                self._log.info(
                    "POST /completion temp=%s n_predict=%s stream=%s grammar=%s",
                    temperature,
                    max_tokens,
                    stream,
                    bool(grammar),
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
