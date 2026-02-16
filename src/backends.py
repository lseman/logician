# agent_core/backends.py
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from .logging_utils import get_logger
from .messages import Message, MessageRole

# ===================== Optional vLLM deps =====================
try:
    from vllm import LLM, SamplingParams  # type: ignore

    HAS_VLLM = True
except Exception:  # pragma: no cover
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    HAS_VLLM = False


def format_messages_for_template(
    messages: list[Message],
    template: str,
) -> str:
    """Convert chat messages to a single prompt according to a template."""

    if template == "chatml":
        out: list[str] = []
        for m in messages:
            out.append(f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>")
        out.append("<|im_start|>assistant\n")
        return "\n".join(out)

    if template == "llama2":
        segs: list[str] = []
        sys = [m for m in messages if m.role == MessageRole.SYSTEM]
        if sys:
            segs.append(f"<s>[INST] <<SYS>>\n{sys[-1].content}\n<</SYS>>\n")
        for m in messages:
            if m.role == MessageRole.USER:
                segs.append(f"{m.content} [/INST] ")
            elif m.role == MessageRole.ASSISTANT:
                segs.append(f"{m.content} </s><s>[INST] ")
        return "".join(segs)

    if template == "zephyr":
        out: list[str] = []
        for m in messages:
            out.append(f"<|{m.role.value}|>\n{m.content}</s>")
        out.append("<|assistant|>\n")
        return "\n".join(out)

    # Fallback: simple role-tagged text
    return (
        "".join(f"{m.role.value.upper()}: {m.content}\n" for m in messages)
        + "ASSISTANT: "
    )


# =====================================================================
# Llama.cpp Client (HTTP)
# =====================================================================
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
        client: httpx.Client,
        method: str,
        url: str,
        **kw: Any,
    ) -> httpx.Response:
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
    ) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            if self.use_chat:
                payload = {
                    "messages": [
                        {"role": m.role.value, "content": m.content} for m in messages
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": self.stop,
                    "stream": bool(stream),
                }
                url = f"{self.base_url}/v1/chat/completions"
                self._log.info(
                    "POST /v1/chat/completions temp=%s max_tokens=%s stream=%s",
                    temperature,
                    max_tokens,
                    stream,
                )
                if stream:
                    r = self._request(client, "POST", url, json=payload)
                    full: list[str] = []
                    for line in r.iter_lines():
                        if not line:
                            continue
                        data = self._parse_stream_json_line(line)
                        if not data:
                            continue
                        delta = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if delta:
                            full.append(delta)
                            if on_token:
                                on_token(delta)
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
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
                    r = self._request(client, "POST", url, json=payload)
                    full: list[str] = []
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
                    return "".join(full).strip()
                else:
                    r = self._request(client, "POST", url, json=payload)
                    data = r.json()
                    return data.get("content", "").strip()


# =====================================================================
# vLLM Client (in-process Python backend)
# =====================================================================
class VLLMClient:
    """
    In-process vLLM backend.

    Notes:
    - Uses the Python API (no HTTP).
    - For simplicity, streaming here calls `on_token` once with the full text.
    """

    def __init__(
        self,
        model: str,
        chat_template: str,
        stop: Iterable[str],
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
    ) -> None:
        if not HAS_VLLM:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        self.template = chat_template
        self.stop = list(stop)
        self._log = get_logger("agent.vllm")
        self._log.info(
            "Loading vLLM model=%s tp=%d mem=%.2f dtype=%s",
            model,
            tensor_parallel_size,
            gpu_memory_utilization,
            dtype,
        )
        extra_kwargs: dict[str, Any] = {}
        if dtype != "auto":
            extra_kwargs["dtype"] = dtype
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **extra_kwargs,
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        prompt = format_messages_for_template(messages, self.template)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self.stop,
        )
        self._log.info(
            "vLLM.generate temp=%s max_tokens=%s stream=%s",
            temperature,
            max_tokens,
            stream,
        )

        outputs = self.llm.generate(
            [prompt], sampling_params, use_tqdm=False, stream=False
        )
        text = outputs[0].outputs[0].text
        text = text.strip()
        if on_token:
            on_token(text)
        return text
