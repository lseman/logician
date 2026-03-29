from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..logging_utils import get_logger
from ..messages import Message
from .common import count_tokens_local, format_messages_for_template

try:
    from vllm import LLM, SamplingParams  # type: ignore

    HAS_VLLM = True
except Exception:  # pragma: no cover
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    HAS_VLLM = False


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
        tools: list[dict[str, Any]] | None = None,  # accepted but ignored
        grammar: str | None = None,  # accepted but ignored (no constrained decoding in vllm path)
        tool_choice: str | dict[str, Any] | None = None,  # accepted but ignored
    ) -> str:
        del tools, grammar, tool_choice
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

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) when available.

        Falls back to char//4 heuristic if tiktoken is not installed.
        """
        return count_tokens_local(text)
