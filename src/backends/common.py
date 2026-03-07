from __future__ import annotations

from ..messages import Message, MessageRole

try:
    import tiktoken as _tiktoken

    # cl100k_base is used by GPT-4 / ChatGPT — a good universal proxy for
    # modern transformer models when the exact tokenizer is unavailable.
    _TK_ENC = _tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except Exception:
    _tiktoken = None  # type: ignore
    _TK_ENC = None
    HAS_TIKTOKEN = False


def count_tokens_local(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) when available.

    This gives an accurate-enough local estimate without an HTTP round-trip to
    the server's /tokenize endpoint.  Falls back to the traditional char//4
    heuristic when tiktoken is not installed.
    """
    if HAS_TIKTOKEN and _TK_ENC is not None:
        try:
            return max(1, len(_TK_ENC.encode(text, disallowed_special=())))
        except Exception:
            pass
    return max(1, len(text) // 4)


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

