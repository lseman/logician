from __future__ import annotations

from typing import Any

from ..messages import Message, MessageRole

try:
    import tiktoken as _tiktoken
    HAS_TIKTOKEN = True
except Exception:
    _tiktoken = None  # type: ignore
    HAS_TIKTOKEN = False

_TK_ENC = None  # loaded on first use


def _get_tk_enc() -> Any | None:
    global _TK_ENC
    if _TK_ENC is not None:
        return _TK_ENC
    if not HAS_TIKTOKEN or _tiktoken is None:
        return None
    try:
        _TK_ENC = _tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TK_ENC = None
    return _TK_ENC


def count_tokens_local(text: str) -> int:
    enc = _get_tk_enc()
    if enc is not None:
        try:
            return max(1, len(enc.encode(text, disallowed_special=())))
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


def normalize_chat_api_messages(
    messages: list[Message],
    *,
    collapse_system_to_first: bool = False,
) -> list[Message]:
    """Normalize message order/content for strict chat templates.

    Some server-side Jinja templates (including llama.cpp variants) require
    that there is at most one ``system`` message and it appears at index 0.
    When ``collapse_system_to_first`` is True, all system messages are merged
    (in original order) into a single leading system message.
    """
    if not collapse_system_to_first or not messages:
        return list(messages)

    system_parts: list[str] = []
    non_system: list[Message] = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            text = str(msg.content or "").strip()
            if text:
                system_parts.append(text)
            continue
        non_system.append(msg)

    if not system_parts:
        return list(messages)

    merged_system = Message(
        role=MessageRole.SYSTEM,
        content="\n\n".join(system_parts),
    )
    return [merged_system, *non_system]
