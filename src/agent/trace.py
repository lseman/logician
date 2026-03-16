from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..messages import Message
from ..tools import ToolCall


def _vprint_block(title: str, s: str, limit: int = 2000) -> None:
    s = s or ""
    if limit > 0 and len(s) > limit:
        s = s[:limit] + f"\n... [truncated {len(s) - limit} chars] ..."
    print(f"\n[{title}]\n{s}")


@dataclass
class AgentResponse:
    messages: list[Message]
    tool_calls: list[ToolCall]
    iterations: int
    final_response: str
    debug: dict[str, Any] = field(default_factory=dict)
    trace_md: str = ""
    thinking_log: list[str] = field(default_factory=list)


def plot_tool_calls_by_iteration(
    response: AgentResponse | dict[str, Any],
    *,
    save_path: str | Path = "tool_calls_by_iteration.png",
    show: bool = False,
    dpi: int = 150,
    title: str = "Agent Tool Calls by Iteration",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    debug = response.get("debug", {}) if isinstance(response, dict) else response.debug
    if not isinstance(debug, dict):
        raise ValueError("Could not find debug payload on response.")

    events = debug.get("events", [])
    rows: list[tuple[int, str, float]] = []
    for ev in events:
        if ev.get("kind") != "parsed_tool_call":
            continue
        iter_id = ev.get("iteration")
        if not isinstance(iter_id, int):
            continue
        name = str(ev.get("name", "unknown_tool"))
        t = float(ev.get("t", 0.0))
        rows.append((iter_id, name, t))

    if not rows:
        raise ValueError("No parsed_tool_call events found. Enable debug_trace and rerun.")

    rows.sort(key=lambda x: (x[0], x[2]))
    iter_to_label: dict[int, str] = {}
    for iter_id, name, _ in rows:
        prev = iter_to_label.get(iter_id)
        iter_to_label[iter_id] = f"{prev}\n{name}" if prev else name

    iterations = sorted(iter_to_label.keys())
    labels = [iter_to_label[i] for i in iterations]
    max_lines = max(lbl.count("\n") + 1 for lbl in labels) if labels else 1

    fig_w = max(8.0, 1.0 * max(6, len(iterations)))
    fig_h = max(3.2, 1.1 + 0.45 * max_lines)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.scatter(iterations, [1] * len(iterations), marker="s", s=140, color="#2563EB")
    for x, lbl in zip(iterations, labels):
        ax.text(x, 1.04, lbl, ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_yticks([])
    ax.set_ylim(0.85, 1.3)
    ax.set_xticks(iterations)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()

    out = Path(save_path)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


@dataclass
class _TraceCollector:
    enabled: bool
    started_ts: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        self.events.append(
            {
                "t": round(time.perf_counter() - self.started_ts, 6),
                "kind": kind,
                **data,
            }
        )

    def total_duration_s(self) -> float:
        return round(time.perf_counter() - self.started_ts, 6)

    def build_debug_payload(
        self,
        *,
        sid: str,
        iterations: int,
        tool_calls: list[ToolCall],
        temp: float,
        n_tok: int,
        config: Config,
    ) -> dict[str, Any]:
        return {
            "session_id": sid,
            "iterations": iterations,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ],
            "events": self.events if self.enabled else [],
            "config": {
                "temperature": temp,
                "max_tokens": n_tok,
                "backend": config.backend,
                "use_chat_api": config.use_chat_api,
                "chat_template": config.chat_template,
                "stream": config.stream,
                "debug_trace": self.enabled,
            },
            "timings": {"total_duration_s": self.total_duration_s()},
        }


def _truncate_text(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + f"\n\n[... truncated {len(s) - limit} chars ...]"


def _render_context_snapshot(
    convo: list[Message],
    *,
    max_messages: int = 8,
    max_chars_per_message: int = 500,
) -> str:
    if max_messages < 1:
        max_messages = 1
    tail = convo[-max_messages:]
    lines: list[str] = []
    for i, m in enumerate(tail, start=1):
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        header = f"{i:02d}. {role}"
        if m.name:
            header += f"({m.name})"
        content = _truncate_text((m.content or "").strip(), max_chars_per_message)
        lines.append(header)
        lines.append(content if content else "<empty>")
    return "\n".join(lines)


def _tool_call_signature(call: ToolCall) -> tuple[str, str]:
    return (call.name, json.dumps(call.arguments, sort_keys=True))


def _render_tool_progress_reminder(
    tool_calls: list[ToolCall],
    tool_result_preview_by_sig: dict[tuple[str, str], str],
    *,
    max_items: int = 8,
) -> str:
    if not tool_calls:
        return ""

    seen: set[tuple[str, str]] = set()
    ordered_unique: list[tuple[str, str]] = []
    for call in tool_calls:
        sig = _tool_call_signature(call)
        if sig in seen:
            continue
        seen.add(sig)
        ordered_unique.append(sig)

    recent = ordered_unique[-max(1, int(max_items)) :]
    lines = [
        "[Iteration memory]",
        "Already executed tool calls in this run (latest unique first):",
    ]
    for i, (name, args_json) in enumerate(reversed(recent), start=1):
        preview = tool_result_preview_by_sig.get((name, args_json), "")
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + " ..."
        lines.append(f"{i}. {name} args={args_json}")
        if preview:
            lines.append(f"   result={preview}")

    lines.append("Do NOT call the exact same tool with the exact same arguments again.")
    lines.append(
        "If a prior tool returned empty/insufficient output, pick a different next-best tool or provide a final answer from current evidence."
    )
    return "\n".join(lines)


def render_trace_markdown(debug: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append(f"### Agent Trace — session `{debug['session_id'][:8]}`")
    lines.append("")
    lines.append(
        f"- **Iterations**: {debug['iterations']} | "
        f"**Total**: {debug['timings']['total_duration_s']} s"
    )

    cfg = debug.get("config", {})
    lines.append(
        f"- **Cfg**: temp={cfg.get('temperature')} max_tokens={cfg.get('max_tokens')} "
        f"chat_api={cfg.get('use_chat_api')} template=`{cfg.get('chat_template')}` "
        f"stream={cfg.get('stream')} backend={cfg.get('backend')} debug={cfg.get('debug_trace')}"
    )
    lines.append("")

    tool_calls = debug.get("tool_calls", [])
    if tool_calls:
        first = tool_calls[0]
        lines.append("**Agent command (first tool call):**")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "tool_call": {
                        "name": first["name"],
                        "arguments": first["arguments"],
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
    else:
        lines.append("_No tool call emitted by model._")
    lines.append("")

    lines.append("**Timeline**")
    lines.append("")
    iters: dict[int, dict[str, Any]] = {}
    for ev in debug.get("events", []):
        t = f"{ev['t']:.3f}s"
        kind = ev["kind"]
        iter_id = ev.get("iteration")

        if isinstance(iter_id, int):
            bucket = iters.setdefault(iter_id, {})
            if kind == "iter_context":
                bucket["context"] = ev.get("context_snapshot", "")
                bucket["total_messages"] = ev.get("total_messages", 0)
            elif kind == "llm_response_raw":
                bucket["assistant"] = ev.get("sample", "")
            elif kind == "parsed_tool_call":
                bucket["tool_name"] = ev.get("name")
                bucket["tool_args"] = ev.get("arguments", {})
            elif kind == "tool_result":
                bucket["tool_result"] = ev.get("result_preview", "")

        if kind == "user_message":
            lines.append(f"- [{t}] user → “{ev.get('message_preview', '')[:80]}”")
        elif kind == "thinking_begin":
            lines.append(f"- [{t}] thinking_begin: order={ev.get('mode')}")
        elif kind == "thinking_end":
            lines.append(f"- [{t}] thinking_end: “{ev.get('preview', '')[:80]}”")
        elif kind == "rag_retrieval":
            lines.append(
                f"- [{t}] rag_retrieval: {ev.get('n_results')} docs "
                f"(preview: {ev.get('preview', '')[:80]}…)"
            )
        elif kind == "llm_request_begin":
            lines.append(
                f"- [{t}] llm_request (temp={ev.get('temperature')}, max_tokens={ev.get('max_tokens')})"
            )
        elif kind == "llm_response_raw":
            sample = (ev.get("sample", "")[:80]).replace("\n", " ")
            lines.append(f"- [{t}] llm_response → “{sample}”")
        elif kind == "parsed_tool_call":
            ap = json.dumps(ev.get("arguments", {}), ensure_ascii=False)[:80]
            lines.append(f"- [{t}] parsed_tool_call: {ev.get('name')}({ap}…)")
        elif kind == "tool_result":
            rp = (ev.get("result_preview", "")[:80]).replace("\n", " ")
            lines.append(f"- [{t}] tool_result {ev.get('name')} → “{rp}”")
        elif kind == "final_answer":
            lines.append(
                f"- [{t}] final_answer → “{ev.get('content_preview', '')[:80]}”"
            )
        elif kind == "guardrail_stop":
            lines.append(f"- [{t}] guardrail_stop: {ev.get('reason')}")

    if iters:
        lines.append("")
        lines.append("**Rendered Iterations**")
        lines.append("")
        for i in sorted(iters.keys()):
            row = iters[i]
            lines.append(f"#### Iteration {i}")
            lines.append("")
            lines.append(
                f"- Context messages in prompt: {row.get('total_messages', 'n/a')}"
            )
            lines.append("")
            lines.append("Context passed to LLM:")
            lines.append("```text")
            lines.append((row.get("context") or "<none>").strip())
            lines.append("```")
            lines.append("")
            lines.append("Agent answer (raw):")
            lines.append("```text")
            lines.append((row.get("assistant") or "<none>").strip())
            lines.append("```")
            if row.get("tool_name"):
                lines.append("")
                lines.append("Tool call:")
                lines.append("```json")
                lines.append(
                    json.dumps(
                        {
                            "name": row.get("tool_name"),
                            "arguments": row.get("tool_args", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                lines.append("```")
            if row.get("tool_result"):
                lines.append("")
                lines.append("Tool result (context):")
                lines.append("```text")
                lines.append((row.get("tool_result") or "").strip())
                lines.append("```")
            lines.append("")

    lines.append("")
    return "\n".join(lines)


__all__ = [
    "AgentResponse",
    "_TraceCollector",
    "_render_context_snapshot",
    "_render_tool_progress_reminder",
    "_tool_call_signature",
    "_truncate_text",
    "_vprint_block",
    "plot_tool_calls_by_iteration",
    "render_trace_markdown",
]
