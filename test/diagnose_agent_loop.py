#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src import AgentResponse, create_agent


DEFAULT_PROMPTS = [
    (
        "Inspect `rust-cli/src`. Read the most relevant files needed to understand "
        "the structure, then summarize the architecture briefly."
    ),
    (
        "Based on what you found in `rust-cli/src`, how would you improve it? "
        "Focus on the highest-leverage changes first."
    ),
]


def _normalize_text(text: str) -> str:
    collapsed = " ".join((text or "").strip().lower().split())
    return collapsed[:4000]


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(a=_normalize_text(left), b=_normalize_text(right)).ratio()


def _extract_batch_sizes(events: list[dict[str, Any]]) -> list[int]:
    by_iteration: dict[int, int] = {}
    for event in events:
        if event.get("kind") != "parsed_tool_call":
            continue
        iteration = event.get("iteration")
        if not isinstance(iteration, int):
            continue
        by_iteration[iteration] = by_iteration.get(iteration, 0) + 1
    return [count for _, count in sorted(by_iteration.items())]


def _summarize_response(prompt: str, response: AgentResponse) -> dict[str, Any]:
    debug = response.debug if isinstance(response.debug, dict) else {}
    events = debug.get("events", []) if isinstance(debug.get("events"), list) else []
    tool_names = [call.name for call in response.tool_calls]
    tool_signatures = [
        (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))
        for call in response.tool_calls
    ]
    duplicate_tool_calls = sum(count - 1 for count in Counter(tool_signatures).values() if count > 1)
    guardrail_events = [
        event
        for event in events
        if str(event.get("kind", "")).startswith("guardrail_")
        or str(event.get("kind", "")).endswith("_nudge")
    ]
    return {
        "prompt": prompt,
        "iterations": response.iterations,
        "tool_call_count": len(response.tool_calls),
        "tool_names": tool_names,
        "duplicate_tool_calls": duplicate_tool_calls,
        "batch_sizes": _extract_batch_sizes(events),
        "guardrail_event_count": len(guardrail_events),
        "guardrail_events": [event.get("kind") for event in guardrail_events],
        "final_response": response.final_response,
        "final_response_chars": len(response.final_response),
        "trace_event_count": len(events),
        "debug": debug,
    }


def _turn_score(turns: list[dict[str, Any]]) -> dict[str, Any]:
    repeated_pairs: list[dict[str, Any]] = []
    for idx in range(1, len(turns)):
        prev = turns[idx - 1]
        curr = turns[idx]
        similarity = _text_similarity(prev["final_response"], curr["final_response"])
        if similarity >= 0.88:
            repeated_pairs.append(
                {
                    "turn_a": idx,
                    "turn_b": idx + 1,
                    "similarity": round(similarity, 4),
                }
            )
    total_iterations = sum(int(turn["iterations"]) for turn in turns)
    total_tool_calls = sum(int(turn["tool_call_count"]) for turn in turns)
    total_guardrail_events = sum(int(turn["guardrail_event_count"]) for turn in turns)
    multi_call_turns = sum(1 for turn in turns if any(size > 1 for size in turn["batch_sizes"]))
    duplicate_tool_turns = sum(1 for turn in turns if int(turn["duplicate_tool_calls"]) > 0)
    return {
        "turn_count": len(turns),
        "total_iterations": total_iterations,
        "total_tool_calls": total_tool_calls,
        "total_guardrail_events": total_guardrail_events,
        "multi_call_turns": multi_call_turns,
        "duplicate_tool_turns": duplicate_tool_turns,
        "near_duplicate_answer_pairs": repeated_pairs,
    }


def _render_summary(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Scenario: {report['scenario_name']}")
    lines.append(f"Runs: {report['runs']}")
    lines.append(f"Model URL: {report['llm_url']}")
    lines.append("")
    for run_idx, run in enumerate(report["results"], start=1):
        score = run["score"]
        lines.append(f"Run {run_idx}: session={run['session_id']}")
        lines.append(
            "  turns={turn_count} iterations={total_iterations} tools={total_tool_calls} "
            "guardrails={total_guardrail_events} multi_call_turns={multi_call_turns} "
            "duplicate_tool_turns={duplicate_tool_turns}".format(**score)
        )
        if score["near_duplicate_answer_pairs"]:
            sims = ", ".join(
                f"{pair['turn_a']}->{pair['turn_b']} ({pair['similarity']})"
                for pair in score["near_duplicate_answer_pairs"]
            )
            lines.append(f"  near-duplicate answers: {sims}")
        for turn_idx, turn in enumerate(run["turns"], start=1):
            lines.append(
                f"    Turn {turn_idx}: iterations={turn['iterations']} "
                f"tools={turn['tool_call_count']} batches={turn['batch_sizes']} "
                f"guardrails={turn['guardrail_event_count']}"
            )
            if turn["tool_names"]:
                lines.append(f"      tool_names={', '.join(turn['tool_names'])}")
            preview = turn["final_response"].replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + " ..."
            lines.append(f"      final={preview}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _make_agent(llm_url: str, db_path: Path, vector_path: Path):
    agent = create_agent(
        llm_url=llm_url,
        db_path=str(db_path),
        config_overrides={
            "rag_enabled": False,
            "vector_path": str(vector_path),
            "debug_trace": True,
            "pre_turn_thinking": False,
            "append_quality_checklist": False,
            "enable_reflection": False,
            "stream": False,
            "max_iterations": 6,
            "strict_iteration_budget": True,
            "allow_multi_tool_calls": True,
            "multi_tool_call_max_calls": 4,
        },
    )
    return agent


def _run_scenario(
    *,
    llm_url: str,
    prompts: list[str],
    session_id: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        agent = _make_agent(
            llm_url=llm_url,
            db_path=tmp_path / "agent_sessions.db",
            vector_path=tmp_path / "message_history.vector",
        )
        turns: list[dict[str, Any]] = []
        tool_callback_events: list[dict[str, Any]] = []

        def _tool_callback(name: str, arguments: dict[str, Any], envelope: dict[str, Any]) -> None:
            tool_callback_events.append(
                {
                    "name": name,
                    "arguments": arguments,
                    **dict(envelope or {}),
                }
            )

        for idx, prompt in enumerate(prompts):
            fresh = idx == 0
            started = time.perf_counter()
            response = agent.run(
                prompt,
                session_id=session_id,
                fresh_session=fresh,
                tool_callback=_tool_callback,
            )
            turn = _summarize_response(prompt, response)
            turn["elapsed_s"] = round(time.perf_counter() - started, 3)
            turns.append(turn)

        return {
            "session_id": session_id,
            "turns": turns,
            "tool_callback_events": tool_callback_events,
            "score": _turn_score(turns),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a repeatable multi-turn live-agent diagnostics scenario."
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8080",
        help="Base URL for the local LLM server.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to replay the prompt sequence with fresh sessions.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to add to the scenario. Repeat this flag for multiple turns.",
    )
    parser.add_argument(
        "--scenario-name",
        default="rust-cli-review",
        help="Label stored in the report.",
    )
    parser.add_argument(
        "--output",
        default="tmp/agent_loop_report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--summary-output",
        default="tmp/agent_loop_report.txt",
        help="Where to write the text summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = list(args.prompts or DEFAULT_PROMPTS)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "scenario_name": args.scenario_name,
        "llm_url": args.llm_url,
        "runs": int(args.runs),
        "prompts": prompts,
        "results": [],
        "created_at_epoch": time.time(),
    }

    for run_idx in range(max(1, int(args.runs))):
        session_id = f"diag-{run_idx + 1}-{uuid.uuid4().hex[:8]}"
        run_result = _run_scenario(
            llm_url=args.llm_url,
            prompts=prompts,
            session_id=session_id,
        )
        report["results"].append(run_result)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = _render_summary(report)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Saved JSON report to {output_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
