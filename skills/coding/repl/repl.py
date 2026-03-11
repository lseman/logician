from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "REPL",
    "description": "Use for short interactive experiments and in-process runtime inspection.",
    "aliases": ["scratch runtime", "quick experiment", "interactive python"],
    "triggers": [
        "evaluate this expression",
        "experiment with this logic",
        "inspect runtime state quickly",
    ],
    "preferred_tools": ["repl_exec", "repl_eval"],
    "example_queries": [
        "try this parser on a sample input",
        "evaluate this expression in the live session",
        "inspect the current REPL state",
    ],
    "when_not_to_use": [
        "a normal shell or standalone Python command is clearer and stateless execution is fine"
    ],
    "next_skills": ["shell", "quality"],
    "workflow": [
        "Use REPL for tight feedback loops and stateful experimentation.",
        "Keep experiments small and explicit.",
        "Promote validated logic into real code or tests rather than leaving it only in REPL state.",
    ],
}

# ---------------------------------------------------------------------------
# Persistent REPL namespace
# ---------------------------------------------------------------------------

_repl_ns: dict[str, Any] = {}
"""Variables survive across repl_exec / repl_eval calls until repl_reset()."""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def repl_exec(code: str) -> str:
    """Use when: Run multi-line Python code, define functions/classes, or perform stateful computations.

    Triggers: run python, execute code, try this code, test snippet, define function, calculate, compute,
              run this, evaluate expression, quick script.
    Avoid when: You need a subprocess with a separate venv — use run_python or run_shell instead.
    Inputs:
      code (str): Python source code to execute. May be multi-line.
    Returns: JSON with {status, stdout, stderr, error, traceback_lines}.
    Side effects: Mutates the shared REPL namespace (_repl_ns); imported modules and defined names persist.
    """
    global _repl_ns  # noqa: PLW0603
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error: str | None = None
    tb_lines: list[str] = []

    # Inject safe helpers if not already in namespace
    for _name, _val in (("json", __import__("json")),):
        _repl_ns.setdefault(_name, _val)

    try:
        compiled = compile(code, "<repl>", "exec")
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
            stderr_buf
        ):
            exec(compiled, _repl_ns)  # noqa: S102
    except SystemExit as exc:
        error = f"SystemExit({exc.code})"
        tb_lines = [error]
    except Exception:
        error = traceback.format_exc().splitlines()[-1]
        tb_lines = traceback.format_exc().splitlines()

    stdout_text = stdout_buf.getvalue()
    stderr_text = stderr_buf.getvalue()

    return _safe_json(
        {
            "status": "ok" if error is None else "error",
            "stdout": stdout_text[-6000:] if len(stdout_text) > 6000 else stdout_text,
            "stderr": stderr_text[-2000:] if len(stderr_text) > 2000 else stderr_text,
            "error": error,
            "traceback_lines": tb_lines[-30:] if tb_lines else [],
        }
    )


@tool
def repl_eval(expr: str) -> str:
    """Use when: Inspect the value of a variable or evaluate a simple expression after repl_exec.

    Triggers: what is the value of, print, inspect variable, check result, show me, evaluate.
    Avoid when: You need to run statements (assignments, loops, def) — use repl_exec instead.
    Inputs:
      expr (str): A single Python expression (no newlines, no statements).
    Returns: JSON with {status, result, repr, type}.
    Side effects: Read-only (no side effects unless the expression itself causes them).
    """
    global _repl_ns  # noqa: PLW0603
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            result = eval(expr, _repl_ns)  # noqa: S307
        result_repr = repr(result)
        # cap repr length
        if len(result_repr) > 4000:
            result_repr = result_repr[:4000] + "... (truncated)"
        return _safe_json(
            {
                "status": "ok",
                "result": result_repr,
                "type": type(result).__name__,
                "stdout": stdout_buf.getvalue(),
            }
        )
    except Exception:
        return _safe_json(
            {
                "status": "error",
                "error": traceback.format_exc().splitlines()[-1],
                "traceback_lines": traceback.format_exc().splitlines()[-15:],
            }
        )


@tool
def repl_reset() -> str:
    """Use when: Start fresh in the REPL (e.g. after a messy experiment or namespace pollution).

    Triggers: reset repl, clear namespace, fresh start, clean repl, restart python session.
    Avoid when: You want to keep existing variables for follow-up code.
    Inputs: None.
    Returns: JSON with {status, cleared_names}.
    Side effects: All names in _repl_ns are deleted.
    """
    global _repl_ns  # noqa: PLW0603
    cleared = list(_repl_ns.keys())
    _repl_ns.clear()
    return _safe_json({"status": "ok", "cleared_names": cleared, "count": len(cleared)})


@tool
def repl_state() -> str:
    """Use when: Inspect what's currently in the REPL before running more code.

    Triggers: what's in the repl, show variables, list namespace, what did I define, repl state, show imports.
    Avoid when: The namespace is empty (use repl_exec to define things first).
    Inputs: None.
    Returns: JSON list of {name, type, preview} for each item in the namespace.
    Side effects: Read-only.
    """
    items = []
    for name, value in _repl_ns.items():
        if name.startswith("__"):
            continue
        preview = repr(value)
        if len(preview) > 120:
            preview = preview[:120] + "..."
        items.append(
            {
                "name": name,
                "type": type(value).__name__,
                "preview": preview,
            }
        )
    items.sort(key=lambda x: x["name"])
    return _safe_json({"status": "ok", "count": len(items), "variables": items})


__tools__ = [repl_exec, repl_eval, repl_reset, repl_state]
