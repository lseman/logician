"""AST-based safety validator and restricted code compiler for EoH-generated policies."""

from __future__ import annotations

import ast
import math
import random
from dataclasses import dataclass
from typing import Any

from .hooks import HeuristicHooks, _wrap_hook
from .types import HOOKS

# ---------------------------------------------------------------------------
# Deny-list for restricted execution environment
# ---------------------------------------------------------------------------

DENY_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "help",
    }
)

DENY_ATTR_PREFIX = "__"

# ---------------------------------------------------------------------------
# SafetyVisitor – AST visitor that rejects dangerous constructs
# ---------------------------------------------------------------------------


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def fail(self, msg: str) -> None:
        if self.error is None:
            self.error = msg

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.fail("imports not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.fail("imports not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.attr, str) and node.attr.startswith(DENY_ATTR_PREFIX):
            self.fail(f"dunder attribute not allowed: {node.attr}")
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in DENY_NAMES:
            self.fail(f"forbidden name: {node.id}")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in DENY_NAMES:
            self.fail(f"forbidden call: {node.func.id}(...)")
            return
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public validation helper
# ---------------------------------------------------------------------------


def validate_code_ast(code: str) -> str | None:
    """
    Parse *code* and run :class:`SafetyVisitor`.
    Returns an error string, or ``None`` if the code is acceptable.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    v = SafetyVisitor()
    v.visit(tree)
    return v.error


# ---------------------------------------------------------------------------
# Restricted builtins available to generated policies
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "float": float,
    "int": int,
    "bool": bool,
    "any": any,
    "all": all,
}

# ---------------------------------------------------------------------------
# compile_hooks – the main entry point for turning LLM code into HeuristicHooks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecResult:
    hooks: HeuristicHooks | None
    error: str | None


def compile_hooks(code: str, *, max_chars: int) -> ExecResult:
    """
    Validate, exec, and assemble *code* into a :class:`HeuristicHooks` object.

    The code runs in a heavily restricted sandbox:
    - No imports
    - No dunder attributes
    - No dangerous builtins
    - Only ``math`` and ``random`` modules available
    """
    if len(code) > max_chars:
        return ExecResult(hooks=None, error="code too long")

    ast_error = validate_code_ast(code)
    if ast_error:
        return ExecResult(hooks=None, error=ast_error)

    env: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "random": random,
        "math": math,
    }

    try:
        exec(code, env, env)  # noqa: S102
    except Exception as exc:
        return ExecResult(hooks=None, error=f"exec failed: {exc}")

    build = env.get("build_hooks")
    if not callable(build):
        return ExecResult(hooks=None, error="missing callable build_hooks()")

    try:
        hook_dict = build()
    except Exception as exc:
        return ExecResult(hooks=None, error=f"build_hooks() failed: {exc}")

    if not isinstance(hook_dict, dict):
        return ExecResult(hooks=None, error="build_hooks() must return a dict")

    hh = HeuristicHooks()
    any_ok = False
    for k, v in hook_dict.items():
        if k not in HOOKS:
            continue
        if callable(v):
            setattr(hh, k, _wrap_hook(v))
            any_ok = True

    if not any_ok:
        return ExecResult(hooks=None, error="no valid hooks provided")

    return ExecResult(hooks=hh, error=None)
