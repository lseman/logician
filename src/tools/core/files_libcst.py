"""LibCST-based file editing tools for syntax-safe, structure-aware edits.
This module provides AST-level file editing using LibCST to avoid the fragile
string-matching problems of purely textual editors.

Key benefits over string-based editing:
- Syntax-aware: edits respect Python syntax rules.
- Formatting-preserving: whitespace, comments, indentation, and line endings
  are preserved.
- Structure-aware: code is matched and modified by syntax tree rather than
  brittle substrings.
- Safer refactors: function bodies, docstrings, decorators, and call arguments
  can be updated reliably.

Example:
    from src.tools.core.files_libcst import (
        edit_file_libcst,
        replace_docstring,
        replace_function_body,
    )
    edit_file_libcst("src/foo.py", "return x + y", "return x * 2")
    replace_docstring("src/foo.py", "my_function", "New docstring with Args.")
"""

from __future__ import annotations

import ast
import difflib
import os
import tempfile
from pathlib import Path
from typing import Any

import libcst as cst
from libcst import matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

__all__ = [
    "edit_file_libcst",
    "replace_function_body",
    "replace_docstring",
    "replace_decorators",
    "replace_argument",
    "insert_after_function",
    "delete_function",
    "find_function_by_name",
    "find_class_by_name",
]


# ---------------------------------------------------------------------
# helpers (unchanged – they are already excellent)
# ---------------------------------------------------------------------
def _ok(**kwargs: Any) -> dict[str, Any]:
    """Build a successful tool result payload."""
    return {"status": "ok", **kwargs}


def _err(error: str, **kwargs: Any) -> dict[str, Any]:
    """Build an error tool result payload."""
    return {"status": "error", "error": error, **kwargs}


def _atomic_write(p: Path, content: str) -> dict[str, Any]:
    """Write content to a file atomically."""
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        Path(tmp).replace(p)
        return _ok(path=str(p), bytes_written=len(content.encode("utf-8")))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _validate_syntax(p: Path, content: str) -> dict[str, Any] | None:
    """Validate Python syntax for generated content."""
    if p.suffix.lower() != ".py":
        return None
    if not content.strip():
        return None
    try:
        ast.parse(content)
        return None
    except SyntaxError as exc:
        return {
            "line": exc.lineno,
            "offset": exc.offset,
            "message": str(exc),
            "hint": "The generated file contains a syntax error; review before running.",
        }


def _unified_diff(original: str, updated: str, label: str = "file") -> str:
    """Compute a unified diff between two text blobs."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"{label} (before)",
        tofile=f"{label} (after)",
        lineterm="",
    )
    return "".join(diff)


def _read_python_file(path: str) -> tuple[Path, str] | dict[str, Any]:
    """Read a source file from disk."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return _err(f"File not found: {path}")
    try:
        return p, p.read_text(encoding="utf-8")
    except OSError as e:
        return _err(f"Cannot read file: {e}")


def _parse_module(path: str) -> tuple[Path, str, cst.Module] | dict[str, Any]:
    """Read and parse a Python source file as a LibCST module."""
    res = _read_python_file(path)
    if isinstance(res, dict):
        return res
    p, content = res
    try:
        module = cst.parse_module(content)
    except Exception as e:
        return _err(f"Could not parse Python file with LibCST: {e}")
    return p, content, module


def _apply_transformer(
    p: Path,
    old_content: str,
    module: cst.Module,
    transformer: cst.CSTTransformer,
    *,
    edits_applied: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a LibCST transformer, write the result, and build a tool payload."""
    try:
        new_module = module.visit(transformer)
        new_content = new_module.code
    except Exception as e:
        return _err(str(e), path=str(p))

    if new_content == old_content:
        return _err("No changes were applied.", path=str(p), edits_applied=0)

    write_result = _atomic_write(p, new_content)
    result: dict[str, Any] = dict(write_result)

    applied = edits_applied
    if applied is None:
        applied = getattr(transformer, "modified_count", None)
    if applied is None:
        applied = 1 if getattr(transformer, "modified", False) else 0
    result["edits_applied"] = applied
    result["diff"] = _unified_diff(old_content, new_content, str(p))

    syntax_err = _validate_syntax(p, new_content)
    if syntax_err:
        result["syntax_error"] = syntax_err
    if extra:
        result.update(extra)

    return result


def _parse_statement_or_suite(code: str) -> list[cst.BaseStatement]:
    """Parse source text into one or more statements."""
    mod = cst.parse_module(code if code.endswith("\n") else code + "\n")
    return list(mod.body)


def _parse_expression(code: str) -> cst.BaseExpression:
    """Parse source text into a single Python expression."""
    return cst.parse_expression(code)


def _parse_decorator(text: str) -> cst.Decorator:
    """Parse decorator text into a LibCST decorator node."""
    raw = text.strip()
    if raw.startswith("@"):
        raw = raw[1:].strip()
    return cst.Decorator(decorator=cst.parse_expression(raw))


def _make_docstring_stmt(doc: str) -> cst.SimpleStatementLine:
    """Create a valid Python docstring statement from raw text."""
    return cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString(repr(doc)))])


def _is_docstring_stmt(stmt: cst.BaseStatement) -> bool:
    """Check whether a statement is a docstring statement."""
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    return isinstance(inner, cst.Expr) and isinstance(inner.value, cst.SimpleString)


def _position_dict(
    wrapper: MetadataWrapper,
    node: cst.CSTNode,
) -> dict[str, int]:
    """Build a simple line/column position payload for a CST node."""
    pos = wrapper.resolve(PositionProvider)[node]
    return {
        "line": pos.start.line,
        "column": pos.start.column,
        "end_line": pos.end.line,
        "end_column": pos.end.column,
    }


def _function_definition_code(node: cst.FunctionDef) -> str:
    """Render a function definition node back to source code."""
    return cst.Module(body=[node]).code


def _class_definition_code(node: cst.ClassDef) -> str:
    """Render a class definition node back to source code."""
    return cst.Module(body=[node]).code


# ---------------------------------------------------------------------
# Modern base transformer (eliminates duplication)
# ---------------------------------------------------------------------
class _BaseTransformer(cst.CSTTransformer):
    """Reusable base for all targeted transformers.

    Provides `modified_count` and a clean `_increment()` helper.
    All concrete transformers inherit from this.
    """

    def __init__(self) -> None:
        self.modified_count = 0

    def _increment(self) -> None:
        """Call after a successful edit."""
        self.modified_count += 1


# ---------------------------------------------------------------------
# generic structural edit by exact normalized AST semantics
# ---------------------------------------------------------------------
def edit_file_libcst(
    path: str,
    old_pattern: str,
    new_code: str,
    case_sensitive: bool = True,
) -> dict[str, Any]:
    """Replace Python code by AST structure instead of raw text matching.

    Use when
    --------
    You need a Python-only replacement that should survive harmless formatting
    differences such as whitespace, comments, or quote style.

    Required call shape
    -------------------
    - `path`: Python file path
    - `old_pattern`: valid Python statement or expression to find
    - `new_code`: valid Python statement(s) or expression to replace it with

    Agent guidance
    --------------
    - Use this for Python when `edit_file` is too brittle.
    - `old_pattern` must be valid Python on its own.
    - `new_code` must also be valid Python on its own.
    - This matches by normalized AST semantics, not exact source bytes.
    - Prefer more targeted tools like `replace_function_body` or
      `replace_docstring` when the edit is about a named symbol.

    Example
    -------
        edit_file_libcst(
            "src/foo.py",
            "return x + y",
            "return x * 2",
        )
    """
    del case_sensitive  # kept for API compatibility

    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    # Original robust AST-normalized matching (kept – it's still the best
    # way to do "exact structural replace" for arbitrary snippets)
    try:
        ast.parse(old_content)
    except SyntaxError as e:
        return _err(
            f"File has syntax error at line {e.lineno}: {e.msg}",
            path=str(p),
        )

    old_stmt_dump = None
    old_expr_dump = None
    try:
        old_stmt_ast = ast.parse(old_pattern)
        old_stmt_dump = ast.dump(old_stmt_ast, include_attributes=False)
    except SyntaxError:
        pass
    try:
        old_expr_ast = ast.parse(old_pattern, mode="eval")
        old_expr_dump = ast.dump(old_expr_ast, include_attributes=False)
    except SyntaxError:
        pass

    if old_stmt_dump is None and old_expr_dump is None:
        return _err("Pattern is not valid Python.", path=str(p))

    replacement_stmts: list[cst.BaseStatement] | None = None
    replacement_expr: cst.BaseExpression | None = None
    try:
        replacement_stmts = _parse_statement_or_suite(new_code)
    except Exception:
        pass
    try:
        replacement_expr = _parse_expression(new_code)
    except Exception:
        pass

    if replacement_stmts is None and replacement_expr is None:
        return _err("Replacement code is not valid Python.", path=str(p))

    wrapper = MetadataWrapper(module)

    class MatcherAndRewriter(_BaseTransformer):
        """Modernized matcher using the original robust AST comparison logic."""

        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self) -> None:
            super().__init__()
            self.statement_locations: list[dict[str, int]] = []
            self.expression_locations: list[dict[str, int]] = []
            self._stmt_mode = False
            self._expr_mode = False

        def _stmt_matches(self, node: cst.BaseStatement) -> bool:
            if old_stmt_dump is None:
                return False
            try:
                code = cst.Module(body=[node]).code
                parsed_stmt = ast.parse(code)
                return ast.dump(parsed_stmt, include_attributes=False) == old_stmt_dump
            except Exception:
                return False

        def _expr_matches(self, node: cst.BaseExpression) -> bool:
            if old_expr_dump is None:
                return False
            try:
                parsed_expr = ast.parse(cst.Module([]).code_for_node(node), mode="eval")
                return ast.dump(parsed_expr, include_attributes=False) == old_expr_dump
            except Exception:
                return False

        # Statement replacements (FlattenSentinel for multi-line)
        def leave_SimpleStatementLine(
            self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        # (All other statement types use the exact same pattern – kept for fidelity)
        def leave_If(
            self, original_node: cst.If, updated_node: cst.If
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_For(
            self, original_node: cst.For, updated_node: cst.For
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_While(
            self, original_node: cst.While, updated_node: cst.While
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_With(
            self, original_node: cst.With, updated_node: cst.With
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_Try(
            self, original_node: cst.Try, updated_node: cst.Try
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        def leave_ClassDef(
            self, original_node: cst.ClassDef, updated_node: cst.ClassDef
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if replacement_stmts is None or not self._stmt_matches(original_node):
                return updated_node
            self.statement_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._stmt_mode = True
            return cst.FlattenSentinel(replacement_stmts)

        # Expression replacements
        def leave_Expr(
            self, original_node: cst.Expr, updated_node: cst.Expr
        ) -> cst.BaseSmallStatement:
            if replacement_expr is None or not self._expr_matches(original_node.value):
                return updated_node
            self.expression_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._expr_mode = True
            return updated_node.with_changes(value=replacement_expr)

        def leave_BinaryOperation(
            self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation
        ) -> cst.BaseExpression:
            if replacement_expr is None or self._stmt_mode or not self._expr_matches(original_node):
                return updated_node
            self.expression_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._expr_mode = True
            return replacement_expr

        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
            if replacement_expr is None or self._stmt_mode or not self._expr_matches(original_node):
                return updated_node
            self.expression_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._expr_mode = True
            return replacement_expr

        def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
            if replacement_expr is None or self._stmt_mode or not self._expr_matches(original_node):
                return updated_node
            self.expression_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._expr_mode = True
            return replacement_expr

        def leave_Attribute(
            self, original_node: cst.Attribute, updated_node: cst.Attribute
        ) -> cst.BaseExpression:
            if replacement_expr is None or self._stmt_mode or not self._expr_matches(original_node):
                return updated_node
            self.expression_locations.append(_position_dict(wrapper, original_node))
            self._increment()
            self._expr_mode = True
            return replacement_expr

    transformer = MatcherAndRewriter()
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] != "ok":
        return result

    locations = transformer.statement_locations or transformer.expression_locations
    result["matches_found"] = transformer.modified_count
    result["locations"] = locations

    if transformer.modified_count == 0:
        return _err(
            "Pattern not found in file. 0 matches.",
            path=str(p),
            matches_found=0,
            locations=[],
        )
    return result


# ---------------------------------------------------------------------
# function body replacement
# ---------------------------------------------------------------------
def replace_function_body(
    path: str,
    function_name: str,
    new_body: str,
) -> dict[str, Any]:
    """Replace the body of a function by function name.

    Use when
    --------
    You want to keep the function signature, decorators, and docstring, but
    replace the executable body.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact function name
    - `new_body`: one or more valid Python statements for the body only

    Agent guidance
    --------------
    - Pass only the body statements in `new_body`, not `def ...:`.
    - Indentation is handled by LibCST; do not wrap the body in extra spaces.
    - If multiple functions share the same name in different scopes, every
      matching definition may be updated.

    Example
    -------
        replace_function_body(
            "src/foo.py",
            "process_items",
            "result = [normalize(x) for x in items]\\nreturn result",
        )
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    try:
        new_body_statements = _parse_statement_or_suite(new_body)
    except Exception as e:
        return _err(f"Invalid function body syntax: {e}", path=str(p))

    class ReplaceBodyTransformer(_BaseTransformer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            if m.matches(original_node, m.FunctionDef(name=m.Name(self.name))):
                self._increment()
                return updated_node.with_changes(body=cst.IndentedBlock(body=new_body_statements))
            return updated_node

    transformer = ReplaceBodyTransformer(function_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(f"Function '{function_name}' not found", path=str(p))
    return result


# ---------------------------------------------------------------------
# docstring replacement
# ---------------------------------------------------------------------
def replace_docstring(
    path: str,
    function_name: str,
    new_docstring: str,
) -> dict[str, Any]:
    """Replace or insert the docstring of a function, method, or class.

    Use when
    --------
    You only need to update documentation text while preserving code.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact function or class name
    - `new_docstring`: raw docstring text without surrounding triple quotes

    Agent guidance
    --------------
    - Pass plain text in `new_docstring`; do not include `\"\"\"`.
    - This inserts a docstring if one does not already exist.
    - The target may be either a function or a class with that name.

    Example
    -------
        replace_docstring(
            "src/foo.py",
            "build_report",
            "Build the summary report.\\n\\nArgs:\\n    rows: Input records.",
        )
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    doc_stmt = _make_docstring_stmt(new_docstring)

    def _replace_body_with_docstring(body: cst.BaseSuite) -> cst.BaseSuite:
        if not isinstance(body, cst.IndentedBlock):
            return body
        items = list(body.body)
        if items and _is_docstring_stmt(items[0]):
            items[0] = doc_stmt
        else:
            items.insert(0, doc_stmt)
        return body.with_changes(body=items)

    class ReplaceDocstringTransformer(_BaseTransformer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            if m.matches(original_node, m.FunctionDef(name=m.Name(self.name))):
                self._increment()
                return updated_node.with_changes(
                    body=_replace_body_with_docstring(updated_node.body)
                )
            return updated_node

        def leave_ClassDef(
            self, original_node: cst.ClassDef, updated_node: cst.ClassDef
        ) -> cst.ClassDef:
            if m.matches(original_node, m.ClassDef(name=m.Name(self.name))):
                self._increment()
                return updated_node.with_changes(
                    body=_replace_body_with_docstring(updated_node.body)
                )
            return updated_node

    transformer = ReplaceDocstringTransformer(function_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(f"Function or class '{function_name}' not found", path=str(p))
    return result


# ---------------------------------------------------------------------
# decorator replacement
# ---------------------------------------------------------------------
def replace_decorators(
    path: str,
    function_name: str,
    new_decorators: list[str],
) -> dict[str, Any]:
    """Replace all decorators on a function by name.

    Use when
    --------
    You need to add, remove, or reorder decorators on a Python function without
    rewriting the function body.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact function name
    - `new_decorators`: list of decorator expressions, with or without `@`

    Agent guidance
    --------------
    - Pass decorators as strings like `\"classmethod\"` or `\"cache(ttl=30)\"`.
    - To remove all decorators, pass an empty list.
    - This only targets function definitions, not classes.

    Example
    -------
        replace_decorators(
            "src/foo.py",
            "cached_lookup",
            ["staticmethod", "cache(ttl=30)"],
        )
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    try:
        decorators = [_parse_decorator(d) for d in new_decorators]
    except Exception as e:
        return _err(f"Invalid decorator syntax: {e}", path=str(p))

    class ReplaceDecoratorsTransformer(_BaseTransformer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.FunctionDef:
            if m.matches(original_node, m.FunctionDef(name=m.Name(self.name))):
                self._increment()
                return updated_node.with_changes(decorators=decorators)
            return updated_node

    transformer = ReplaceDecoratorsTransformer(function_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(f"Function '{function_name}' not found", path=str(p))
    return result


# ---------------------------------------------------------------------
# call argument replacement
# ---------------------------------------------------------------------
def replace_argument(
    path: str,
    function_name: str,
    arg_name: str,
    new_value: str,
) -> dict[str, Any]:
    """Replace a keyword argument value in calls to a target function.

    Use when
    --------
    You want to update call sites like `foo(timeout=10)` without editing the
    surrounding source text manually.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: called function name
    - `arg_name`: keyword argument name to replace
    - `new_value`: valid Python expression for the new value

    Agent guidance
    --------------
    - This only replaces keyword arguments, not positional arguments.
    - `new_value` must be a valid Python expression such as `30`, `None`, or
      `settings.DEFAULT_TIMEOUT`.
    - Calls matched include both `foo(...)` and `obj.foo(...)`.

    Example
    -------
        replace_argument(
            "src/foo.py",
            "request",
            "timeout",
            "30",
        )
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    try:
        new_expr = _parse_expression(new_value)
    except Exception as e:
        return _err(f"Invalid new argument value: {e}", path=str(p))

    class ReplaceArgTransformer(_BaseTransformer):
        def __init__(self, func_name: str, kw_name: str) -> None:
            super().__init__()
            self.func_name = func_name
            self.kw_name = kw_name

        def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
            if not m.matches(
                original_node.func,
                m.OneOf(
                    m.Name(self.func_name),
                    m.Attribute(attr=m.Name(self.func_name)),
                ),
            ):
                return updated_node

            changed = False
            new_args: list[cst.Arg] = []
            for arg in updated_node.args:
                if arg.keyword and arg.keyword.value == self.kw_name:
                    new_args.append(arg.with_changes(value=new_expr))
                    changed = True
                else:
                    new_args.append(arg)

            if changed:
                self._increment()
                return updated_node.with_changes(args=new_args)
            return updated_node

    transformer = ReplaceArgTransformer(function_name, arg_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(
            f"No calls to '{function_name}' with keyword '{arg_name}' were found",
            path=str(p),
        )
    return result


# ---------------------------------------------------------------------
# insert after function
# ---------------------------------------------------------------------
def insert_after_function(
    path: str,
    function_name: str,
    new_code: str,
) -> dict[str, Any]:
    """Insert top-level Python code immediately after a top-level function.

    Use when
    --------
    You need to add a helper, assignment, or class after a known top-level
    function definition.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact top-level function name after which to insert
    - `new_code`: one or more valid top-level Python statements

    Agent guidance
    --------------
    - `new_code` must be top-level code, not an indented function body.
    - This only targets top-level functions in the module body.
    - If the named function is nested, this tool will not match it.

    Example
    -------
        insert_after_function(
            "src/foo.py",
            "main",
            "HELPER_TABLE = {\\n    'a': 1,\\n    'b': 2,\\n}",
        )
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    try:
        new_statements = _parse_statement_or_suite(new_code)
    except Exception as e:
        return _err(f"Invalid code syntax: {e}", path=str(p))

    class InsertAfterTransformer(_BaseTransformer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
            new_body: list[cst.BaseStatement] = []
            inserted = False
            for stmt in updated_node.body:
                new_body.append(stmt)
                if (
                    isinstance(stmt, cst.FunctionDef)
                    and m.matches(stmt, m.FunctionDef(name=m.Name(self.name)))
                    and not inserted
                ):
                    new_body.extend(new_statements)
                    self._increment()
                    inserted = True
            return updated_node.with_changes(body=new_body)

    transformer = InsertAfterTransformer(function_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(f"Top-level function '{function_name}' not found", path=str(p))
    return result


# ---------------------------------------------------------------------
# delete function
# ---------------------------------------------------------------------
def delete_function(
    path: str,
    function_name: str,
) -> dict[str, Any]:
    """Delete function definitions by name.

    Use when
    --------
    You want to remove one or more Python functions cleanly by symbol name.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact function name to remove

    Agent guidance
    --------------
    - This removes matching function definitions structurally.
    - If more than one function with the same name exists, each matching
      function definition may be removed.
    - Use `find_function_by_name` first if you need to confirm the target.

    Example
    -------
        delete_function("src/foo.py", "legacy_helper")
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    p, old_content, module = parsed

    class DeleteFunctionTransformer(_BaseTransformer):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
        ) -> cst.RemovalSentinel | cst.FunctionDef:
            if m.matches(original_node, m.FunctionDef(name=m.Name(self.name))):
                self._increment()
                return cst.RemoveFromParent()
            return updated_node

    transformer = DeleteFunctionTransformer(function_name)
    result = _apply_transformer(
        p, old_content, module, transformer, edits_applied=transformer.modified_count
    )
    if result["status"] == "ok" and transformer.modified_count == 0:
        return _err(f"Function '{function_name}' not found", path=str(p))
    return result


# ---------------------------------------------------------------------
# lookup helpers
# ---------------------------------------------------------------------
def find_function_by_name(path: str, function_name: str) -> dict[str, Any]:
    """Find a function definition by name and return its source and location.

    Use when
    --------
    You need to inspect a Python function before editing it with a symbol-aware
    tool.

    Required call shape
    -------------------
    - `path`: Python file path
    - `function_name`: exact function name

    Returns
    -------
    A dict containing function name, line/column positions, async status, and
    the rendered definition source.
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    _, _, module = parsed
    wrapper = MetadataWrapper(module)

    class FindFunctionVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, name: str) -> None:
            self.name = name
            self.result: dict[str, Any] | None = None

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if self.result is not None:
                return
            if m.matches(node, m.FunctionDef(name=m.Name(self.name))):
                pos = wrapper.resolve(PositionProvider)[node]
                self.result = {
                    "function_name": node.name.value,
                    "line": pos.start.line,
                    "column": pos.start.column,
                    "end_line": pos.end.line,
                    "end_column": pos.end.column,
                    "is_async": node.asynchronous is not None,
                    "definition": _function_definition_code(node),
                }

    visitor = FindFunctionVisitor(function_name)
    wrapper.visit(visitor)
    if visitor.result is not None:
        return _ok(**visitor.result)
    return _err(f"Function '{function_name}' not found")


def find_class_by_name(path: str, class_name: str) -> dict[str, Any]:
    """Find a class definition by name and return its source and location.

    Use when
    --------
    You need to inspect a Python class before making a symbol-aware edit.

    Required call shape
    -------------------
    - `path`: Python file path
    - `class_name`: exact class name

    Returns
    -------
    A dict containing class name, line/column positions, and the rendered
    class definition source.
    """
    parsed = _parse_module(path)
    if isinstance(parsed, dict):
        return parsed
    _, _, module = parsed
    wrapper = MetadataWrapper(module)

    class FindClassVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, name: str) -> None:
            self.name = name
            self.result: dict[str, Any] | None = None

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            if self.result is not None:
                return
            if m.matches(node, m.ClassDef(name=m.Name(self.name))):
                pos = wrapper.resolve(PositionProvider)[node]
                self.result = {
                    "class_name": node.name.value,
                    "line": pos.start.line,
                    "column": pos.start.column,
                    "end_line": pos.end.line,
                    "end_column": pos.end.column,
                    "definition": _class_definition_code(node),
                }

    visitor = FindClassVisitor(class_name)
    wrapper.visit(visitor)
    if visitor.result is not None:
        return _ok(**visitor.result)
    return _err(f"Class '{class_name}' not found")
