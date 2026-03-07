#!/usr/bin/env python3
"""Validate SOUL.md tool references against registered skill tools.

Rules:
1) Parse tool names from Python skills by detecting functions decorated with @llm.tool(...).
2) Parse SOUL references from bullet lines that start with `tool_name` in backticks.
3) Fail if SOUL references a tool that is not registered.
4) Report duplicate tool names across skill modules as warnings.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


def _is_llm_tool_decorator(decorator: ast.expr) -> bool:
    """Return True when decorator matches @llm.tool or @llm.tool(...)."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(target, ast.Attribute):
        return False
    if target.attr != "tool":
        return False
    return isinstance(target.value, ast.Name) and target.value.id == "llm"


def collect_skill_tools(skills_dir: Path) -> dict[str, list[Path]]:
    """Collect tool_name -> [module_paths] from skill Python files."""
    tool_files: dict[str, list[Path]] = defaultdict(list)

    for py_file in sorted(skills_dir.rglob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_is_llm_tool_decorator(dec) for dec in node.decorator_list):
                tool_files[node.name].append(py_file)

    return dict(tool_files)


def collect_soul_tool_refs(soul_path: Path) -> list[tuple[str, int]]:
    """Collect (tool_name, line_no) from SOUL bullet lines.

    Example accepted line:
      - `run_shell`
      - `run_shell` (with additional text)
    """
    refs: list[tuple[str, int]] = []
    bullet_tool = re.compile(r"^\s*-\s+`([A-Za-z_][A-Za-z0-9_]*)`(?:\s|$)")

    for line_no, line in enumerate(
        soul_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = bullet_tool.search(line)
        if match:
            refs.append((match.group(1), line_no))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soul",
        type=Path,
        default=Path("SOUL.md"),
        help="Path to SOUL markdown file (default: SOUL.md)",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path("skills"),
        help="Skills directory to scan (default: skills)",
    )
    parser.add_argument(
        "--fail-on-duplicate",
        action="store_true",
        help="Fail when duplicate tool names are found across skill modules.",
    )
    args = parser.parse_args()

    soul_path = args.soul.resolve()
    skills_dir = args.skills_dir.resolve()

    if not soul_path.exists():
        print(f"ERROR: SOUL file not found: {soul_path}", file=sys.stderr)
        return 2
    if not skills_dir.exists():
        print(f"ERROR: skills dir not found: {skills_dir}", file=sys.stderr)
        return 2

    try:
        tool_files = collect_skill_tools(skills_dir)
    except SyntaxError as exc:
        print(
            f"ERROR: failed parsing skill file {exc.filename}:{exc.lineno}: {exc.msg}",
            file=sys.stderr,
        )
        return 2

    registered_tools = set(tool_files)
    refs = collect_soul_tool_refs(soul_path)

    missing: list[tuple[str, int]] = []
    for tool_name, line_no in refs:
        if tool_name not in registered_tools:
            missing.append((tool_name, line_no))

    duplicates = {
        name: paths for name, paths in sorted(tool_files.items()) if len(paths) > 1
    }

    if missing:
        print("SOUL tool reference errors:")
        for tool_name, line_no in missing:
            print(f"  - {tool_name} (line {line_no}) is not a registered tool")
        print(
            f"FAILED: {len(missing)} invalid reference(s), "
            f"{len(refs)} SOUL tool reference(s) checked."
        )
    else:
        print(
            f"OK: {len(refs)} SOUL tool reference(s) validated against "
            f"{len(registered_tools)} registered tool(s)."
        )

    if duplicates:
        print("Duplicate registered tool names (warning):")
        for name, paths in duplicates.items():
            where = ", ".join(str(p.relative_to(skills_dir.parent)) for p in paths)
            print(f"  - {name}: {where}")
        if args.fail_on_duplicate:
            print("FAILED: duplicate tools found and --fail-on-duplicate is set.")
            return 1

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
