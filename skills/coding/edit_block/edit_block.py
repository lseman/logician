"""SOTA Search/Replace block tool.

Provides an Aider-style Search/Replace block application tool, which is more
robust for LLMs than unified diffs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import json as _json_mod
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "Edit Block",
    "description": "Use for precise block-level edits and targeted code replacements inside files.",
    "aliases": ["block replace", "targeted replace", "exact code block edit"],
    "triggers": [
        "replace this function body",
        "update this exact block",
        "make a targeted code replacement",
    ],
    "preferred_tools": ["apply_edit_block"],
    "example_queries": [
        "replace this method without touching nearby code",
        "update this exact block in place",
        "make a minimal targeted change here",
    ],
    "when_not_to_use": [
        "the change is broad, regex-driven, or better handled as a patch or multi-file edit"
    ],
    "next_skills": ["quality"],
    "workflow": [
        "Use exact surrounding anchors to avoid collateral edits.",
        "Keep replacements minimal and local.",
        "Re-read the file if anchors may have drifted.",
    ],
}

if "_safe_json" not in globals():
    def _safe_json(obj: Any) -> str:
        try:
            return _json_mod.dumps(obj)
        except Exception as e:
            return _json_mod.dumps({"status": "error", "error": f"JSON encode error: {e}"})

def _parse_blocks(text: str) -> list[tuple[str, str]]:
    """Parse multiple <<<< SEARCH ==== >>>> REPLACE blocks."""
    blocks = []
    pattern = re.compile(
        r"<{4,}\s*SEARCH\s*\n(.*?)\n?={4,}\s*\n(.*?)\n?>{4,}\s*REPLACE",
        re.DOTALL
    )
    for match in pattern.finditer(text):
        search_text = match.group(1)
        replace_text = match.group(2)
        blocks.append((search_text, replace_text))
    return blocks

def _fuzzy_find_and_replace(content: str, search_text: str, replace_text: str) -> tuple[bool, str, str]:
    """Returns (Success, NewContent, ErrorMessage)."""
    # 1. Exact match
    if content.count(search_text) == 1:
        return True, content.replace(search_text, replace_text), ""
    elif content.count(search_text) > 1:
        return False, content, "Search block is not unique."

    # 2. Try exact match but ignoring leading/trailing empty lines in search_text
    search_stripped = search_text.strip("\r\n")
    if search_stripped and content.count(search_stripped) == 1:
        return True, content.replace(search_stripped, replace_text.strip("\r\n")), ""

    # 3. Line-by-line fuzzy match (ignoring exact indentation and whitespace, but matching relative)
    content_lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines()

    # Remove leading/trailing empty lines from search
    while search_lines and not search_lines[0].strip():
        search_lines.pop(0)
    while search_lines and not search_lines[-1].strip():
        search_lines.pop()

    if not search_lines:
        return False, content, "Search block is empty or only whitespace."

    # Find windows where stripped lines match
    matches = []
    for i in range(len(content_lines) - len(search_lines) + 1):
        match = True
        for j, s_line in enumerate(search_lines):
            c_line = content_lines[i + j]
            if s_line.strip() != c_line.strip():
                match = False
                break
        if match:
            matches.append(i)

    if len(matches) == 0:
        return False, content, "Search block not found exactly or with fuzzy whitespace matching."
    if len(matches) > 1:
        return False, content, "Search block is not unique (fuzzy match found multiple)."

    # Exactly one fuzzy match!
    match_idx = matches[0]

    # Compute indentation difference based on first non-empty line
    orig_indent = len(content_lines[match_idx]) - len(content_lines[match_idx].lstrip())
    search_indent = len(search_lines[0]) - len(search_lines[0].lstrip())
    indent_diff = orig_indent - search_indent

    # Adjust replace_text indentation
    replace_lines = replace_text.splitlines(keepends=True)
    if indent_diff > 0:
        prefix = " " * indent_diff
        replace_lines = [(prefix + line if line.strip() else line) for line in replace_lines]
    elif indent_diff < 0:
        prefix_len = -indent_diff
        replace_lines = [(line[prefix_len:] if len(line) - len(line.lstrip()) >= prefix_len else line.lstrip("\t ")) for line in replace_lines]

    # Ensure the last line of replace_lines has a newline if it originally didn't, but the replaced block did
    if replace_lines and not replace_lines[-1].endswith("\n"):
        last_replaced = content_lines[match_idx + len(search_lines) - 1]
        if last_replaced.endswith("\n"):
            replace_lines[-1] = replace_lines[-1] + "\n"

    # Reconstruct content
    new_content_lines = content_lines[:match_idx] + replace_lines + content_lines[match_idx + len(search_lines):]
    return True, "".join(new_content_lines), ""


@tool
def apply_edit_block(path: str, blocks: str) -> str:
    """Use when: You need to modify a file using SEARCH/REPLACE blocks (SOTA coding approach).

    Triggers: search replace block, ader block, edit block, modify file block.
    Avoid when: The file is brand new (use write_file).
    Inputs:
      path (str, required): The target file path.
      blocks (str, required): A string containing one or more blocks like:
        <<<< SEARCH
        def old_code():
        ====
        def new_code():
        >>>> REPLACE
    Returns: JSON with status and number of blocks applied.
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return _safe_json({"status": "error", "error": f"File not found: {path}"})

        parsed = _parse_blocks(blocks)
        if not parsed:
            return _safe_json({"status": "error", "error": "No valid SEARCH/REPLACE blocks found in the input. Remember to use <<<< SEARCH, ====, and >>>> REPLACE lines."})

        content = p.read_text(encoding="utf-8", errors="replace")

        applied = 0
        for i, (search_text, replace_text) in enumerate(parsed):
            if not search_text and replace_text:
                continue

            ok, content, err = _fuzzy_find_and_replace(content, search_text, replace_text)
            if not ok:
                return _safe_json(
                    {
                        "status": "error",
                        "error": f"Block {i+1} failed: {err}"
                    }
                )
            applied += 1

        p.write_text(content, encoding="utf-8")
        return _safe_json({"status": "ok", "path": str(p), "blocks_applied": applied})
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


# Keep the skill metadata/grammar export, but avoid registering a second
# implementation for the canonical core apply_edit_block tool.
__tools__: list = []


_APPLY_EDIT_BLOCK_GRAMMAR = r"""root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"apply_edit_block\", \"arguments\": " args "}}"
args      ::= "{\"path\": " string ", \"blocks\": " string "}"
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" escape
escape    ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

__grammars__: dict[str, str] = {
    "apply_edit_block": _APPLY_EDIT_BLOCK_GRAMMAR,
}
