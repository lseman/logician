"""File Ops skill — routing metadata and grammar hints.

All tool implementations live under src/tools/core/*Tool/ and are registered
as always-on core tools. This module provides only:
  - __skill__  : routing/trigger metadata for skill selection
  - __grammars__: constrained-decoding hints for write_file / edit_file
"""

from __future__ import annotations

_CODING_METADATA_ONLY = True

__skill__ = {
    "name": "File Ops",
    "description": (
        "Use for local filesystem reads, writes, path listing, "
        "single-file edits, and git-aware operations."
    ),
    "aliases": [
        "files",
        "file editing",
        "file reads",
        "single file edit",
        "git status",
        "git diff",
    ],
    "triggers": [
        "read this file",
        "write this file",
        "create a new file",
        "list this directory",
        "edit this file",
        "patch this file",
        "fix this function",
        "git status",
        "git diff",
        "show git changes",
        "what did I change",
        "find in file",
        "search file",
        "where is function",
    ],
    "preferred_tools": [
        "search_file",
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "get_git_status",
        "get_git_diff",
    ],
    "example_queries": [
        "open the config file and inspect it",
        "create a new module with this content",
        "list the files in src/db",
        "find the definition of process_batch in agent.py",
        "what's the git status?",
        "show the diff between my changes and HEAD",
        "edit the function foo to return 42 instead of 0",
    ],
    "when_not_to_use": [
        "the task needs coordinated edits across several files — use search_replace or multi_edit instead",
    ],
    "next_skills": ["explore", "multi_edit", "quality"],
    "preferred_sequence": ["search_file", "read_file", "edit_file", "quality"],
    "entry_criteria": [
        "The exact file is known or the user gave a specific path.",
        "The job is a one-file read, write, or tightly scoped edit.",
    ],
    "decision_rules": [
        "Use read_file or search_file before edit_file when the exact block is not already known.",
        "Use write_file for new files or intentional full-file rewrites.",
        "For Python symbol-level edits, prefer structural tools when they are available.",
    ],
    "workflow": [
        "1. search_file(path, pattern) to find the exact text including indentation.",
        "2. edit_file(path, old_string, new_string) to apply the change.",
        "   - On failure: read closest_matches in the error and retry with corrected old_string.",
        "   - For full rewrites: write_file(path, content) instead.",
    ],
    "failure_recovery": [
        "If edit_file says the block was not found or was not unique, re-read the file and include more surrounding context.",
        "If quoting or newline formatting looks wrong, keep the entire payload in one string field instead of splitting it across extra keys.",
    ],
    "exit_criteria": [
        "The intended one-file change is applied without unrelated edits.",
        "A follow-up verification step is chosen when the change affects behavior.",
    ],
    "anti_patterns": [
        "Rewriting an existing file with write_file for a tiny local change.",
        "Splitting code across extra argument keys instead of using content, old_string, or new_string.",
    ],
}

# ---------------------------------------------------------------------------
# Constrained-decoding grammar hints
# ---------------------------------------------------------------------------
# These help local LLM backends (llama.cpp, vllm) generate well-formed
# JSON tool calls for the two most commonly mis-formatted tools.

_JSON_STRING_RULES = r"""
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" escape
escape    ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

_WRITE_FILE_GRAMMAR = (
    r"""root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"write_file\", \"arguments\": " args "}}"
args      ::= "{\"path\": " string ", \"content\": " string opt-mode "}"
opt-mode  ::= "" | ", \"mode\": " mode
mode      ::= "\"w\"" | "\"a\""
"""
    + _JSON_STRING_RULES
)

_EDIT_FILE_GRAMMAR = (
    r"""root      ::= tool-call
tool-call ::= "{\"tool_call\": {\"name\": \"edit_file\", \"arguments\": " args "}}"
args      ::= "{\"path\": " string ", \"old_string\": " string ", \"new_string\": " string "}"
"""
    + _JSON_STRING_RULES
)

__grammars__: dict[str, str] = {
    "write_file": _WRITE_FILE_GRAMMAR,
    "edit_file": _EDIT_FILE_GRAMMAR,
}
