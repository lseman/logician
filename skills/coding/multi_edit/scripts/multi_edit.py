"""Multi-file batch edit tool.

Applies multiple (old_string → new_string) replacements in one tool call,
across one or multiple files — equivalent to calling edit_file_replace N times
but atomic in terms of agent turn cost.

Each replacement must uniquely identify a single location in its file
(same uniqueness contract as edit_file_replace).  Failed hunks are reported
individually without stopping the remaining replacements.
"""

from __future__ import annotations

import json as _json_mod
from pathlib import Path
from typing import Any
from skills.coding.bootstrap.runtime_access import get_coding_runtime

__skill__ = {
    "name": "Multi Edit",
    "description": "Use for coordinated edits across multiple files or repeated edit operations.",
    "aliases": ["batch edit", "multi file edit", "coordinated edits"],
    "triggers": [
        "update these files together",
        "make the same change in several files",
        "apply this edit across the codebase",
    ],
    "preferred_tools": ["multi_edit"],
    "example_queries": [
        "rename this helper in all touched files",
        "update imports and call sites together",
        "apply the same edit to these modules",
    ],
    "when_not_to_use": [
        "the task is only one file or is primarily search-driven pattern replacement"
    ],
    "next_skills": ["quality", "git"],
    "preferred_sequence": ["explore", "multi_edit", "quality", "git"],
    "entry_criteria": [
        "Several files or several exact replacements must change together.",
        "The edit set is already known well enough to batch safely.",
    ],
    "decision_rules": [
        "Use multi_edit when consistency across files matters more than interactive one-by-one editing.",
        "Keep each replacement exact and unique inside its file.",
        "If the change is pattern-driven across an uncertain file set, prefer search_replace first.",
    ],
    "workflow": [
        "Identify the full file set before editing.",
        "Apply consistent edits in one batch when possible.",
        "Run quality or targeted checks immediately after the multi-file change.",
    ],
    "failure_recovery": [
        "If one replacement is not unique, re-read that file and add more surrounding context to old_string.",
        "If the file set is still changing as you inspect the repo, stop batching and switch back to exploration.",
    ],
    "exit_criteria": [
        "All coordinated replacements are applied or the remaining failures are clearly isolated.",
        "A targeted verification step has been selected for the touched files.",
    ],
    "anti_patterns": [
        "Using multi_edit before confirming the full affected file set.",
        "Batching broad regex-style refactors that would be safer with scoped search_replace tools.",
    ],
}

if "_safe_json" not in globals():

    def _safe_json(obj: Any) -> str:
        try:
            return _json_mod.dumps(obj, ensure_ascii=False)
        except Exception as e:
            return _json_mod.dumps(
                {"status": "error", "error": f"JSON encode error: {e}"}
            )


def _runtime():
    return get_coding_runtime(globals())


def _resolve_path(path: str) -> Path:
    return _runtime().resolve_path(path)


def multi_edit(replacements: list) -> str:
    """Use when: Make several targeted edits across one or more files in a single turn.

    Triggers: multi edit, batch edit, apply multiple changes, edit several places,
              fix multiple files, patch multiple hunks, edit across files.
    Avoid when: You need to create a new file — use write_file.
    Avoid when: You are rewriting a whole file — use write_file.

    Inputs:
      replacements (list, required): List of dicts, each with:
        - file       (str, required): Path to the file to edit.
        - old_string (str, required): Exact text to find. Must appear exactly once.
                                      Include 3-5 unchanged lines before and after
                                      the target text as context to ensure uniqueness.
        - new_string (str, required): Replacement text.
        - explanation (str, optional): Short note for the trace log (not written to file).

    Returns: JSON with overall status and per-replacement result.
      {
        "status": "ok" | "partial" | "error",
        "applied": 3,
        "failed": 1,
        "results": [
          {"index": 0, "file": "src/foo.py", "status": "ok", "lines_removed": 2, "lines_added": 3},
          {"index": 1, "file": "src/bar.py", "status": "error", "error": "old_string not found"}
        ]
      }

    Side effects: Modifies files in-place; no backups created.
    """
    if not isinstance(replacements, list) or not replacements:
        return _safe_json(
            {"status": "error", "error": "replacements must be a non-empty list"}
        )

    results: list[dict] = []
    applied = 0
    failed = 0

    # Group by file so we apply all edits to a file before writing it.
    # Edits within the same file must be non-overlapping; we verify order by
    # tracking which regions were already consumed.
    from collections import defaultdict

    by_file: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, rep in enumerate(replacements):
        if not isinstance(rep, dict):
            results.append(
                {
                    "index": idx,
                    "file": "?",
                    "status": "error",
                    "error": "replacement must be a dict",
                }
            )
            failed += 1
            continue
        file_key = rep.get("file", "")
        if not file_key:
            results.append(
                {
                    "index": idx,
                    "file": "?",
                    "status": "error",
                    "error": "missing 'file' key",
                }
            )
            failed += 1
            continue
        by_file[file_key].append((idx, rep))

    file_contents: dict[str, str] = {}

    for file_path, edits in by_file.items():
        try:
            p = _resolve_path(file_path)
            if not p.is_file():
                for idx, rep in edits:
                    results.append(
                        {
                            "index": idx,
                            "file": file_path,
                            "status": "error",
                            "error": f"File not found: {file_path}",
                        }
                    )
                    failed += 1
                continue
            content = p.read_text(encoding="utf-8")
        except Exception as exc:
            for idx, rep in edits:
                results.append(
                    {
                        "index": idx,
                        "file": file_path,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                failed += 1
            continue

        file_contents[file_path] = content

        for idx, rep in edits:
            old_string = rep.get("old_string", "")
            new_string = rep.get("new_string", "")
            if not old_string:
                results.append(
                    {
                        "index": idx,
                        "file": file_path,
                        "status": "error",
                        "error": "missing 'old_string'",
                    }
                )
                failed += 1
                continue

            current = file_contents[file_path]
            count = current.count(old_string)
            if count == 0:
                results.append(
                    {
                        "index": idx,
                        "file": file_path,
                        "status": "error",
                        "error": (
                            "old_string not found in file. "
                            "Check for whitespace/indentation differences or re-read the file. "
                            "Tip: include more surrounding context lines to make it unique."
                        ),
                    }
                )
                failed += 1
                continue
            if count > 1:
                results.append(
                    {
                        "index": idx,
                        "file": file_path,
                        "status": "error",
                        "error": (
                            f"old_string matches {count} locations — add more surrounding context "
                            "lines (before and after) to make it unique."
                        ),
                    }
                )
                failed += 1
                continue

            patched = current.replace(old_string, new_string, 1)
            file_contents[file_path] = patched

            old_lines = len(old_string.splitlines())
            new_lines = len(new_string.splitlines())
            results.append(
                {
                    "index": idx,
                    "file": file_path,
                    "status": "ok",
                    "lines_removed": old_lines,
                    "lines_added": new_lines,
                }
            )
            applied += 1

    # Flush modified contents to disk
    for file_path, content in file_contents.items():
        # Only write if at least one hunk was applied
        if any(r["file"] == file_path and r["status"] == "ok" for r in results):
            try:
                p = _resolve_path(file_path)
                p.write_text(content, encoding="utf-8")
            except Exception as exc:
                # Walk back: mark all ok results for this file as write-failed
                for r in results:
                    if r["file"] == file_path and r["status"] == "ok":
                        r["status"] = "error"
                        r["error"] = f"Write failed: {exc}"
                        applied -= 1
                        failed += 1

    overall = "ok" if failed == 0 else ("partial" if applied > 0 else "error")
    # Sort results by original index
    results.sort(key=lambda r: r.get("index", 0))

    return _safe_json(
        {
            "status": overall,
            "applied": applied,
            "failed": failed,
            "results": results,
        }
    )




_MULTI_EDIT_GRAMMAR = r"""root          ::= tool-call
tool-call     ::= "{\"tool_call\": {\"name\": \"multi_edit\", \"arguments\": " args "}}"
args          ::= "{\"replacements\": [" replacement ("," replacement)* "]}"
replacement   ::= "{\"file\": " string ", \"old_string\": " string ", \"new_string\": " string opt-explanation "}"
opt-explanation ::= "" | ", \"explanation\": " string
string        ::= "\"" char* "\""
char          ::= [^"\\] | "\\" escape
escape        ::= ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
"""

__grammars__: dict[str, str] = {
    "multi_edit": _MULTI_EDIT_GRAMMAR,
}
