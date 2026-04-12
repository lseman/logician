"""Coding web skill routing metadata.

The concrete web tools live in skills/qol/web/web.py so they can be shared
outside the coding leaf-skill grouping as well. This module stays metadata-only
to avoid shadowing the richer fetch/search implementations with an older local
copy.
"""

from __future__ import annotations

_CODING_METADATA_ONLY = True

__skill__ = {
    "name": "Web",
    "description": "Use for external documentation, package references, GitHub source lookups, and other web context needed for coding tasks.",
    "aliases": ["docs lookup", "documentation", "api docs", "external reference", "web docs"],
    "triggers": [
        "look up the docs",
        "fetch this url",
        "read the package documentation",
        "search the web for this error",
    ],
    "preferred_tools": ["web_search", "fetch_url", "pypi_info", "github_read_file"],
    "example_queries": [
        "fetch the Python docs page for pathlib",
        "search for this stack trace online",
        "read the API reference for this package",
        "inspect a public GitHub source file for this library",
    ],
    "when_not_to_use": [
        "the answer is already available in the repo or only local files need to be inspected"
    ],
    "next_skills": ["explore", "file_ops", "shell"],
    "preferred_sequence": ["web_search", "fetch_url", "explore", "file_ops"],
    "entry_criteria": [
        "The task depends on external documentation, an exact URL, or current package reference material.",
        "Local repo inspection is insufficient to answer the implementation question.",
    ],
    "decision_rules": [
        "Prefer exact URLs when the source is already known.",
        "Search first when the problem is broad and source selection matters.",
        "Bring back only the technical facts needed for the next local edit or command.",
    ],
    "workflow": [
        "Prefer exact URLs when the source is known.",
        "Use search when the problem is broad or source selection matters.",
        "Bring back only the relevant technical facts, then continue locally.",
    ],
    "failure_recovery": [
        "If a page is noisy or badly rendered, switch to a better source or a richer web skill instead of over-trusting the extracted text.",
        "If external docs conflict with the repo, verify the local code path before editing.",
    ],
    "exit_criteria": [
        "The needed external fact has been reduced to a short actionable takeaway.",
        "The task is ready to continue locally in file_ops, shell, or explore.",
    ],
    "anti_patterns": [
        "Copying large chunks of documentation into context.",
        "Using external docs when the repo already contains the answer.",
    ],
}
