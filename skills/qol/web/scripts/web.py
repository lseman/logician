"""Shared web skill routing metadata.

The concrete web tools now live in src/tools/core/web.py so they are always-on
core capabilities. This module stays metadata-only to keep QoL routing hints
without registering shadow copies.
"""

from __future__ import annotations

_QOL_METADATA_ONLY = True

__skill__ = {
    "name": "Web",
    "description": "Tools for interacting with the internet, searching the web, and fetching URL contents.",
    "aliases": ["docs lookup", "web search", "url fetch", "package docs"],
    "triggers": [
        "look up the docs",
        "fetch this url",
        "search the web",
        "read the package documentation",
    ],
    "preferred_tools": ["fetch_url", "web_search", "pypi_info", "github_read_file"],
    "example_queries": [
        "fetch the Python docs page for pathlib",
        "search for this stack trace online",
        "read the source file on GitHub",
    ],
    "when_not_to_use": [
        "the answer is already available in the local repo",
        "a dedicated crawler like firecrawl is the better fit",
    ],
    "next_skills": ["coding/explore", "firecrawl"],
    "workflow": [
        "Prefer exact URLs when known.",
        "Use search for broader discovery.",
        "Bring back only the relevant external facts.",
    ],
}
