---
name: Web
description: Use for external documentation and web content retrieval, especially when the answer depends on live URLs, package docs, or public API references.
aliases:
  - docs lookup
  - web search
  - url fetch
triggers:
  - search the web
  - fetch documentation
  - read a web page
preferred_tools:
  - fetch_url
  - web_search
  - pypi_info
  - github_read_file
example_queries:
  - fetch the package docs for `numpy`
  - search GitHub source for this API
  - read the webpage for this library and summarize it
when_not_to_use:
  - content is already available in the local repo
  - the task only requires offline code edits
next_skills:
  - coding/explore
  - qol/firecrawl
workflow:
  - Use `fetch_url` for single-page lookups and `web_search` for broader documentation queries.
  - Use `pypi_info` for package metadata and `github_read_file` for source-level inspection.
---

## Role

This skill routes the agent to shared web tools for retrieving public content, documentation, and package references. It complements local search and code inspection by letting the model access live web resources.

## Tools

- `fetch_url(url)`
- `web_search(query)`
- `pypi_info(package)`
- `github_read_file(path)`

## Implementation

The skill metadata is maintained in `skills/qol/web/scripts/web.py`.
