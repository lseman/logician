from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "Firecrawl",
    "description": "Use for deep web crawling, full-site scraping, and structured content ingestion via the Firecrawl API.",
    "aliases": ["crawl", "site scrape", "web extraction", "firecrawl ingest"],
    "triggers": [
        "crawl this site",
        "scrape all pages under this URL",
        "extract structured content from this domain",
        "fetch javascript-rendered page",
    ],
    "preferred_tools": ["firecrawl_search", "firecrawl_crawl", "firecrawl_scrape"],
    "example_queries": [
        "crawl the docs site and ingest all pages",
        "scrape this React-rendered docs page into markdown",
        "search the docs site before crawling",
    ],
    "when_not_to_use": [
        "a single static page that fetch_url handles fine",
        "content is already in the local repo or RAG index",
        "the site blocks crawlers",
    ],
    "next_skills": ["rag", "coding/explore"],
    "workflow": [
        "Use search first when site scope is unclear.",
        "Crawl only the needed URL prefixes.",
        "Prefer markdown output for downstream LLM and RAG use.",
    ],
}

_FIRECRAWL_BASE_URL = os.environ.get("FIRECRAWL_URL", "http://localhost:3002")
_FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")


@tool
def firecrawl_search(query: str, limit: int = 5, scrape_content: bool = True) -> str:
    """Use when: Search the web via the self-hosted Firecrawl API and return clean markdown content for the top results.

    Triggers: search the web, look up online, find information about, scrape website, crawl page, search internet, google, look online, fetch url, read webpage.
    Avoid when: The user is asking about data already loaded in the context..
    Inputs: query (str, required): Search query string; limit (int, optional): Maximum number of results to return (default 5); scrape_content (bool, optional): If True, include scraped markdown body for each result (default True).
    Returns: JSON with search results including URLs, titles, and content.
    Side effects: performs external HTTP calls to the configured Firecrawl endpoint.
    """
    url = f"{_FIRECRAWL_BASE_URL.rstrip('/')}/v1/search"
    limit = max(1, min(int(limit), 20))
    payload: dict[str, Any] = {"query": str(query), "limit": limit}
    if scrape_content:
        payload["scrapeOptions"] = {"formats": ["markdown"]}

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return _safe_json(
            {
                "status": "error",
                "error": f"HTTP {exc.code}: {exc.reason}",
                "detail": err_body[:400],
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})

    if not result.get("success", True) and "error" in result:
        return _safe_json({"status": "error", "error": result["error"]})

    raw_data = result.get("data") or result.get("results") or []
    results_out = []
    for item in raw_data:
        entry = {
            "url": item.get("url") or item.get("sourceURL", ""),
            "title": item.get("title") or item.get("metadata", {}).get("title", ""),
            "description": item.get("description")
            or item.get("metadata", {}).get("description", ""),
        }
        md = item.get("markdown") or item.get("content") or ""
        if scrape_content and md:
            entry["content"] = md[:4000].strip() + (" ..." if len(md) > 4000 else "")
        results_out.append(entry)

    return _safe_json(
        {
            "status": "ok",
            "query": query,
            "count": len(results_out),
            "results": results_out,
        }
    )


@tool
def firecrawl_scrape(
    url: str, max_chars: int = 6000, include_links: bool = False
) -> str:
    """Use when: Scrape a single URL via the self-hosted Firecrawl API and return its content as clean markdown.

    Triggers: search the web, look up online, find information about, scrape website, crawl page, search internet, google, look online, fetch url, read webpage.
    Avoid when: No internet access is needed and the answer can be derived from existing data..
    Inputs: url (str, required): The URL to scrape; max_chars (int, optional): Maximum content characters to return (default 6000); include_links (bool, optional): Include extracted links section (default False).
    Returns: JSON with page title, description, and markdown content.
    Side effects: performs external HTTP calls to the configured Firecrawl endpoint.
    """
    endpoint = f"{_FIRECRAWL_BASE_URL.rstrip('/')}/v1/scrape"
    formats = ["markdown"]
    if include_links:
        formats.append("links")

    payload = {"url": str(url), "formats": formats}

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return _safe_json(
            {
                "status": "error",
                "error": f"HTTP {exc.code}: {exc.reason}",
                "detail": err_body[:400],
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})

    if not result.get("success", True) and "error" in result:
        return _safe_json({"status": "error", "error": result["error"]})

    page = result.get("data") or result
    metadata = page.get("metadata") or {}
    md = page.get("markdown") or page.get("content") or ""
    max_chars = max(200, int(max_chars))
    truncated = len(md) > max_chars
    md_out = md[:max_chars].strip() + (" ..." if truncated else "")

    out: dict[str, Any] = {
        "status": "ok",
        "url": url,
        "title": metadata.get("title") or page.get("title", ""),
        "description": metadata.get("description") or "",
        "content": md_out,
        "truncated": truncated,
    }
    if include_links:
        out["links"] = page.get("links") or []

    return _safe_json(out)


@tool
def firecrawl_crawl(url: str, limit: int = 10, include_paths: str = "") -> str:
    """Use when: Crawl a website starting from a root URL via the self-hosted Firecrawl API, collecting content from multiple pages.

    Triggers: search the web, look up online, find information about, scrape website, crawl page, search internet, google, look online, fetch url, read webpage.
    Avoid when: The user is asking about data already loaded in the context..
    Inputs: url (str, required): Root URL to start crawling from; limit (int, optional): Maximum number of pages to crawl (default 10, max 50); include_paths (str, optional): Comma-separated include path filters.
    Returns: JSON with crawled pages including titles and content snippets.
    Side effects: performs external HTTP calls to the configured Firecrawl endpoint.
    """
    endpoint = f"{_FIRECRAWL_BASE_URL.rstrip('/')}/v1/crawl"
    limit = max(1, min(int(limit), 50))

    payload: dict[str, Any] = {
        "url": str(url),
        "limit": limit,
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if include_paths.strip():
        payload["includePaths"] = [
            part.strip() for part in include_paths.split(",") if part.strip()
        ]

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return _safe_json(
            {
                "status": "error",
                "error": f"HTTP {exc.code}: {exc.reason}",
                "detail": err_body[:400],
            }
        )
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})

    if not result.get("success", True) and "error" in result:
        return _safe_json({"status": "error", "error": result["error"]})

    pages = result.get("data") or result.get("pages") or []
    out_pages = []
    for item in pages:
        md = item.get("markdown") or item.get("content") or ""
        out_pages.append(
            {
                "url": item.get("url") or item.get("sourceURL", ""),
                "title": item.get("title") or item.get("metadata", {}).get("title", ""),
                "content": md[:2500].strip() + (" ..." if len(md) > 2500 else ""),
            }
        )

    return _safe_json(
        {"status": "ok", "url": url, "count": len(out_pages), "pages": out_pages}
    )


__all__ = ["firecrawl_search", "firecrawl_scrape", "firecrawl_crawl"]


__tools__ = [firecrawl_search, firecrawl_scrape, firecrawl_crawl]
