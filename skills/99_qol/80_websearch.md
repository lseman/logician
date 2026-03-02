---
name: Web Search
summary: Search the web and scrape URLs using a self-hosted Firecrawl instance, returning clean markdown content for analysis or context.
triggers:
  - search the web
  - look up online
  - find information about
  - scrape website
  - crawl page
  - search internet
  - google
  - look online
  - fetch url
  - read webpage
anti_triggers:
  - load csv
  - forecast horizon
  - analyze the series
aliases:
  - search
  - web
  - scrape
  - crawl
  - lookup
  - browse
  - internet
preferred_tools:
  - firecrawl_search
  - firecrawl_scrape
  - firecrawl_crawl
example_queries:
  - Search the web for recent papers on time series anomaly detection.
  - Scrape the content of https://example.com and summarize it.
  - Look up the latest statsmodels release notes online.
when_not_to_use:
  - The user is asking about data already loaded in the context.
  - No internet access is needed and the answer can be derived from existing data.
next_skills:
  - analysis
  - recommendations
---

## Bootstrap: firecrawl_config

**Description:**
Sets `_FIRECRAWL_BASE_URL` and `_FIRECRAWL_API_KEY` in the tool execution globals.
The URL defaults to `http://localhost:3002` and can be overridden with the
`FIRECRAWL_URL` environment variable. The API key defaults to an empty string
(not required for self-hosted instances) and can be set via `FIRECRAWL_API_KEY`.

**Implementation:**
```python
import os as _os

_FIRECRAWL_BASE_URL = _os.environ.get("FIRECRAWL_URL", "http://localhost:3002")
_FIRECRAWL_API_KEY = _os.environ.get("FIRECRAWL_API_KEY", "")
```

---

## Tool: firecrawl_search

**Description:** Search the web via the self-hosted Firecrawl API and return clean markdown content for the top results.

**Parameters:**
- query (str, required): Search query string
- limit (int, optional): Maximum number of results to return (default 5)
- scrape_content (bool, optional): If True, include scraped markdown body for each result (default True)

**Returns:** JSON with search results including URLs, titles, and content

**Implementation:**
```python
def firecrawl_search(query, limit=5, scrape_content=True):
    """Search the web using the self-hosted Firecrawl API."""
    import urllib.request
    import urllib.error

    base_url = _FIRECRAWL_BASE_URL
    url = f"{base_url.rstrip('/')}/v1/search"

    limit = max(1, min(int(limit), 20))
    payload = {
        "query": str(query),
        "limit": limit,
    }
    if scrape_content:
        payload["scrapeOptions"] = {"formats": ["markdown"]}

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return _safe_json({
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
            "detail": err_body[:400],
        })
    except Exception as e:
        return _safe_json({"status": "error", "error": str(e)})

    if not result.get("success", True) and "error" in result:
        return _safe_json({"status": "error", "error": result["error"]})

    # Normalise the Firecrawl response shape
    raw_data = result.get("data") or result.get("results") or []
    results_out = []
    for item in raw_data:
        entry = {
            "url": item.get("url") or item.get("sourceURL", ""),
            "title": item.get("title") or item.get("metadata", {}).get("title", ""),
            "description": item.get("description") or item.get("metadata", {}).get("description", ""),
        }
        md = item.get("markdown") or item.get("content") or ""
        if scrape_content and md:
            # Truncate long pages to keep context manageable
            entry["content"] = md[:4000].strip() + (" ..." if len(md) > 4000 else "")
        results_out.append(entry)

    return _safe_json({
        "status": "ok",
        "query": query,
        "count": len(results_out),
        "results": results_out,
    })
```

---

## Tool: firecrawl_scrape

**Description:** Scrape a single URL via the self-hosted Firecrawl API and return its content as clean markdown.

**Parameters:**
- url (str, required): The URL to scrape
- max_chars (int, optional): Maximum content characters to return (default 6000)
- include_links (bool, optional): Include extracted links section (default False)

**Returns:** JSON with page title, description, and markdown content

**Implementation:**
```python
def firecrawl_scrape(url, max_chars=6000, include_links=False):
    """Scrape a URL using the self-hosted Firecrawl API."""
    import urllib.request
    import urllib.error

    base_url = _FIRECRAWL_BASE_URL
    endpoint = f"{base_url.rstrip('/')}/v1/scrape"

    formats = ["markdown"]
    if include_links:
        formats.append("links")

    payload = {
        "url": str(url),
        "formats": formats,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return _safe_json({
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
            "detail": err_body[:400],
        })
    except Exception as e:
        return _safe_json({"status": "error", "error": str(e)})

    if not result.get("success", True) and "error" in result:
        return _safe_json({"status": "error", "error": result["error"]})

    page = result.get("data") or result
    metadata = page.get("metadata") or {}
    md = page.get("markdown") or page.get("content") or ""
    max_chars = max(200, int(max_chars))
    truncated = len(md) > max_chars
    md_out = md[:max_chars].strip()
    if truncated:
        md_out += " ..."

    out = {
        "status": "ok",
        "url": url,
        "title": metadata.get("title") or page.get("title", ""),
        "description": metadata.get("description") or "",
        "content": md_out,
        "truncated": truncated,
        "original_length": len(md),
    }
    if include_links:
        out["links"] = page.get("links") or []

    return _safe_json(out)
```

---

## Tool: firecrawl_crawl

**Description:** Crawl a website starting from a root URL via the self-hosted Firecrawl API, collecting content from multiple pages. Useful for reading docs or multi-page sites.

**Parameters:**
- url (str, required): Root URL to start crawling from
- max_pages (int, optional): Maximum number of pages to crawl (default 5, max 20)
- max_chars_per_page (int, optional): Maximum content characters per page (default 2000)

**Returns:** JSON with crawled pages including titles and content summaries

**Implementation:**
```python
def firecrawl_crawl(url, max_pages=5, max_chars_per_page=2000):
    """Crawl a site using the self-hosted Firecrawl API (synchronous map+scrape)."""
    import urllib.request
    import urllib.error
    import time

    base_url = _FIRECRAWL_BASE_URL
    max_pages = max(1, min(int(max_pages), 20))
    max_chars_per_page = max(200, int(max_chars_per_page))

    # --- Step 1: map the site to get a list of URLs ---
    map_endpoint = f"{base_url.rstrip('/')}/v1/map"
    map_payload = {"url": str(url), "limit": max_pages}

    def _post(endpoint, payload, timeout=30):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        if _FIRECRAWL_API_KEY:
            req.add_header("Authorization", f"Bearer {_FIRECRAWL_API_KEY}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        map_result = _post(map_endpoint, map_payload)
    except Exception as e:
        return _safe_json({"status": "error", "error": f"Map failed: {e}"})

    links = map_result.get("links") or map_result.get("urls") or []
    # Ensure the root is included
    if str(url) not in links:
        links = [str(url)] + list(links)
    links = links[:max_pages]

    if not links:
        return _safe_json({"status": "error", "error": "No pages discovered by Firecrawl map."})

    # --- Step 2: scrape each discovered URL ---
    scrape_endpoint = f"{base_url.rstrip('/')}/v1/scrape"
    pages = []
    for page_url in links:
        try:
            result = _post(scrape_endpoint, {"url": page_url, "formats": ["markdown"]}, timeout=20)
            page = result.get("data") or result
            metadata = page.get("metadata") or {}
            md = page.get("markdown") or page.get("content") or ""
            truncated = len(md) > max_chars_per_page
            pages.append({
                "url": page_url,
                "title": metadata.get("title") or page.get("title", ""),
                "content": md[:max_chars_per_page].strip() + (" ..." if truncated else ""),
                "truncated": truncated,
            })
        except Exception as e:
            pages.append({"url": page_url, "error": str(e)})
        time.sleep(0.1)  # polite pause between requests

    return _safe_json({
        "status": "ok",
        "root_url": url,
        "pages_crawled": len(pages),
        "pages": pages,
    })
```

---
