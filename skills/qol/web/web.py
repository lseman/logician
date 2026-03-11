from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from skills.coding.bootstrap.runtime_access import tool

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Strip HTML tags and collect visible text."""

    _SKIP = {"script", "style", "head", "noscript", "svg", "iframe"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _html_to_text(raw_html: str) -> str:
    """Convert raw HTML to plain text."""
    parser = _TextExtractor()
    parser.feed(raw_html)
    text = parser.get_text()
    text = html.unescape(text)
    # Collapse runs of 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _http_get(
    url: str, timeout: int = 15, headers: dict | None = None
) -> tuple[int, str]:
    """Return (status_code, response_text). Raises on network error."""
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (compatible; CodingAgent/1.0; +https://github.com)",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = "utf-8"
        ct = resp.headers.get_content_charset()
        if ct:
            charset = ct
        return resp.status, resp.read().decode(charset, errors="replace")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def fetch_url(url: str, timeout: int = 15, max_chars: int = 12000) -> str:
    """Use when: Read the contents of a webpage to answer a question or extract information.

    Triggers: fetch url, open url, read page, get webpage, browse, look up website, what does this page say.
    Avoid when: You only need a web search — use web_search instead.
    Inputs:
      url (str): Full URL including scheme (https://...).
      timeout (int, optional): Request timeout in seconds (default 15).
      max_chars (int, optional): Truncate text output to this many characters (default 12000).
    Returns: JSON with {url, status_code, text, truncated}.
    Side effects: Outbound HTTP request.
    """
    try:
        status, body = _http_get(url, timeout=timeout)
        text = _html_to_text(body)
        truncated = len(text) > max_chars
        return _safe_json(
            {
                "url": url,
                "status_code": status,
                "text": text[:max_chars],
                "truncated": truncated,
                "char_count": len(text),
            }
        )
    except Exception as exc:
        return _safe_json({"url": url, "status": "error", "error": str(exc)})


@tool
def web_search(query: str, n: int = 6) -> str:
    """Use when: Need up-to-date information, documentation, or to look up errors online.

    Triggers: search, look up, find online, google, duckduckgo, web search, stackoverflow, what is, how to.
    Avoid when: You already have the URL — use fetch_url instead.
    Inputs:
      query (str): Search query string.
      n (int, optional): Max number of results to return (default 6, max 20).
    Returns: JSON list of {title, url, snippet}.
    Side effects: Outbound HTTP request to duckduckgo.com.
    """
    n = min(max(1, n), 20)
    encoded = urllib.parse.quote_plus(query)
    ddg_url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    try:
        _, body = _http_get(ddg_url, timeout=15)
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc), "query": query})

    # DDG lite returns an HTML table; extract result rows
    results: list[dict] = []

    # DDG Lite uses single-quoted attributes and places href before class.
    # Match the full opening tag of result links, then extract href separately.
    link_tag_re = re.compile(
        r"<a\b([^>]*?\bclass=['\"]result-link['\"][^>]*)>(.*?)</a>",
        re.DOTALL | re.IGNORECASE,
    )
    href_attr_re = re.compile(r'\bhref="([^"]+)"')
    snippet_re = re.compile(
        r"<td\b[^>]*\bclass=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        re.DOTALL | re.IGNORECASE,
    )

    links = []
    for attrs, title_raw in link_tag_re.findall(body):
        m = href_attr_re.search(attrs)
        if m:
            links.append((m.group(1), title_raw))

    def _snippet_text(raw: str) -> str:
        # Collapse newlines so bold-wrapped words don't produce line breaks
        return re.sub(r"\s+", " ", html.unescape(_html_to_text(raw))).strip()

    snippets = [_snippet_text(s) for s in snippet_re.findall(body)]

    for i, (href, title_raw) in enumerate(links[:n]):
        # DDG lite wraps results in redirect URLs; extract the real URL
        if "uddg=" in href:
            m = re.search(r"uddg=([^&]+)", href)
            if m:
                href = urllib.parse.unquote(m.group(1))
        title = html.unescape(_html_to_text(title_raw))
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({"title": title, "url": href, "snippet": snippet})

    if not results:
        # Fallback: generic link extraction
        generic_re = re.compile(r'href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        seen: set[str] = set()
        for href, txt in generic_re.findall(body):
            if "duckduckgo.com" in href:
                continue
            if href in seen:
                continue
            seen.add(href)
            results.append(
                {"title": html.unescape(_html_to_text(txt)), "url": href, "snippet": ""}
            )
            if len(results) >= n:
                break

    return _safe_json(
        {"status": "ok", "query": query, "count": len(results), "results": results}
    )


@tool
def pypi_info(package_name: str) -> str:
    """Use when: Need to check if a package exists, what version is latest, or its description.

    Triggers: pypi, pip install, package info, what version, latest version, does package exist, check package.
    Avoid when: You want to install the package — use install_packages instead.
    Inputs:
      package_name (str): PyPI package name (e.g. "numpy", "requests").
    Returns: JSON with {name, version, summary, home_url, requires_python, license, releases_count}.
    Side effects: Outbound HTTP request to pypi.org.
    """
    url = f"https://pypi.org/pypi/{urllib.parse.quote(package_name)}/json"
    try:
        _, body = _http_get(url, timeout=10)
        data = json.loads(body)
        info = data.get("info", {})
        return _safe_json(
            {
                "status": "ok",
                "name": info.get("name"),
                "version": info.get("version"),
                "summary": info.get("summary"),
                "home_url": info.get("home_page") or info.get("project_url"),
                "requires_python": info.get("requires_python"),
                "license": info.get("license"),
                "author": info.get("author"),
                "keywords": info.get("keywords"),
                "releases_count": len(data.get("releases", {})),
                "project_urls": info.get("project_urls") or {},
            }
        )
    except Exception as exc:
        return _safe_json(
            {"status": "error", "package": package_name, "error": str(exc)}
        )


@tool
def github_read_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "main",
) -> str:
    """Use when: Need to read source code from a public GitHub repo to understand an API or copy a snippet.

    Triggers: github, read file from repo, look at source, check github, raw file, repo file, open source code.
    Avoid when: The repo is private (no auth support) or you need to clone the full repo.
    Inputs:
      owner (str): GitHub username or organisation.
      repo (str): Repository name.
      path (str): File path within the repo (e.g. "src/utils.py").
      ref (str, optional): Branch, tag, or commit SHA (default "main").
    Returns: JSON with {url, content, lines, truncated}.
    Side effects: Outbound HTTP request to raw.githubusercontent.com.
    """
    url = (
        f"https://raw.githubusercontent.com/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}/"
        f"{urllib.parse.quote(ref)}/{path}"
    )
    try:
        status, content = _http_get(url, timeout=15)
        if status == 404:
            # Try "master" as fallback if ref was "main"
            if ref == "main":
                fallback_url = url.replace("/main/", "/master/")
                status, content = _http_get(fallback_url, timeout=15)
                if status == 200:
                    url = fallback_url
                else:
                    return _safe_json(
                        {
                            "status": "not_found",
                            "owner": owner,
                            "repo": repo,
                            "path": path,
                            "ref": ref,
                        }
                    )
        max_chars = 20000
        truncated = len(content) > max_chars
        lines = content.count("\n")
        return _safe_json(
            {
                "status": "ok",
                "url": url,
                "lines": lines,
                "content": content[:max_chars],
                "truncated": truncated,
            }
        )
    except Exception as exc:
        return _safe_json(
            {
                "status": "error",
                "owner": owner,
                "repo": repo,
                "path": path,
                "error": str(exc),
            }
        )


__tools__ = [fetch_url, web_search, pypi_info, github_read_file]
