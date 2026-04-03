"""Core web and external-reference tools.

These are always-on, read-only network helpers for documentation lookup,
package metadata, and public GitHub source inspection.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any


class _TextExtractor(HTMLParser):
    """Strip HTML tags and collect visible text."""

    _SKIP = {"script", "style", "head", "noscript", "svg", "iframe"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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
    parser = _TextExtractor()
    parser.feed(raw_html)
    text = parser.get_text()
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _http_get(
    url: str,
    *,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (compatible; LogicianCore/1.0; +https://github.com/lseman/logician)",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.status, resp.read().decode(charset, errors="replace")


def fetch_url(url: str, timeout: int = 15, max_chars: int = 12000) -> dict[str, Any]:
    """Read the text content of a webpage."""
    if max_chars <= 0:
        return {"status": "error", "error": "max_chars must be >= 1", "url": url}
    try:
        status, body = _http_get(url, timeout=timeout)
        text = _html_to_text(body)
        truncated = len(text) > max_chars
        return {
            "status": "ok",
            "url": url,
            "status_code": status,
            "text": text[:max_chars],
            "truncated": truncated,
            "char_count": len(text),
        }
    except Exception as exc:
        return {"status": "error", "url": url, "error": str(exc)}


def web_search(query: str, n: int = 6) -> dict[str, Any]:
    """Search the web for up-to-date documentation or errors."""
    limit = min(max(1, n), 20)
    encoded = urllib.parse.quote_plus(query)
    ddg_url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    try:
        _, body = _http_get(ddg_url, timeout=15)
    except Exception as exc:
        return {"status": "error", "query": query, "error": str(exc)}

    results: list[dict[str, str]] = []
    link_tag_re = re.compile(
        r"<a\b([^>]*?\bclass=['\"]result-link['\"][^>]*)>(.*?)</a>",
        re.DOTALL | re.IGNORECASE,
    )
    href_attr_re = re.compile(r'\bhref="([^"]+)"')
    snippet_re = re.compile(
        r"<td\b[^>]*\bclass=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        re.DOTALL | re.IGNORECASE,
    )

    links: list[tuple[str, str]] = []
    for attrs, title_raw in link_tag_re.findall(body):
        match = href_attr_re.search(attrs)
        if match:
            links.append((match.group(1), title_raw))

    def _snippet_text(raw: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(_html_to_text(raw))).strip()

    snippets = [_snippet_text(item) for item in snippet_re.findall(body)]

    for index, (href, title_raw) in enumerate(links[:limit]):
        if "uddg=" in href:
            match = re.search(r"uddg=([^&]+)", href)
            if match:
                href = urllib.parse.unquote(match.group(1))
        title = html.unescape(_html_to_text(title_raw))
        snippet = snippets[index] if index < len(snippets) else ""
        results.append({"title": title, "url": href, "snippet": snippet})

    if not results:
        generic_re = re.compile(r'href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        seen: set[str] = set()
        for href, txt in generic_re.findall(body):
            if "duckduckgo.com" in href or href in seen:
                continue
            seen.add(href)
            results.append(
                {
                    "title": html.unescape(_html_to_text(txt)),
                    "url": href,
                    "snippet": "",
                }
            )
            if len(results) >= limit:
                break

    return {"status": "ok", "query": query, "count": len(results), "results": results}


def pypi_info(package_name: str) -> dict[str, Any]:
    """Fetch package metadata from PyPI."""
    url = f"https://pypi.org/pypi/{urllib.parse.quote(package_name)}/json"
    try:
        _, body = _http_get(url, timeout=10)
        data = json.loads(body)
        info = data.get("info", {})
        return {
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
    except Exception as exc:
        return {"status": "error", "package": package_name, "error": str(exc)}


def github_read_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "main",
) -> dict[str, Any]:
    """Read a text file from a public GitHub repository."""
    url = (
        f"https://raw.githubusercontent.com/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}/"
        f"{urllib.parse.quote(ref)}/{path}"
    )
    try:
        status, content = _http_get(url, timeout=15)
        if status == 404 and ref == "main":
            fallback_url = url.replace("/main/", "/master/")
            status, content = _http_get(fallback_url, timeout=15)
            if status == 200:
                url = fallback_url
            else:
                return {
                    "status": "not_found",
                    "owner": owner,
                    "repo": repo,
                    "path": path,
                    "ref": ref,
                }
        max_chars = 20000
        truncated = len(content) > max_chars
        return {
            "status": "ok",
            "url": url,
            "lines": content.count("\n"),
            "content": content[:max_chars],
            "truncated": truncated,
        }
    except Exception as exc:
        return {
            "status": "error",
            "owner": owner,
            "repo": repo,
            "path": path,
            "error": str(exc),
        }
