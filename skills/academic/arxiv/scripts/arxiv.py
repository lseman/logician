"""arXiv academic provider helper — raw HTTP (no arxiv pip package).

Uses the arXiv Atom API directly via httpx. No external arxiv package needed.
See https://info.arxiv.org/help/api/basics.html

Atom namespaces used:
  atom   — http://www.w3.org/2005/Atom
  arxiv  — http://arxiv.org/schemas/atom
  opensearch — http://a9.com/-/spec/opensearch/1.1/
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_norm_arxiv_id = _common._norm_arxiv_id
_norm_space = _common._norm_space
_year_from_any = _common._year_from_any

__tools__ = ["arxiv_search"]

__skill__ = {
    "name": "arXiv",
    "description": "Provider-specific arXiv academic search helper (raw HTTP).",
}

# ── XML namespaces ──────────────────────────────────────────────────
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# ── API constants ───────────────────────────────────────────────────
API_URL = "https://export.arxiv.org/api/query"
# arXiv asks clients making repeated calls to wait three seconds.
_RATE_MIN_GAP = 3.0


def arxiv_search(
    query: str,
    limit: int = 10,
    sort: str = "submitted",
) -> dict[str, Any]:
    """Search arXiv for preprints matching a query."""
    source = ArxivSource()
    papers = source.search(query=query, limit=limit, sort=sort)
    return {
        "status": "ok",
        "source": "arxiv",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


# ── Rate-limited client ─────────────────────────────────────────────
_last_request_time: float = 0.0


def _rate_limited_get(
    client: Any,
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = 3,
    base_backoff: float = 1.0,
) -> bytes:
    """GET with per-request backoff + retry on 429/5xx.

    arXiv asks clients to wait three seconds between repeated calls.
    On transient failures we back off exponentially up to max_retries.
    """
    global _last_request_time
    for attempt in range(max_retries + 1):
        # Enforce minimum gap between requests
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < _RATE_MIN_GAP:
            time.sleep(_RATE_MIN_GAP - elapsed)

        try:
            r = client.get(url, params=params)
            _last_request_time = time.monotonic()
        except Exception:
            _last_request_time = time.monotonic()
            if attempt >= max_retries:
                raise
            time.sleep(base_backoff * (2 ** attempt))
            continue

        if r.status_code == 200:
            return r.content

        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            backoff = base_backoff * (2 ** attempt)
            time.sleep(backoff)
            continue

        r.raise_for_status()
        return r.content  # unreachable but keeps type checker happy


# ── XML helpers ─────────────────────────────────────────────────────
def _xml_text(el: Optional[ET.Element]) -> Optional[str]:
    if el is None:
        return None
    return (el.text or "").strip() or None


def _xml_children(el: Optional[ET.Element], tag: str, ns: str) -> list[ET.Element]:
    if el is None:
        return []
    return el.findall(f"{{{ns}}}{tag}")


# ── Source ──────────────────────────────────────────────────────────
class ArxivSource(BaseHTTPSource):
    name = "arxiv"

    def __init__(self, *, timeout: float = 25.0):
        super().__init__(timeout=timeout)

    # ── sort mapping ──────────────────────────────────────────────
    _SORT_MAP = {
        "relevance": ("relevance", "descending"),
        "lastupdateddate": ("lastUpdatedDate", "descending"),
        "submitteddate": ("submittedDate", "descending"),
    }

    # ── search ────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        sort: str = "submitted",
    ) -> list[Paper]:
        max_r = max(1, min(int(limit), 200))
        sort_key, sort_order = self._SORT_MAP.get(
            sort.lower(),
            ("submittedDate", "descending"),
        )

        # Build search_query parameter per arXiv spec
        # "all:term" searches title/abstract/keywords/comments
        sq = self._build_search_query(query)

        params: dict[str, Any] = {
            "search_query": sq,
            "start": 0,
            "max_results": max_r,
            "sortBy": sort_key,
            "sortOrder": sort_order,
        }

        return self._fetch_and_parse(params, max_r)

    # ── get by ID ─────────────────────────────────────────────────
    def get_by_id(self, arxiv_id: str) -> Optional[Paper]:
        normalized = _norm_arxiv_id(arxiv_id)
        if not normalized:
            return None
        params = {"id_list": normalized, "start": 0, "max_results": 1}
        results = self._fetch_and_parse(params, 1)
        return results[0] if results else None

    # ── internal ──────────────────────────────────────────────────
    def _build_search_query(self, query: str) -> str:
        """Convert a natural-language query into arXiv search_query syntax.

        arXiv supports: all:term, ti:term, au:term, co:term, jr:term, cat:term,
        and boolean operators (AND, OR, NOT). We wrap each word in all: for broad
        recall, then fall back to simple term matching.
        """
        normalized = _norm_space(query)
        if not normalized:
            raise ValueError("arXiv query must not be empty")
        # Preserve callers' explicit arXiv field/boolean syntax.
        if re.search(r"\b(?:all|ti|au|abs|co|jr|cat|rn|id):", normalized, re.I):
            return normalized
        terms = shlex.split(normalized)
        return " AND ".join(
            f'all:"{term}"' if " " in term else f"all:{term}"
            for term in terms
        )

    def _fetch_and_parse(
        self,
        params: dict[str, Any],
        max_results: int,
    ) -> list[Paper]:
        """Execute the HTTP request and parse Atom XML into Paper list."""
        content = _rate_limited_get(self._client, API_URL, params)
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []

        entries = root.findall("atom:entry", _NS)
        if len(entries) == 1:
            title = _xml_text(entries[0].find("atom:title", _NS)) or ""
            entry_id = _xml_text(entries[0].find("atom:id", _NS)) or ""
            if title.lower() == "error" or "/api/errors" in entry_id:
                detail = _xml_text(entries[0].find("atom:summary", _NS))
                raise ValueError(detail or "arXiv API rejected the query")
        papers: list[Paper] = []

        for entry in entries[:max_results]:
            paper = self._parse_entry(entry)
            if paper and paper.title:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Paper]:
        """Parse a single Atom <entry> into a Paper."""
        # id: http://arxiv.org/abs/2301.00001v1 → 2301.00001v1
        id_el = entry.find("atom:id", _NS)
        id_text = _xml_text(id_el)
        arxiv_id = None
        if id_text:
            candidate = id_text.split("/abs/", 1)[-1].rstrip("/")
            candidate = re.sub(r"v\d+$", "", candidate)
            arxiv_id = _norm_arxiv_id(candidate)

        title_el = entry.find("atom:title", _NS)
        title = _norm_space(_xml_text(title_el) or "")
        if not title:
            return None

        # Summary (abstract)
        summary_el = entry.find("atom:summary", _NS)
        abstract = _norm_space(_xml_text(summary_el))
        if abstract:
            # Strip LaTeX formatting artifacts for readability
            abstract = abstract.replace("\n", " ").replace("\\", "")

        # Published year
        published_el = entry.find("atom:published", _NS)
        pub_text = _xml_text(published_el)
        year = _year_from_any(pub_text) if pub_text else None

        # Authors
        authors: list[str] = []
        author_elems = entry.findall("atom:author", _NS)
        for author_el in author_elems:
            name_el = author_el.find("atom:name", _NS)
            name = _xml_text(name_el)
            if name:
                authors.append(name)

        # Links: alternate (=abs page), related (=pdf)
        # arXiv Atom format: pdf links look like
        #   <link rel="related" href="http://arxiv.org/pdf/1706.03762v1" type="application/pdf"/>
        # No .pdf extension — detect via /pdf/ path segment
        url = None
        pdf_url = None
        links = entry.findall("atom:link", _NS)
        for link in links:
            href = link.get("href")
            rel = link.get("rel", "")
            if rel == "alternate" and href:
                url = href
            elif rel == "related" and href and "/pdf/" in href:
                pdf_url = href

        # Fallback URL / PDF if not in links
        if not url and arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Categories
        categories: list[str] = []
        for cat in entry.findall("atom:category", _NS):
            term = cat.get("term")
            if term:
                categories.append(term)
        primary = entry.find("arxiv:primary_category", _NS)
        # term is an ATTRIBUTE on arxiv:primary_category, not a child element
        primary_cat = primary.get("term") if primary is not None else None

        # arXiv-specific metadata
        comment = _xml_text(entry.find("arxiv:comment", _NS))
        journal_ref = _xml_text(entry.find("arxiv:journal_ref", _NS))
        doi = _xml_text(entry.find("arxiv:doi", _NS))

        return Paper(
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=abstract,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
            pdf_url=pdf_url,
            source="arxiv",
            is_open_access=True,
            citation_count=None,
            extra={
                "categories": categories,
                "primary_category": primary_cat,
                "comment": comment,
                "journal_ref": journal_ref,
            },
        )


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Search the official arXiv Atom API")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="search arXiv")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--sort",
        choices=["submitted", "submittedDate", "lastUpdatedDate", "relevance"],
        default="submitted",
    )
    get_parser = subparsers.add_parser("get", help="retrieve one arXiv ID")
    get_parser.add_argument("arxiv_id")
    args = parser.parse_args(argv)

    if args.command == "search":
        payload = arxiv_search(args.query, limit=args.limit, sort=args.sort)
    else:
        paper = ArxivSource().get_by_id(args.arxiv_id)
        payload = {
            "status": "ok",
            "source": "arxiv",
            "query": args.arxiv_id,
            "results": [paper.to_dict()] if paper else [],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
