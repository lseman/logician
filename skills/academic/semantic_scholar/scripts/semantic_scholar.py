"""Semantic Scholar academic provider helper.

This module contains the Semantic Scholar source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_norm_space = _common._norm_space
_norm_doi = _common._norm_doi
_norm_arxiv_id = _common._norm_arxiv_id
_year_from_any = _common._year_from_any
_safe_int = _common._safe_int
_simplify_query_for_s2 = _common._simplify_query_for_s2
_s2_fallback_queries = _common._s2_fallback_queries


__tools__ = ["semantic_scholar_search"]

__skill__ = {
    "name": "Semantic Scholar",
    "description": "Provider-specific Semantic Scholar academic search helper.",
}


def semantic_scholar_search(
    query: str,
    limit: int = 50,
    offset: int = 0,
    open_access_only: bool = False,
    normalize_query: bool = True,
) -> dict[str, Any]:
    """Search Semantic Scholar for papers matching a query."""
    source = SemanticScholarSource()
    papers = source.search(
        query=query,
        limit=limit,
        offset=offset,
        open_access_only=open_access_only,
        normalize_query=normalize_query,
    )
    return {
        "status": "ok",
        "source": "semantic_scholar",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


class SemanticScholarSource(BaseHTTPSource):
    name = "s2"

    def __init__(self, *, timeout: float = 25.0, api_key_env: str = "S2_API_KEY"):
        super().__init__(timeout=timeout)
        self._api_key = os.getenv(api_key_env) or None

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
        open_access_only: bool = False,
        normalize_query: bool = True,
        retry_with_broad_query: bool = True,
        debug_raw: bool = False,
        debug_on_empty: bool = True,
        debug_body_preview_chars: int = 1200,
        fields: str = (
            "title,authors,venue,year,publicationDate,externalIds,url,openAccessPdf,"
            "abstract,citationCount,isOpenAccess,fieldsOfStudy,s2FieldsOfStudy"
        ),
    ) -> list[Paper]:
        lim = max(1, min(int(limit), 100))
        off = max(0, int(offset))
        base = "https://api.semanticscholar.org/graph/v1"
        url = f"{base}/paper/search"

        effective_query = _simplify_query_for_s2(query) if normalize_query else _norm_space(query)
        params: dict[str, Any] = {
            "query": effective_query,
            "limit": lim,
            "offset": off,
            "fields": fields,
        }
        if open_access_only:
            params["openAccessPdf"] = "true"

        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        r = self._client.get(url, params=params, headers=headers)
        request_url = str(r.request.url)

        raw: dict[str, Any] = {}
        try:
            payload = r.json()
            raw = payload if isinstance(payload, dict) else {}
        except Exception:
            raw = {}

        items = raw.get("data") or []

        if r.is_success and retry_with_broad_query and len(items) == 0 and normalize_query:
            for q2 in _s2_fallback_queries(effective_query):
                params2 = dict(params)
                params2["query"] = q2
                r2 = self._client.get(url, params=params2, headers=headers)
                raw2: dict[str, Any] = {}
                try:
                    payload2 = r2.json()
                    raw2 = payload2 if isinstance(payload2, dict) else {}
                except Exception:
                    raw2 = {}
                items2 = raw2.get("data") or []
                if r2.is_success and len(items2) > 0:
                    r, raw, items = r2, raw2, items2
                    request_url = str(r.request.url)
                    effective_query = q2
                    break

        should_debug = bool(debug_raw) or (bool(debug_on_empty) and len(items) == 0)
        if should_debug:
            dbg = {
                "source": "s2",
                "query": query,
                "effective_query": effective_query,
                "api_key_present": bool(self._api_key),
                "status_code": r.status_code,
                "request_url": request_url,
                "returned_count": len(items),
                "total_hint": raw.get("total"),
                "response_headers": {
                    "content-type": r.headers.get("content-type"),
                    "x-ratelimit-limit": r.headers.get("x-ratelimit-limit"),
                    "x-ratelimit-remaining": r.headers.get("x-ratelimit-remaining"),
                    "retry-after": r.headers.get("retry-after"),
                },
                "body_preview": r.text[: max(200, int(debug_body_preview_chars))],
            }
            print("[S2 DEBUG]", json.dumps(dbg, indent=2, ensure_ascii=False))

        if r.status_code >= 400:
            return []

        out: list[Paper] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            title = _norm_space(it.get("title") or "")
            year = it.get("year") or _year_from_any(it.get("publicationDate"))
            venue = it.get("venue") or None
            abstract = it.get("abstract")

            ext = it.get("externalIds") or {}
            doi = _norm_doi(ext.get("DOI")) if isinstance(ext, dict) else None
            arx = _norm_arxiv_id(ext.get("ArXiv")) if isinstance(ext, dict) else None

            authors_raw = it.get("authors") or []
            authors: list[str] = []
            for a in authors_raw:
                if isinstance(a, dict) and a.get("name"):
                    authors.append(a["name"])
                elif a:
                    authors.append(str(a))

            pdf_url = None
            oapdf = it.get("openAccessPdf") or {}
            if isinstance(oapdf, dict):
                pdf_url = oapdf.get("url") or None
            if (not pdf_url) and arx:
                pdf_url = f"https://arxiv.org/pdf/{arx}.pdf"

            is_oa = it.get("isOpenAccess")
            if is_oa is None and pdf_url:
                is_oa = True

            out.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=_year_from_any(year),
                    venue=venue,
                    abstract=abstract,
                    doi=doi,
                    arxiv_id=arx,
                    url=it.get("url"),
                    pdf_url=pdf_url,
                    source="s2",
                    is_open_access=bool(is_oa) if is_oa is not None else None,
                    citation_count=_safe_int(it.get("citationCount")),
                    extra={
                        "s2_paperId": it.get("paperId"),
                        "fieldsOfStudy": it.get("fieldsOfStudy") or [],
                        "s2FieldsOfStudy": it.get("s2FieldsOfStudy") or [],
                    },
                )
            )
        return out
