"""IEEE Xplore academic provider helper.

This module contains the IEEE Xplore source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_norm_doi = _common._norm_doi
_norm_space = _common._norm_space
_safe_bool = _common._safe_bool
_safe_int = _common._safe_int

__tools__ = ["ieee_search"]

__skill__ = {
    "name": "IEEE Xplore",
    "description": "Provider-specific IEEE Xplore academic search helper.",
}


def ieee_search(
    query: str,
    limit: int = 50,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
) -> dict[str, Any]:
    """Search IEEE Xplore for articles matching a query."""
    source = IEEEXploreSource()
    papers = source.search(query=query, limit=limit, from_year=from_year, to_year=to_year)
    return {
        "status": "ok",
        "source": "ieee",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


class IEEEXploreSource(BaseHTTPSource):
    name = "ieee"

    def __init__(self, *, timeout: float = 25.0, api_key_env: str = "IEEE_API_KEY"):
        super().__init__(timeout=timeout)
        self._api_key = os.getenv(api_key_env) or None
        if not self._api_key:
            print(
                "Warning: IEEE Xplore API key not found (set IEEE_API_KEY env var). IEEE will return []."
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        **kwargs,
    ) -> list[Paper]:
        if not self._api_key:
            return []

        url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        params: dict[str, Any] = {
            "apikey": self._api_key,
            "querytext": query,
            "rows": min(int(limit), 200),
            "start": 1,
            "sortfield": "publication_year",
            "sortorder": "desc",
        }
        if from_year or to_year:
            yr_from = from_year or 1800
            yr_to = to_year or 2099
            params["ranges"] = f"{yr_from}_{yr_to}_Year"

        out: list[Paper] = []
        try:
            data = self._get_json(url, params=params, max_retries=2)
            articles = data.get("articles", []) or []
            for art in articles[:limit]:
                title = _norm_space(art.get("title", ""))
                year = _safe_int(art.get("publication_year"))
                authors_raw = (art.get("authors") or {}).get("authors", [])
                authors: list[str] = []
                if isinstance(authors_raw, list):
                    for a in authors_raw:
                        if isinstance(a, dict) and a.get("preferredName"):
                            authors.append(str(a.get("preferredName")))
                        elif isinstance(a, str) and a.strip():
                            authors.append(a.strip())
                doi = _norm_doi(art.get("doi"))
                venue = art.get("publication_title") or art.get("publication_number", "")
                abstract = art.get("abstract")
                article_number = art.get("article_number")
                url2 = (
                    f"https://ieeexplore.ieee.org/document/{article_number}"
                    if article_number
                    else art.get("html_url")
                )
                pdf_url = art.get("pdf_url") or None
                is_oa = _safe_bool(art.get("open_access"))

                out.append(
                    Paper(
                        title=title,
                        authors=authors,
                        year=year,
                        venue=venue,
                        abstract=abstract,
                        doi=doi,
                        url=url2,
                        pdf_url=pdf_url,
                        source="ieee",
                        is_open_access=is_oa,
                        citation_count=_safe_int(art.get("citing_paper_count")),
                        extra={"article_number": article_number},
                    )
                )
        except httpx.HTTPStatusError as e:
            print(f"IEEE Xplore HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            print(f"IEEE Xplore error: {e}")

        return out
