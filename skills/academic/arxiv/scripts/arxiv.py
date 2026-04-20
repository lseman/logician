"""arXiv academic provider helper.

This module contains the arXiv source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_norm_arxiv_id = _common._norm_arxiv_id
_norm_space = _common._norm_space
_year_from_any = _common._year_from_any

try:
    import arxiv

    HAS_ARXIV = True
except Exception:
    HAS_ARXIV = False

__tools__ = ["arxiv_search"]

__skill__ = {
    "name": "arXiv",
    "description": "Provider-specific arXiv academic search helper.",
}


def arxiv_search(query: str, limit: int = 10, sort: str = "submitted") -> dict[str, Any]:
    """Search arXiv for preprints matching a query."""
    source = ArxivSource()
    papers = source.search(query=query, limit=limit, sort=sort)
    return {
        "status": "ok",
        "source": "arxiv",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


def _run_async(awaitable: Any) -> Any:
    try:
        return asyncio.run(awaitable)
    except RuntimeError as exc:
        if "cannot be called from a running event loop" in str(exc):
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(awaitable)
        raise


def _iterable_to_list(iterator: Any) -> list[Any]:
    if inspect.isawaitable(iterator):
        return list(_run_async(iterator))
    if inspect.isasyncgen(iterator):

        async def _drain() -> list[Any]:
            return [item async for item in iterator]

        return _run_async(_drain())
    return list(iterator)


class ArxivSource(BaseHTTPSource):
    name = "arxiv"

    def __init__(self, *, timeout: float = 25.0):
        super().__init__(timeout=timeout)
        if not HAS_ARXIV:
            raise RuntimeError("arxiv is not installed. pip install arxiv")

    def search(self, query: str, *, limit: int = 50, sort: str = "submitted") -> list[Paper]:
        max_r = max(1, min(int(limit), 200))

        sort_by = None
        if hasattr(arxiv, "SortCriterion"):
            sc = arxiv.SortCriterion
            sort_by = {
                "submitted": getattr(sc, "SubmittedDate", None),
                "updated": getattr(sc, "LastUpdatedDate", None),
                "relevance": getattr(sc, "Relevance", None),
            }.get(sort, getattr(sc, "SubmittedDate", None))

        if hasattr(arxiv, "Search"):
            search = (
                arxiv.Search(query=query, max_results=max_r, sort_by=sort_by)
                if sort_by
                else arxiv.Search(query=query, max_results=max_r)
            )

            try:
                if hasattr(arxiv, "Client"):
                    client = arxiv.Client()
                    results = _iterable_to_list(client.results(search))
                else:
                    results = _iterable_to_list(search.results())
            except Exception:
                results = _iterable_to_list(search.results())
        else:
            query_fn = getattr(arxiv, "query", None) or getattr(arxiv, "search", None)
            if query_fn is None:
                raise RuntimeError(
                    "Unsupported arxiv package version. Install the maintained `arxiv` package."
                )

            try:
                results = query_fn(query=query, max_results=max_r, sort_by=sort_by)
            except TypeError:
                results = query_fn(query=query, max_results=max_r)
            results = _iterable_to_list(results)

        out: list[Paper] = []
        for r in results[:max_r]:
            title = _norm_space(getattr(r, "title", "") or "")
            year = getattr(getattr(r, "published", None), "year", None) or _year_from_any(
                getattr(r, "published", None)
            )
            entry_id = getattr(r, "entry_id", None) or None
            arx = _norm_arxiv_id(entry_id.split("/")[-1] if entry_id else None) or _norm_arxiv_id(
                getattr(r, "arxiv_id", None)
            )
            authors_obj = getattr(r, "authors", []) or []
            authors = [
                (getattr(a, "name", None) or str(a)).strip()
                for a in authors_obj
                if (getattr(a, "name", None) or str(a)).strip()
            ]

            pdf_url = getattr(r, "pdf_url", None)
            if not pdf_url and arx:
                pdf_url = f"https://arxiv.org/pdf/{arx}.pdf"

            url = entry_id or (f"https://arxiv.org/abs/{arx}" if arx else None)

            out.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    abstract=getattr(r, "summary", None),
                    doi=None,
                    arxiv_id=arx,
                    url=url,
                    pdf_url=pdf_url,
                    source="arxiv",
                    is_open_access=True,
                    citation_count=None,
                    extra={},
                )
            )
        return out
