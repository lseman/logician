"""Crossref academic provider helper.

This module contains the Crossref source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_norm_doi = _common._norm_doi
_norm_space = _common._norm_space
_year_from_any = _common._year_from_any

try:
    from habanero import Crossref

    HAS_CROSSREF = True
except Exception:
    HAS_CROSSREF = False

__tools__ = ["crossref_search"]

__skill__ = {
    "name": "Crossref",
    "description": "Provider-specific Crossref academic search helper.",
}


def crossref_search(
    query: str,
    limit: int = 50,
    filter: dict[str, str] | None = None,
    sort: str | None = None,
    order: str = "desc",
) -> dict[str, Any]:
    """Search Crossref for works matching a query."""
    source = CrossrefSource()
    papers = source.search(query=query, limit=limit, filter=filter, sort=sort, order=order)
    return {
        "status": "ok",
        "source": "crossref",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


class CrossrefSource(BaseHTTPSource):
    name = "crossref"

    def __init__(self, *, timeout: float = 35.0):
        super().__init__(timeout=timeout)
        if not HAS_CROSSREF:
            raise RuntimeError("habanero is not installed. pip install habanero")
        self._cr = Crossref()

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        filter: Optional[dict[str, str]] = None,
        sort: Optional[str] = None,
        order: str = "desc",
        max_retries: int = 3,
    ) -> list[Paper]:
        rows = max(1, min(int(limit), 200))
        params: dict[str, Any] = {"query": query, "rows": rows}
        if filter:
            params["filter"] = filter
        if sort:
            params["sort"] = sort
            params["order"] = order if order in ("asc", "desc") else "desc"

        res: dict[str, Any] = {}
        try:
            raw = self._cr.works(**params)
            res = raw if isinstance(raw, dict) else {}
        except Exception as e:
            print(f"[WARN] Crossref via habanero failed: {e}. Retrying via HTTP API...")
            row_attempts = [rows, min(rows, 100), min(rows, 50)]
            row_attempts = list(dict.fromkeys(row_attempts))
            last_err: Optional[Exception] = None
            for rws in row_attempts:
                try:
                    p2 = dict(params)
                    p2["rows"] = rws
                    res = self._get_json(
                        "https://api.crossref.org/works",
                        params=p2,
                        max_retries=max_retries,
                        base_backoff_s=1.0,
                    )
                    if (res or {}).get("message", {}).get("items"):
                        break
                except Exception as e2:
                    last_err = e2
            if not res:
                print(f"Crossref error: {last_err or e}")
                return []

        items = (res or {}).get("message", {}).get("items", []) or []
        out: list[Paper] = []
        for it in items:
            title = (
                (it.get("title") or [""])[0]
                if isinstance(it.get("title"), list)
                else (it.get("title") or "")
            )
            title = _norm_space(title)

            names: list[str] = []
            for a in it.get("author") or []:
                given = (a or {}).get("given")
                family = (a or {}).get("family")
                nm = " ".join([x for x in [given, family] if x])
                if nm.strip():
                    names.append(nm.strip())

            year = None
            for key in ("published-print", "published-online", "created", "issued"):
                dparts = (it.get(key) or {}).get("date-parts")
                if dparts and dparts[0] and dparts[0][0]:
                    year = int(dparts[0][0])
                    break

            venue = (
                (it.get("container-title") or [""])[0]
                if isinstance(it.get("container-title"), list)
                else (it.get("container-title") or None)
            )
            venue = _norm_space(venue) if venue else None

            doi = _norm_doi(it.get("DOI"))
            url = it.get("URL") or (f"https://doi.org/{doi}" if doi else None)

            out.append(
                Paper(
                    title=title,
                    authors=names,
                    year=_year_from_any(year),
                    venue=venue,
                    abstract=None
                    if it.get("abstract") is None
                    else _norm_space(it.get("abstract"))
                    if isinstance(it.get("abstract"), str)
                    else None,
                    doi=doi,
                    arxiv_id=None,
                    url=url,
                    pdf_url=None,
                    source="crossref",
                    is_open_access=None,
                    citation_count=None,
                    extra={"type": it.get("type")},
                )
            )
        return out
