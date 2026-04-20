"""OpenAlex academic provider helper.

This module contains the OpenAlex source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
Paper = _common.Paper
_coalesce = _common._coalesce
_norm_doi = _common._norm_doi
_norm_space = _common._norm_space
_safe_int = _common._safe_int
_year_from_any = _common._year_from_any

__tools__ = ["openalex_search"]

__skill__ = {
    "name": "OpenAlex",
    "description": "Provider-specific OpenAlex academic search helper.",
}


def openalex_search(
    query: str,
    limit: int = 50,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search OpenAlex for works matching a query."""
    source = OpenAlexSource()
    search_params = OpenAlexParams(**(params or {}))
    papers = source.search(query=query, limit=limit, params=search_params, **kwargs)
    return {
        "status": "ok",
        "source": "openalex",
        "query": query,
        "results": [paper.to_dict() for paper in papers],
    }


@dataclass
class OpenAlexParams:
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    sort: str = "relevance"
    mailto_env: str = "OPENALEX_MAILTO"

    concept_ids: list[str] = field(default_factory=list)
    concept_names: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    language: Optional[str] = "en"
    has_abstract: bool = True
    open_access_only: bool = False

    max_pages: int = 2
    per_page: int = 200

    require_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)


def _normalize_openalex_type(t: str) -> str:
    raw = _norm_space(str(t or "")).lower()
    if not raw:
        return raw
    mapping = {
        "journal-article": "article",
        "proceedings-article": "article",
        "posted-content": "preprint",
        "book-chapter": "book-chapter",
        "book": "book",
        "dataset": "dataset",
        "report": "report",
        "dissertation": "dissertation",
        "review": "review",
        "article": "article",
        "preprint": "preprint",
    }
    return mapping.get(raw, raw)


def _openalex_reconstruct_abstract(inv_idx: Optional[dict[str, list[int]]]) -> Optional[str]:
    if not inv_idx or not isinstance(inv_idx, dict):
        return None
    positions: dict[int, str] = {}
    try:
        for w, pos_list in inv_idx.items():
            if not isinstance(pos_list, list):
                continue
            for p in pos_list:
                if isinstance(p, int) and p >= 0 and p not in positions:
                    positions[p] = w
        if not positions:
            return None
        max_pos = max(positions.keys())
        words = [positions.get(i, "") for i in range(max_pos + 1)]
        txt = " ".join([w for w in words if w])
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt or None
    except Exception:
        return None


class OpenAlexSource(BaseHTTPSource):
    name = "openalex"

    def __init__(self, *, timeout: float = 25.0, mailto_env: str = "OPENALEX_MAILTO"):
        super().__init__(timeout=timeout)
        self._mailto = os.getenv(mailto_env)
        self._api_key = os.getenv("OPENALEX_API_KEY") or None

    def _concept_search(self, name: str, *, limit: int = 10) -> list[tuple[str, str]]:
        url = "https://api.openalex.org/concepts"
        params: dict[str, Any] = {"search": name, "per-page": max(1, min(limit, 200))}
        if self._mailto:
            params["mailto"] = self._mailto
        if self._api_key:
            params["api_key"] = self._api_key
        raw = self._get_json(url, params=params, max_retries=2)
        out: list[tuple[str, str]] = []
        for it in raw.get("results") or []:
            cid = it.get("id")
            disp = it.get("display_name")
            if cid and disp:
                m = re.search(r"(C\d+)$", str(cid))
                out.append((m.group(1) if m else str(cid), str(disp)))
        return out

    def resolve_concepts(self, names: list[str], *, pick_top: int = 1) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for nm in names:
            nm2 = _norm_space(nm)
            if not nm2:
                continue
            hits = self._concept_search(nm2, limit=10)
            if hits:
                resolved[nm2] = (
                    hits[0][0] if pick_top == 1 else hits[min(pick_top - 1, len(hits) - 1)][0]
                )
        return resolved

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        params: Optional[OpenAlexParams] = None,
        **kwargs,
    ) -> list[Paper]:
        p = params or OpenAlexParams()
        p.per_page = max(1, min(int(p.per_page), 200))
        p.max_pages = max(1, int(p.max_pages))

        concept_ids = list(p.concept_ids)
        if p.concept_names:
            resolved = self.resolve_concepts(p.concept_names)
            concept_ids.extend(resolved.values())

        flt: list[str] = []
        if p.from_year is not None:
            flt.append(f"from_publication_date:{int(p.from_year)}-01-01")
        if p.to_year is not None:
            flt.append(f"to_publication_date:{int(p.to_year)}-12-31")
        if concept_ids:
            flt.append(
                "concept.id:" + "|".join([str(c).strip() for c in concept_ids if str(c).strip()])
            )
        if p.types:
            types_norm = [_normalize_openalex_type(t) for t in p.types if str(t).strip()]
            types_norm = [t for t in types_norm if t]
            if types_norm:
                flt.append("type:" + "|".join(types_norm))
        if p.language:
            flt.append(f"language:{p.language}")
        if p.has_abstract:
            flt.append("has_abstract:true")
        if p.open_access_only:
            flt.append("is_oa:true")
        for t in p.require_terms:
            tt = _norm_space(t)
            if tt:
                flt.append(f"title.search:{tt}")
        for t in p.exclude_terms:
            tt = _norm_space(t)
            if tt:
                flt.append(f"!title.search:{tt}")

        url = "https://api.openalex.org/works"
        out: list[Paper] = []
        seen_ids: set[str] = set()

        per_page = max(1, min(int(p.per_page), 200))
        max_needed = max(1, min(int(limit), 2000))
        pages = min(p.max_pages, max(1, math.ceil(max_needed / per_page)))

        for page in range(1, pages + 1):
            params_http: dict[str, Any] = {
                "search": query,
                "per-page": per_page,
                "page": page,
            }
            if self._mailto:
                params_http["mailto"] = self._mailto
            if self._api_key:
                params_http["api_key"] = self._api_key
            if flt:
                params_http["filter"] = ",".join(flt)

            if p.sort == "date":
                params_http["sort"] = "publication_date:desc"
            elif p.sort == "cited":
                params_http["sort"] = "cited_by_count:desc"

            raw = self._get_json(url, params=params_http, max_retries=2)
            items = raw.get("results") or []

            for it in items:
                oid = it.get("id")
                if oid and oid in seen_ids:
                    continue
                if oid:
                    seen_ids.add(oid)

                title = _norm_space(it.get("title") or "")
                year = it.get("publication_year") or _year_from_any(it.get("publication_date"))
                doi = _norm_doi(it.get("doi"))

                venue = None
                primary_loc = it.get("primary_location") or {}
                if isinstance(primary_loc, dict):
                    src_meta = primary_loc.get("source") or {}
                    if isinstance(src_meta, dict):
                        venue = src_meta.get("display_name") or None
                if not venue:
                    host = it.get("host_venue") or {}
                    if isinstance(host, dict):
                        venue = host.get("display_name") or None
                if not venue:
                    for loc in it.get("locations") or []:
                        if not isinstance(loc, dict):
                            continue
                        src_meta = loc.get("source") or {}
                        if isinstance(src_meta, dict) and src_meta.get("display_name"):
                            venue = src_meta.get("display_name")
                            break

                authors: list[str] = []
                for a in it.get("authorships") or []:
                    aa = (a or {}).get("author") or {}
                    nm = aa.get("display_name") if isinstance(aa, dict) else None
                    if nm:
                        authors.append(nm)

                oa = it.get("open_access") or {}
                is_oa = oa.get("is_oa") if isinstance(oa, dict) else None
                best_oa = it.get("best_oa_location") or {}
                pdf_url = None
                if isinstance(best_oa, dict):
                    pdf_url = best_oa.get("pdf_url") or best_oa.get("url") or None
                if not pdf_url and isinstance(primary_loc, dict):
                    pdf_url = primary_loc.get("pdf_url") or None

                abstract = _openalex_reconstruct_abstract(it.get("abstract_inverted_index"))

                canonical_url = _coalesce(
                    it.get("doi"),
                    (best_oa.get("landing_page_url") if isinstance(best_oa, dict) else None),
                    (
                        primary_loc.get("landing_page_url")
                        if isinstance(primary_loc, dict)
                        else None
                    ),
                    oid,
                )

                concepts = []
                for c in it.get("concepts") or []:
                    if isinstance(c, dict) and c.get("display_name"):
                        concepts.append(
                            {
                                "display_name": c.get("display_name"),
                                "score": c.get("score"),
                                "id": c.get("id"),
                            }
                        )

                out.append(
                    Paper(
                        title=title,
                        authors=authors,
                        year=_year_from_any(year),
                        venue=venue,
                        abstract=abstract,
                        doi=doi,
                        arxiv_id=None,
                        url=canonical_url,
                        pdf_url=pdf_url,
                        source="openalex",
                        is_open_access=bool(is_oa) if is_oa is not None else None,
                        citation_count=_safe_int(it.get("cited_by_count")),
                        extra={"openalex_id": oid, "concepts": concepts},
                    )
                )

            if len(out) >= max_needed:
                break

        return out[:max_needed]
