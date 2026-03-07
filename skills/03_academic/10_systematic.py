# systematic_review.py
# -*- coding: utf-8 -*-
"""
Systematic review helper (CS/ML friendly):
- Multi-source search: Semantic Scholar, OpenAlex, arXiv, Crossref, IEEE Xplore
- NEW: DBLP (CS bib), Hugging Face Papers (ML discovery)
- Stronger dedupe: DOI/arXiv/url + fuzzy title fallback
- PRISMA-style flow accounting
- Review-style plots

Optional enrichers:
- Unpaywall (OA + PDF links by DOI)  -> requires UNPAYWALL_EMAIL (or plan.unpaywall_email)
- OpenCitations (citation_count by DOI)

Notes:
- OpenAlex abstracts: reconstructed from inverted index.
- Minimal deps: httpx; optional: arxiv, habanero, pandas, matplotlib, networkx, rapidfuzz.
"""

from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from html import unescape
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx

# Optional deps
try:
    import arxiv  # pip install arxiv

    HAS_ARXIV = True
except Exception:
    HAS_ARXIV = False

try:
    from habanero import Crossref  # pip install habanero

    HAS_CROSSREF = True
except Exception:
    HAS_CROSSREF = False

try:
    import pandas as pd  # pip install pandas

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt  # pip install matplotlib

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    from rapidfuzz import fuzz  # pip install rapidfuzz

    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _norm_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I).strip()
    return d.lower() if d else None


def _norm_arxiv_id(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = str(x).strip()
    s = s.replace("arXiv:", "").replace("ARXIV:", "").strip()
    return s.lower() if s else None


def _year_from_any(s: Any) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, int):
        return s
    m = re.search(r"(\d{4})", str(s))
    if not m:
        return None
    y = int(m.group(1))
    return y if 1800 <= y <= 2100 else None


def _title_key(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_keywords(text: str) -> List[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [x for x in t.split(" ") if 3 <= len(x) <= 30]
    stop = {
        "the",
        "and",
        "or",
        "but",
        "for",
        "with",
        "from",
        "into",
        "using",
        "use",
        "used",
        "based",
        "via",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "by",
        "at",
        "as",
        "we",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "which",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "having",
        "not",
        "no",
        "yes",
        "than",
        "then",
        "also",
        "very",
        "more",
        "most",
        "less",
        "least",
        "many",
        "much",
        "few",
        "several",
        "all",
        "any",
        "each",
        "every",
        "while",
        "across",
        "through",
        "however",
        "including",
        "include",
        "includes",
        "within",
        "among",
        "between",
        "toward",
        "towards",
        "therefore",
        "thus",
        "overall",
        "both",
        "either",
        "neither",
        "other",
        "another",
        "such",
        "same",
        "study",
        "paper",
        "approach",
        "method",
        "methods",
        "results",
        "model",
        "models",
        "data",
        "analysis",
        "system",
        "systems",
        "time",
        "series",
    }
    return [w for w in toks if w not in stop]


def _simplify_query_for_s2(query: str) -> str:
    q = _norm_space(query)
    if not q:
        return q
    q = q.replace('"', " ")
    q = q.replace("(", " ").replace(")", " ")
    q = re.sub(r"(^|\s)-\w+", " ", q)
    q = re.sub(r"(^|\s)\+\w+", " ", q)
    q = re.sub(r"\b(OR|AND|NOT)\b", " ", q, flags=re.I)
    q = re.sub(r"[^a-zA-Z0-9\s\-]", " ", q)
    return _norm_space(q)


def _s2_fallback_queries(query: str) -> List[str]:
    q = _simplify_query_for_s2(query)
    if not q:
        return []
    toks = [t for t in q.split(" ") if t]
    if not toks:
        return []
    uniq: List[str] = []
    seen = set()
    for t in toks:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        uniq.append(t)
    fallbacks: List[str] = []
    if len(uniq) > 12:
        fallbacks.append(" ".join(uniq[:12]))
    if len(uniq) > 8:
        fallbacks.append(" ".join(uniq[:8]))
    if len(uniq) > 5:
        fallbacks.append(" ".join(uniq[:5]))
    return fallbacks


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float) and math.isfinite(x):
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
        return None
    except Exception:
        return None


def _extract_citation_count(
    record: Dict[str, Any], *candidate_keys: str
) -> Optional[int]:
    """Best-effort citation count extraction from heterogeneous API payloads."""
    if not isinstance(record, dict):
        return None

    keys = candidate_keys or (
        "citation_count",
        "citationCount",
        "cited_by_count",
        "citing_paper_count",
        "num_citations",
        "n_citations",
        "citations",
    )

    for key in keys:
        val = _safe_int(record.get(key))
        if val is not None:
            return val
    return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    return []


def _safe_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
    return None


def _strip_markup(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    txt = re.sub(r"<[^>]+>", " ", str(text))
    txt = unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt or None


def _coalesce(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Paper:
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None

    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    url: Optional[str] = None
    pdf_url: Optional[str] = None

    source: str = "unknown"
    is_open_access: Optional[bool] = None
    citation_count: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def ref_id(self) -> str:
        d = _norm_doi(self.doi)
        if d:
            return f"doi:{d}"
        a = _norm_arxiv_id(self.arxiv_id)
        if a:
            return f"arxiv:{a}"
        if self.url:
            return f"url:{self.url}"
        return f"title:{_title_key(self.title)}:{self.year or ''}"

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["doi"] = _norm_doi(self.doi)
        out["arxiv_id"] = _norm_arxiv_id(self.arxiv_id)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Source interface + base
# ──────────────────────────────────────────────────────────────────────────────


class Source(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 50, **kwargs) -> List[Paper]: ...


class BaseHTTPSource:
    name: str = "base"

    def __init__(self, *, timeout: float = 25.0):
        headers = {"User-Agent": "SystematicReviewLib/0.3"}
        self._client = httpx.Client(
            timeout=timeout, headers=headers, follow_redirects=True
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 2,
        base_backoff_s: float = 0.8,
    ) -> Dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_exc: Optional[Exception] = None

        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                r = self._client.get(url, params=params, headers=headers)
                if r.status_code in retry_statuses and attempt < max_retries:
                    time.sleep(base_backoff_s * (2**attempt))
                    continue
                r.raise_for_status()
                payload = r.json()
                return payload if isinstance(payload, dict) else {}
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(base_backoff_s * (2**attempt))

        if last_exc:
            raise last_exc
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────


class ArxivSource(BaseHTTPSource):
    name = "arxiv"

    def __init__(self, *, timeout: float = 25.0):
        super().__init__(timeout=timeout)
        if not HAS_ARXIV:
            raise RuntimeError("arxiv is not installed. pip install arxiv")

    def search(
        self, query: str, *, limit: int = 50, sort: str = "submitted"
    ) -> List[Paper]:
        max_r = max(1, min(int(limit), 200))

        sort_by = None
        if hasattr(arxiv, "SortCriterion"):
            sc = arxiv.SortCriterion
            sort_by = {
                "submitted": getattr(sc, "SubmittedDate", None),
                "updated": getattr(sc, "LastUpdatedDate", None),
                "relevance": getattr(sc, "Relevance", None),
            }.get(sort, getattr(sc, "SubmittedDate", None))

        search = (
            arxiv.Search(query=query, max_results=max_r, sort_by=sort_by)
            if sort_by
            else arxiv.Search(query=query, max_results=max_r)
        )

        try:
            client = arxiv.Client()
            results = list(client.results(search))
        except Exception:
            results = list(search.results())

        out: List[Paper] = []
        for r in results[:max_r]:
            title = _norm_space(getattr(r, "title", "") or "")
            year = getattr(
                getattr(r, "published", None), "year", None
            ) or _year_from_any(getattr(r, "published", None))
            entry_id = getattr(r, "entry_id", None) or None
            arx = _norm_arxiv_id(
                entry_id.split("/")[-1] if entry_id else None
            ) or _norm_arxiv_id(getattr(r, "arxiv_id", None))
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


class SemanticScholarSource(BaseHTTPSource):
    name = "s2"

    def __init__(self, *, timeout: float = 25.0, api_key_env: str = "S2_API_KEY"):
        super().__init__(timeout=timeout)
        self._api_key = os.getenv(api_key_env) or None  # do NOT hardcode

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
    ) -> List[Paper]:
        lim = max(1, min(int(limit), 100))
        off = max(0, int(offset))
        base = "https://api.semanticscholar.org/graph/v1"
        url = f"{base}/paper/search"

        effective_query = (
            _simplify_query_for_s2(query) if normalize_query else _norm_space(query)
        )
        params: Dict[str, Any] = {
            "query": effective_query,
            "limit": lim,
            "offset": off,
            "fields": fields,
        }
        if open_access_only:
            params["openAccessPdf"] = "true"

        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        r = self._client.get(url, params=params, headers=headers)
        request_url = str(r.request.url)

        raw: Dict[str, Any] = {}
        try:
            payload = r.json()
            raw = payload if isinstance(payload, dict) else {}
        except Exception:
            raw = {}

        items = raw.get("data") or []

        if (
            r.is_success
            and retry_with_broad_query
            and len(items) == 0
            and normalize_query
        ):
            for q2 in _s2_fallback_queries(effective_query):
                params2 = dict(params)
                params2["query"] = q2
                r2 = self._client.get(url, params=params2, headers=headers)
                raw2: Dict[str, Any] = {}
                try:
                    payload2 = r2.json()
                    raw2 = payload2 if isinstance(payload2, dict) else {}
                except Exception:
                    raw2 = {}
                items2 = raw2.get("data") or []
                if debug_raw:
                    dbg2 = {
                        "source": "s2",
                        "fallback": True,
                        "fallback_query": q2,
                        "status_code": r2.status_code,
                        "request_url": str(r2.request.url),
                        "returned_count": len(items2),
                        "total_hint": raw2.get("total"),
                    }
                    print("[S2 DEBUG]", json.dumps(dbg2, indent=2, ensure_ascii=False))
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

        out: List[Paper] = []
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
            authors: List[str] = []
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
        filter: Optional[Dict[str, str]] = None,
        sort: Optional[str] = None,
        order: str = "desc",
        max_retries: int = 3,
    ) -> List[Paper]:
        rows = max(1, min(int(limit), 200))
        params: Dict[str, Any] = {"query": query, "rows": rows}
        if filter:
            params["filter"] = filter
        if sort:
            params["sort"] = sort
            params["order"] = order if order in ("asc", "desc") else "desc"

        res: Dict[str, Any] = {}
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
        out: List[Paper] = []
        for it in items:
            title = (
                (it.get("title") or [""])[0]
                if isinstance(it.get("title"), list)
                else (it.get("title") or "")
            )
            title = _norm_space(title)

            names: List[str] = []
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
                    abstract=_strip_markup(it.get("abstract")),
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


def _openalex_reconstruct_abstract(
    inv_idx: Optional[Dict[str, List[int]]],
) -> Optional[str]:
    if not inv_idx or not isinstance(inv_idx, dict):
        return None
    positions: Dict[int, str] = {}
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


@dataclass
class OpenAlexParams:
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    sort: str = "relevance"  # "relevance" | "date" | "cited"
    mailto_env: str = "OPENALEX_MAILTO"

    concept_ids: List[str] = field(default_factory=list)
    concept_names: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    language: Optional[str] = "en"
    has_abstract: bool = True
    open_access_only: bool = False

    max_pages: int = 2
    per_page: int = 200

    require_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)


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


class OpenAlexSource(BaseHTTPSource):
    name = "openalex"

    def __init__(self, *, timeout: float = 25.0, mailto_env: str = "OPENALEX_MAILTO"):
        super().__init__(timeout=timeout)
        self._mailto = os.getenv(mailto_env)
        self._api_key = os.getenv("OPENALEX_API_KEY") or None  # optional

    def _concept_search(self, name: str, *, limit: int = 10) -> List[Tuple[str, str]]:
        url = "https://api.openalex.org/concepts"
        params: Dict[str, Any] = {"search": name, "per-page": max(1, min(limit, 200))}
        if self._mailto:
            params["mailto"] = self._mailto
        if self._api_key:
            params["api_key"] = self._api_key
        raw = self._get_json(url, params=params, max_retries=2)
        out: List[Tuple[str, str]] = []
        for it in raw.get("results") or []:
            cid = it.get("id")
            disp = it.get("display_name")
            if cid and disp:
                m = re.search(r"(C\d+)$", str(cid))
                out.append((m.group(1) if m else str(cid), str(disp)))
        return out

    def resolve_concepts(
        self, names: List[str], *, pick_top: int = 1
    ) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        for nm in names:
            nm2 = _norm_space(nm)
            if not nm2:
                continue
            hits = self._concept_search(nm2, limit=10)
            if hits:
                resolved[nm2] = (
                    hits[0][0]
                    if pick_top == 1
                    else hits[min(pick_top - 1, len(hits) - 1)][0]
                )
        return resolved

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        params: Optional[OpenAlexParams] = None,
        **kwargs,
    ) -> List[Paper]:
        p = params or OpenAlexParams()
        p.per_page = max(1, min(int(p.per_page), 200))
        p.max_pages = max(1, int(p.max_pages))

        concept_ids = list(p.concept_ids)
        if p.concept_names:
            resolved = self.resolve_concepts(p.concept_names)
            concept_ids.extend(resolved.values())

        flt: List[str] = []
        if p.from_year is not None:
            flt.append(f"from_publication_date:{int(p.from_year)}-01-01")
        if p.to_year is not None:
            flt.append(f"to_publication_date:{int(p.to_year)}-12-31")
        if concept_ids:
            flt.append(
                "concept.id:"
                + "|".join([str(c).strip() for c in concept_ids if str(c).strip()])
            )
        if p.types:
            types_norm = [
                _normalize_openalex_type(t) for t in p.types if str(t).strip()
            ]
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
        out: List[Paper] = []
        seen_ids: set[str] = set()

        per_page = max(1, min(int(p.per_page), 200))
        max_needed = max(1, min(int(limit), 2000))
        pages = min(p.max_pages, max(1, math.ceil(max_needed / per_page)))

        for page in range(1, pages + 1):
            params_http: Dict[str, Any] = {
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
                year = it.get("publication_year") or _year_from_any(
                    it.get("publication_date")
                )
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

                authors: List[str] = []
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

                abstract = _openalex_reconstruct_abstract(
                    it.get("abstract_inverted_index")
                )

                canonical_url = _coalesce(
                    it.get("doi"),
                    (
                        best_oa.get("landing_page_url")
                        if isinstance(best_oa, dict)
                        else None
                    ),
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
    ) -> List[Paper]:
        if not self._api_key:
            return []

        url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
        params: Dict[str, Any] = {
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

        out: List[Paper] = []
        try:
            data = self._get_json(url, params=params, max_retries=2)
            articles = data.get("articles", []) or []
            for art in articles[:limit]:
                title = _norm_space(art.get("title", ""))
                year = _safe_int(art.get("publication_year"))
                authors_raw = (art.get("authors") or {}).get("authors", [])
                authors: List[str] = []
                if isinstance(authors_raw, list):
                    for a in authors_raw:
                        if isinstance(a, dict) and a.get("preferredName"):
                            authors.append(str(a.get("preferredName")))
                        elif isinstance(a, str) and a.strip():
                            authors.append(a.strip())
                doi = _norm_doi(art.get("doi"))
                venue = art.get("publication_title") or art.get(
                    "publication_number", ""
                )
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
            print(
                f"IEEE Xplore HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            )
        except Exception as e:
            print(f"IEEE Xplore error: {e}")

        return out


# ──────────────────────────────────────────────────────────────────────────────
# NEW: DBLP (CS/ML friendly)
# ──────────────────────────────────────────────────────────────────────────────


class DBLPSource(BaseHTTPSource):
    """
    DBLP publication search.
    Endpoint:
      https://dblp.org/search/publ/api?q=...&format=json&h=...&f=...
    """

    name = "dblp"

    def __init__(self, *, timeout: float = 25.0):
        super().__init__(timeout=timeout)

    def search(
        self, query: str, *, limit: int = 50, offset: int = 0, **kwargs
    ) -> List[Paper]:
        h = max(1, min(int(limit), 1000))
        f = max(0, int(offset))
        url = "https://dblp.org/search/publ/api"
        queries = [query] + [q for q in _s2_fallback_queries(query) if q != query]

        hits: List[Any] = []
        for q in queries:
            params = {"q": q, "format": "json", "h": h, "f": f}
            try:
                raw = self._get_json(url, params=params, max_retries=2)
            except Exception as e:
                print(f"[WARN] DBLP request failed for query='{q}': {e}")
                continue

            hit_obj = (((raw or {}).get("result") or {}).get("hits") or {}).get("hit")
            hits = _as_list(hit_obj)
            if hits:
                if q != query:
                    print(f"[INFO] DBLP fallback query succeeded: '{q}'")
                break

        if not hits:
            return []

        out: List[Paper] = []
        for hitem in hits:
            if not isinstance(hitem, dict):
                continue
            info = (hitem or {}).get("info") or {}
            if not isinstance(info, dict):
                continue
            title = _strip_markup(info.get("title")) or _norm_space(
                str(info.get("title") or "")
            )
            year = _year_from_any(info.get("year"))
            venue = _strip_markup(info.get("venue")) or info.get("venue") or None
            url2 = info.get("url") or None

            # authors can be {"author": "X"} or {"author": ["X","Y"]} or nested dicts
            authors: List[str] = []
            auth = info.get("authors") or {}
            auth2 = auth.get("author") if isinstance(auth, dict) else None
            if isinstance(auth2, str):
                authors = [_norm_space(auth2)]
            elif isinstance(auth2, list):
                authors = [_norm_space(str(a)) for a in auth2 if _norm_space(str(a))]
            elif isinstance(auth2, dict) and auth2.get("text"):
                authors = [_norm_space(str(auth2.get("text")))]

            # DBLP sometimes exposes DOI as "doi": "10...."
            doi = _norm_doi(info.get("doi")) if info.get("doi") else None

            citation_count = _extract_citation_count(
                info,
                "citation_count",
                "citationCount",
                "citations",
                "cited_by",
                "citedBy",
            )

            out.append(
                Paper(
                    title=_norm_space(title),
                    authors=authors,
                    year=year,
                    venue=_norm_space(venue) if venue else None,
                    abstract=None,
                    doi=doi,
                    url=url2,
                    pdf_url=None,
                    source="dblp",
                    is_open_access=None,
                    citation_count=citation_count,
                    extra={"dblp_type": info.get("type")},
                )
            )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Hugging Face Papers (CS/ML discovery)
# ──────────────────────────────────────────────────────────────────────────────


class HuggingFacePapersSource(BaseHTTPSource):
    """
    Hugging Face papers discovery.

    Preferred endpoint (may require auth):
      https://huggingface.co/papers/search?q=...&page=...

    Fallback endpoint (public HTML):
      https://huggingface.co/papers?query=...
    """

    name = "hf_papers"

    def __init__(self, *, timeout: float = 25.0, token_env: str = "HF_TOKEN"):
        super().__init__(timeout=timeout)
        self._token = os.getenv(token_env) or None

    def search(
        self, query: str, *, limit: int = 50, page: int = 1, **kwargs
    ) -> List[Paper]:
        endpoint_candidates = [
            "https://huggingface.co/papers/search",
            "https://huggingface.co/api/papers",
        ]
        page = max(1, int(page))
        max_pages = max(1, math.ceil(max(1, limit) / 20))

        headers: Dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        def _extract_results(payload: Any) -> List[Dict[str, Any]]:
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)]
            if not isinstance(payload, dict):
                return []
            for key in ("results", "items", "data", "papers"):
                vals = payload.get(key)
                if isinstance(vals, list):
                    return [x for x in vals if isinstance(x, dict)]
            return []

        out: List[Paper] = []
        attempts = [query] + [q for q in _s2_fallback_queries(query) if q != query]

        for current_query in attempts:
            endpoint_error: Optional[str] = None

            for base_url in endpoint_candidates:
                any_results_this_endpoint = False

                for p in range(page, page + max_pages):
                    params = {"q": current_query, "query": current_query, "page": p}
                    try:
                        r = self._client.get(base_url, params=params, headers=headers)
                        if r.status_code >= 400:
                            endpoint_error = f"HTTP {r.status_code}"
                            break

                        ct = (r.headers.get("content-type") or "").lower()
                        final_url = str(r.url)
                        if "json" not in ct:
                            endpoint_error = (
                                f"non-json content-type '{ct}' at {final_url}"
                            )
                            break

                        payload = r.json()
                        results = _extract_results(payload)
                    except Exception as e:
                        endpoint_error = str(e)
                        break

                    if not results:
                        break

                    any_results_this_endpoint = True

                    for it in results:
                        if not isinstance(it, dict):
                            continue

                        title = _norm_space(
                            it.get("title") or it.get("paper_title") or ""
                        )
                        if not title:
                            continue

                        y = _year_from_any(
                            it.get("published")
                            or it.get("published_date")
                            or it.get("date")
                        )
                        url2 = it.get("url") or it.get("paper_url") or None
                        arx = _norm_arxiv_id(
                            it.get("arxiv_id") or it.get("arxiv") or None
                        )

                        citation_count = _extract_citation_count(
                            it,
                            "citation_count",
                            "citationCount",
                            "citations",
                            "num_citations",
                            "n_citations",
                        )
                        if citation_count is None:
                            paper_info = (
                                it.get("paper")
                                if isinstance(it.get("paper"), dict)
                                else None
                            )
                            if isinstance(paper_info, dict):
                                citation_count = _extract_citation_count(
                                    paper_info,
                                    "citation_count",
                                    "citationCount",
                                    "citations",
                                    "num_citations",
                                    "n_citations",
                                )

                        out.append(
                            Paper(
                                title=title,
                                authors=[],
                                year=y,
                                venue=it.get("conference") or it.get("venue") or None,
                                abstract=None,
                                doi=_norm_doi(it.get("doi")) if it.get("doi") else None,
                                arxiv_id=arx,
                                url=url2,
                                pdf_url=None,
                                source="hf_papers",
                                is_open_access=None,
                                citation_count=citation_count,
                                extra={
                                    "hf_paper_id": it.get("id"),
                                    "tasks": it.get("tasks"),
                                    "repositories": it.get("repositories"),
                                    "endpoint": base_url,
                                },
                            )
                        )
                        if len(out) >= limit:
                            return out[:limit]

                if any_results_this_endpoint:
                    if current_query != query:
                        print(
                            f"[INFO] HF papers fallback query succeeded: '{current_query}'"
                        )
                    return out[:limit]

            if endpoint_error and current_query == query:
                print(
                    f"[WARN] HF papers JSON endpoints unavailable for query='{current_query}': {endpoint_error}"
                )

        # HTML fallback: parse /papers links from public pages
        html_seen: set[str] = set()
        query_tokens = set(_tokenize_keywords(query))

        for p in range(page, page + max_pages):
            params = {"query": query, "page": p}
            try:
                r = self._client.get("https://huggingface.co/papers", params=params)
                if r.status_code >= 400:
                    break
                html = r.text if isinstance(r.text, str) else ""
            except Exception as e:
                print(f"[WARN] HF papers HTML fallback failed: {e}")
                break

            ids = re.findall(r'href="/papers/([0-9]{4}\.[0-9]{4,6})"', html)
            ids = list(dict.fromkeys(ids))
            if not ids:
                break

            for hid in ids:
                if hid in html_seen:
                    continue
                html_seen.add(hid)
                paper_url = f"https://huggingface.co/papers/{hid}"
                try:
                    rp = self._client.get(paper_url)
                    if rp.status_code >= 400:
                        continue
                    ph = rp.text if isinstance(rp.text, str) else ""
                except Exception:
                    continue

                m_title = re.search(
                    r'<meta\s+property="og:title"\s+content="([^"]+)"',
                    ph,
                    flags=re.I,
                )
                title = _norm_space(_strip_markup(m_title.group(1)) if m_title else "")
                if not title:
                    continue

                if query_tokens:
                    title_tokens = set(_tokenize_keywords(title))
                    if title_tokens and len(title_tokens & query_tokens) == 0:
                        continue

                m_desc = re.search(
                    r'<meta\s+name="description"\s+content="([^"]+)"',
                    ph,
                    flags=re.I,
                )
                abstract = _strip_markup(m_desc.group(1)) if m_desc else None

                yy = _safe_int(hid[:2])
                year = 2000 + yy if yy is not None and 0 <= yy <= 99 else None

                out.append(
                    Paper(
                        title=title,
                        authors=[],
                        year=year,
                        venue="Hugging Face Papers",
                        abstract=abstract,
                        doi=None,
                        arxiv_id=_norm_arxiv_id(hid),
                        url=paper_url,
                        pdf_url=f"https://arxiv.org/pdf/{hid}.pdf",
                        source="hf_papers",
                        is_open_access=True,
                        citation_count=None,
                        extra={"hf_paper_id": hid, "endpoint": "html_fallback"},
                    )
                )

                if len(out) >= limit:
                    return out[:limit]

        if out:
            print(f"[INFO] HF papers HTML fallback returned {len(out)} results.")

        return out[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# Optional enrichers (DOI-based)
# ──────────────────────────────────────────────────────────────────────────────


class UnpaywallEnricher(BaseHTTPSource):
    """
    Unpaywall OA + PDF lookup by DOI:
      https://api.unpaywall.org/v2/{doi}?email=...
    """

    name = "unpaywall"

    def __init__(self, *, timeout: float = 25.0, email: Optional[str] = None):
        super().__init__(timeout=timeout)
        self._email = (
            email
            or os.getenv("UNPAYWALL_EMAIL")
            or os.getenv("UNPAYWALL_MAILTO")
            or None
        )

    def lookup(self, doi: str) -> Optional[Dict[str, Any]]:
        if not self._email:
            return None
        d = _norm_doi(doi)
        if not d:
            return None
        url = f"https://api.unpaywall.org/v2/{d}"
        try:
            return self._get_json(url, params={"email": self._email}, max_retries=2)
        except Exception:
            return None


class OpenCitationsEnricher(BaseHTTPSource):
    """
    OpenCitations citation-count by DOI:
      https://opencitations.net/index/api/v1/citation-count/doi:{doi}
    Response is usually a list like: [{"doi":"...","count":"123"}]
    """

    name = "opencitations"

    def __init__(self, *, timeout: float = 25.0):
        super().__init__(timeout=timeout)

    def citation_count(self, doi: str) -> Optional[int]:
        d = _norm_doi(doi)
        if not d:
            return None
        url = f"https://opencitations.net/index/api/v1/citation-count/doi:{d}"
        try:
            r = self._client.get(url)
            if r.status_code >= 400:
                return None
            payload = r.json()
            if isinstance(payload, list) and payload:
                cnt = payload[0].get("count")
                return _safe_int(cnt)
            return None
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Query builder
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryPlan:
    base: str
    phrases: List[str] = field(default_factory=list)
    any_terms: List[str] = field(default_factory=list)
    all_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)

    def render(self) -> str:
        parts: List[str] = []
        b = _norm_space(self.base)
        if b:
            parts.append(b)

        for ph in self.phrases:
            ph2 = _norm_space(ph)
            if ph2:
                parts.append(f'"{ph2}"')

        if self.all_terms:
            for t in self.all_terms:
                t2 = _norm_space(t)
                if t2:
                    parts.append(t2)

        if self.any_terms:
            ors = []
            for t in self.any_terms:
                t2 = _norm_space(t)
                if t2:
                    ors.append(t2)
            if ors:
                parts.append("(" + " OR ".join(ors) + ")")

        for t in self.exclude_terms:
            t2 = _norm_space(t)
            if t2:
                parts.append(f"-{t2}")

        return _norm_space(" ".join(parts))


# ──────────────────────────────────────────────────────────────────────────────
# PRISMA-style screening
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScreeningPlan:
    include_any: List[str] = field(default_factory=list)
    include_all: List[str] = field(default_factory=list)
    exclude_any: List[str] = field(default_factory=list)
    require_year: bool = False
    require_abstract: bool = False

    def decision(self, p: Paper) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        text = (p.title or "") + "\n" + (p.abstract or "")

        def _match_any(patterns: List[str]) -> bool:
            for pat in patterns:
                if not pat:
                    continue
                if re.search(pat, text, flags=re.I):
                    return True
            return False

        def _match_all(patterns: List[str]) -> bool:
            for pat in patterns:
                if not pat:
                    continue
                if not re.search(pat, text, flags=re.I):
                    return False
            return True

        if self.require_year and p.year is None:
            reasons.append("missing_year")
        if self.require_abstract and not (p.abstract and p.abstract.strip()):
            reasons.append("missing_abstract")

        if self.include_any and not _match_any(self.include_any):
            reasons.append("failed_include_any")
        if self.include_all and not _match_all(self.include_all):
            reasons.append("failed_include_all")
        if self.exclude_any and _match_any(self.exclude_any):
            reasons.append("matched_exclude_any")

        ok = len(reasons) == 0
        return ok, reasons


@dataclass
class PrismaFlow:
    identified: int = 0
    duplicates_removed: int = 0
    screened: int = 0
    excluded_title_abstract: int = 0
    eligible: int = 0
    excluded_fulltext: int = 0
    included: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Review plan
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchPlan:
    query: str
    per_source_limit: int = 50
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    open_access_only: bool = False

    query_plan: Optional[QueryPlan] = None
    openalex: OpenAlexParams = field(default_factory=OpenAlexParams)
    screening: ScreeningPlan = field(default_factory=ScreeningPlan)

    source_order: Optional[List[str]] = None
    source_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # NEW: enrichment controls
    enrich_unpaywall: bool = False
    unpaywall_email: Optional[str] = None
    enrich_opencitations: bool = False

    # NEW: enable/disable new CS/ML sources
    enable_dblp: bool = True
    enable_hf_papers: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Systematic Review engine
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    query: str
    papers: List[Paper]
    by_source: Dict[str, int]
    added_new: int
    prisma: PrismaFlow


class SystematicReview:
    def __init__(self, sources: Optional[List[Source]] = None):
        self.sources: List[Source] = sources or self._default_sources()
        self._papers: List[Paper] = []
        self._seen: set[str] = set()
        self._title_seen: List[Tuple[str, Optional[int]]] = []
        self._decisions: Dict[str, Dict[str, Any]] = {}
        self._prisma: PrismaFlow = PrismaFlow()

        # enrichment caches
        self._unpaywall_cache: Dict[str, Dict[str, Any]] = {}
        self._opencit_cache: Dict[str, Optional[int]] = {}

    def _default_sources(self) -> List[Source]:
        srcs: List[Source] = []
        if HAS_ARXIV:
            srcs.append(ArxivSource())
        srcs.append(SemanticScholarSource())
        if HAS_CROSSREF:
            srcs.append(CrossrefSource())
        srcs.append(OpenAlexSource())
        srcs.append(IEEEXploreSource())
        # NEW:
        srcs.append(DBLPSource())
        srcs.append(HuggingFacePapersSource())
        return srcs

    def close(self) -> None:
        for s in self.sources:
            if isinstance(s, BaseHTTPSource):
                s.close()

    @property
    def papers(self) -> List[Paper]:
        return list(self._papers)

    @property
    def prisma(self) -> PrismaFlow:
        return PrismaFlow(**self._prisma.to_dict())

    def reset(self) -> "SystematicReview":
        self._papers.clear()
        self._seen.clear()
        self._title_seen.clear()
        self._decisions.clear()
        self._prisma = PrismaFlow()
        self._unpaywall_cache.clear()
        self._opencit_cache.clear()
        return self

    def add_source(self, source: Source) -> "SystematicReview":
        self.sources.append(source)
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Dedupe
    # ──────────────────────────────────────────────────────────────────────

    def _strong_key(self, p: Paper) -> Optional[str]:
        d = _norm_doi(p.doi)
        if d:
            return f"doi:{d}"
        a = _norm_arxiv_id(p.arxiv_id)
        if a:
            return f"arxiv:{a}"
        if p.url:
            return f"url:{p.url}"
        return None

    def _normalize_paper(self, p: Paper) -> None:
        p.doi = _norm_doi(p.doi)
        p.arxiv_id = _norm_arxiv_id(p.arxiv_id)
        p.title = _norm_space(p.title)
        p.year = _year_from_any(p.year)
        p.venue = _norm_space(p.venue) if p.venue else None

    def _apply_post_filters(
        self,
        papers: List[Paper],
        *,
        from_year: Optional[int],
        to_year: Optional[int],
        open_access_only: bool,
    ) -> List[Paper]:
        out = papers
        if from_year is not None:
            out = [p for p in out if (p.year is None) or (p.year >= from_year)]
        if to_year is not None:
            out = [p for p in out if (p.year is None) or (p.year <= to_year)]
        if open_access_only:
            out = [
                p for p in out if p.is_open_access is True or (p.pdf_url is not None)
            ]
        return out

    def _fuzzy_dup(self, p: Paper, *, threshold: int = 93) -> bool:
        tk = _title_key(p.title)
        if not tk:
            return False

        if not HAS_RAPIDFUZZ:
            return any(
                (tk == t and (p.year is None or y is None or p.year == y))
                for (t, y) in self._title_seen
            )

        y = p.year
        for t_prev, y_prev in self._title_seen:
            if y is not None and y_prev is not None and abs(y - y_prev) > 1:
                continue
            score = fuzz.token_set_ratio(tk, t_prev)
            if score >= threshold:
                return True
        return False

    def _add_papers(self, papers: List[Paper]) -> Tuple[int, int]:
        added = 0
        dupes = 0

        for p in papers:
            self._normalize_paper(p)

            sk = self._strong_key(p)
            if sk is not None:
                if sk in self._seen:
                    dupes += 1
                    continue
                self._seen.add(sk)
            else:
                if self._fuzzy_dup(p):
                    dupes += 1
                    continue

            self._papers.append(p)
            self._title_seen.append((_title_key(p.title), p.year))
            added += 1

        return added, dupes

    # ──────────────────────────────────────────────────────────────────────
    # Enrichment (optional)
    # ──────────────────────────────────────────────────────────────────────

    def enrich_with_unpaywall(
        self, *, email: Optional[str] = None, max_workers: int = 8
    ) -> int:
        """
        For papers with a DOI: fill pdf_url / is_open_access when missing.
        Returns how many papers were updated.
        """
        enr = UnpaywallEnricher(email=email)
        dois = []
        for p in self._papers:
            d = _norm_doi(p.doi)
            if not d:
                continue
            # Only enrich if useful
            if (p.pdf_url is None) or (p.is_open_access is None):
                dois.append(d)

        dois = list(dict.fromkeys(dois))
        if not dois:
            enr.close()
            return 0

        updated = 0

        def one(d: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            if d in self._unpaywall_cache:
                return d, self._unpaywall_cache[d]
            payload = enr.lookup(d)
            if isinstance(payload, dict):
                self._unpaywall_cache[d] = payload
            return d, payload

        try:
            workers = max(1, min(int(max_workers), 16))
            if workers == 1:
                results = [one(d) for d in dois]
            else:
                results = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(one, d) for d in dois]
                    for f in as_completed(futs):
                        results.append(f.result())

            m = {d: payload for d, payload in results if isinstance(payload, dict)}
            for p in self._papers:
                d = _norm_doi(p.doi)
                if not d:
                    continue
                payload = m.get(d)
                if not payload:
                    continue
                # Unpaywall structure: best_oa_location.url_for_pdf / url, is_oa, oa_status
                best = payload.get("best_oa_location") or {}
                if p.pdf_url is None:
                    pdf = (
                        (best.get("url_for_pdf") or best.get("url"))
                        if isinstance(best, dict)
                        else None
                    )
                    if pdf:
                        p.pdf_url = pdf
                        updated += 1
                if p.is_open_access is None:
                    is_oa = payload.get("is_oa")
                    if is_oa is not None:
                        p.is_open_access = bool(is_oa)
                        updated += 1
                p.extra["unpaywall"] = {
                    "is_oa": payload.get("is_oa"),
                    "oa_status": payload.get("oa_status"),
                    "best_oa_host_type": (
                        best.get("host_type") if isinstance(best, dict) else None
                    ),
                }
        finally:
            enr.close()

        return updated

    def enrich_with_opencitations(self, max_workers: int = 8) -> int:
        """
        For papers with DOI and missing citation_count: fill citation_count.
        Returns how many papers were updated.
        """
        enr = OpenCitationsEnricher()
        dois = []
        for p in self._papers:
            d = _norm_doi(p.doi)
            if not d:
                continue
            if p.citation_count is None:
                dois.append(d)
        dois = list(dict.fromkeys(dois))
        if not dois:
            enr.close()
            return 0

        updated = 0

        def one(d: str) -> Tuple[str, Optional[int]]:
            if d in self._opencit_cache:
                return d, self._opencit_cache[d]
            cnt = enr.citation_count(d)
            self._opencit_cache[d] = cnt
            return d, cnt

        try:
            workers = max(1, min(int(max_workers), 16))
            if workers == 1:
                results = [one(d) for d in dois]
            else:
                results = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(one, d) for d in dois]
                    for f in as_completed(futs):
                        results.append(f.result())

            m = {d: cnt for d, cnt in results}
            for p in self._papers:
                d = _norm_doi(p.doi)
                if not d:
                    continue
                if p.citation_count is None and m.get(d) is not None:
                    p.citation_count = m[d]
                    updated += 1
                    p.extra["opencitations"] = {"citation_count": m[d]}
        finally:
            enr.close()

        return updated

    # ──────────────────────────────────────────────────────────────────────
    # Search + screening
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        per_source_limit: int = 50,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        open_access_only: bool = False,
        source_order: Optional[List[str]] = None,
        source_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        openalex_params: Optional[OpenAlexParams] = None,
        enable_dblp: bool = True,
        enable_hf_papers: bool = True,
    ) -> Tuple[List[Paper], Dict[str, int], int, int]:
        before = len(self._papers)
        by_source: Dict[str, int] = {}
        merged: List[Paper] = []

        srcs = list(self.sources)

        # optionally disable new sources
        if not enable_dblp:
            srcs = [s for s in srcs if getattr(s, "name", "") != "dblp"]
        if not enable_hf_papers:
            srcs = [s for s in srcs if getattr(s, "name", "") != "hf_papers"]

        if source_order:
            order = {n: i for i, n in enumerate(source_order)}
            srcs = sorted(srcs, key=lambda s: order.get(getattr(s, "name", ""), 10_000))

        skw = source_kwargs or {}

        def _search_one_source(src: Source) -> Tuple[str, List[Paper], Optional[str]]:
            name = getattr(src, "name", "unknown")
            kwargs = dict(skw.get(name, {}))

            if name == "s2":
                kwargs.setdefault("open_access_only", open_access_only)

            if name == "openalex":
                oa_seed = openalex_params or OpenAlexParams()
                oa = OpenAlexParams(**asdict(oa_seed))
                oa.from_year = oa.from_year if oa.from_year is not None else from_year
                oa.to_year = oa.to_year if oa.to_year is not None else to_year
                oa.open_access_only = oa.open_access_only or open_access_only
                kwargs.setdefault("params", oa)

            if name == "ieee":
                kwargs.setdefault("from_year", from_year)
                kwargs.setdefault("to_year", to_year)

            try:
                papers = src.search(query, limit=per_source_limit, **kwargs)
                papers = self._apply_post_filters(
                    papers,
                    from_year=from_year,
                    to_year=to_year,
                    open_access_only=open_access_only,
                )
                return name, papers, None
            except Exception as e:
                return name, [], str(e)

        source_results: Dict[str, List[Paper]] = {}
        source_errors: Dict[str, str] = {}

        if len(srcs) <= 1:
            for src in srcs:
                name, papers, err = _search_one_source(src)
                source_results[name] = papers
                if err is not None:
                    source_errors[name] = err
        else:
            max_workers = min(10, max(1, len(srcs)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_search_one_source, src) for src in srcs]
                for future in as_completed(futures):
                    name, papers, err = future.result()
                    source_results[name] = papers
                    if err is not None:
                        source_errors[name] = err

        for src in srcs:
            name = getattr(src, "name", "unknown")
            if name in source_errors:
                print(f"[WARN] Source '{name}' failed: {source_errors[name]}")
            papers = source_results.get(name, [])
            by_source[name] = len(papers)
            merged.extend(papers)

        added, dupes = self._add_papers(merged)
        after = len(self._papers)
        return merged, by_source, (after - before), dupes

    def screen(self, screening: ScreeningPlan) -> Tuple[List[Paper], PrismaFlow]:
        flow = self._prisma
        flow.screened = len(self._papers)

        included: List[Paper] = []
        reasons_counter = Counter()

        for p in self._papers:
            ok, reasons = screening.decision(p)
            rid = p.ref_id()
            self._decisions[rid] = {
                "include": ok,
                "reasons": reasons,
                "source": p.source,
            }
            if ok:
                included.append(p)
            else:
                flow.excluded_title_abstract += 1
                for r in reasons:
                    reasons_counter[r] += 1

        flow.eligible = len(included)
        flow.included = len(included)
        flow.exclusion_reasons = dict(reasons_counter)

        return included, PrismaFlow(**flow.to_dict())

    def run(self, plan: SearchPlan) -> SearchResult:
        q = plan.query_plan.render() if plan.query_plan else plan.query
        q = _norm_space(q)

        merged, by_source, added_new, dupes = self.search(
            q,
            per_source_limit=plan.per_source_limit,
            from_year=plan.from_year,
            to_year=plan.to_year,
            open_access_only=plan.open_access_only,
            source_order=plan.source_order,
            source_kwargs=plan.source_kwargs,
            openalex_params=plan.openalex,
            enable_dblp=plan.enable_dblp,
            enable_hf_papers=plan.enable_hf_papers,
        )

        self._prisma.identified += len(merged)
        self._prisma.duplicates_removed += dupes

        # Optional enrichers (operate on internal corpus before screening)
        if plan.enrich_unpaywall:
            upd = self.enrich_with_unpaywall(email=plan.unpaywall_email)
            print(
                f"[INFO] Unpaywall enrichment updated fields on ~{upd} assignments (pdf_url/is_open_access)."
            )
        if plan.enrich_opencitations:
            upd = self.enrich_with_opencitations()
            print(
                f"[INFO] OpenCitations enrichment updated citation_count for {upd} papers."
            )

        screened, prisma = self.screen(plan.screening)

        return SearchResult(
            query=q,
            papers=screened,
            by_source=by_source,
            added_new=added_new,
            prisma=prisma,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Sorting, filtering, metrics, export, plots (same as your version)
    # ──────────────────────────────────────────────────────────────────────

    def sort(
        self,
        papers: Optional[List[Paper]] = None,
        *,
        by: str = "year",
        descending: bool = True,
    ) -> List[Paper]:
        ps = list(papers if papers is not None else self._papers)

        def key(p: Paper):
            if by == "year":
                return p.year if p.year is not None else -1
            if by == "citations":
                return p.citation_count if p.citation_count is not None else -1
            if by == "title":
                return _title_key(p.title)
            if by == "source":
                return p.source or ""
            if by == "venue":
                return p.venue or ""
            return p.year if p.year is not None else -1

        ps.sort(key=key, reverse=descending)
        return ps

    def filter_year(
        self,
        *,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        papers: Optional[List[Paper]] = None,
    ) -> List[Paper]:
        ps = list(papers if papers is not None else self._papers)
        if from_year is not None:
            ps = [p for p in ps if p.year is not None and p.year >= from_year]
        if to_year is not None:
            ps = [p for p in ps if p.year is not None and p.year <= to_year]
        return ps

    def metrics(self, papers: Optional[List[Paper]] = None) -> Dict[str, Any]:
        ps = papers if papers is not None else self._papers

        years = [p.year for p in ps if p.year is not None]
        venues = [p.venue for p in ps if p.venue]
        sources = [p.source for p in ps if p.source]
        oa_flags = [p.is_open_access for p in ps if p.is_open_access is not None]
        citations = [p.citation_count for p in ps if isinstance(p.citation_count, int)]

        by_year = Counter(years)
        by_venue = Counter(venues)
        by_source = Counter(sources)

        author_counter = Counter()
        for p in ps:
            for a in p.authors[:50]:
                aa = _norm_space(a)
                if aa:
                    author_counter[aa] += 1

        oa_rate = None
        if oa_flags:
            oa_rate = sum(1 for x in oa_flags if x) / float(len(oa_flags))

        citation_stats = None
        if citations:
            c_sorted = sorted(citations)
            citation_stats = {
                "count_with_citations": len(citations),
                "min": c_sorted[0],
                "median": c_sorted[len(c_sorted) // 2],
                "p90": c_sorted[int(0.9 * (len(c_sorted) - 1))],
                "max": c_sorted[-1],
                "mean": sum(c_sorted) / float(len(c_sorted)),
            }

        kw_counter = Counter()
        for p in ps:
            text = (p.title or "") + " " + (p.abstract or "")
            kw_counter.update(_tokenize_keywords(text))
        top_keywords = kw_counter.most_common(30)

        return {
            "n_papers": len(ps),
            "n_with_year": len(years),
            "year_range": [min(years), max(years)] if years else None,
            "by_year": dict(sorted(by_year.items())),
            "by_source": by_source.most_common(),
            "top_venues": by_venue.most_common(15),
            "top_authors": author_counter.most_common(20),
            "top_keywords": top_keywords,
            "open_access_rate": oa_rate,
            "citation_stats": citation_stats,
            "prisma": self._prisma.to_dict(),
        }

    def to_rows(self, papers: Optional[List[Paper]] = None) -> List[Dict[str, Any]]:
        ps = papers if papers is not None else self._papers
        return [p.to_dict() for p in ps]

    def to_dataframe(self, papers: Optional[List[Paper]] = None):
        if not HAS_PANDAS:
            raise RuntimeError("pandas is not installed. pip install pandas")
        return pd.DataFrame(self.to_rows(papers))

    def save_jsonl(self, path: str, papers: Optional[List[Paper]] = None) -> str:
        rows = self.to_rows(papers)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    def save_csv(self, path: str, papers: Optional[List[Paper]] = None) -> str:
        df = self.to_dataframe(papers)
        df.to_csv(path, index=False)
        return path

    def plots(self) -> "ReviewPlots":
        if not HAS_MPL:
            raise RuntimeError("matplotlib is not installed. pip install matplotlib")
        return ReviewPlots(self)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (kept same structure; copied from your version with no behavior changes)
# ──────────────────────────────────────────────────────────────────────────────


class ReviewPlots:
    def __init__(self, sr: SystematicReview):
        self.sr = sr
        self._palette = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#72B7B2",
            "#B279A2",
            "#FF9DA6",
            "#9D755D",
            "#BAB0AC",
        ]
        self._apply_paper_style()

    def _clip_label(self, text: str, max_len: int = 42) -> str:
        t = _norm_space(str(text or ""))
        if len(t) <= max_len:
            return t
        if max_len <= 1:
            return "…"
        return t[: max_len - 1].rstrip() + "…"

    def _apply_paper_style(self):
        plt.rcParams.update(
            {
                "figure.dpi": 140,
                "savefig.dpi": 320,
                "savefig.bbox": "tight",
                "font.size": 10.5,
                "axes.titlesize": 12,
                "axes.titleweight": "bold",
                "axes.labelsize": 10.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#4A4A4A",
                "axes.linewidth": 0.8,
                "grid.color": "#D9D9D9",
                "grid.linestyle": "-",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.55,
                "legend.frameon": False,
                "legend.fontsize": 8.5,
            }
        )

    def _fig(self, title: str, figsize: Tuple[float, float] = (8.4, 4.8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc="left")
        ax.set_facecolor("#FCFCFD")
        fig.patch.set_facecolor("white")
        return fig

    def _finalize_axes(
        self,
        ax,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xgrid: bool = False,
        ygrid: bool = True,
    ):
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(False)
        if ygrid:
            ax.yaxis.grid(True)
        if xgrid:
            ax.xaxis.grid(True)

    def prisma_flow(self, *, save: Optional[str] = None):
        f = self.sr.prisma
        labels = [
            "Identified",
            "Duplicates removed",
            "Screened",
            "Excluded (title/abstract)",
            "Eligible",
            "Included",
        ]
        vals = [
            f.identified,
            f.duplicates_removed,
            f.screened,
            f.excluded_title_abstract,
            f.eligible,
            f.included,
        ]

        fig = self._fig("PRISMA flow (counts)", figsize=(9.2, 5.0))
        ax = plt.gca()
        ax.axis("off")

        from matplotlib.patches import FancyBboxPatch

        x0 = 0.05
        w = 0.58
        y = 0.92
        dy = 0.125
        for i, (lab, v) in enumerate(zip(labels, vals)):
            color = self._palette[i % len(self._palette)]
            rect = FancyBboxPatch(
                (x0, y - 0.06),
                w,
                0.085,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.0,
                edgecolor=color,
                facecolor="#FFFFFF",
                alpha=0.96,
            )
            ax.add_patch(rect)
            ax.text(x0 + 0.02, y - 0.02, lab, fontsize=10.5, va="center")
            ax.text(
                x0 + w - 0.02,
                y - 0.02,
                f"{v:,}",
                fontsize=11.5,
                va="center",
                ha="right",
                fontweight="bold",
                color=color,
            )
            if i < len(labels) - 1:
                ax.annotate(
                    "",
                    xy=(x0 + w / 2.0, y - 0.078),
                    xytext=(x0 + w / 2.0, y - dy + 0.018),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#7A7A7A"),
                )
            y -= dy

        if f.exclusion_reasons:
            ax.text(
                0.69,
                0.90,
                "Top exclusion reasons",
                fontsize=10.5,
                va="top",
                fontweight="bold",
            )
            y_reason = 0.84
            for k, v in sorted(f.exclusion_reasons.items(), key=lambda x: -x[1])[:6]:
                ax.text(
                    0.69,
                    y_reason,
                    f"• {k}: {v}",
                    fontsize=9.5,
                    va="top",
                    color="#444444",
                )
                y_reason -= 0.08

        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def year_histogram(
        self, papers: Optional[List[Paper]] = None, *, save: Optional[str] = None
    ):
        ps = papers if papers is not None else self.sr.papers
        years = [p.year for p in ps if p.year is not None]
        fig = self._fig("Publications by year", figsize=(8.2, 4.5))
        if years:
            by_year = Counter(years)
            xs = sorted(by_year.keys())
            ys = [by_year[y] for y in xs]
            bars = plt.bar(
                xs,
                ys,
                width=0.75,
                color=self._palette[0],
                edgecolor="#2E4A6B",
                linewidth=0.6,
            )
            if len(xs) <= 20:
                plt.xticks(xs)
            self._finalize_axes(plt.gca(), xlabel="Year", ylabel="Count", ygrid=True)
            from matplotlib.ticker import MaxNLocator

            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            for b in bars:
                h = b.get_height()
                if h > 0:
                    plt.gca().text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.08,
                        f"{int(h)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#2E4A6B",
                    )
        else:
            plt.text(0.5, 0.5, "No year data available", ha="center", va="center")
            plt.gca().axis("off")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def sources_bar(
        self, papers: Optional[List[Paper]] = None, *, save: Optional[str] = None
    ):
        ps = papers if papers is not None else self.sr.papers
        c = Counter([p.source for p in ps if p.source])
        fig = self._fig("Included papers by source", figsize=(7.8, 4.4))
        if c:
            labels, vals = zip(*c.most_common())
            colors = [self._palette[i % len(self._palette)] for i in range(len(labels))]
            plt.bar(labels, vals, color=colors, edgecolor="#2F2F2F", linewidth=0.55)
            plt.xticks(rotation=30, ha="right")
            self._finalize_axes(plt.gca(), ylabel="Count", ygrid=True)
        else:
            plt.text(0.5, 0.5, "No source data", ha="center", va="center")
            plt.gca().axis("off")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def venues_bar(
        self,
        papers: Optional[List[Paper]] = None,
        top_k: int = 15,
        *,
        save: Optional[str] = None,
    ):
        ps = papers if papers is not None else self.sr.papers
        c = Counter([p.venue for p in ps if p.venue])
        fig = self._fig(f"Top venues (k={top_k})", figsize=(8.8, 5.2))
        if c:
            items = c.most_common(top_k)
            labels, vals = zip(*items)
            ylabels = [self._clip_label(lb, max_len=42) for lb in reversed(labels)]
            xvals = list(reversed(vals))
            bars = plt.barh(
                ylabels,
                xvals,
                color=self._palette[2],
                edgecolor="#2E4A6B",
                linewidth=0.55,
            )
            self._finalize_axes(plt.gca(), xlabel="Count", xgrid=True, ygrid=False)
            for b in bars:
                w = b.get_width()
                plt.gca().text(
                    w + 0.1,
                    b.get_y() + b.get_height() / 2.0,
                    f"{int(w)}",
                    va="center",
                    fontsize=8,
                )
        else:
            plt.text(0.5, 0.5, "No venue data", ha="center", va="center")
            plt.gca().axis("off")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def citations_distribution(
        self, papers: Optional[List[Paper]] = None, *, save: Optional[str] = None
    ):
        ps = papers if papers is not None else self.sr.papers
        cites = [p.citation_count for p in ps if isinstance(p.citation_count, int)]
        fig = self._fig("Citations distribution", figsize=(8.2, 4.6))
        if cites:
            bins = max(12, min(36, int(math.sqrt(len(cites)) * 2)))
            plt.hist(
                cites,
                bins=bins,
                color=self._palette[1],
                edgecolor="#6B3A0A",
                linewidth=0.55,
                alpha=0.9,
            )
            self._finalize_axes(
                plt.gca(), xlabel="Citations", ylabel="Count", ygrid=True
            )
            med = sorted(cites)[len(cites) // 2]
            plt.axvline(
                med,
                color="#7A7A7A",
                linestyle="--",
                linewidth=1.1,
                label=f"median={med}",
            )
            plt.legend(loc="upper right")
        else:
            plt.text(0.5, 0.5, "No citation data available", ha="center", va="center")
            plt.gca().axis("off")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        return fig

    def keyword_cooccurrence(
        self,
        papers: Optional[List[Paper]] = None,
        top_k: int = 25,
        *,
        save: Optional[str] = None,
    ):
        ps = papers if papers is not None else self.sr.papers

        docs_tokens: List[List[str]] = []
        kw_counter = Counter()
        for p in ps:
            toks = _tokenize_keywords((p.title or "") + " " + (p.abstract or ""))
            toks = list(dict.fromkeys(toks))
            if toks:
                docs_tokens.append(toks)
                kw_counter.update(toks)

        top = [w for w, _ in kw_counter.most_common(top_k)]
        idx = {w: i for i, w in enumerate(top)}
        n = len(top)
        mat = [[0] * n for _ in range(n)]

        for toks in docs_tokens:
            keep = [t for t in toks if t in idx]
            for i in range(len(keep)):
                for j in range(i, len(keep)):
                    a = idx[keep[i]]
                    b = idx[keep[j]]
                    mat[a][b] += 1
                    if a != b:
                        mat[b][a] += 1

        fig = self._fig(f"Keyword co-occurrence (top {top_k})", figsize=(9.0, 6.6))
        if n == 0:
            plt.text(0.5, 0.5, "Insufficient keyword data", ha="center", va="center")
            plt.gca().axis("off")
        else:
            plt.imshow(mat, aspect="auto", cmap="magma")
            plt.xticks(range(n), top, rotation=90, fontsize=7)
            plt.yticks(range(n), top, fontsize=7)
            plt.colorbar(label="Co-occurrence count")
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=220, bbox_inches="tight")
        return fig

    def save_all(self, out_dir: str, papers: Optional[List[Paper]] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        ps = papers if papers is not None else self.sr.papers
        self.prisma_flow(save=os.path.join(out_dir, "prisma_flow.pdf"))
        self.year_histogram(ps, save=os.path.join(out_dir, "by_year.pdf"))
        self.sources_bar(ps, save=os.path.join(out_dir, "by_source.pdf"))
        self.venues_bar(ps, save=os.path.join(out_dir, "top_venues.pdf"))
        self.citations_distribution(ps, save=os.path.join(out_dir, "citations.pdf"))
        self.keyword_cooccurrence(
            ps, save=os.path.join(out_dir, "keyword_cooccurrence.pdf")
        )
        return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@llm.tool(
    description="Search multiple academic databases (Semantic Scholar, OpenAlex, arXiv, Crossref, IEEE, DBLP) and deduplicate results."
)
def run_systematic_review(query: str, limit_per_source: int = 20, from_year: Optional[int] = None) -> str:
    """Use when: You need to perform a systematic literature review across multiple sources.

    Inputs:
      query (str, required): The search query (e.g. 'time series forecasting').
      limit_per_source (int, optional): Max results per source (default 20).
      from_year (int, optional): Filter papers published on or after this year.
    Returns: JSON summary of the search results including PRISMA metrics and top papers.
    """
    try:
        sr = SystematicReview()
        plan = SearchPlan(
            query=query,
            per_source_limit=limit_per_source,
            from_year=from_year,
            open_access_only=False,
            # Just basic search, disable complex filtering for generic tool
            screening=ScreeningPlan(require_year=False, require_abstract=False),
        )
        res = sr.run(plan)
        metrics = sr.metrics(res.papers)
        
        top = sr.sort(res.papers, by="citations")[:15]
        top_list = []
        for p in top:
            top_list.append({
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "venue": p.venue,
                "citations": p.citation_count,
                "url": p.url or p.pdf_url,
                "source": p.source
            })
            
        return json.dumps({
            "status": "ok",
            "added_new": res.added_new,
            "by_source": res.by_source,
            "prisma": res.prisma.to_dict(),
            "metrics": metrics,
            "top_papers": top_list
        }, ensure_ascii=False)
    except Exception as exc:
        import traceback
        return json.dumps({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})
