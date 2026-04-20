from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from html import unescape
from typing import Any, Dict, List, Optional, Protocol

import httpx


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


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is not None and v != "":
            return v
    return None


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


class Source(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 50, **kwargs) -> List[Paper]: ...


class BaseHTTPSource:
    name = "base"

    def __init__(self, *, timeout: float = 25.0):
        headers = {"User-Agent": "SystematicReviewLib/0.3"}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        base_backoff_s: float = 1.0,
    ) -> Dict[str, Any]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_exc: Optional[Exception] = None
        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                r = self._client.get(url, params=params)
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
