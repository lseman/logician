# academic_agent.py
# -*- coding: utf-8 -*-
"""
AcademicAgent — Literature search + RAG over papers (with debug output)

New architecture:
- PHASE 1 (search): programmatic multi-source search (no LLM, no tools).
- PHASE 2 (ingest): programmatic ingestion into vector DB (no LLM).
- PHASE 3 (ask): build retrieval context and send a single prompt to the LLM.
  The LLM does NOT call tools; it only receives the context + question.

Core features:
- Token-aware chunking aligned with the embedding model (Sentence-Transformers/HF),
- Fast sentence splitting via BlingFire (with regex fallback),
- Robust DOI->PDF resolution with a hard timeout budget,
- Semantic Scholar, arXiv, Crossref search,
- Ingestion into a Chroma-based DocumentDB,
- RAG-based question answering with citations and bibliography.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Any

import json
import os
import re
import time
import uuid

import httpx
import numpy as np

# API SDKs
import arxiv
from habanero import Crossref

try:
    from pypdf import PdfReader  # noqa: F401

    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# Framework imports
from core import Agent, ToolParameter, DocumentDB  # ToolParameter kept for compatibility

# How long we're willing to spend trying to turn a DOI into a direct PDF URL
DOI_RESOLVE_TIMEOUT_S: float = 8.0

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight stopwords for query simplification (Semantic Scholar)
# ──────────────────────────────────────────────────────────────────────────────

_S2_STOPWORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "are",
    "is",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "with",
    "latest",
    "recent",
    "new",
    "newest",
    "current",
    "state",
    "art",
    "paper",
    "papers",
    "article",
    "articles",
    "about",
    "regarding",
    "related",
    "using",
    "use",
    "based",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
}


# ──────────────────────────────────────────────────────────────────────────────
# Generic utilities
# ──────────────────────────────────────────────────────────────────────────────


def _year_from_date(s: str | None) -> int | None:
    if not s:
        return None
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


def _simplify_query(q: str) -> str:
    """
    Heuristically simplify a natural-language question into a keyword-style
    query that works better with Semantic Scholar's /paper/search.

    Example:
        "What are the latest transformer architectures for NLP?"
        -> "transformer architectures nlp"
    """
    if not q:
        return q

    q_clean = q.strip().lower()
    q_clean = re.sub(r"[?!\.\,;:\s]+$", "", q_clean)

    tokens = re.findall(r"[a-z0-9]+", q_clean)
    if not tokens:
        return q.strip()

    filtered = [t for t in tokens if t not in _S2_STOPWORDS]
    if not filtered:
        filtered = tokens

    simplified = " ".join(filtered)
    if len(simplified) > len(q.strip()):
        return q.strip()

    return simplified or q.strip()


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _parse_crossref_filter(s: str) -> dict[str, str]:
    """
    Accepts:
      - 'from-pub-date:2022-01-01,until-pub-date:2025-12-31,type:journal-article'
      - or a single raw DOI like '10.1007/s10489-023-04789-3'
    """
    filt: dict[str, str] = {}
    if not s:
        return filt
    s = s.strip()
    if s.startswith("10.") and "/" in s and ":" not in s:
        filt["doi"] = s
        return filt
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            k, v = k.strip(), v.strip()
            if k and v:
                filt[k] = v
    return filt


def _parse_indices_arg(s: str, upper_bound: int) -> list[int]:
    """
    Parse index selections into a list of ints (supports JSON, lists, and ranges).
    Values are clipped and deduped.
    """
    s = (s or "").strip()
    if not s:
        return []

    # Try JSON first
    try:
        parsed = json.loads(s)
        cand = parsed if isinstance(parsed, list) else []
    except Exception:
        cand = []

    # Fallback: comma / whitespace separated, with range support
    if not cand:
        parts = re.split(r"[,\s]+", s)
        cand = []
        for p in parts:
            if not p:
                continue
            m = re.match(r"^(\d+)\s*(?:\-|\..)\s*(\d+)$", p)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a <= b:
                    cand.extend(range(a, b + 1))
                else:
                    cand.extend(range(b, a + 1))
            else:
                try:
                    cand.append(int(p))
                except Exception:
                    pass

    cand = [i for i in cand if isinstance(i, int)]
    cand = [min(max(i, 0), max(upper_bound - 1, 0)) for i in cand]
    return sorted(dict.fromkeys(cand))


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer & sentence splitting
# ──────────────────────────────────────────────────────────────────────────────

try:
    import blingfire

    HAS_BLINGFIRE = True
except Exception:
    HAS_BLINGFIRE = False

try:
    from transformers import AutoTokenizer

    HAS_HF = True
except Exception:
    HAS_HF = False


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    """
    Use the same tokenizer as the embedding model to make chunk sizes
    match the embedding model’s token budget.
    """
    if not HAS_HF:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        return tok
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if HAS_BLINGFIRE:
        s = blingfire.text_to_sentences(text)  # one sentence per line
        return [t.strip() for t in s.splitlines() if t.strip()]
    # Lightweight regex fallback
    return [t.strip() for t in re.split(r"(?<=[.!?])\s+", text) if t.strip()]


def _chunk_text(
    text: str,
    chunk_size: int = 1200,  # legacy chars-based size (fallback path)
    overlap: int = 150,  # legacy chars-based overlap (fallback path)
    *,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """
    Prefer token-aware chunking using the same HF tokenizer as the embedding model.
    If the tokenizer is unavailable, fall back to a sentence-aware char-based packer.
    """
    tok = _get_tokenizer(embedding_model_name)
    if tok is None:
        # Char-based fallback (keeps old behavior)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= chunk_size:
            return [text]
        sents = _split_sentences(text) or [text]
        chunks, cur = [], ""
        for s in sents:
            nxt = (cur + " " + s).strip() if cur else s
            if len(nxt) <= chunk_size:
                cur = nxt
            else:
                if cur:
                    chunks.append(cur)
                cur = s
                while len(cur) > chunk_size:
                    chunks.append(cur[:chunk_size])
                    cur = cur[chunk_size:]
        if cur:
            chunks.append(cur)
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prepend = chunks[i - 1][-overlap:]
                chunks[i] = (prepend + chunks[i]).strip()
        return [c for c in chunks if c]

    # Token-aware path
    if max_tokens is None:
        lower = embedding_model_name.lower()
        if "mini" in lower or "small" in lower:
            max_tokens = 240
        elif "base" in lower or "mpnet" in lower or "e5" in lower:
            max_tokens = 350
        else:
            max_tokens = 300
    if overlap_tokens is None:
        overlap_tokens = max(16, max_tokens // 6)  # ~15–20%

    def tok_len(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=False))

    sents = _split_sentences(text) or [text]
    chunks: list[str] = []
    cur_sents: list[str] = []
    cur_ids_len = 0

    for s in sents:
        s_len = tok_len(s)
        if s_len > max_tokens:
            # Flush current
            if cur_sents:
                chunks.append(" ".join(cur_sents).strip())
                cur_sents, cur_ids_len = [], 0
            # Hard-split this long sentence by tokens
            ids = tok.encode(s, add_special_tokens=False)
            start = 0
            while start < len(ids):
                end = min(start + max_tokens, len(ids))
                piece = tok.decode(ids[start:end], skip_special_tokens=True).strip()
                if piece:
                    chunks.append(piece)
                if end >= len(ids):
                    break
                start = max(0, end - overlap_tokens)
            continue

        # Try to add to current chunk
        if cur_ids_len + s_len <= max_tokens:
            cur_sents.append(s)
            cur_ids_len += s_len
        else:
            if cur_sents:
                chunk_text = " ".join(cur_sents).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if overlap_tokens > 0:
                    prev_ids = tok.encode(chunk_text, add_special_tokens=False)
                    keep_ids = (
                        prev_ids[-overlap_tokens:]
                        if overlap_tokens <= len(prev_ids)
                        else prev_ids
                    )
                    prefix = tok.decode(keep_ids, skip_special_tokens=True).strip()
                    cur_sents = [prefix] if prefix else []
                    cur_ids_len = tok_len(prefix) if prefix else 0
                else:
                    cur_sents, cur_ids_len = [], 0
            if s_len <= max_tokens:
                cur_sents.append(s)
                cur_ids_len = tok_len(" ".join(cur_sents))

    if cur_sents:
        chunk_text = " ".join(cur_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return [c for c in chunks if c]


# ──────────────────────────────────────────────────────────────────────────────
# PDF + HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────

try:
    import fitz  # PyMuPDF

    HAS_FITZ = True
except Exception:
    HAS_FITZ = False


def _read_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    if not HAS_FITZ:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        txt = []
        for page in doc:
            try:
                txt.append(page.get_text() or "")
            except Exception:
                txt.append("")
        doc.close()
        return "\n".join(txt)
    except Exception:
        return ""


def _http_get_bytes(
    client: httpx.Client,
    url: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Fetch URL with retries."""
    backoff = 0.6
    for attempt in range(3):
        try:
            r = client.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            )
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(backoff * (attempt + 1))
    return b""


def _http_head(
    client: httpx.Client,
    url: str,
    timeout: float | httpx.Timeout,
    headers: dict[str, str] | None = None,
) -> httpx.Response | None:
    """HEAD helper with graceful fallback (some servers block HEAD)."""
    try:
        to = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        r = client.head(url, timeout=to, follow_redirects=True, headers=headers)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _http_get_response(
    client: httpx.Client,
    url: str,
    timeout: float | httpx.Timeout,
    headers: dict[str, str] | None = None,
) -> httpx.Response | None:
    """GET helper that returns the Response object (for header inspection)."""
    try:
        to = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
        r = client.get(url, timeout=to, follow_redirects=True, headers=headers)
        r.raise_for_status()
        return r
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# arXiv & Crossref PDF helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    s = str(arxiv_id).strip()
    s = s.replace("arXiv:", "").replace("ARXIV:", "").strip()
    return s


def _arxiv_pdf_url(entry_id: str | None, arxiv_id: str | None) -> str | None:
    """Return a robust arXiv PDF URL regardless of arxiv lib version."""
    aid = _normalize_arxiv_id(arxiv_id)
    if aid:
        return f"https://arxiv.org/pdf/{aid}.pdf"
    if entry_id:
        url = entry_id.replace("http://", "https://")
        if "/abs/" in url:
            url = url.replace("/abs/", "/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
        return url
    return None


def _crossref_pick_pdf_link(item: dict[str, Any]) -> str | None:
    """
    Prefer official Crossref 'link' entries with content-type application/pdf.
    Heuristics:
      1) link[].content-type == 'application/pdf' (prefer version=='vor')
      2) any link URL ending with .pdf
      3) None (let DOI resolution try later)
    """
    links = item.get("link") or []
    if not isinstance(links, list):
        return None

    def _is_pdf_link(l: dict[str, Any]) -> bool:
        ctype = (l or {}).get("content-type", "") or ""
        url = (l or {}).get("URL", "") or ""
        return ("application/pdf" in ctype.lower()) or url.lower().endswith(".pdf")

    pdf_links = [l for l in links if _is_pdf_link(l)]
    if not pdf_links:
        return None

    vor = [l for l in pdf_links if (l or {}).get("version", "").lower() == "vor"]
    pick = vor[0] if vor else pdf_links[0]
    return (pick or {}).get("URL") or None


def _resolve_doi_to_pdf_url(
    client: httpx.Client,
    doi: str,
    timeout: float = DOI_RESOLVE_TIMEOUT_S,
) -> str | None:
    """
    Resolve DOI -> direct PDF URL using 'Accept: application/pdf' with a hard deadline.
    Strategy:
      - Try HEAD first (fast). If Content-Type=application/pdf, return final URL.
      - If inconclusive and time remains, do GET with same Accept header.
    """
    if not doi:
        return None

    base = f"https://doi.org/{doi}"
    headers = {"Accept": "application/pdf", "User-Agent": "AcademicAgent/1.0"}

    # Hard deadline across both attempts
    deadline = time.monotonic() + max(0.1, float(timeout))

    def _remaining_timeout() -> httpx.Timeout | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        remaining = max(0.1, remaining)
        connect = min(3.0, max(0.05, 0.3 * remaining))
        read = max(0.05, 0.6 * remaining)
        write = min(2.0, max(0.05, 0.1 * remaining))
        return httpx.Timeout(connect=connect, read=read, write=write, pool=read)

    # Attempt HEAD (fast path)
    to = _remaining_timeout()
    if to is not None:
        r_head = _http_head(client, base, timeout=to, headers=headers)
        if r_head is not None:
            ctype = (r_head.headers.get("Content-Type") or "").lower()
            if "application/pdf" in ctype:
                return str(r_head.url)

    # Attempt GET if we still have time
    to = _remaining_timeout()
    if to is None:
        return None

    r_get = _http_get_response(client, base, timeout=to, headers=headers)
    if r_get is not None:
        ctype = (r_get.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ctype:
            return str(r_get.url)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Metadata sanitizers
# ──────────────────────────────────────────────────────────────────────────────


def _to_python_scalar(x):
    try:
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
    except Exception:
        pass
    return x


def _sanitize_value(v):
    if v is None:
        return None
    v = _to_python_scalar(v)
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, dict, tuple)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    if "authors" in meta:
        v = meta.get("authors")
        if isinstance(v, list):
            clean["authors"] = ", ".join(str(a) for a in v)
            clean["authors_json"] = json.dumps(v, ensure_ascii=False)
        elif v is not None:
            clean["authors"] = str(v)

    for k, v in meta.items():
        if k == "authors":
            continue
        if k == "year":
            if v is not None:
                try:
                    vv = _to_python_scalar(v)
                    clean["year"] = int(vv)
                except Exception:
                    pass
            continue
        sv = _sanitize_value(v)
        if sv is not None:
            clean[k] = sv

    return clean


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Scholar request helper (shared retry logic)
# ──────────────────────────────────────────────────────────────────────────────


def _s2_request(
    client: httpx.Client,
    path: str,
    params: dict[str, Any],
    *,
    timeout: float = 20.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Generic Semantic Scholar Graph API GET with simple 429 handling.

    - Uses S2_API_URL or SEMANTIC_SCHOLAR_API_URL if set, otherwise default.
    - Retries on 429 a limited number of times with exponential-ish backoff.
    """
    base_env = (
        os.getenv("S2_API_URL")
        or os.getenv("SEMANTIC_SCHOLAR_API_URL")
        or "https://api.semanticscholar.org/graph/v1"
    )
    base = base_env.rstrip("/")
    url = f"{base}/{path.lstrip('/')}"
    backoff = 0.8

    for attempt in range(max_retries + 1):
        try:
            print(f"   [S2 API] Attempt {attempt + 1}: {url}")
            print(f"   [S2 API] Params: {params}")
            r = client.get(url, params=params, timeout=timeout, follow_redirects=True)
            print(f"   [S2 API] Status: {r.status_code}")

            if r.status_code == 429 and attempt < max_retries:
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        sleep_for = max(0.5, float(ra))
                    except Exception:
                        sleep_for = backoff * (attempt + 1)
                else:
                    sleep_for = backoff * (attempt + 1)
                sleep_for = min(sleep_for, 5.0)
                print(f"   [S2 API] Rate limited, sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue

            r.raise_for_status()
            text_preview = r.text[:200]
            print(text_preview + ("..." if len(r.text) > 200 else ""))
            return r.json()
        except Exception as e:
            print(f"   [S2 API] Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            if attempt == max_retries:
                print("   [S2 API] All retries failed, returning empty dict")
                return {}
            time.sleep(backoff * (attempt + 1))
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Paper:
    """Normalized paper metadata across all sources."""

    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    pdf_url: str | None
    source: str  # "arxiv" | "s2" | "crossref"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "source": self.source,
            **(
                {"pdf_via_doi": self.extra.get("pdf_via_doi")}
                if "pdf_via_doi" in self.extra
                else {}
            ),
        }

    def ref_id(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        if self.url:
            return f"url:{self.url}"
        if self.title:
            return f"title:{self.title.strip().lower()}:{self.year or ''}"
        return f"uuid:{uuid.uuid4()}"


@dataclass
class AcademicContext:
    """Shared state for academic agent (like Context in TimeSeriesAgent)."""

    papers: list[Paper] = field(default_factory=list)
    ingested_ids: list[str] = field(default_factory=list)
    last_query: str | None = None
    last_search_source: str | None = None
    total_chunks: int = 0

    def add_papers(self, new_papers: list[Paper]) -> int:
        seen = {p.ref_id() for p in self.papers}
        added = 0
        for p in new_papers:
            rid = p.ref_id()
            if rid not in seen:
                self.papers.append(p)
                seen.add(rid)
                added += 1
        return added

    def get_summary(self) -> str:
        return (
            f"Papers: {len(self.papers)} | "
            f"Ingested: {len(self.ingested_ids)} chunks | "
            f"Last query: {self.last_query or 'None'}"
        )


# ─── LLM tool echo policy (still used for debugging/summary payloads) ───
MAX_TITLES_ECHO = 12


def _titles_payload(papers, added_new: int) -> str:
    titles = [
        p.title[:160] for p in papers[:MAX_TITLES_ECHO] if getattr(p, "title", "")
    ]
    return json.dumps(
        {"count": len(papers), "added_new": added_new, "titles": titles},
        ensure_ascii=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tool-like classes (used programmatically, not by the LLM)
# ──────────────────────────────────────────────────────────────────────────────


class SearchTools:
    """Paper search across multiple sources (used programmatically)."""

    __tools__ = [
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "auto_multi_search",
    ]

    def __init__(self, ctx: AcademicContext):
        self.ctx = ctx

        # Small HTTP client; add S2 API key if present
        s2_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"User-Agent": "AcademicAgent/1.0"}
        if s2_key:
            headers["x-api-key"] = s2_key
        self.http = httpx.Client(headers=headers)

    def auto_multi_search(
        self,
        query: str,
        per_source_limit: str = "10",
        order: str = "s2,arxiv,crossref",
        open_access_only: str = "false",
    ) -> str:
        """
        Run multi-source search (no ingestion). Returns counts and top titles per source.
        """
        print(
            f"\n[MultiSearch] query={query!r} per_source_limit={per_source_limit} "
            f"order={order}"
        )

        lim = str(max(1, min(int(per_source_limit), 20)))
        srcs = [s.strip().lower() for s in (order or "").split(",") if s.strip()]
        srcs = [s for s in srcs if s in ("s2", "arxiv", "crossref")]
        if not srcs:
            srcs = ["s2", "arxiv", "crossref"]

        summary: dict[str, Any] = {"query": query, "sources": []}
        before_count = len(self.ctx.papers)

        for s in srcs:
            try:
                if s == "s2":
                    print(f"[MultiSearch] S2(limit={lim})")
                    payload = self.semantic_scholar_search(
                        query,
                        lim,
                        open_access_only=open_access_only,
                    )
                    payload_obj = json.loads(payload)
                    summary["sources"].append({"name": "s2", **payload_obj})
                elif s == "arxiv":
                    print(f"[MultiSearch] arXiv(max_results={lim})")
                    payload = self.arxiv_search(
                        query,
                        lim,
                        arxiv.SortCriterion.SubmittedDate,
                        "0",
                    )
                    payload_obj = json.loads(payload)
                    summary["sources"].append({"name": "arxiv", **payload_obj})
                elif s == "crossref":
                    print(f"[MultiSearch] Crossref(rows={lim})")
                    payload = self.crossref_search(
                        query,
                        lim,
                        "",
                        "relevance",
                        "desc",
                    )
                    payload_obj = json.loads(payload)
                    summary["sources"].append({"name": "crossref", **payload_obj})
            except Exception as e:
                print(f"[MultiSearch] {s} error: {e}")
                summary["sources"].append(
                    {
                        "name": s,
                        "error": str(e),
                        "count": 0,
                        "added_new": 0,
                        "titles": [],
                    }
                )

        after_count = len(self.ctx.papers)
        summary["new_total_added"] = after_count - before_count
        return json.dumps(summary, ensure_ascii=False)

    # ——— arXiv ———
    def arxiv_search(
        self,
        query: str,
        max_results: str = "10",
        sort_by: str = arxiv.SortCriterion.SubmittedDate,
        start: str = "0",
    ) -> str:
        print(f"\n[arXiv] Searching for: '{query}'")
        print(f"   Parameters: max_results={max_results}, sort_by={sort_by}")

        max_r = max(1, min(int(max_results), 50))

        sort_val = None
        if hasattr(arxiv, "SortCriterion"):
            try:
                sc = arxiv.SortCriterion
                sort_val = {
                    "relevance": getattr(sc, "Relevance", None),
                    "lastUpdatedDate": getattr(sc, "LastUpdatedDate", None),
                    "submittedDate": getattr(sc, "SubmittedDate", None),
                }.get(sort_by, None)
            except Exception:
                sort_val = None

        try:
            if sort_val is not None:
                search = arxiv.Search(query=query, max_results=max_r, sort_by=sort_val)
            else:
                search = arxiv.Search(query=query, max_results=max_r)
            try:
                client = arxiv.Client()
                results = list(client.results(search))
            except Exception:
                results = list(search.results())
            print(f"   Found {len(results)} papers")
        except Exception as e:
            print(f"   Error: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        papers: list[Paper] = []
        for i, result in enumerate(results[:max_r], 1):
            year = getattr(
                getattr(result, "published", None), "year", None
            ) or _year_from_date(str(getattr(result, "published", "")))
            entry_id = getattr(result, "entry_id", None) or ""
            arxiv_id_from_entry = entry_id.split("/")[-1] if entry_id else None
            arxiv_id = _normalize_arxiv_id(
                arxiv_id_from_entry
                or getattr(result, "get_short_id", lambda: None)()
                or getattr(result, "arxiv_id", None)
            )
            auth_objs = getattr(result, "authors", []) or []
            authors = [
                getattr(a, "name", None) or str(a)
                for a in auth_objs
                if (getattr(a, "name", None) or str(a))
            ]

            raw_pdf_url = getattr(result, "pdf_url", None)
            pdf_url = raw_pdf_url or _arxiv_pdf_url(entry_id, arxiv_id)

            papers.append(
                Paper(
                    title=getattr(result, "title", "") or "",
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    abstract=getattr(result, "summary", None),
                    doi=None,
                    arxiv_id=arxiv_id,
                    url=entry_id
                    or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
                    pdf_url=pdf_url,
                    source="arxiv",
                )
            )
            print(
                f"   [{i}] {papers[-1].title[:80]} ({year}) "
                f"{'PDF' if pdf_url else 'no-PDF'}"
            )

        added = self.ctx.add_papers(papers)
        self.ctx.last_query = query
        self.ctx.last_search_source = "arxiv"
        print(f"   Added {added} new papers to context (total: {len(self.ctx.papers)})")

        return _titles_payload(papers, added)

    # ——— Semantic Scholar (Graph API /paper/search) ———
    def semantic_scholar_search(
        self,
        query: str,
        limit: str = "10",
        fields: str = (
            "title,authors,venue,year,publicationDate,publicationTypes,"
            "externalIds,url,openAccessPdf,abstract"
        ),
        open_access_only: str = "false",
        offset: str = "0",
    ) -> str:
        """
        Paper relevance search via Semantic Scholar Graph API.

        Matches the swagger for:
            GET /graph/v1/paper/search

        - `query` (required): plain-text search query
        - `limit`, `offset`: pagination (limit <= 100)
        - `fields`: comma-separated list as per swagger
        - `open_access_only`: if truthy, adds `openAccessPdf` param
        """
        query_orig = query
        query = _simplify_query(query)
        print(f"\n[SemanticScholar/API] Searching for: '{query}' (from: {query_orig!r})")
        print(
            f"   Parameters: limit={limit}, offset={offset}, "
            f"open_access_only={open_access_only}"
        )

        lim = max(1, min(int(limit), 100))
        off = max(0, int(offset))

        # Build fields param (swagger: single string, not repeated)
        fields_list = [f.strip() for f in (fields or "").split(",") if f.strip()]
        fields_param = (
            ",".join(fields_list)
            if fields_list
            else (
                "paperId,title,authors,venue,year,publicationDate,"
                "externalIds,url,openAccessPdf,abstract"
            )
        )

        want_open = str(open_access_only).lower() in ("true", "1", "yes")

        params: dict[str, Any] = {
            "query": query,
            "limit": lim,
            "offset": off,
            "fields": fields_param,
        }
        if want_open:
            # Presence of this parameter (no value) filters to open access
            params["openAccessPdf"] = ""

        raw = _s2_request(
            self.http,
            "paper/search",
            params=params,
            timeout=20.0,
            max_retries=1,
        )

        if not isinstance(raw, dict) or not raw:
            print("   Unexpected or empty S2 response")
            return json.dumps(
                {
                    "error": "S2 request failed or returned non-dict response",
                    "count": 0,
                    "added_new": 0,
                    "titles": [],
                },
                ensure_ascii=False,
            )

        total = raw.get("total")
        offset_val = raw.get("offset")
        items = raw.get("data")

        # IMPORTANT: the live API may omit `data` when total == 0
        if items is None:
            if isinstance(total, int) and total == 0:
                print(
                    "   S2 reports total=0 and no 'data' field; treating as empty result set."
                )
                items = []
            else:
                # Fallback: try alternate keys just in case
                for key in ("papers", "results", "items"):
                    if isinstance(raw.get(key), list):
                        items = raw[key]
                        print(f"   Using fallback key '{key}' for results")
                        break

        if not isinstance(items, list):
            print("   No usable result list in S2 response; raw payload:")
            try:
                print(json.dumps(raw, indent=2)[:2000])
            except Exception:
                print(str(raw)[:2000])

            return json.dumps(
                {
                    "error": "S2 response has no result list; see logs",
                    "count": 0,
                    "added_new": 0,
                    "titles": [],
                },
                ensure_ascii=False,
            )

        print(f"   Found {len(items)} papers (raw) (total={total}, offset={offset_val})")

        papers: list[Paper] = []

        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue

            title = (item.get("title") or "").strip()
            venue = (item.get("venue") or None) or None
            year = item.get("year") or _year_from_date(item.get("publicationDate"))

            abs_ = item.get("abstract")
            url_paper = item.get("url")

            ext = item.get("externalIds") or {}
            doi = ext.get("DOI")
            arx = _normalize_arxiv_id(ext.get("ArXiv")) if ext else None

            # Authors (swagger: authors[*].authorId, name, etc.)
            auth_raw = item.get("authors") or []
            authors: list[str] = []
            for a in auth_raw:
                nm = a.get("name") if isinstance(a, dict) else str(a)
                if nm:
                    authors.append(nm)

            # PDF URL (if openAccessPdf field is present)
            pdf_url = None
            oapdf = item.get("openAccessPdf") or {}
            if isinstance(oapdf, dict):
                pdf_url = oapdf.get("url") or pdf_url
            if (not pdf_url) and arx:
                pdf_url = _arxiv_pdf_url(None, arx)

            # Note: since we already used `openAccessPdf` at the API level,
            # `want_open` should *normally* imply pdf_url is not None, but we keep
            # this defensive check anyway.
            if want_open and not pdf_url:
                continue

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=abs_,
                    doi=doi,
                    arxiv_id=arx,
                    url=url_paper,
                    pdf_url=pdf_url,
                    source="s2",
                    extra={
                        "publicationTypes": item.get("publicationTypes"),
                        "s2_paperId": item.get("paperId"),
                    },
                )
            )
            print(
                f"   [{i}] {title[:80]} ({year}) "
                f"{'PDF' if pdf_url else 'no-PDF'}"
            )

        added = self.ctx.add_papers(papers)
        self.ctx.last_query = query
        self.ctx.last_search_source = "s2"
        print(
            f"   Added {added} new papers to context (total: {len(self.ctx.papers)})"
        )

        return _titles_payload(papers, added)

    # ——— Crossref ———
    def crossref_search(
        self,
        query: str,
        rows: str = "10",
        filter: str = "",
        sort: str = "",
        order: str = "desc",
    ) -> str:
        print(f"\n[Crossref] Searching for: '{query}'")
        print(
            f"   Parameters: rows={rows}, filter='{filter}', "
            f"sort='{sort}', order='{order}'"
        )

        cr = Crossref()
        r = max(1, min(int(rows), 20))

        params: dict[str, Any] = {"query": query, "rows": r}
        filt = _parse_crossref_filter(filter)
        if filt:
            params["filter"] = filt
        if sort:
            params["sort"] = sort
        if order in ("asc", "desc"):
            params["order"] = order

        try:
            works_result = cr.works(**params)
            works = works_result["message"]["items"]
        except Exception as e:
            print(f"   Error: {e}")
            try:
                params.pop("filter", None)
                params.pop("sort", None)
                params.pop("order", None)
                works_result = cr.works(**params)
                works = works_result["message"]["items"]
            except Exception as e2:
                print(f"   Fallback also failed: {e2}")
                return json.dumps({"error": str(e)}, ensure_ascii=False)

        print(f"   Found {len(works)} papers")

        papers: list[Paper] = []
        for i, item in enumerate(works, 1):
            title = (item.get("title") or [""])[0]
            names: list[str] = []
            for a in item.get("author") or []:
                given = (a or {}).get("given")
                family = (a or {}).get("family")
                nm = " ".join([x for x in [given, family] if x])
                if nm:
                    names.append(nm)

            year = None
            for key in ("published-print", "published-online", "created", "issued"):
                d = (item.get(key) or {}).get("date-parts")
                if d and d[0]:
                    year = d[0][0]
                    break

            venue = (item.get("container-title") or [""])[0] or None
            doi = item.get("DOI")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)
            abstract = item.get("abstract")

            pdf_url = _crossref_pick_pdf_link(item)

            papers.append(
                Paper(
                    title=title,
                    authors=names,
                    year=year,
                    venue=venue,
                    abstract=abstract,
                    doi=doi,
                    arxiv_id=None,
                    url=url,
                    pdf_url=pdf_url,
                    source="crossref",
                    extra={"type": item.get("type")},
                )
            )
            print(f"   [{i}] {title[:80]} ({year}) {'PDF' if pdf_url else 'no-PDF'}")

        added = self.ctx.add_papers(papers)
        self.ctx.last_query = query
        self.ctx.last_search_source = "crossref"
        print(f"   Added {added} new papers to context (total: {len(self.ctx.papers)})")

        return _titles_payload(papers, added)


class IngestionTools:
    """Paper ingestion into RAG DocumentDB (used programmatically)."""

    __tools__ = ["ingest_papers", "ingest_all"]

    def __init__(self, ctx: AcademicContext, doc_db: DocumentDB):
        self.ctx = ctx
        self.doc_db = doc_db

    def ingest_papers(
        self,
        paper_indices: str,
        use_pdf: str = "true",
        chunk_size: str = "1200",
        overlap: str = "150",
        max_workers: str = "4",
    ) -> str:
        """Ingest specific papers by index into RAG. Use 'all' to ingest all fetched papers.

        The heavy per-paper work (PDF download, extraction, chunking) is done in
        parallel threads; the final collection.add() calls are done in the main
        thread for safety.
        """
        print(f"\n[Ingestion] Starting paper ingestion (parallel)...")
        print(
            f"   Parameters: paper_indices={paper_indices}, use_pdf={use_pdf}, "
            f"chunk_size={chunk_size}, overlap={overlap}, max_workers={max_workers}"
        )

        # ---- Select papers ---------------------------------------------------
        if isinstance(paper_indices, str) and paper_indices.strip().lower() == "all":
            papers = self.ctx.papers
            print(f"   Ingesting all {len(papers)} papers")
        else:
            try:
                idxs = _parse_indices_arg(
                    str(paper_indices),
                    upper_bound=len(self.ctx.papers),
                )
                papers = [
                    self.ctx.papers[i] for i in idxs if 0 <= i < len(self.ctx.papers)
                ]
                print(f"   Ingesting {len(papers)} selected papers (indices: {idxs})")
            except Exception as e:
                print(f"   Error parsing indices: {e}")
                return json.dumps(
                    {"error": "Invalid paper_indices format"},
                    ensure_ascii=False,
                )

        if not papers:
            print("   No papers to ingest")
            return json.dumps(
                {"error": "No papers to ingest"},
                ensure_ascii=False,
            )

        # ---- Common config ---------------------------------------------------
        use_pdf_bool = str(use_pdf).lower() in ("true", "1", "yes")
        cs = int(chunk_size)
        ov = int(overlap)
        try:
            mw = max(1, int(max_workers))
        except Exception:
            mw = 4

        added_ids: list[str] = []
        total_chunks = 0

        print(f"   Processing {len(papers)} papers with {mw} worker(s)...")

        # We share a single httpx.Client across threads (thread-safe for GETs).
        with httpx.Client(
            headers={"User-Agent": "AcademicAgent/1.0"},
            timeout=30.0,
            follow_redirects=True,
        ) as client:

            def _process_single_paper(args):
                idx, paper = args
                try:
                    print(
                        f"\n   [worker] [{idx}/{len(papers)}] Processing: "
                        f"{paper.title[:120]}"
                    )

                    # Ensure arXiv PDF if missing
                    if (
                        (not paper.pdf_url)
                        and (paper.source in ("arxiv", "s2"))
                        and paper.arxiv_id
                    ):
                        paper.pdf_url = _arxiv_pdf_url(paper.url, paper.arxiv_id)

                    # If item lacks PDF but has DOI, try DOI→PDF resolution (works for s2 and crossref)
                    if (not paper.pdf_url) and paper.doi:
                        print("       • (worker) Resolving DOI to PDF...")
                        try:
                            resolved = _resolve_doi_to_pdf_url(
                                client,
                                paper.doi,
                                timeout=DOI_RESOLVE_TIMEOUT_S,
                            )
                            if resolved:
                                paper.pdf_url = resolved
                                paper.extra["pdf_via_doi"] = True
                                print(
                                    f"       • (worker) DOI resolved to PDF URL: {resolved}"
                                )
                            else:
                                print(
                                    "       • (worker) DOI resolution did not yield a PDF URL"
                                )
                        except Exception as e:
                            print(f"       • (worker) DOI resolution error: {e}")

                    # Build text from metadata + abstract
                    text = self._build_paper_text(paper)
                    print(f"       • (worker) Built base text ({len(text)} chars)")

                    # Optionally fetch PDF
                    if use_pdf_bool and paper.pdf_url:
                        print(
                            f"       • (worker) Fetching PDF from "
                            f"{paper.pdf_url[:120]} ..."
                        )
                        try:
                            pdf_start = time.time()
                            pdf_bytes = _http_get_bytes(
                                client,
                                paper.pdf_url,
                                timeout=30.0,
                            )
                            pdf_dur = time.time() - pdf_start

                            if pdf_bytes:
                                print(
                                    f"       • (worker) Downloaded {len(pdf_bytes)} bytes "
                                    f"in {pdf_dur:.1f}s"
                                )
                                print("       • (worker) Extracting text from PDF...")
                                pdf_text = _read_pdf_bytes(pdf_bytes)
                                if pdf_text:
                                    text += f"\n\nFullText:\n{pdf_text}"
                                    print(
                                        f"       • (worker) Extracted "
                                        f"{len(pdf_text)} chars from PDF"
                                    )
                                else:
                                    print(
                                        "       • (worker) Could not extract text "
                                        "(using abstract only)"
                                    )
                            else:
                                print(
                                    "       • (worker) PDF download failed "
                                    "(using abstract only)"
                                )
                        except Exception as e:
                            print(
                                f"       • (worker) PDF error: {e} "
                                "(using abstract only)"
                            )
                    elif paper.pdf_url:
                        print("       • (worker) Skipping PDF (use_pdf=false)")
                    else:
                        print("       • (worker) No PDF URL available")

                    # Chunk and prepare documents/metadatas/ids
                    print(
                        "       • (worker) Chunking text with embedding-aligned tokenizer"
                    )
                    chunks = _chunk_text(
                        text,
                        chunk_size=cs,
                        overlap=ov,
                        embedding_model_name=self.doc_db.embedding_model_name,
                        max_tokens=None,
                        overlap_tokens=None,
                    )
                    if not chunks:
                        print("       • (worker) No chunks generated, skipping")
                        return ([], [], [])

                    print(f"       • (worker) Generated {len(chunks)} chunks")
                    ref_id = paper.ref_id()
                    metas: list[dict[str, Any]] = []
                    for i_chunk in range(len(chunks)):
                        raw_meta = {
                            **paper.to_metadata(),
                            "chunk": int(i_chunk),
                            "ref_id": str(ref_id),
                        }
                        metas.append(_sanitize_metadata(raw_meta))

                    ids = [f"{ref_id}#c{i_chunk}" for i_chunk in range(len(chunks))]

                    # Return to main thread for collection.add()
                    return (ids, chunks, metas)

                except Exception as e:
                    print(
                        f"       • (worker) ERROR processing paper '{paper.title}': {e}"
                    )
                    return ([], [], [])

            # ---- Run workers in parallel ------------------------------------
            futures = []
            with ThreadPoolExecutor(max_workers=mw) as executor:
                for idx, paper in enumerate(papers, 1):
                    futures.append(executor.submit(_process_single_paper, (idx, paper)))

                for fut in as_completed(futures):
                    ids, chunks, metas = fut.result()
                    if not chunks:
                        continue
                    # Do the DB write on the main thread
                    print(f"   [main] Adding {len(chunks)} chunks to vector store...")
                    self.doc_db.collection.add(
                        documents=chunks,
                        metadatas=metas,
                        ids=ids,
                    )
                    added_ids.extend(ids)
                    total_chunks += len(chunks)

        # ---- Final bookkeeping ----------------------------------------------
        self.ctx.ingested_ids.extend(added_ids)
        self.ctx.total_chunks += total_chunks

        print("\n   Ingestion complete!")
        print(f"   • Papers processed: {len(papers)}")
        print(f"   • Chunks added: {total_chunks}")
        print(f"   • Total ingested: {len(self.ctx.ingested_ids)} chunks")

        return json.dumps(
            {
                "ingested_papers": len(papers),
                "added_chunks": total_chunks,
                "total_ingested": len(self.ctx.ingested_ids),
            },
            ensure_ascii=False,
        )

    def ingest_all(
        self,
        use_pdf: str = "true",
        chunk_size: str = "1200",
        overlap: str = "150",
        max_workers: str = "4",
    ) -> str:
        """Convenience: ingest all fetched papers (parallel)."""
        return self.ingest_papers("all", use_pdf, chunk_size, overlap, max_workers)

    def _build_paper_text(self, paper: Paper) -> str:
        """Build base text from paper metadata."""
        parts = [f"{paper.title}"]
        if paper.authors:
            parts.append(f"Authors: {', '.join(paper.authors)}")
        if paper.venue:
            parts.append(f"Venue: {paper.venue}")
        if paper.year:
            parts.append(f"Year: {paper.year}")
        if paper.doi:
            parts.append(f"DOI: {paper.doi}")
        if paper.arxiv_id:
            parts.append(f"arXiv: {paper.arxiv_id}")
        if paper.url:
            parts.append(f"URL: {paper.url}")
        if paper.pdf_url:
            parts.append(f"PDF: {paper.pdf_url}")
        if paper.extra.get("pdf_via_doi"):
            parts.append("PDF-Note: resolved via DOI Accept: application/pdf")
        if paper.extra.get("publicationTypes"):
            parts.append(f"Type: {paper.extra.get('publicationTypes')}")
        parts.append(f"Source: {paper.source}")

        text = "\n".join(parts) + "\n\n"
        if paper.abstract:
            text += f"Abstract: {paper.abstract}\n"
        return text


class QueryTools:
    """RAG-based Q&A over ingested papers (builds context + bibliography)."""

    __tools__ = ["qa_papers", "list_papers"]

    def __init__(self, ctx: AcademicContext, doc_db: DocumentDB):
        self.ctx = ctx
        self.doc_db = doc_db

    def qa_papers(
        self,
        question: str,
        top_k: str = "5",
        max_ctx_chars: str = "4000",
    ) -> str:
        """Answer question using ingested papers with citations."""
        print(f"\n[Q&A] Processing question: '{question}'")
        print(f"   Parameters: top_k={top_k}, max_ctx_chars={max_ctx_chars}")

        k = max(1, min(int(top_k), 12))
        max_chars = int(max_ctx_chars)

        print(f"   Querying vector store for {k} most relevant chunks...")
        query_start = time.time()
        res = self.doc_db.collection.query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        query_dur = time.time() - query_start
        print(f"   Retrieved in {query_dur:.2f}s")

        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or [None] * len(docs)

        print("   Deduplicating by paper (ref_id)...")
        seen = set()
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            ref_id = meta.get("ref_id") or str(uuid.uuid4())
            if ref_id in seen:
                continue
            seen.add(ref_id)
            ranked.append((float(dist) if dist is not None else 0.0, doc, meta))

        ranked.sort(key=lambda x: x[0])
        print(f"   Unique papers after dedupe: {len(ranked)}")

        print(f"   Building context (max {max_chars} chars)...")
        context_parts: list[str] = []
        bibliography: list[dict[str, Any]] = []
        total = 0

        for idx, (dist, doc, meta) in enumerate(ranked, start=1):
            title = meta.get("title", "Unknown")
            year = meta.get("year", "n.d.")
            venue = meta.get("venue", "") or meta.get("source", "")

            header = f"[{idx}] {title} ({year}) — {venue}"
            snippet = (doc or "").strip()
            keep = max_chars - total
            if keep <= 0:
                break
            snippet = snippet[:keep]
            total += len(snippet)

            context_parts.append(f"{header}\n{snippet}\n")

            authors_meta = meta.get("authors")
            if isinstance(authors_meta, list):
                authors = ", ".join(authors_meta)
            else:
                authors = str(authors_meta or "")
            doi = meta.get("doi")
            url = meta.get("url") or (f"https://doi.org/{doi}" if doi else "")

            bibliography.append(
                {
                    "index": idx,
                    "title": title,
                    "authors": authors,
                    "venue": venue,
                    "year": year,
                    "doi": doi,
                    "url": url,
                }
            )
            print(f"   [{idx}] {title[:50]}... (distance: {dist if dists else 'n/a'})")

        context = "\n\n".join(context_parts)
        print(
            f"   Built context with {len(bibliography)} citations "
            f"({len(context)} chars)"
        )

        return json.dumps(
            {
                "question": question,
                "context": context,
                "citations_count": len(bibliography),
                "bibliography": bibliography,
                "instruction": (
                    "Use the context above to answer the question. "
                    "Cite papers with [1], [2], etc. End with a "
                    "Bibliography section."
                ),
            },
            ensure_ascii=False,
        )

    def list_papers(self) -> str:
        """List all fetched papers with indices."""
        print("\n[List] Listing all papers...")

        if not self.ctx.papers:
            print("   No papers in context")
            return json.dumps(
                {"papers": [], "count": 0},
                ensure_ascii=False,
            )

        papers_info: list[dict[str, Any]] = []
        for i, p in enumerate(self.ctx.papers):
            papers_info.append(
                {
                    "index": i,
                    "title": p.title,
                    "authors": p.authors[:3],
                    "year": p.year,
                    "source": p.source,
                    "has_pdf": bool(p.pdf_url),
                }
            )
            print(
                f"   [{i}] {'PDF' if p.pdf_url else 'ABS'} "
                f"{p.title[:80]} ({p.year}) [{p.source}]"
            )

        print(f"   Total: {len(papers_info)} papers")
        print(f"   Summary: {self.ctx.get_summary()}")

        return json.dumps(
            {
                "count": len(papers_info),
                "papers": papers_info,
                "summary": self.ctx.get_summary(),
            },
            ensure_ascii=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# AcademicAgent with separated phases (no tool-calling LLM)
# ──────────────────────────────────────────────────────────────────────────────


class AcademicAgent:
    """
    Academic literature agent with explicit separated phases:

      PHASE 1: search(...)          → multi-source search (no LLM)
      PHASE 2: ingest_all(...),     → ingestion into vector DB (no LLM)
               ingest_selection(...)
      PHASE 3: ask(...)             → build context via RAG and query LLM

    The LLM no longer calls search/ingest tools. You call these phases
    programmatically from Python, then only send a single prompt with
    context + question to the LLM in ask().
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        chat_template: str = "chatml",
        use_chat_api: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        rag_collection: str = "papers",
    ):
        print(f"\nInitializing AcademicAgent...")
        print(f"   LLM: {llm_url}")
        print(f"   Chat template: {chat_template}")
        print(f"   Embedding model: {embedding_model}")
        print(f"   RAG collection: {rag_collection}")

        self.ctx = AcademicContext()

        print("   Setting up DocumentDB...")
        self.doc_db = DocumentDB(
            chroma_path="rag_docs.chroma",
            embedding_model_name=embedding_model,
            collection_name=rag_collection,
        )
        print("   DocumentDB ready")

        print("   Creating LLM client (no tools)...")
        self.llm = Agent(
            llm_url=llm_url,
            system_prompt=self._get_system_prompt(),
            use_chat_api=use_chat_api,
            chat_template=chat_template,
        )
        print("   LLM client ready")

        # Internal helpers for phases (used programmatically)
        self._search_tools = SearchTools(self.ctx)
        self._ingestion_tools = IngestionTools(self.ctx, self.doc_db)
        self._query_tools = QueryTools(self.ctx, self.doc_db)

        print("\nAcademicAgent ready!\n")

    # ─── Phase 1: SEARCH ───
    def search(
        self,
        query: str,
        per_source_limit: int = 10,
        order: str = "s2,arxiv,crossref",
        open_access_only: bool = False,
        multi: bool = True,
    ):
        """
        Run multi-source search programmatically (no LLM).

        - If multi=True, use auto_multi_search (S2, arXiv, Crossref).
        - If multi=False, you can call the specific methods yourself later
          (e.g., ._search_tools.semantic_scholar_search, etc.).
        """
        print("=" * 80)
        print(f"[PHASE 1] SEARCH: query={query!r}")
        print("=" * 80)

        if multi:
            payload = self._search_tools.auto_multi_search(
                query=query,
                per_source_limit=str(per_source_limit),
                order=order,
                open_access_only=str(open_access_only).lower(),
            )
            print("[PHASE 1] Multi-source search summary:")
            print(payload)
        else:
            # Manual: only S2 by default
            self._search_tools.semantic_scholar_search(
                query,
                limit=str(per_source_limit),
                open_access_only=str(open_access_only).lower(),
            )

        print(f"[PHASE 1] Context summary: {self.ctx.get_summary()}")
        return self.ctx.papers

    # ─── Phase 2: INGEST ───
    def ingest_all(
        self,
        use_pdf: bool = True,
        chunk_size: int = 1200,
        overlap: int = 150,
        max_workers: int = 4,
    ):
        """
        Ingest all papers currently in context into the vector store.
        No LLM involved.
        """
        print("=" * 80)
        print("[PHASE 2] INGEST ALL")
        print("=" * 80)
        result = self._ingestion_tools.ingest_all(
            use_pdf=str(use_pdf).lower(),
            chunk_size=str(chunk_size),
            overlap=str(overlap),
            max_workers=str(max_workers),
        )
        print(result)
        return json.loads(result)

    def ingest_selection(
        self,
        indices: list[int],
        use_pdf: bool = True,
        chunk_size: int = 1200,
        overlap: int = 150,
        max_workers: int = 4,
    ):
        """
        Ingest only selected papers by indices.
        """
        print("=" * 80)
        print(f"[PHASE 2] INGEST SELECTION: indices={indices}")
        print("=" * 80)
        s = ",".join(str(i) for i in indices)
        result = self._ingestion_tools.ingest_papers(
            paper_indices=s,
            use_pdf=str(use_pdf).lower(),
            chunk_size=str(chunk_size),
            overlap=str(overlap),
            max_workers=str(max_workers),
        )
        print(result)
        return json.loads(result)

    # ─── Phase 3: ASK (RAG + LLM) ───
    def ask(
        self,
        question: str,
        top_k: int = 5,
        max_ctx_chars: int = 4000,
        extra_system_instruction: str | None = None,
        verbose: bool = False,
    ) -> str:
        """
        Build RAG context from ingested papers and query the LLM.

        The LLM receives:
          - Context (snippets from papers),
          - The question,
          - An instruction on how to answer and cite.

        The LLM DOES NOT call tools. This is a single chat() call.
        """
        print("=" * 80)
        print(f"[PHASE 3] ASK: {question!r}")
        print("=" * 80)

        # Build RAG context via QueryTools
        payload_json = self._query_tools.qa_papers(
            question=question,
            top_k=str(top_k),
            max_ctx_chars=str(max_ctx_chars),
        )
        payload = json.loads(payload_json)

        system_prefix = ""
        if extra_system_instruction:
            system_prefix = extra_system_instruction.strip() + "\n\n"

        prompt = (
            f"{system_prefix}"
            f"Context:\n{payload['context']}\n\n"
            f"Question: {payload['question']}\n\n"
            f"{payload['instruction']}"
        )

        if verbose:
            print("[PHASE 3] Prompt sent to LLM:")
            print("-" * 80)
            print(prompt[:4000])  # avoid flooding logs
            print("-" * 80)

        # Single chat call (no tools)
        answer = self.llm.chat(prompt, verbose=verbose)
        return answer

    # ─── Inspect / helpers ───
    def list_papers(self) -> list[dict[str, Any]]:
        """
        List current papers in context (for inspecting search results).
        """
        print("\n[AcademicAgent] list_papers()")
        out: list[dict[str, Any]] = []
        for i, p in enumerate(self.ctx.papers):
            out.append(
                {
                    "index": i,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "source": p.source,
                    "has_pdf": bool(p.pdf_url),
                }
            )
        print(f"   Total: {len(out)} papers")
        return out

    def get_papers(self) -> list[Paper]:
        return self.ctx.papers

    def get_summary(self) -> str:
        return self.ctx.get_summary()

    def reset(self):
        print("\nResetting AcademicAgent context (search/ingest state only)...")
        self.ctx = AcademicContext()
        self._search_tools = SearchTools(self.ctx)
        self._ingestion_tools = IngestionTools(self.ctx, self.doc_db)
        self._query_tools = QueryTools(self.ctx, self.doc_db)
        print("   Reset complete\n")
        return self

    # ─── Internal system prompt for LLM ───
    def _get_system_prompt(self) -> str:
        return (
            "You are an expert academic research assistant.\n"
            "You will receive:\n"
            "  - A 'Context' section (snippets from academic papers),\n"
            "  - A 'Question'.\n\n"
            "Use ONLY the information in the Context to answer.\n"
            "If something is not supported by the Context, explicitly say so.\n"
            "Cite papers using [1], [2], ... according to the bracket numbers\n"
            "in the Context headers, and end with a 'Bibliography' section listing\n"
            "the cited references with their titles and venues.\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory Function
# ──────────────────────────────────────────────────────────────────────────────


def create_academic_agent(
    llm_url: str = "http://localhost:8080",
    chat_template: str = "chatml",
    use_chat_api: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    rag_collection: str = "papers",
) -> AcademicAgent:
    """Create an AcademicAgent with default settings."""
    return AcademicAgent(
        llm_url=llm_url,
        chat_template=chat_template,
        use_chat_api=use_chat_api,
        embedding_model=embedding_model,
        rag_collection=rag_collection,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Example Usage (manual phases, no tool-calling LLM)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = create_academic_agent()

    # PHASE 1 — SEARCH (no LLM)
    agent.search(
        "What are the latest transformer architectures for NLP?",
        per_source_limit=8,
        order="s2,arxiv,crossref",
        open_access_only=False,
    )

    print("\nPAPERS AFTER SEARCH:")
    for p in agent.list_papers()[:5]:
        print(f"  [{p['index']}] {p['title']} ({p['year']}) [{p['source']}]")

    # PHASE 2 — INGEST (no LLM)
    agent.ingest_all(use_pdf=True, chunk_size=512, overlap=128, max_workers=4)

    # PHASE 3 — ASK (RAG + LLM, single prompt)
    answer = agent.ask(
        "What are the latest transformer architectures for NLP?",
        top_k=5,
        max_ctx_chars=4000,
    )
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
