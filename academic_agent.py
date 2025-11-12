# academic_agent.py
# -*- coding: utf-8 -*-
"""
AcademicAgent — Literature search + RAG over papers (with debug output)
"""
from __future__ import annotations
import io
import json
import re
import time
import uuid
import inspect
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable

import httpx
import numpy as np

# API SDKs
import arxiv
from semanticscholar import SemanticScholar  # noqa: F401  (kept for future use)
from habanero import Crossref

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# Framework imports
from core import Agent, ToolParameter, DocumentDB


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def _year_from_date(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


# NLTK sentence tokenizer (best-effort)
import nltk
nltk.download('punkt', quiet=True)

def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        sentences = [text]
    chunks, current = [], ''
    for sent in sentences:
        if len(current + ' ' + sent) <= chunk_size:
            current = (current + ' ' + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
            if len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size:]
    if current:
        chunks.append(current)
    if overlap > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prepend = chunks[i-1][-overlap:]
            chunks[i] = prepend + chunks[i]
    return chunks

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
def _http_get_bytes(client: httpx.Client, url: str, timeout: float, headers: Optional[Dict[str, str]] = None) -> bytes:
    """Fetch URL with retries."""
    backoff = 0.6
    for attempt in range(3):
        try:
            r = client.get(url, timeout=timeout, follow_redirects=True, headers=headers)
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


def _http_head(client: httpx.Client, url: str, timeout: float, headers: Optional[Dict[str, str]] = None) -> Optional[httpx.Response]:
    """HEAD helper with graceful fallback (some servers block HEAD)."""
    try:
        r = client.head(url, timeout=timeout, follow_redirects=True, headers=headers)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _http_get_response(client: httpx.Client, url: str, timeout: float, headers: Optional[Dict[str, str]] = None) -> Optional[httpx.Response]:
    """GET helper that returns the Response object (for header inspection)."""
    try:
        r = client.get(url, timeout=timeout, follow_redirects=True, headers=headers)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _as_dictlike(x) -> Dict[str, Any]:
    """Return a dict-like view for dicts OR objects (attribute access)."""
    if isinstance(x, dict):
        return x

    class _Proxy(dict):
        def get(self, k, default=None):
            if isinstance(x, dict):
                return x.get(k, default)
            return getattr(x, k, default)

        def __contains__(self, k):
            if isinstance(x, dict):
                return k in x
            return hasattr(x, k)

        def __getitem__(self, k):
            if isinstance(x, dict):
                return x[k]
            val = getattr(x, k)
            if val is None:
                raise KeyError(k)
            return val

    return _Proxy()


def _ensure_list_items(raw):
    """Semantic Scholar may return a list or {'data': [...]}."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    return []


def _parse_crossref_filter(s: str) -> Dict[str, str]:
    """
    Accepts:
      - 'from-pub-date:2022-01-01,until-pub-date:2025-12-31,type:journal-article'
      - or a single raw DOI like '10.1007/s10489-023-04789-3'
    """
    filt: Dict[str, str] = {}
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


def _parse_indices_arg(s: str, upper_bound: int) -> List[int]:
    """
    Parse index selections into a list of ints (supports JSON, lists, and ranges).
    Values are clipped and deduped.
    """
    s = (s or "").strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        cand = parsed if isinstance(parsed, list) else []
    except Exception:
        cand = []
    if not cand:
        parts = re.split(r"[,\s]+", s)
        cand = []
        for p in parts:
            if not p:
                continue
            m = re.match(r"^(\d+)\s*(?:\-|\.\.)\s*(\d+)$", p)
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
# arXiv & Crossref PDF helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    s = str(arxiv_id).strip()
    s = s.replace("arXiv:", "").replace("ARXIV:", "").strip()
    return s

def _arxiv_pdf_url(entry_id: Optional[str], arxiv_id: Optional[str]) -> Optional[str]:
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

def _crossref_pick_pdf_link(item: Dict[str, Any]) -> Optional[str]:
    """
    Prefer official Crossref 'link' entries with content-type application/pdf.
    Heuristics:
      1) link[].content-type == 'application/pdf'
         - Prefer version=='vor' (version of record) if available
      2) any link URL ending with .pdf
      3) None (let DOI resolution try later)
    """
    links = item.get("link") or []
    if not isinstance(links, list):
        return None

    def _is_pdf_link(l: Dict[str, Any]) -> bool:
        ctype = (l or {}).get("content-type", "") or ""
        url = (l or {}).get("URL", "") or ""
        return ("application/pdf" in ctype.lower()) or url.lower().endswith(".pdf")

    pdf_links = [l for l in links if _is_pdf_link(l)]
    if not pdf_links:
        return None

    # Prefer VOR (version of record) when present
    vor = [l for l in pdf_links if (l or {}).get("version", "").lower() == "vor"]
    pick = (vor[0] if vor else pdf_links[0])
    return (pick or {}).get("URL") or None

def _resolve_doi_to_pdf_url(client: httpx.Client, doi: str, timeout: float = 30.0) -> Optional[str]:
    """
    Try to resolve DOI to a PDF URL by requesting with Accept: application/pdf.
    Strategy:
      - HEAD https://doi.org/<doi> with Accept: application/pdf (follow redirects)
      - If HEAD blocked or inconclusive, GET with same header and check content-type
      - If final response has 'application/pdf' in Content-Type, return final URL
    """
    if not doi:
        return None
    base = f"https://doi.org/{doi}"
    accept_pdf = {"Accept": "application/pdf", "User-Agent": "AcademicAgent/1.0"}

    # HEAD first
    r_head = _http_head(client, base, timeout=timeout, headers=accept_pdf)
    if r_head is not None:
        ctype = (r_head.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ctype:
            return str(r_head.url)

    # Fallback to GET (may download content; acceptable since we'll parse anyway)
    r_get = _http_get_response(client, base, timeout=timeout, headers=accept_pdf)
    if r_get is not None:
        ctype = (r_get.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ctype:
            return str(r_get.url)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Strong metadata sanitizers
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


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}

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
# Data Models
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Paper:
    """Normalized paper metadata across all sources."""
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    source: str  # "arxiv" | "s2" | "crossref"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
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
            **({"pdf_via_doi": self.extra.get("pdf_via_doi")} if "pdf_via_doi" in self.extra else {}),
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
    papers: List[Paper] = field(default_factory=list)
    ingested_ids: List[str] = field(default_factory=list)
    last_query: Optional[str] = None
    last_search_source: Optional[str] = None
    total_chunks: int = 0

    def add_papers(self, new_papers: List[Paper]) -> int:
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

# ──────────────────────────────────────────────────────────────────────────────
# Tool Classes (with debug output)
# ──────────────────────────────────────────────────────────────────────────────
class SearchTools:
    """Paper search across multiple sources."""

    __tools__ = ["arxiv_search", "crossref_search"]

    def __init__(self, ctx: AcademicContext):
        self.ctx = ctx

    def arxiv_search(
        self,
        query: str,
        max_results: str = "10",
        sort_by: str = arxiv.SortCriterion.SubmittedDate,
        start: str = "0"
    ) -> str:
        """Search arXiv for papers. Returns normalized Paper list as JSON."""
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

        papers = []
        for i, result in enumerate(results[:max_r], 1):
            year = getattr(getattr(result, "published", None), "year", None) or _year_from_date(str(getattr(result, "published", "")))
            entry_id = getattr(result, "entry_id", None) or ""
            arxiv_id_from_entry = entry_id.split("/")[-1] if entry_id else None
            arxiv_id = _normalize_arxiv_id(arxiv_id_from_entry or getattr(result, "get_short_id", lambda: None)() or getattr(result, "arxiv_id", None))
            auth_objs = getattr(result, "authors", []) or []
            authors = [getattr(a, "name", None) or str(a) for a in auth_objs if (getattr(a, "name", None) or str(a))]

            raw_pdf_url = getattr(result, "pdf_url", None)
            pdf_url = raw_pdf_url or _arxiv_pdf_url(entry_id, arxiv_id)

            papers.append(Paper(
                title=getattr(result, "title", "") or "",
                authors=authors,
                year=year,
                venue="arXiv",
                abstract=getattr(result, "summary", None),
                doi=None,
                arxiv_id=arxiv_id,
                url=entry_id or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
                pdf_url=pdf_url,
                source="arxiv",
            ))
            print(f"   [{i}] {papers[-1].title[:80]} ({year}) {'PDF' if pdf_url else 'no-PDF'}")

        added = self.ctx.add_papers(papers)
        self.ctx.last_query = query
        self.ctx.last_search_source = "arxiv"
        print(f"   Added {added} new papers to context (total: {len(self.ctx.papers)})")

        return json.dumps({
            "found": len(papers),
            "added_new": added,
            "papers": [asdict(p) for p in papers]
        }, ensure_ascii=False)

    def crossref_search(
        self,
        query: str,
        rows: str = "10",
        filter: str = "",
        sort: str = "",
        order: str = "desc"
    ) -> str:
        """Search Crossref for papers. Returns normalized Paper list as JSON."""
        print(f"\n[Crossref] Searching for: '{query}'")
        print(f"   Parameters: rows={rows}, filter='{filter}', sort='{sort}', order='{order}'")

        cr = Crossref()
        r = max(1, min(int(rows), 20))

        params: Dict[str, Any] = {"query": query, "rows": r}
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

        papers = []
        for i, item in enumerate(works, 1):
            title = (item.get("title") or [""])[0]
            names = []
            for a in (item.get("author") or []):
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

            # NEW: Try to populate PDF from Crossref itself
            pdf_url = _crossref_pick_pdf_link(item)

            papers.append(Paper(
                title=title,
                authors=names,
                year=year,
                venue=venue,
                abstract=abstract,
                doi=doi,
                arxiv_id=None,
                url=url,
                pdf_url=pdf_url,  # may be None; ingestion can still resolve via DOI later
                source="crossref",
                extra={"type": item.get("type")}
            ))
            print(f"   [{i}] {title[:80]} ({year}) {'PDF' if pdf_url else 'no-PDF'}")

        added = self.ctx.add_papers(papers)
        self.ctx.last_query = query
        self.ctx.last_search_source = "crossref"
        print(f"   Added {added} new papers to context (total: {len(self.ctx.papers)})")

        return json.dumps({
            "found": len(papers),
            "added_new": added,
            "papers": [asdict(p) for p in papers]
        }, ensure_ascii=False)


class IngestionTools:
    """Paper ingestion into RAG DocumentDB."""

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
    ) -> str:
        """Ingest specific papers by index into RAG. Use 'all' to ingest all fetched papers."""
        print(f"\n[Ingestion] Starting paper ingestion...")
        print(f"   Parameters: paper_indices={paper_indices}, use_pdf={use_pdf}, chunk_size={chunk_size}, overlap={overlap}")

        if isinstance(paper_indices, str) and paper_indices.strip().lower() == "all":
            papers = self.ctx.papers
            print(f"   Ingesting all {len(papers)} papers")
        else:
            try:
                idxs = _parse_indices_arg(str(paper_indices), upper_bound=len(self.ctx.papers))
                papers = [self.ctx.papers[i] for i in idxs if 0 <= i < len(self.ctx.papers)]
                print(f"   Ingesting {len(papers)} selected papers (indices: {idxs})")
            except Exception as e:
                print(f"   Error parsing indices: {e}")
                return json.dumps({"error": "Invalid paper_indices format"}, ensure_ascii=False)

        if not papers:
            print(f"   No papers to ingest")
            return json.dumps({"error": "No papers to ingest"}, ensure_ascii=False)

        use_pdf_bool = str(use_pdf).lower() in ("true", "1", "yes")
        cs = int(chunk_size)
        ov = int(overlap)

        added_ids = []
        total_chunks = 0

        print(f"   Processing {len(papers)} papers...")
        with httpx.Client(headers={"User-Agent": "AcademicAgent/1.0"}, timeout=30.0, follow_redirects=True) as client:
            for idx, paper in enumerate(papers, 1):
                print(f"\n   [{idx}/{len(papers)}] Processing: {paper.title[:120]}")

                # Ensure arXiv PDF if missing
                if (not paper.pdf_url) and (paper.source == "arxiv"):
                    paper.pdf_url = _arxiv_pdf_url(paper.url, paper.arxiv_id)

                # NEW: If Crossref item lacks PDF but has DOI, try DOI→PDF resolution
                if (not paper.pdf_url) and paper.source == "crossref" and paper.doi:
                    print("       • Resolving DOI to PDF (Accept: application/pdf)...")
                    try:
                        resolved = _resolve_doi_to_pdf_url(client, paper.doi, timeout=30.0)
                        if resolved:
                            paper.pdf_url = resolved
                            paper.extra["pdf_via_doi"] = True
                            print(f"       • DOI resolved to PDF URL: {resolved}")
                        else:
                            print("       • DOI resolution did not yield a PDF URL")
                    except Exception as e:
                        print(f"       • DOI resolution error: {e}")

                # Build text from metadata + abstract
                text = self._build_paper_text(paper)
                print(f"       • Built base text ({len(text)} chars)")

                # Optionally fetch PDF
                if use_pdf_bool and paper.pdf_url:
                    print(f"       • Fetching PDF from {paper.pdf_url[:120]} ...")
                    try:
                        pdf_start = time.time()
                        # If PDF came from DOI resolution, we already negotiated Accept header.
                        # For direct fetching now, plain GET is fine.
                        pdf_bytes = _http_get_bytes(client, paper.pdf_url, timeout=30.0)
                        pdf_dur = time.time() - pdf_start

                        if pdf_bytes:
                            print(f"       • Downloaded {len(pdf_bytes)} bytes in {pdf_dur:.1f}s")
                            print(f"       • Extracting text from PDF...")
                            pdf_text = _read_pdf_bytes(pdf_bytes)
                            if pdf_text:
                                text += f"\n\nFullText:\n{pdf_text}"
                                print(f"       • Extracted {len(pdf_text)} chars from PDF")
                            else:
                                print(f"       • Could not extract text (using abstract only)")
                        else:
                            print(f"       • PDF download failed (using abstract only)")
                    except Exception as e:
                        print(f"       • PDF error: {e} (using abstract only)")
                elif paper.pdf_url:
                    print(f"       • Skipping PDF (use_pdf=false)")
                else:
                    print(f"       • No PDF URL available")

                # Chunk and add to vector store
                print(f"       • Chunking text (size={cs}, overlap={ov})")
                chunks = _chunk_text(text, chunk_size=cs, overlap=ov)
                if not chunks:
                    print(f"       • No chunks generated, skipping")
                    continue

                print(f"       • Generated {len(chunks)} chunks")
                ref_id = paper.ref_id()
                metas = []
                for i in range(len(chunks)):
                    raw_meta = {**paper.to_metadata(), "chunk": int(i), "ref_id": str(ref_id)}
                    metas.append(_sanitize_metadata(raw_meta))

                ids = [f"{ref_id}#c{i}" for i in range(len(chunks))]

                print(f"       • Adding to vector store...")
                self.doc_db.collection.add(documents=chunks, metadatas=metas, ids=ids)
                added_ids.extend(ids)
                total_chunks += len(chunks)
                print(f"       • Added {len(chunks)} chunks to vector store")

        self.ctx.ingested_ids.extend(added_ids)
        self.ctx.total_chunks += total_chunks

        print(f"\n   Ingestion complete!")
        print(f"   • Papers processed: {len(papers)}")
        print(f"   • Chunks added: {total_chunks}")
        print(f"   • Total ingested: {len(self.ctx.ingested_ids)} chunks")

        return json.dumps({
            "ingested_papers": len(papers),
            "added_chunks": total_chunks,
            "total_ingested": len(self.ctx.ingested_ids),
        }, ensure_ascii=False)

    def ingest_all(self, use_pdf: str = "true", chunk_size: str = "1200", overlap: str = "150") -> str:
        """Convenience: ingest all fetched papers."""
        return self.ingest_papers("all", use_pdf, chunk_size, overlap)

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
            parts.append(f"PDF-Note: resolved via DOI Accept: application/pdf")
        parts.append(f"Source: {paper.source}")

        text = "\n".join(parts) + "\n\n"
        if paper.abstract:
            text += f"Abstract: {paper.abstract}\n"
        return text


class QueryTools:
    """RAG-based Q&A over ingested papers."""

    __tools__ = ["qa_papers", "list_papers"]

    def __init__(self, ctx: AcademicContext, doc_db: DocumentDB):
        self.ctx = ctx
        self.doc_db = doc_db

    def qa_papers(self, question: str, top_k: str = "5", max_ctx_chars: str = "4000") -> str:
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

        print(f"   Deduplicating by paper (ref_id)...")
        seen = set()
        ranked: List[Tuple[float, str, Dict[str, Any]]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            ref_id = meta.get("ref_id") or str(uuid.uuid4())
            if ref_id in seen:
                continue
            seen.add(ref_id)
            ranked.append(((float(dist) if dist is not None else 0.0), doc, meta))

        ranked.sort(key=lambda x: x[0])
        print(f"   Unique papers after dedupe: {len(ranked)}")

        print(f"   Building context (max {max_chars} chars)...")
        context_parts = []
        bibliography = []
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

            bibliography.append({
                "index": idx,
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "doi": doi,
                "url": url,
            })
            print(f"   [{idx}] {title[:50]}... (distance: {dist if dists else 'n/a'})")

        context = "\n\n".join(context_parts)
        print(f"   Built context with {len(bibliography)} citations ({len(context)} chars)")

        return json.dumps({
            "question": question,
            "context": context,
            "citations_count": len(bibliography),
            "bibliography": bibliography,
            "instruction": "Use the context above to answer the question. Cite papers with [1], [2], etc. End with a Bibliography section."
        }, ensure_ascii=False)

    def list_papers(self) -> str:
        """List all fetched papers with indices."""
        print(f"\n[List] Listing all papers...")

        if not self.ctx.papers:
            print(f"   No papers in context")
            return json.dumps({"papers": [], "count": 0}, ensure_ascii=False)

        papers_info = []
        for i, p in enumerate(self.ctx.papers):
            papers_info.append({
                "index": i,
                "title": p.title,
                "authors": p.authors[:3],
                "year": p.year,
                "source": p.source,
                "has_pdf": bool(p.pdf_url),
            })
            print(f"   [{i}] {'PDF' if p.pdf_url else 'ABS'} {p.title[:80]} ({p.year}) [{p.source}]")

        print(f"   Total: {len(papers_info)} papers")
        print(f"   Summary: {self.ctx.get_summary()}")

        return json.dumps({
            "count": len(papers_info),
            "papers": papers_info,
            "summary": self.ctx.get_summary()
        }, ensure_ascii=False)


class OrchestrationTools:
    """
    Optional "one-shot" pipeline to reduce error surface:
      search → ingest → qa, in a single tool call.
    """

    __tools__ = ["search_ingest_qa"]

    def __init__(self, ctx: AcademicContext, doc_db: DocumentDB):
        self.ctx = ctx
        self.doc_db = doc_db
        self.search = SearchTools(ctx)
        self.ingest = IngestionTools(ctx, doc_db)
        self.query = QueryTools(ctx, doc_db)

    def search_ingest_qa(
        self,
        user_question: str,
        search_query: str = "",
        source_order: str = "s2,arxiv,crossref",
        per_source_limit: str = "10",
        ingest_use_pdf: str = "true",
        ingest_chunk_size: str = "1200",
        ingest_overlap: str = "150",
        qa_top_k: str = "5",
        qa_max_ctx_chars: str = "4000",
    ) -> str:
        """
        Run a full pipeline:
          1) Search multiple sources in order (subset via source_order).
          2) Ingest all fetched papers (with or without PDFs).
          3) Run QA and return the QA payload.
        """
        print("\n[Orchestrator] Starting search → ingest → qa pipeline")
        q = search_query or user_question
        lim = str(max(1, min(int(per_source_limit), 20)))
        order = [s.strip().lower() for s in (source_order or "").split(",") if s.strip()]
        order = [s for s in order if s in ("s2", "arxiv", "crossref")]
        if not order:
            order = ["s2", "arxiv", "crossref"]

        for src in order:
            try:
                if src == "arxiv":
                    print(f"[Orchestrator] Searching arXiv (max_results={lim})")
                    self.search.arxiv_search(q, lim, arxiv.SortCriterion.SubmittedDate, "0")
                elif src == "crossref":
                    print(f"[Orchestrator] Searching Crossref (rows={lim})")
                    self.search.crossref_search(q, lim, "", "relevance", "desc")
            except Exception as e:
                print(f"[Orchestrator] Search error on {src}: {e}")

        try:
            print("[Orchestrator] Ingesting all fetched papers...")
            self.ingest.ingest_all(ingest_use_pdf, ingest_chunk_size, ingest_overlap)
        except Exception as e:
            print(f"[Orchestrator] Ingestion error: {e}")

        try:
            print("[Orchestrator] Running QA...")
            return self.query.qa_papers(user_question, qa_top_k, qa_max_ctx_chars)
        except Exception as e:
            print(f"[Orchestrator] QA error: {e}")
            return json.dumps({
                "question": user_question,
                "context": "",
                "citations_count": 0,
                "bibliography": [],
                "instruction": "No context available due to an error."
            }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# AcademicAgent (TimeSeriesAgent pattern)
# ──────────────────────────────────────────────────────────────────────────────
class AcademicAgent:
    """Academic literature agent with clean .run() / .chat() interface."""

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

        print(f"   Setting up DocumentDB...")
        self.doc_db = DocumentDB(
            chroma_path="rag_docs.chroma",
            embedding_model_name=embedding_model,
            collection_name=rag_collection,
        )
        print(f"   DocumentDB ready")

        print(f"   Creating base Agent...")
        self.agent = Agent(
            llm_url=llm_url,
            system_prompt=self._get_system_prompt(),
            use_chat_api=use_chat_api,
            chat_template=chat_template,
        )
        print(f"   Agent ready")

        self._seen_functions = set()
        self._registered_tools: Dict[str, Any] = {}

        print(f"   Mounting tools...")
        self._mount_tools_auto()
        print(f"   Registered {len(self._registered_tools)} tools: {list(self._registered_tools.keys())}")
        print(f"\nAcademicAgent ready!\n")

    # ——— Public API ———
    def chat(self, message: str, verbose: bool = False) -> str:
        enhanced = self._enhance_message(message)
        return self.agent.chat(enhanced, verbose=verbose)

    def run(self, message: str, verbose: bool = False):
        print(f"\n{'='*80}")
        print(f"Running agent with message: '{message}'")
        print(f"{'='*80}\n")
        enhanced = self._enhance_message(message)
        result = self.agent.run(enhanced, verbose=verbose)
        print(f"\n{'='*80}")
        print(f"Agent execution complete!")
        print(f"   Iterations: {result.iterations}")
        print(f"   Tool calls: {len(result.tool_calls)}")
        print(f"{'='*80}\n")
        return result

    def reset(self):
        print(f"\nResetting agent...")
        self.ctx = AcademicContext()
        self.agent.reset()
        self._seen_functions = set()
        self._registered_tools = {}
        print(f"   Reset complete\n")
        return self

    def get_papers(self) -> List[Paper]:
        return self.ctx.papers

    def get_summary(self) -> str:
        return self.ctx.get_summary()

    # ——— Internals ———
    def _enhance_message(self, message: str) -> str:
        if self.ctx.papers and "papers" not in message.lower():
            summary = self.ctx.get_summary()
            return f"Context: {summary}\n\nUser: {message}"
        return message

    def _get_system_prompt(self) -> str:
        return """You are an expert academic research assistant with literature search and RAG capabilities.

Your workflow for research questions:
1. Search: Use arxiv_search or crossref_search to find relevant papers
   - Aim for 5-15 papers from recent years
   - Use multiple sources for comprehensive coverage

2. Ingest: Call ingest_all or ingest_papers to add papers to the RAG system
   - use_pdf=true for full-text when available
   - This enables citation-backed answers

3. Answer: Use qa_papers with the user's question
   - Returns context and bibliography
   - Cite sources as [1], [2], etc.
   - Always include a Bibliography section at the end

4. Best Practices:
   - Only cite papers that were actually retrieved
   - Be precise and concise
   - Use tables for comparisons when appropriate
   - If no relevant papers found, acknowledge and suggest broader search

For non-research questions, respond naturally without tools.

CRITICAL: Return tool calls as exact JSON on a single line, no extra text:
{"tool_call":{"name":"tool_name","arguments":{"param":"value"}}}

Examples:
- Select specific papers by index:
{"tool_call":{"name":"ingest_papers","arguments":{"paper_indices":"[0,1,2,3]","use_pdf":"true","chunk_size":"512","overlap":"128"}}}
- Ingest all fetched:
{"tool_call":{"name":"ingest_all","arguments":{"use_pdf":"true","chunk_size":"512","overlap":"128"}}}
- One-shot pipeline:
{"tool_call":{"name":"search_ingest_qa","arguments":{"user_question":"<your question here>","per_source_limit":"10","ingest_use_pdf":"true"}}}
"""

    def _mount_tools_auto(self):
        try:
            from helpers import _iter_exposed_methods as _iter_exposed_methods_ext  # type: ignore
            iterator = _iter_exposed_methods_ext
        except Exception:
            raise Warning("Falling back to basic tool discovery")

        instances = [
            SearchTools(self.ctx),
            IngestionTools(self.ctx, self.doc_db),
            QueryTools(self.ctx, self.doc_db),
            OrchestrationTools(self.ctx, self.doc_db),
        ]

        seen_fn_ids = set()
        collected = []
        for inst in instances:
            for tool_name, desc, method, params in iterator(inst):
                fn = getattr(method, "__func__", method)
                fid = id(fn)
                if fid in seen_fn_ids:
                    continue
                seen_fn_ids.add(fid)
                collected.append((inst, tool_name, desc, method, params))

        for inst, tool_name, desc, method, params in collected:
            self._register_tool(inst, tool_name, desc, method, params)

    def _register_tool(self, inst, tool_name: str, desc: str, method, params):
        fn = getattr(method, "__func__", method)
        fid = id(fn)

        if fid in self._seen_functions:
            return

        final_name = tool_name
        if final_name in self._registered_tools and self._registered_tools[final_name] is not method:
            final_name = f"{inst.__class__.__name__}.{tool_name}"

        if final_name in self._registered_tools:
            return

        self.agent.add_tool(final_name, desc, method, params)
        self._registered_tools[final_name] = method
        self._seen_functions.add(fid)


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
# Example Usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = create_academic_agent()
    response = agent.run("What are the latest transformer architectures for NLP?", verbose=True)

    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print("="*80)
    print(response.final_response)
    print("\n" + "="*80)
    print(f"Trace:\n{response.trace_md}")
