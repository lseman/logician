"""Academic provider helpers — shared utilities and core types.

Provider scripts import these directly. Not all symbols in common.py
are re-exported here (see comments in common.py for details).
"""

from __future__ import annotations

# Re-export core types and utilities that provider scripts use.
from .common import (
    BaseHTTPSource,
    Paper,
    _coalesce,
    _norm_arxiv_id,
    _norm_doi,
    _norm_space,
    _safe_bool,
    _safe_int,
    _s2_fallback_queries,
    _simplify_query_for_s2,
    _year_from_any,
)

# Not exported here:
#   Source     — Protocol redefined in systematic.py; unused by provider scripts
#   _title_key — only used by Paper.ref_id() (internal) and systematic.py
#   _strip_markup — only used by systematic.py (DBLPSource, HuggingFacePapersSource)

__all__ = [
    "BaseHTTPSource",
    "Paper",
    "_coalesce",
    "_norm_arxiv_id",
    "_norm_doi",
    "_norm_space",
    "_safe_bool",
    "_safe_int",
    "_s2_fallback_queries",
    "_simplify_query_for_s2",
    "_year_from_any",
]
