"""Unpaywall academic provider helper.

This module contains the Unpaywall source implementation extracted from the
academic systematic review script.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_common = importlib.import_module("scripts.common")
BaseHTTPSource = _common.BaseHTTPSource
_norm_doi = _common._norm_doi

__tools__ = ["unpaywall_lookup"]

__skill__ = {
    "name": "Unpaywall",
    "description": "Provider-specific Unpaywall academic resolution helper.",
}


def unpaywall_lookup(doi: str, email: str | None = None) -> dict[str, Any]:
    """Lookup Open Access metadata for a DOI via Unpaywall.

    Returns a dict with 'status', 'source', 'doi', and 'result' keys.
    'result' is None when the DOI is unresolvable or no email is configured.

    Requirements:
      - A valid DOI
      - An email via the 'email' parameter or UNPAYWALL_EMAIL / UNPAYWALL_MAILTO env var
    """
    source = UnpaywallEnricher(email=email)
    if not source._email:
        return {
            "status": "error",
            "source": "unpaywall",
            "doi": doi,
            "error": "No email configured for Unpaywall. Set email= or UNPAYWALL_EMAIL env var.",
        }
    result = source.lookup(doi=doi)
    if result is None:
        return {"status": "ok", "source": "unpaywall", "doi": doi, "result": None}
    return {"status": "ok", "source": "unpaywall", "doi": doi, "result": result}


class UnpaywallEnricher(BaseHTTPSource):
    name = "unpaywall"

    def __init__(self, *, timeout: float = 25.0, email: Optional[str] = None):
        super().__init__(timeout=timeout)
        self._email = email or os.getenv("UNPAYWALL_EMAIL") or os.getenv("UNPAYWALL_MAILTO")
        if not self._email:
            print(
                "[WARN] Unpaywall requires an email for fair use. "
                "Set email= or the UNPAYWALL_EMAIL environment variable."
            )

    def lookup(self, doi: str) -> Optional[dict[str, Any]]:
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
