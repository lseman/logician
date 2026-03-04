from __future__ import annotations

# =============================================================================
# GLOBAL SKILLS BOOTSTRAP
# Loaded first (lexicographic sort ensures 00_global/ < 00_timeseries/ < 01_coding/ …)
# Defines shared utilities available to every skill module:
#   _json_default, _safe_json, _try_parse_json
# =============================================================================
import json
from datetime import datetime
from typing import Any, Dict

# Optional heavy deps — tolerate absence so this file loads even in lean envs
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    """Extended JSON default that handles numpy, pandas, and datetime types."""
    if _HAS_NUMPY and np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)

    if _HAS_PANDAS and pd is not None:
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, pd.Timedelta):
            return str(o)
        if isinstance(o, pd.Index):
            return o.astype(str).tolist()
        if isinstance(o, pd.Series):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")

    if isinstance(o, datetime):
        return o.isoformat()

    if hasattr(o, "to_dict"):
        try:
            return o.to_dict()
        except Exception:
            pass

    return str(o)


def _safe_json(d: Any) -> str:
    """Serialise *d* to a JSON string, never raising.

    Uses :func:`_json_default` to handle numpy/pandas/datetime types.
    Falls back to a ``{"status": "error", ...}`` envelope on failure.
    """
    try:
        return json.dumps(d, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as exc:
        try:
            return json.dumps(
                {"status": "error", "error": f"JSON serialisation failed: {exc}"},
                indent=2,
                ensure_ascii=False,
            )
        except Exception:
            return (
                '{"status": "error", "error": "JSON serialisation failed completely"}'
            )


def _try_parse_json(obj: Any) -> Dict[str, Any]:
    """Try to parse *obj* as JSON, returning a normalised dict.

    Handles: dict passthrough, JSON string, partial JSON extraction.
    Always returns a dict; never raises.
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {"status": "error", "error": "empty response"}
        try:
            parsed = json.loads(s)
            return (
                parsed if isinstance(parsed, dict) else {"status": "ok", "data": parsed}
            )
        except Exception:
            pass
        # attempt to salvage a partial JSON object
        try:
            start, end = s.find("{"), s.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(s[start : end + 1])
                return (
                    parsed
                    if isinstance(parsed, dict)
                    else {"status": "ok", "data": parsed}
                )
        except Exception:
            pass
        return {"status": "error", "error": "invalid JSON payload", "raw": s[:2000]}

    return {
        "status": "error",
        "error": f"unsupported response type: {type(obj).__name__}",
    }
