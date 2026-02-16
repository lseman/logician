# -*- coding: utf-8 -*-
"""
Time Series Analysis - Core Helpers and Context Only

All tool implementations have been moved to SKILLS.md for dynamic loading.
This module provides only shared helpers and the Context class.

Version: 2.0.1 (Fixes: robust _safe_json, helper injection completeness)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ==================== Optional Dependencies ====================


def check_optional_deps() -> Dict[str, bool]:
    """
    Check which optional dependencies are available.

    Returns:
        Dict mapping package name to availability status
    """
    deps = {
        "scipy": False,
        "statsmodels": False,
        "ruptures": False,
        "sklearn": False,
        "neuralforecast": False,
    }

    try:
        import scipy  # noqa: F401

        deps["scipy"] = True
    except ImportError:
        pass

    try:
        import statsmodels  # noqa: F401

        deps["statsmodels"] = True
    except ImportError:
        pass

    try:
        import ruptures  # noqa: F401

        deps["ruptures"] = True
    except ImportError:
        pass

    try:
        import sklearn  # noqa: F401

        deps["sklearn"] = True
    except ImportError:
        pass

    try:
        import neuralforecast  # noqa: F401

        deps["neuralforecast"] = True
    except ImportError:
        pass

    return deps


# ==================== Core Helper Functions ====================


def _json_default(o: Any) -> Any:
    """
    Best-effort JSON serializer for common scientific Python objects.

    This is used by _safe_json and prevents crashes/NameErrors in tools
    that attempt to dump numpy/pandas objects.
    """
    # numpy scalars/arrays
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()

    # pandas objects
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (pd.Timedelta,)):
        return str(o)
    if isinstance(o, (pd.Index,)):
        return o.astype(str).tolist()

    # datetimes
    if isinstance(o, (datetime,)):
        return o.isoformat()

    # fallbacks
    if hasattr(o, "to_dict"):
        try:
            return o.to_dict()
        except Exception:
            pass

    return str(o)


def _safe_json(d: Union[dict, list]) -> str:
    """
    Safely serialize dict/list to JSON string.

    Args:
        d: Dictionary/list to serialize

    Returns:
        JSON string with pretty formatting
    """
    try:
        return json.dumps(d, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as e:
        # last-resort minimal error payload
        return json.dumps(
            {"status": "error", "error": f"JSON serialization failed: {e}"},
            indent=2,
            ensure_ascii=False,
        )


def _try_parse_json(obj: Any) -> Dict[str, Any]:
    """
    Attempt to parse object as JSON dict.

    Args:
        obj: Object to parse (dict, str, or other)

    Returns:
        Parsed dictionary or error dict
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
            # Try to extract JSON object from string
            try:
                start, end = s.find("{"), s.rfind("}")
                if start != -1 and end != -1 and end >= start:
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


def _infer_freq_safe(index: pd.DatetimeIndex) -> Optional[str]:
    """
    Infer frequency from DatetimeIndex robustly.

    Tries pd.infer_freq first, then falls back to manual detection based on
    common time deltas.

    Args:
        index: DatetimeIndex to analyze

    Returns:
        Frequency string (e.g., 'D', 'H', 'W') or None
    """
    try:
        f = pd.infer_freq(index)
        if f:
            return f
    except Exception:
        pass

    if len(index) >= 3:
        try:
            deltas = (index[1:] - index[:-1]).to_series().dt.total_seconds()
        except Exception:
            deltas = pd.Series(
                [
                    (index[i] - index[i - 1]).total_seconds()
                    for i in range(1, len(index))
                ]
            )

        if deltas.empty:
            return None

        mode = deltas.mode()
        step = float(mode.iloc[0]) if not mode.empty else float(deltas.median())
        day = 86400.0

        # Common frequencies
        if abs(step - day) < 10:
            return "D"  # Daily
        if abs(step - 7 * day) < 10:
            return "W"  # Weekly
        if abs(step - 3600.0) < 5:
            return "H"  # Hourly
        if abs(step - 60.0) < 2:
            return "T"  # Minute
        if abs(step - 30 * day) < 3 * day:
            return "MS"  # Month start

    return None


def _regularize_series(
    df: pd.DataFrame, freq: Optional[str], method: str
) -> pd.DataFrame:
    """
    Regularize series to full frequency grid with interpolation.

    Handles multivariate data (all value columns interpolated).

    Args:
        df: DataFrame with 'date' column + value column(s)
        freq: Target frequency (e.g., 'D', 'H')
        method: Interpolation method ('linear', 'ffill', 'pad')

    Returns:
        Regularized DataFrame with complete date index
    """
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")

    freq = freq or _infer_freq_safe(df.index)

    if freq:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)

        value_cols = [col for col in df.columns if col != "date"]
        for col in value_cols:
            if method == "linear":
                # 'time' interpolation requires DatetimeIndex (we have it)
                df[col] = df[col].interpolate(method="time", limit_direction="both")
            elif method == "ffill":
                df[col] = df[col].ffill().bfill()
            elif method == "pad":
                df[col] = df[col].ffill()
            else:
                # sensible default
                df[col] = df[col].interpolate(method="time", limit_direction="both")

    return df.reset_index().rename(columns={"index": "date"})


@contextmanager
def _nan_guard_ctx(values: np.ndarray):
    """
    Context manager for NaN guarding.

    Interpolates NaN values for safe computation, yields clean array.

    Args:
        values: Array potentially containing NaNs

    Yields:
        Array with NaNs filled
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        series = pd.Series(values).interpolate(limit_direction="both").bfill().ffill()
        yield series.values
    else:
        yield values


def _guess_season_length(y: np.ndarray) -> int:
    """
    Guess seasonal period from ACF peaks.

    Uses statsmodels ACF to find highest local maximum, suggesting
    seasonal period.

    Args:
        y: Time series values

    Returns:
        Estimated seasonal period (default 7 if detection fails)
    """
    y = np.asarray(y, dtype=float)
    try:
        import statsmodels.tsa.api as smt

        nlags = min(365, max(30, len(y) // 2))
        acf = smt.stattools.acf(y, nlags=nlags, fft=True)

        candidates = [
            (lag, acf[lag])
            for lag in range(2, len(acf) - 1)
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1]
        ]
        if candidates:
            return int(max(candidates, key=lambda kv: kv[1])[0])
    except Exception:
        pass

    return 7


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard time series forecast metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict with RMSE, MSE, MAE, MAPE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) == 0:
        return {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan}

    err = y_true[ok] - y_pred[ok]
    mse_val = float(np.mean(err**2))

    return {
        "RMSE": float(np.sqrt(mse_val)),
        "MSE": mse_val,
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(
            np.mean(np.abs(err) / np.maximum(np.abs(y_true[ok]), 1e-8)) * 100.0
        ),
    }


# ==================== Context ====================


@dataclass
class Context:
    """
    Shared context for time series analysis tools.

    This object is injected into the execution scope of all tools,
    allowing them to share state across operations.

    Attributes:
        data: Current DataFrame with 'date' column + value column(s)
        original_data: Backup of original data before transformations
        data_name: Name/identifier for the dataset
        freq_cache: Cached inferred frequency
        anomaly_store: Cached anomaly indices per column (Dict[column, List[int]])
        anomaly_meta: Cached anomaly metadata per column
    """

    data: Optional[pd.DataFrame] = None
    original_data: Optional[pd.DataFrame] = None
    data_name: str = ""
    freq_cache: Optional[str] = None

    # Anomaly caching (keeps prompts small)
    anomaly_store: Dict[str, List[int]] = field(default_factory=dict)
    anomaly_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # NeuralForecast state (if needed)
    nf_best_model: Optional[str] = None
    nf_cv_full: Optional[pd.DataFrame] = None
    nf_pred_col: Optional[str] = None

    @property
    def loaded(self) -> bool:
        """Check if data is loaded."""
        return self.data is not None and len(self.data) > 0

    @property
    def value_columns(self) -> List[str]:
        """Get list of value columns (all except 'date')."""
        if self.data is None:
            return []
        return [col for col in self.data.columns if col != "date"]

    @property
    def is_multivariate(self) -> bool:
        """Check if data has multiple value columns."""
        return len(self.value_columns) > 1

    def reset(self) -> None:
        """Reset context to empty state."""
        self.data = None
        self.original_data = None
        self.data_name = ""
        self.freq_cache = None
        self.anomaly_store.clear()
        self.anomaly_meta.clear()
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None


def get_helpers() -> Dict[str, Any]:
    """
    Get all helper functions for injection into tool execution scope.

    Returns:
        Dict mapping helper names to functions
    """
    return {
        # helpers
        "_safe_json": _safe_json,
        "_try_parse_json": _try_parse_json,
        "_infer_freq_safe": _infer_freq_safe,
        "_regularize_series": _regularize_series,
        "_nan_guard_ctx": _nan_guard_ctx,
        "_guess_season_length": _guess_season_length,
        "_metrics": _metrics,
        "_json_default": _json_default,
        # core types/modules tools often reference
        "Context": Context,
        "np": np,
        "pd": pd,
        "json": json,
    }


# ==================== Module Info ====================

__version__ = "2.0.1"
__all__ = [
    "Context",
    "_safe_json",
    "_try_parse_json",
    "_infer_freq_safe",
    "_guess_season_length",
    "_regularize_series",
    "_nan_guard_ctx",
    "_metrics",
    "check_optional_deps",
    "get_helpers",
]


# Print info on import
def _print_import_info():
    """Print dependency info when module is imported."""
    deps = check_optional_deps()
    available = [k for k, v in deps.items() if v]
    if available:
        print(f"ts_tools v{__version__}: Available deps: {', '.join(available)}")
    else:
        print(f"ts_tools v{__version__}: No optional dependencies (basic mode)")


if __name__ != "__main__":
    _print_import_info()
