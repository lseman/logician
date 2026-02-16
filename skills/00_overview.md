# Time Series Analysis - Complete Skills

All time series analysis tools for dynamic loading via ToolRegistry.

## Setup Instructions

**Important:** `ToolRegistry` must execute the `## Bootstrap: helpers` block before registering tools.

Then you only need to inject `ctx`:

```python
from src.tools import ToolRegistry, Context  # adjust import path if your package name differs

ctx = Context()
registry = ToolRegistry(auto_load_from_skills=True)
registry.install_context(ctx)
```

## Analysis Playbooks

Use these high-level flows to make tool use more deliberate and useful for real time-series work.

1. **Quick Diagnostic Pass**
   `get_data_info` -> `compute_statistics` -> `detect_trend` -> `stationarity_tests` -> `stl_seasonality` -> `detect_anomalies`
2. **Preprocessing Pass**
   `regularize_series` -> `fill_missing` -> `hampel_filter` -> `detrend` -> `transform_series` -> `scale_series`
3. **Forecasting Pass**
   `suggest_horizon` -> `forecast_baselines` or `ensemble_forecast` -> `rolling_eval` -> `plot_forecast`
4. **Visual QA Pass**
   `plot_series` -> `plot_diagnostics` -> `plot_forecast`

These playbooks are guidance only; tools can be used independently.

## Bootstrap: helpers

**Description:**
Core helpers + `Context` definition used by all time-series tools. This block is executed by `ToolRegistry` *before* registering any `## Tool:` sections, and its symbols are injected into tool execution globals (`np`, `pd`, `json`, `_safe_json`, `_infer_freq_safe`, `_guess_season_length`, `_metrics`, `_regularize_series`, `Context`).

**Implementation:**

```python
# =============================================================================
# SKILLS BOOTSTRAP — helpers + Context
# Executed once before Tool sections are loaded
# =============================================================================

import json
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# -------------------------
# JSON helpers
# -------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()

    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (pd.Timedelta,)):
        return str(o)
    if isinstance(o, (pd.Index,)):
        return o.astype(str).tolist()

    if hasattr(o, "to_dict"):
        try:
            return o.to_dict()
        except Exception:
            pass

    return str(o)


def _safe_json(d) -> str:
    try:
        return json.dumps(d, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"JSON serialization failed: {e}"},
            indent=2,
            ensure_ascii=False,
        )


def _try_parse_json(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {"status": "error", "error": "empty response"}

        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"status": "ok", "data": parsed}
        except Exception:
            # attempt to salvage a dict substring
            try:
                start, end = s.find("{"), s.rfind("}")
                if start != -1 and end != -1 and end >= start:
                    parsed = json.loads(s[start:end+1])
                    return parsed if isinstance(parsed, dict) else {"status": "ok", "data": parsed}
            except Exception:
                pass

        return {"status": "error", "error": "invalid JSON payload", "raw": s[:2000]}

    return {"status": "error", "error": f"unsupported response type: {type(obj).__name__}"}


# -------------------------
# Frequency inference
# -------------------------

def _infer_freq_safe(index: pd.DatetimeIndex) -> Optional[str]:
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
            deltas = pd.Series([(index[i] - index[i - 1]).total_seconds() for i in range(1, len(index))])

        if deltas.empty:
            return None

        mode = deltas.mode()
        step = float(mode.iloc[0]) if not mode.empty else float(deltas.median())

        day = 86400.0
        if abs(step - day) < 10:
            return "D"
        if abs(step - 7 * day) < 10:
            return "W"
        if abs(step - 3600.0) < 5:
            return "H"
        if abs(step - 60.0) < 2:
            return "T"
        if abs(step - 30 * day) < 3 * day:
            return "MS"

    return None


# -------------------------
# Regularization
# -------------------------

def _regularize_series(df: pd.DataFrame, freq: Optional[str], method: str) -> pd.DataFrame:
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    freq = freq or _infer_freq_safe(df.index)

    if freq:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)

        for col in df.columns:
            if method == "linear":
                df[col] = df[col].interpolate(method="time", limit_direction="both")
            elif method == "ffill":
                df[col] = df[col].ffill().bfill()
            elif method == "pad":
                df[col] = df[col].ffill()
            else:
                df[col] = df[col].interpolate(method="time", limit_direction="both")

    return df.reset_index().rename(columns={"index": "date"})


# -------------------------
# Season length guess
# -------------------------

def _guess_season_length(y: np.ndarray) -> int:
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


# -------------------------
# Metrics
# -------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true[ok]), 1e-8)) * 100.0),
    }


# -------------------------
# Context
# -------------------------

@dataclass
class Context:
    data: Optional[pd.DataFrame] = None
    original_data: Optional[pd.DataFrame] = None
    data_name: str = ""
    freq_cache: Optional[str] = None

    anomaly_store: Dict[str, List[int]] = field(default_factory=dict)
    anomaly_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    nf_best_model: Optional[str] = None
    nf_cv_full: Optional[pd.DataFrame] = None
    nf_pred_col: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.data is not None and len(self.data) > 0

    @property
    def value_columns(self) -> List[str]:
        if self.data is None:
            return []
        return [c for c in self.data.columns if c != "date"]

    @property
    def is_multivariate(self) -> bool:
        return len(self.value_columns) > 1

    def reset(self) -> None:
        self.data = None
        self.original_data = None
        self.data_name = ""
        self.freq_cache = None
        self.anomaly_store.clear()
        self.anomaly_meta.clear()
        self.nf_best_model = None
        self.nf_cv_full = None
        self.nf_pred_col = None
```

## Context Reference

The `ctx` object is available to all tools:

```python
# ctx attributes:
ctx.data              # pd.DataFrame with 'date' + value columns
ctx.original_data     # Backup DataFrame
ctx.data_name         # str
ctx.freq_cache        # Optional[str]
ctx.anomaly_store     # Dict[str, List[int]] - cached indices
ctx.anomaly_meta      # Dict[str, Dict[str, Any]] - metadata

# ctx properties:
ctx.loaded            # bool
ctx.value_columns     # List[str]
ctx.is_multivariate   # bool
```

---

