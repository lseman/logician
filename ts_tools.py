# -*- coding: utf-8 -*-
"""
Enhanced Time Series Analysis Tools for Agent (Updated Nov 2025).

Key adjustments in this version (per request):
- Anomaly detection now returns ONLY counts/rates (tiny JSON). Full indices are cached
  internally in Context.anomaly_store (per column) with small metadata in Context.anomaly_meta.
- Detrending is strictly polynomial fitting (degree 1 = linear, or any degree>=2).
  No temporal differencing or STL differencing is used for detrending.

Other features preserved:
- Multivariate support (wide format with a "date" column + one or more value columns).
- Robust frequency inference, regularization, missing-value handling.
- Stats, stationarity tests, seasonality indicators, ACF/PACF peaks.
- Baseline forecasting with simple intervals; NeuralForecast-based selection (if installed).
- Plotting utilities (series, diagnostics, STL decomposition, CV overlays).
- All tools are exposed via the @as_tool decorator for agent auto-registration.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Tool exposure decorator (keeps agent-side auto-mount simple, avoids circular imports)
# ──────────────────────────────────────────────────────────────────────────────
def as_tool(name: Optional[str] = None, desc: Optional[str] = None):
    """
    Mark an instance method as a tool. Agent will introspect these and register
    them via _iter_exposed_methods(...).
    """

    def _wrap(fn):
        tool_name = name or fn.__name__
        original_doc = fn.__doc__

        def wrapped(*args, **kwargs):
            # Print tool call with parameters (useful when running headless)
            param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            if len(args) > 1:
                pos_str = ", ".join([str(a) for a in args[1:]])
                if pos_str:
                    param_str = f"{pos_str}, {param_str}" if param_str else pos_str
            print(f"Calling tool '{tool_name}' with parameters: {param_str}")
            result = fn(*args, **kwargs)
            return result

        setattr(wrapped, "_tool_exposed", True)
        if name is not None:
            setattr(wrapped, "_tool_name_override", name)
        if desc is not None:
            setattr(wrapped, "_tool_desc_override", desc)
        wrapped.__doc__ = original_doc
        wrapped.__name__ = fn.__name__
        wrapped.__module__ = fn.__module__
        return wrapped

    return _wrap


# ──────────────────────────────────────────────────────────────────────────────
# Optional deps (safe gating)
# ──────────────────────────────────────────────────────────────────────────────
SCIPY_OK = False
STATSM_OK = False
RUPTURES_OK = False
SKLEARN_OK = False
NF_OK = False
try:
    import scipy.signal as spsig  # noqa
    from scipy.fft import rfft, rfftfreq  # noqa
    from scipy.stats import boxcox
    from scipy.stats import entropy as scipy_entropy  # noqa

    SCIPY_OK = True
except Exception:
    spsig = scipy_entropy = boxcox = rfft = rfftfreq = None
try:
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    from statsmodels.graphics.tsaplots import plot_acf as _plot_acf  # noqa
    from statsmodels.graphics.tsaplots import plot_pacf as _plot_pacf  # noqa
    from statsmodels.tsa.seasonal import STL

    STATSM_OK = True
except Exception:
    sm = smt = STL = _plot_acf = _plot_pacf = None
try:
    import ruptures as rpt

    RUPTURES_OK = True
except Exception:
    rpt = None
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler

    SKLEARN_OK = True
except Exception:
    IsolationForest = RobustScaler = None
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import GRU, LSTM, MLP, NBEATS, NHITS, RNN, TCN, TFT
    from neuralforecast.models import BiTCN, DeepAR, DilatedRNN, FEDformer, Informer
    from neuralforecast.models import NBEATSx, PatchTST, TiDE, TimesNet

    _NEW_MODELS = [PatchTST]
    NF_OK = True
except Exception:
    _NEW_MODELS = []
    NeuralForecast = None


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers & context
# ──────────────────────────────────────────────────────────────────────────────
def _safe_json(d: dict) -> str:
    """Safely serialize dict to JSON string."""
    try:
        return json.dumps(d, indent=2, default=lambda o: str(o))
    except Exception as e:
        return json.dumps(
            {"status": "error", "error": f"JSON serialization failed: {e}"}, indent=2
        )


def _try_parse_json(obj: Any) -> Dict[str, Any]:
    """Attempt to parse object as JSON dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {"status": "error", "error": "empty response"}
        try:
            return json.loads(s)
        except Exception:
            try:
                start, end = s.find("{"), s.rfind("}")
                if start != -1 and end != -1 and end >= start:
                    return json.loads(s[start : end + 1])
            except Exception:
                pass
        return {"status": "error", "error": "invalid JSON payload", "raw": s[:2000]}
    return {
        "status": "error",
        "error": f"unsupported response type: {type(obj).__name__}",
    }


def _infer_freq_safe(index: pd.DatetimeIndex) -> Optional[str]:
    """Infer frequency from DatetimeIndex robustly (improved with pd.infer_freq fallback)."""
    try:
        f = pd.infer_freq(index)
        if f:
            return f
    except Exception:
        pass
    if len(index) >= 3:
        deltas = (index[1:] - index[:-1]).to_series().dt.total_seconds()
        if deltas.empty:
            return None
        step = deltas.mode().iloc[0] if not deltas.mode().empty else deltas.median()
        day = 86400.0
        if abs(step - day) < 10:
            return "D"
        if abs(step - 7 * day) < 10:
            return "W"
        if abs(step - 3600.0) < 5:
            return "H"
        if abs(step - 30 * day) < 3 * day:
            return "MS"
        if abs(step - 60.0) < 2:
            return "T"
    return None


def _regularize_series(
    df: pd.DataFrame, freq: Optional[str], method: str
) -> pd.DataFrame:
    """Regularize series to full index with interpolation (multivariate-safe)."""
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    freq = freq or _infer_freq_safe(df.index)
    if freq:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)
        value_cols = [col for col in df.columns if col != "date"]
        for col in value_cols:
            if method == "linear":
                df[col] = df[col].interpolate(method="time", limit_direction="both")
            elif method == "ffill":
                df[col] = df[col].ffill().bfill()
            elif method == "pad":
                df[col] = df[col].ffill()
    return df.reset_index().rename(columns={"index": "date"})


@contextmanager
def _nan_guard_ctx(values: np.ndarray):
    """Context manager for NaN guarding."""
    if np.isnan(values).any():
        series = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )
        yield series.values
    else:
        yield values


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard time series metrics."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    err = y_true[ok] - y_pred[ok]
    mse_v = float(np.mean(err**2))
    return {
        "RMSE": float(np.sqrt(mse_v)),
        "MSE": mse_v,
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / (np.maximum(np.abs(y_true[ok]), 1e-8))))
        * 100.0,
    }


def _guess_season_length(y: np.ndarray) -> int:
    """Guess seasonal period from ACF peaks."""
    if not STATSM_OK:
        return 7
    try:
        acf = smt.stattools.acf(y, nlags=min(365, max(30, len(y) // 2)), fft=True)
        cands = [
            (lag, acf[lag])
            for lag in range(2, len(acf))
            if lag + 1 < len(acf)
            and acf[lag] > acf[lag - 1]
            and acf[lag] > acf[lag + 1]
        ]
        if cands:
            return int(sorted(cands, key=lambda kv: kv[1], reverse=True)[0][0])
    except Exception:
        pass
    return 7


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class NFState:
    """State for NeuralForecast caching (multivariate-safe)."""

    best_model: Optional[str] = None
    cv_full: Optional[pd.DataFrame] = None
    pred_col: Optional[str] = None
    date_val_start: Optional[pd.Timestamp] = None
    date_test_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    y_norm: Optional[np.ndarray] = None  # For univariate; per-series for multi
    dates: Optional[pd.Series] = None
    title: Optional[str] = None
    unique_ids: Optional[List[str]] = None


@dataclass
class Context:
    """Shared context for tool state (added value_columns)."""

    data: Optional[pd.DataFrame] = None  # columns: date, value(s) or multi-series
    original_data: Optional[pd.DataFrame] = None  # backup
    data_name: str = ""
    freq_cache: Optional[str] = None
    nf: NFState = field(default_factory=NFState)

    # NEW: cache anomaly indices and light metadata so prompts stay small
    anomaly_store: Dict[str, List[int]] = field(default_factory=dict)
    anomaly_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def value_columns(self) -> List[str]:
        if self.data is None:
            return []
        return [col for col in self.data.columns if col != "date"]

    @property
    def is_multivariate(self) -> bool:
        return len(self.value_columns) > 1

    @property
    def loaded(self) -> bool:
        return self.data is not None and len(self.data) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Tool base & concrete tool classes
# ──────────────────────────────────────────────────────────────────────────────
class ToolBase:
    """Base class for all tools, providing context access."""

    def __init__(self, ctx: Context):
        self.ctx = ctx


# ========== Data I/O ==========
class DataTools(ToolBase):
    """Tools for loading and inspecting time series data (multivariate-ready)."""

    @as_tool(
        name="set_numpy",
        desc="Load 1D/2D NumPy array into the agent as a time series (univariate only).",
    )
    def set_numpy(
        self,
        arr: np.ndarray,
        start_date: str = "2018-01-01",
        freq: str = "D",
        name: str = "numpy_series",
    ) -> str:
        """Load 1D/2D numpy array as time series (univariate)."""
        a = np.asarray(arr)
        if a.ndim == 1:
            y = a.astype(float)
        elif a.ndim == 2:
            y = np.nanmean(a, axis=1).astype(float)
        else:
            return _safe_json({"status": "error", "error": "arr must be 1D or 2D"})
        T = len(y)
        if T == 0:
            return _safe_json({"status": "error", "error": "Empty array"})
        idx = pd.date_range(start=start_date, periods=T, freq=freq)
        self.ctx.data = pd.DataFrame({"date": idx, "value": y})
        self.ctx.original_data = self.ctx.data.copy()
        self.ctx.data_name = name
        self.ctx.freq_cache = freq
        return _safe_json(
            {
                "status": "ok",
                "message": "numpy data loaded",
                "n": T,
                "freq": freq,
                "name": name,
            }
        )

    @as_tool(
        name="load_csv_data",
        desc="Load time series from CSV by specifying date/value columns (supports multi-series).",
    )
    def load_csv_data(
        self,
        filepath: str,
        date_column: str,
        value_column: Union[str, List[str]],
        sep: str = ",",
    ) -> str:
        """Load CSV as time series; supports multi-value columns (kept wide)."""
        try:
            df = pd.read_csv(filepath, sep=sep)
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column).rename(columns={date_column: "date"})
            value_cols = (
                [value_column] if isinstance(value_column, str) else value_column
            )
            for col in value_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    return _safe_json(
                        {"status": "error", "error": f"Value column '{col}' not found"}
                    )
            self.ctx.data = df
            self.ctx.original_data = df.copy()
            self.ctx.data_name = filepath
            self.ctx.freq_cache = _infer_freq_safe(df["date"])
            return _safe_json(
                {
                    "status": "ok",
                    "message": "data loaded",
                    "info": self.get_data_info_dict(),
                }
            )
        except Exception as e:
            return _safe_json({"status": "error", "error": str(e)})

    @as_tool(
        name="create_sample_data",
        desc="Create synthetic data: 'trend'|'seasonal'|'random'|'anomaly'|'stationary'|'cyclic_trend' (univariate).",
    )
    def create_sample_data(
        self, pattern: str, n_points: Union[int, str] = 200, noise_level: float = 1.0
    ) -> str:
        """Create synthetic time series data with various patterns (univariate)."""
        try:
            n = int(n_points)
            if n < 10:
                return _safe_json(
                    {"status": "error", "error": "n_points must be >= 10"}
                )
            dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
            rng = np.random.default_rng(42)
            if pattern == "trend":
                values = np.linspace(10, 50, n) + rng.normal(0, noise_level, n)
            elif pattern == "seasonal":
                t = np.arange(n)
                values = (
                    20
                    + 6 * np.sin(2 * np.pi * t / 7)
                    + 3 * np.sin(2 * np.pi * t / 30)
                    + rng.normal(0, noise_level, n)
                )
            elif pattern == "anomaly":
                values = 25 + 2 * rng.normal(size=n, scale=noise_level)
                n_spikes = max(3, n // 60)
                spikes = rng.choice(n, size=n_spikes, replace=False)
                values[spikes] += rng.choice([-1, 1], size=n_spikes) * rng.uniform(
                    8, 15, size=n_spikes
                )
            elif pattern == "stationary":
                values = 25 + rng.normal(0, noise_level, n)
            elif pattern == "cyclic_trend":
                t = np.arange(n)
                values = (
                    20
                    + 0.1 * t
                    + 5 * np.sin(2 * np.pi * t / 365)
                    + rng.normal(0, noise_level, n)
                )
            else:
                values = 25 + 3 * rng.normal(size=n, scale=noise_level)
            self.ctx.data = pd.DataFrame({"date": dates, "value": values})
            self.ctx.original_data = self.ctx.data.copy()
            self.ctx.data_name = f"sample_{pattern}"
            self.ctx.freq_cache = "D"
            return _safe_json(
                {
                    "status": "ok",
                    "message": "sample created",
                    "n": n,
                    "pattern": pattern,
                }
            )
        except Exception as e:
            return _safe_json({"status": "error", "error": str(e)})

    def get_data_info_dict(self) -> Dict[str, Any]:
        """Get dictionary of data info (multivariate-aware)."""
        if not self.ctx.loaded:
            return {"loaded": False}
        df = self.ctx.data
        dr = (df["date"].min(), df["date"].max())
        value_cols = self.ctx.value_columns
        info = {
            "loaded": True,
            "name": self.ctx.data_name,
            "n_records": int(len(df)),
            "date_range": [str(dr[0]), str(dr[1])],
            "value_columns": value_cols,
            "is_multivariate": self.ctx.is_multivariate,
            "freq_inferred": self.ctx.freq_cache or _infer_freq_safe(df["date"]),
        }
        for col in value_cols:
            v = df[col]
            info[f"{col}_missing"] = int(v.isna().sum())
            info[f"{col}_min"] = float(np.nanmin(v.values))
            info[f"{col}_max"] = float(np.nanmax(v.values))
        return info

    @as_tool(
        name="get_data_info",
        desc="Return metadata of the currently loaded series (includes multivariate info).",
    )
    def get_data_info(self) -> str:
        """Get JSON string of data info."""
        return _safe_json(self.get_data_info_dict())

    @as_tool(
        name="restore_original",
        desc="Restore the original data before any transformations.",
    )
    def restore_original(self) -> str:
        """Restore original data."""
        if self.ctx.original_data is None:
            return _safe_json(
                {"status": "error", "error": "No original data to restore"}
            )
        self.ctx.data = self.ctx.original_data.copy()
        return _safe_json({"status": "ok", "message": "Original data restored"})


# ========== Hygiene ==========
class HygieneTools(ToolBase):
    """Tools for data cleaning and regularization (multivariate-safe)."""

    @as_tool(
        name="regularize_series",
        desc="Resample to a regular frequency and fill small gaps (handles multi-columns).",
    )
    def regularize_series(self, freq: str = "", method: str = "linear") -> str:
        """Regularize to full frequency grid."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        f = freq or self.ctx.freq_cache
        if not f:
            return _safe_json(
                {"status": "error", "error": "Frequency required or inferable"}
            )
        self.ctx.data = _regularize_series(self.ctx.data, f, method)
        self.ctx.freq_cache = f or _infer_freq_safe(self.ctx.data["date"])
        return _safe_json(
            {
                "status": "ok",
                "freq": self.ctx.freq_cache,
                "filled_method": method,
                "n_cols": len(self.ctx.value_columns),
            }
        )

    @as_tool(
        name="fill_missing",
        desc="Fill NaNs via 'time' interpolation (default) or 'ffill' (multi-column).",
    )
    def fill_missing(self, method: str = "time") -> str:
        """Fill missing values (per column)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        df = self.ctx.data.sort_values("date").copy()
        value_cols = self.ctx.value_columns
        if not value_cols:
            return _safe_json({"status": "error", "error": "No value columns"})
        remaining_nans = {}
        for col in value_cols:
            if method == "ffill":
                df[col] = df[col].ffill().bfill()
            else:
                df_temp = df.set_index("date")
                df_temp[col] = (
                    df_temp[col]
                    .interpolate(method="time", limit_direction="both")
                    .ffill()
                    .bfill()
                )
                df[col] = df_temp[col].reset_index(drop=True)
            remaining_nans[col] = int(df[col].isna().sum())
        self.ctx.data = df
        return _safe_json(
            {"status": "ok", "method": method, "remaining_nans": remaining_nans}
        )

    @as_tool(
        name="hampel_filter",
        desc="Apply Hampel filter (robust outlier suppression, per column).",
    )
    def hampel_filter(
        self, window: Union[int, str] = 15, n_sigmas: Union[float, str] = 3.0
    ) -> str:
        """Apply Hampel filter for outlier detection/replacement (multi-column)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        w = int(window)
        w = w if w % 2 == 1 else w + 1
        ns = float(n_sigmas)
        replaced = {}
        for col in self.ctx.value_columns:
            s = self.ctx.data[col].copy()
            med = s.rolling(w, center=True).median()
            mad = (s - med).abs().rolling(w, center=True).median()
            k = 1.4826
            mask = (s - med).abs() > ns * k * mad.replace(0, np.nan)
            s_filt = s.copy()
            s_filt[mask] = med[mask]
            self.ctx.data[col] = s_filt
            replaced[col] = int(mask.sum())
        return _safe_json(
            {"status": "ok", "replaced": replaced, "window": w, "n_sigmas": ns}
        )


# ========== Transform ==========
class TransformTools(ToolBase):
    """Tools for series transformations (per column for multi)."""

    @as_tool(
        name="detrend",
        desc="Remove linear or polynomial trend from series via numpy.polyfit (handles multi).",
    )
    def detrend(
        self,
        method: str = "linear",
        degree: Union[int, str] = 2,
        column: Optional[str] = None,
    ) -> str:
        """Remove trend via polynomial fitting. 'linear' is degree=1; else specify degree>=2."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        cols = [column] if column else self.ctx.value_columns
        if not cols:
            return _safe_json({"status": "error", "error": "No columns to detrend"})
        deg_out = 1
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            with _nan_guard_ctx(y) as y_guarded:
                y = y_guarded
            x = np.arange(len(y))
            if method == "linear":
                coeffs = np.polyfit(x, y, 1)
                trend = np.polyval(coeffs, x)
                deg_out = 1
            else:
                deg = max(3, int(degree))
                coeffs = np.polyfit(x, y, deg)
                trend = np.polyval(coeffs, x)
                deg_out = deg
            self.ctx.data[col] = y - trend
        
        return _safe_json(
            {"status": "ok", "method": "polyfit", "degree": deg_out, "columns": cols}
        )

    @as_tool(
        name="transform_series",
        desc="Apply log, sqrt, or Box-Cox transformation (per column).",
    )
    def transform_series(self, method: str, column: Optional[str] = None) -> str:
        """Apply log, sqrt, or boxcox transform (specific or all columns)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        cols = [column] if column else self.ctx.value_columns
        results = {}
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            if np.any(y <= 0) and method in ["log", "boxcox"]:
                results[col] = "error: requires positive values"
                continue
            try:
                if method == "log":
                    self.ctx.data[col] = np.log(y)
                    results[col] = "ok"
                elif method == "sqrt":
                    if np.any(y < 0):
                        results[col] = "error: requires non-negative values"
                        continue
                    self.ctx.data[col] = np.sqrt(y)
                    results[col] = "ok"
                elif method == "boxcox":
                    if not SCIPY_OK:
                        results[col] = "error: scipy not available"
                        continue
                    transformed, lmbda = boxcox(y)
                    self.ctx.data[col] = transformed
                    results[col] = {"ok": True, "lambda": float(lmbda)}
                else:
                    results[col] = f"error: unknown method {method}"
            except Exception as e:
                results[col] = f"error: {str(e)}"
        return _safe_json({"status": "ok", "method": method, "results": results})

    @as_tool(
        name="scale_series",
        desc="Standardize (z-score) or normalize (min-max) or robust (default).",
    )
    def scale_series(self, method: str = "robust", column: Optional[str] = None) -> str:
        """Scale series (handles multi; robust default)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        cols = [column] if column else self.ctx.value_columns
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            with _nan_guard_ctx(y) as y_guarded:
                y = y_guarded
            if method == "standard":
                scaled = (y - np.mean(y)) / (np.std(y) + 1e-12)
            elif method == "minmax":
                vmin, vmax = np.min(y), np.max(y)
                scaled = (y - vmin) / (vmax - vmin + 1e-12)
            elif method == "robust":
                if not SKLEARN_OK:
                    return _safe_json(
                        {
                            "status": "error",
                            "error": "sklearn not available for robust scaling",
                        }
                    )
                scaler = RobustScaler()
                scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                return _safe_json(
                    {"status": "error", "error": f"Unknown method: {method}"}
                )
            self.ctx.data[col] = scaled
        return _safe_json({"status": "ok", "method": method, "columns": cols})


# ========== Stats & Seasonality ==========
class StatsTools(ToolBase):
    """Basic statistical tools (multivariate extensions)."""

    @as_tool(
        name="compute_statistics",
        desc="Mean/median/std/min/max/q25/q75 (per column or aggregate).",
    )
    def compute_statistics(self, aggregate: bool = False) -> str:
        """Compute descriptive statistics (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        value_cols = self.ctx.value_columns
        if aggregate:
            v = pd.concat([self.ctx.data[col] for col in value_cols], axis=0).dropna()
            stats = {
                "mean": float(v.mean()),
                "median": float(v.median()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)),
            }
            return _safe_json(stats)
        stats = {}
        for col in value_cols:
            v = pd.to_numeric(self.ctx.data[col], errors="coerce")
            stats[col] = {
                "mean": float(v.mean()),
                "median": float(v.median()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)),
                "skew": float(v.skew()) if SCIPY_OK else None,
            }
        return _safe_json(stats)

    @as_tool(
        name="stationarity_tests", desc="ADF and Ljung–Box (if available, per column)."
    )
    def stationarity_tests(
        self, lags: Union[int, str] = 0, column: Optional[str] = None
    ) -> str:
        """ADF and Ljung-Box tests (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        cols = [column] if column else self.ctx.value_columns
        out = {}
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            col_out = {"adf": None, "ljung_box": None, "notes": []}
            if STATSM_OK:
                try:
                    adf = sm.tsa.stattools.adfuller(y, autolag="AIC")
                    col_out["adf"] = {
                        "stat": float(adf[0]),
                        "pvalue": float(adf[1]),
                        "stationary": adf[1] < 0.05,
                    }
                except Exception as e:
                    col_out["notes"].append(f"ADF failed: {e}")
                try:
                    lb_lags = int(lags) if lags else min(40, max(10, len(y) // 10))
                    lb = sm.stats.acorr_ljungbox(y, lags=[lb_lags], return_df=True)
                    col_out["ljung_box"] = {
                        "lags": lb_lags,
                        "stat": float(lb["lb_stat"].iloc[0]),
                        "pvalue": float(lb["lb_pvalue"].iloc[0]),
                    }
                except Exception as e:
                    col_out["notes"].append(f"Ljung-Box failed: {e}")
            else:
                col_out["notes"].append("statsmodels not available")
            out[col] = col_out
        return _safe_json(out)

    @as_tool(name="acf_pacf_peaks", desc="Top ACF/PACF lags by magnitude (per column).")
    def acf_pacf_peaks(
        self,
        nlags: Union[int, str] = 40,
        topk: Union[int, str] = 5,
        column: Optional[str] = None,
    ) -> str:
        """Top ACF/PACF peaks (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        N = int(nlags)
        K = int(topk)
        cols = [column] if column else self.ctx.value_columns
        out = {}
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            col_out = {"acf_top": [], "pacf_top": [], "notes": []}
            if STATSM_OK:
                try:
                    acf = smt.stattools.acf(y, nlags=N, fft=True)
                    pacf = smt.stattools.pacf(y, nlags=N, method="yw")
                    lag_acf = [
                        (lag, float(abs(val)))
                        for lag, val in enumerate(acf[1:], start=1)
                    ]
                    lag_pacf = [
                        (lag, float(abs(val)))
                        for lag, val in enumerate(pacf[1:], start=1)
                    ]
                    col_out["acf_top"] = sorted(
                        lag_acf, key=lambda x: x[1], reverse=True
                    )[:K]
                    col_out["pacf_top"] = sorted(
                        lag_pacf, key=lambda x: x[1], reverse=True
                    )[:K]
                except Exception as e:
                    col_out["notes"].append(f"ACF/PACF failed: {e}")
            else:
                col_out["notes"].append("statsmodels not available")
            out[col] = col_out
        return _safe_json(out)

    @as_tool(
        name="compute_multivariate_stats",
        desc="Compute correlations and per-series stats for multi-column data.",
    )
    def compute_multivariate_stats(self) -> str:
        """Correlations + per-series stats for multivariate data."""
        if not self.ctx.is_multivariate:
            return _safe_json(
                {
                    "status": "error",
                    "error": "Not multivariate data (need >=2 value columns)",
                }
            )
        df_num = self.ctx.data[self.ctx.value_columns].dropna()
        corr = df_num.corr().to_dict()
        per_series = {}
        for col in self.ctx.value_columns:
            v = pd.to_numeric(self.ctx.data[col], errors="coerce")
            per_series[col] = {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "adf_pvalue": float(sm.tsa.stattools.adfuller(v.dropna())[1])
                if STATSM_OK
                else None,
            }
        return _safe_json({"correlations": corr, "per_series": per_series})


class SeasonalityTools(ToolBase):
    """Seasonality detection tools (per column)."""

    @as_tool(
        name="detect_trend",
        desc="Linear slope and percent change (per column or aggregate).",
    )
    def detect_trend(self, column: Optional[str] = None) -> str:
        """Detect overall trend direction (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        cols = [column] if column else self.ctx.value_columns
        out = {}
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            x = np.arange(len(self.ctx.data))
            y = self.ctx.data[col].values.astype(float)
            coeffs = np.polyfit(x, y, 1)
            slope = float(coeffs[0])
            start_val, end_val = float(y[0]), float(y[-1])
            pct_change = float(
                ((end_val - start_val) / (start_val if abs(start_val) > 1e-12 else 1.0))
                * 100.0
            )
            trend = (
                "flat" if abs(slope) < 1e-3 else ("upward" if slope > 0 else "downward")
            )
            out[col] = {
                "trend": trend,
                "slope": slope,
                "percent_change": pct_change,
                "start_value": start_val,
                "end_value": end_val,
            }
        return _safe_json(out)

    @as_tool(
        name="stl_seasonality",
        desc="STL seasonal strength and period guess (per column).",
    )
    def stl_seasonality(
        self, period: Union[int, str] = 0, column: Optional[str] = None
    ) -> str:
        """STL-based seasonality strength (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        if not STATSM_OK:
            return _safe_json({"status": "error", "error": "statsmodels not available"})
        cols = [column] if column else self.ctx.value_columns
        out = {}
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            per = int(period) if period else _guess_season_length(y)
            try:
                stl = STL(y, period=max(2, per), robust=True).fit()
                var = np.var
                seas_strength = 1.0 - (
                    var(stl.resid) / (var(stl.resid + stl.seasonal) + 1e-12)
                )
                s = float(max(0.0, min(1.0, seas_strength)))
                out[col] = {
                    "period_used": int(per),
                    "seasonal_strength": s,
                    "interpretation": "Strong"
                    if s > 0.64
                    else "Moderate"
                    if s > 0.36
                    else "Weak",
                }
            except Exception as e:
                out[col] = {"status": "error", "error": str(e), "period_attempted": per}
        return _safe_json(out)


# ========== Anomaly & Change points ==========
class AnomalyTools(ToolBase):
    """Anomaly detection tools (vectorized for multi)."""

    @as_tool(
        name="detect_anomalies",
        desc="zscore|iqr|hampel|stl_resid|iforest — returns only counts/rates; indices cached internally.",
    )
    def detect_anomalies(
        self,
        method: str,
        threshold: Union[float, str] = 3.0,
        period: Union[int, str] = 0,
        column: Optional[str] = None,
    ) -> str:
        """
        Detect anomalies using various methods (multi-aware).
        Returns a tiny JSON per column with {'n_anomalies', 'anomaly_rate'} only.
        Full anomaly indices are stored in ctx.anomaly_store[column] and metadata in
        ctx.anomaly_meta[column], so the agent can retrieve them later without bloating prompts.
        """
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})

        def _to_float(x, default):
            try:
                return float(x)
            except Exception:
                return float(default)

        def _to_int(x, default):
            try:
                return int(float(x))
            except Exception:
                return int(default)

        mth = (method or "").strip().lower()
        cols = [column] if column else self.ctx.value_columns
        if not cols:
            return _safe_json({"status": "error", "error": "No value columns"})

        per = _to_int(period, 0)
        summary: Dict[str, Any] = {}

        for col in cols:
            if col not in self.ctx.data.columns:
                summary[col] = {"status": "error", "error": f"Column '{col}' not found"}
                continue

            y = pd.to_numeric(self.ctx.data[col], errors="coerce").values.astype(float)
            mask = np.zeros_like(y, dtype=bool)
            notes: List[str] = []

            try:
                if mth == "zscore":
                    thr = _to_float(threshold, 3.0)
                    mu = np.nanmean(y)
                    sd = np.nanstd(y) + 1e-12
                    mask = np.abs((y - mu) / sd) > thr

                elif mth == "iqr":
                    thr = _to_float(threshold, 1.5)
                    q1, q3 = np.nanpercentile(y, 25), np.nanpercentile(y, 75)
                    iqr = q3 - q1
                    lb, ub = q1 - thr * iqr, q3 + thr * iqr
                    mask = (y < lb) | (y > ub)

                elif mth == "hampel":
                    w_default, ns_default = 15, 3.0
                    w, ns = w_default, ns_default
                    if isinstance(threshold, str):
                        parts = [p.strip() for p in threshold.split(",") if p.strip()]
                        if len(parts) >= 1:
                            w = _to_int(parts[0], w_default)
                        if len(parts) >= 2:
                            ns = _to_float(parts[1], ns_default)
                    else:
                        w = _to_int(threshold, w_default)
                        ns = ns_default
                    w = w if w % 2 == 1 else w + 1
                    s = pd.Series(y)
                    med = s.rolling(w, center=True).median()
                    mad = (s - med).abs().rolling(w, center=True).median()
                    k = 1.4826
                    denom = (k * mad.replace(0, np.nan)).values
                    mask = (np.abs(s.values - med.values) > ns * np.nan_to_num(denom, nan=np.inf))

                elif mth == "stl_resid":
                    if not STATSM_OK:
                        notes.append("statsmodels not available; cannot use STL")
                        mask = np.zeros_like(y, dtype=bool)
                    else:
                        p = per if per > 0 else _guess_season_length(y)
                        from statsmodels.tsa.seasonal import STL as _STL
                        res = _STL(y, period=max(2, int(p)), robust=True).fit().resid
                        q1, q3 = np.nanpercentile(res, 25), np.nanpercentile(res, 75)
                        iqr = q3 - q1
                        lb, ub = q1 - 3 * iqr, q3 + 3 * iqr
                        mask = (res < lb) | (res > ub)

                elif mth == "iforest":
                    if not SKLEARN_OK:
                        notes.append("sklearn not available; cannot use IsolationForest")
                        mask = np.zeros_like(y, dtype=bool)
                    else:
                        X = y.reshape(-1, 1)
                        labels = IsolationForest(
                            n_estimators=200, contamination="auto", random_state=42
                        ).fit_predict(X)
                        mask = labels == -1

                else:
                    notes.append(f"Unknown method: {method}")
                    mask = np.zeros_like(y, dtype=bool)

            except Exception as e:
                notes.append(f"error: {e}")
                mask = np.zeros_like(y, dtype=bool)

            idxs = np.where(mask)[0].astype(int)
            n = int(mask.sum())
            rate = float(100.0 * (np.mean(mask) if len(y) else 0.0))

            # Cache indices + small metadata in Context (agent memory, not prompt)
            self.ctx.anomaly_store[col] = idxs.tolist()
            self.ctx.anomaly_meta[col] = {
                "method": mth,
                "threshold": threshold,
                "period": per if mth == "stl_resid" else None,
                "anomaly_rate": rate,
                "n": n,
                "notes": notes,
            }

            # Return only compact stats
            summary[col] = {"n_anomalies": n, "anomaly_rate": rate}

        return _safe_json({"status": "ok", "summary": summary})

    @as_tool(
        name="get_cached_anomalies",
        desc="Return cached anomaly indices per column (compact).",
    )
    def get_cached_anomalies(self, column: Optional[str] = None) -> str:
        """Retrieve cached anomaly indices (for internal/agent use)."""
        if column:
            return _safe_json(
                {
                    "indices": self.ctx.anomaly_store.get(column, []),
                    "meta": self.ctx.anomaly_meta.get(column, {}),
                }
            )
        return _safe_json(
            {"indices": self.ctx.anomaly_store, "meta": self.ctx.anomaly_meta}
        )

    @as_tool(
        name="change_points",
        desc="Detect structural breaks (ruptures PELT L2, per column).",
    )
    def change_points(
        self, penalty: Union[float, str] = 0, column: Optional[str] = None
    ) -> str:
        """Detect change points using PELT (multi-aware)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        if not RUPTURES_OK:
            return _safe_json({"status": "error", "error": "ruptures not available"})
        cols = [column] if column else self.ctx.value_columns
        out = {}
        pen = float(penalty) if penalty else None
        for col in cols:
            if col not in self.ctx.data.columns:
                continue
            y = self.ctx.data[col].values.astype(float)
            try:
                algo = rpt.Pelt(model="l2").fit(y)
                bks = algo.predict(pen=pen)
                n = len(y)
                cp = [int(b) for b in bks if b < n]
                cps = [
                    {"index": c, "date": str(self.ctx.data["date"].iloc[c])} for c in cp
                ]
                out[col] = {"status": "ok", "change_points": cps, "penalty": pen}
            except Exception as e:
                out[col] = {"status": "error", "error": str(e)}
        return _safe_json(out)


# ========== Forecast baselines ==========
class ForecastBaselineTools(ToolBase):
    """Classical forecasting baselines (with intervals via bootstrap)."""

    def _bootstrap_intervals(
        self,
        fc_vals: np.ndarray,
        y_hist: np.ndarray,
        n_boot: int = 100,
        level: float = 80,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap simple intervals for baselines."""
        if len(y_hist) < 10:
            return np.full_like(fc_vals, np.nan), np.full_like(fc_vals, np.nan)
        resids = y_hist[-min(50, len(y_hist)) :] - np.mean(
            y_hist[-min(50, len(y_hist)) :]
        )
        boots = np.array(
            [
                fc_vals + np.random.choice(resids, size=len(fc_vals))
                for _ in range(n_boot)
            ]
        )
        lower = np.percentile(boots, (100 - level) / 2, axis=0)
        upper = np.percentile(boots, 100 - (100 - level) / 2, axis=0)
        return lower, upper

    @as_tool(
        name="simple_forecast",
        desc="moving_average|linear_trend — kept for backward-compat (with intervals).",
    )
    def simple_forecast(
        self, method: str, periods: Union[int, str], with_intervals: bool = True
    ) -> str:
        """Simple one-value extrapolation forecasts (with optional intervals)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        n_periods = int(periods)
        if n_periods <= 0:
            return _safe_json({"status": "error", "error": "periods must be positive"})
        values = self.ctx.data["value"].values.astype(float)  # Univariate
        last_date = self.ctx.data["date"].iloc[-1]
        freq = self.ctx.freq_cache or _infer_freq_safe(self.ctx.data["date"]) or "D"
        if method == "moving_average":
            window = min(10, len(values))
            forecast_value = float(values[-window:].mean())
            fc = np.repeat(forecast_value, n_periods)
        elif method == "linear_trend":
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            fc = np.polyval(coeffs, np.arange(len(values), len(values) + n_periods))
        else:
            return _safe_json({"status": "error", "error": "Unknown method"})
        future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[
            1:
        ]
        lower, upper = (
            self._bootstrap_intervals(fc, values) if with_intervals else (None, None)
        )
        forecast_list = [
            {"date": str(d), "value": float(v)} for d, v in zip(future_dates, fc)
        ]
        if with_intervals:
            for i in range(len(forecast_list)):
                forecast_list[i]["lower"] = float(lower[i])
                forecast_list[i]["upper"] = float(upper[i])
        return _safe_json(
            {"method": method, "periods": n_periods, "forecast": forecast_list}
        )

    @as_tool(
        name="forecast_baselines",
        desc="naive|snaive|moving_avg|holt_winters|sarimax (with intervals).",
    )
    def forecast_baselines(
        self,
        method: str,
        periods: Union[int, str],
        season_length: Union[int, str] = 0,
        with_intervals: bool = True,
    ) -> str:
        """Advanced baselines: naive, seasonal naive, MA, ETS, SARIMAX (with optional intervals)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        h = int(periods)
        s = int(season_length) if season_length else None
        y = self.ctx.data["value"].values.astype(float)  # Univariate
        last_date = self.ctx.data["date"].iloc[-1]
        freq = self.ctx.freq_cache or _infer_freq_safe(self.ctx.data["date"]) or "D"

        def fc_naive(train):
            return np.repeat(train[-1], h)

        def fc_snaive(train, s_):
            if len(train) < s_:
                return np.repeat(train[-1], h)
            pattern = train[-s_:]
            reps = int(np.ceil(h / s_))
            return np.tile(pattern, reps)[:h]

        def fc_ma(train, k=7):
            return np.repeat(np.mean(train[-min(k, len(train)) :]), h)

        def fc_ets(train, s_):
            if not STATSM_OK:
                raise RuntimeError("statsmodels unavailable")
            train_ = train + (1e-9 if np.std(train) == 0 else 0)
            trend = "add" if len(train_) > 3 else None
            seasonal = "add" if (s_ and s_ >= 2) else None
            model = sm.tsa.ExponentialSmoothing(
                train_,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=s_ if seasonal else None,
            )
            fit = model.fit(optimized=True, use_brute=True)
            return fit.forecast(h)

        def fc_sarimax(train, s_):
            if not STATSM_OK:
                raise RuntimeError("statsmodels unavailable")
            p = d = q = 1
            P = D = Q = 1 if (s_ and s_ >= 2) else 0
            seas_order = (P, D, Q, s_) if (P or Q or D) else None
            fit = sm.tsa.statespace.SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=seas_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            return fit.forecast(h)

        try:
            if method == "naive":
                fc_vals = fc_naive(y)
            elif method == "snaive":
                s_eff = s or _guess_season_length(y)
                fc_vals = fc_snaive(y, s_eff)
            elif method == "moving_avg":
                fc_vals = fc_ma(y, k=s or 7)
            elif method == "holt_winters":
                s_eff = s or _guess_season_length(y)
                fc_vals = fc_ets(y, s_eff)
            elif method == "sarimax":
                s_eff = s or _guess_season_length(y)
                fc_vals = fc_sarimax(y, s_eff)
            else:
                return _safe_json(
                    {"status": "error", "error": f"Unknown method: {method}"}
                )
        except Exception as e:
            return _safe_json({"status": "error", "error": str(e)})
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]
        lower, upper = (
            self._bootstrap_intervals(fc_vals, y) if with_intervals else (None, None)
        )
        forecast_list = [
            {"date": str(d), "value": float(v)} for d, v in zip(future_dates, fc_vals)
        ]
        if with_intervals:
            for i in range(len(forecast_list)):
                forecast_list[i]["lower"] = float(lower[i])
                forecast_list[i]["upper"] = float(upper[i])
        return _safe_json(
            {"status": "ok", "method": method, "horizon": h, "forecast": forecast_list}
        )

    @as_tool(
        name="ensemble_forecast",
        desc="Average an ensemble of baselines with auto-weights from CV (intelligent).",
    )
    def ensemble_forecast(
        self, methods: List[str], periods: Union[int, str], with_intervals: bool = True
    ) -> str:
        """Ensemble of baselines (auto-weighted by rolling CV MAE)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        h = int(periods)
        # Auto-weights via quick rolling eval
        weights = []
        for m in methods:
            eval_json = self.rolling_eval(m, h, min_train=50)  # Short eval for speed
            parsed = _try_parse_json(eval_json)
            mae = parsed.get("metrics", {}).get("mae", np.inf)
            weights.append(1.0 / (mae + 1e-6))  # Inverse MAE
        weights = np.array(weights) / np.sum(weights)
        forecasts = []
        for i, m in enumerate(methods):
            fc_json = self.forecast_baselines(m, h, with_intervals=False)
            parsed = _try_parse_json(fc_json)
            if parsed.get("status") == "ok":
                fc_vals = [f["value"] for f in parsed["forecast"]]
                forecasts.append(np.array(fc_vals) * weights[i])
        if not forecasts:
            return _safe_json(
                {"status": "error", "error": "No valid forecasts for ensemble"}
            )
        ensemble_fc = np.sum(forecasts, axis=0)
        last_date = self.ctx.data["date"].iloc[-1]
        freq = self.ctx.freq_cache or _infer_freq_safe(self.ctx.data["date"]) or "D"
        future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]
        y_hist = self.ctx.data["value"].values.astype(float)  # Univariate
        lower, upper = (
            self._bootstrap_intervals(ensemble_fc, y_hist)
            if with_intervals
            else (None, None)
        )
        forecast_list = [
            {"date": str(d), "value": float(v)}
            for d, v in zip(future_dates, ensemble_fc)
        ]
        if with_intervals:
            for i in range(len(forecast_list)):
                forecast_list[i]["lower"] = float(lower[i])
                forecast_list[i]["upper"] = float(upper[i])
        return _safe_json(
            {
                "status": "ok",
                "method": "auto_weighted_ensemble",
                "methods": methods,
                "weights": weights.tolist(),
                "horizon": h,
                "forecast": forecast_list,
            }
        )

    @as_tool(
        name="suggest_horizon",
        desc="Suggest a reasonable short-term horizon (per series guess).",
    )
    def suggest_horizon(self) -> str:
        """Suggest forecast horizon based on season length (multi-aware aggregate)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        seasons = []
        for col in self.ctx.value_columns:
            y = self.ctx.data[col].values.astype(float)
            seasons.append(_guess_season_length(y))
        s = int(np.mean(seasons)) if seasons else 7
        h = int(min(max(s, 7), max(14, 2 * s), 365))
        return _safe_json({"suggested_horizon": h, "avg_season_length_guess": s})

    @as_tool(
        name="rolling_eval",
        desc="Light rolling-origin evaluation of a baseline (univariate).",
    )
    def rolling_eval(
        self,
        method: str,
        horizon: Union[int, str],
        min_train: Union[int, str] = 100,
        season_length: Union[int, str] = 0,
    ) -> str:
        """Rolling window evaluation of baseline (univariate)."""
        if not self.ctx.loaded or self.ctx.is_multivariate:
            return _safe_json(
                {
                    "status": "error",
                    "error": "Univariate data required for rolling eval",
                }
            )
        h = int(horizon)
        m = int(min_train)
        y = self.ctx.data["value"].values.astype(float)
        if len(y) < m + h:
            return _safe_json(
                {"status": "error", "error": "Data too short for rolling eval"}
            )
        s = int(season_length) if season_length else _guess_season_length(y)

        def make_forecaster():
            if method == "naive":
                return lambda tr: np.repeat(tr[-1], h)
            if method == "snaive":
                return lambda tr: np.tile(
                    tr[-s:] if len(tr) >= s else np.repeat(tr[-1], s), (h + s - 1) // s
                )[:h]
            if method == "moving_avg":
                k = max(3, min(14, s))
                return lambda tr: np.repeat(np.mean(tr[-min(k, len(tr)) :]), h)
            if method == "holt_winters" and STATSM_OK:

                def f(tr):
                    trend = "add" if len(tr) > 3 else None
                    seasonal = "add" if (s and s >= 2) else None
                    fit = sm.tsa.ExponentialSmoothing(
                        tr,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=s if seasonal else None,
                    ).fit(optimized=True, use_brute=True)
                    return fit.forecast(h)

                return f
            if method == "sarimax" and STATSM_OK:

                def f(tr):
                    p = d = q = 1
                    P = D = Q = 1 if (s and s >= 2) else 0
                    seas_order = (P, D, Q, s) if (P or D or Q) else None
                    fit = sm.tsa.statespace.SARIMAX(
                        tr,
                        order=(p, d, q),
                        seasonal_order=seas_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                    return fit.forecast(h)

                return f
            return lambda tr: np.repeat(tr[-1], h)

        maes, rmses, mapes = [], [], []
        step = max(1, h // 2)  # Overlapping windows for better eval
        for start in range(m, len(y) - h, step):
            tr = y[:start]
            true = y[start : start + h]
            pred = make_forecaster()(tr)
            if len(pred) < h:
                pred = np.pad(pred, (0, h - len(pred)), mode="edge")
            err = true - pred[:h]
            maes.append(np.mean(np.abs(err)))
            rmses.append(np.sqrt(np.mean(err**2)))
            mapes.append(np.mean(np.abs(err) / np.maximum(np.abs(true), 1e-8)) * 100.0)
        metrics = {
            "mae": float(np.mean(maes)) if maes else np.nan,
            "rmse": float(np.mean(rmses)) if rmses else np.nan,
            "mape": float(np.mean(mapes)) if mapes else np.nan,
            "n_windows": len(maes),
        }
        return _safe_json({"method": method, "horizon": h, "metrics": metrics})


# ========== Plotting ==========
class PlotTools(ToolBase):
    """Visualization tools with notebook-aware display (multi-series support)."""

    def _in_notebook(self) -> bool:
        """Return True if running inside an IPython/Jupyter-like kernel (robust)."""
        try:
            from IPython import get_ipython

            ip = get_ipython()
            if ip is None:
                return False
            if hasattr(ip, "kernel"):
                return True
            cfg = getattr(ip, "config", {})
            return bool(cfg)
        except Exception:
            return False

    def _save_or_show(
        self,
        fig: plt.Figure,
        save_path: Optional[str] = None,
        force_show: Optional[bool] = None,
    ) -> None:
        """
        Save plot if path provided, else show.
        Robust across Jupyter, VSCode, Molten. If unsure, we still try to show.
        """
        import matplotlib.pyplot as plt

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        is_nb = self._in_notebook()
        must_show = (force_show is True) or (save_path is None) or is_nb
        try:
            if is_nb:
                try:
                    from IPython.display import display

                    fig.canvas.draw()
                    display(fig)
                except Exception:
                    plt.show(block=False)
                    plt.pause(0.01)
            else:
                if must_show:
                    fig.canvas.draw()
                    plt.show(block=False)
                    plt.pause(0.05)
        finally:
            plt.close(fig)

    @as_tool(
        name="plot_series",
        desc="Plot the loaded time series data (overlay for multivariate).",
    )
    def plot_series(
        self, column: Optional[str] = None, save_path: Optional[str] = None
    ) -> str:
        """Plot time series (handles multi-columns via overlay)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        os.makedirs("plots", exist_ok=True)
        if save_path is None:
            save_path = "plots/series.pdf"
        cols = [column] if column else self.ctx.value_columns
        if not cols:
            return _safe_json({"status": "error", "error": "No columns to plot"})
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in cols:
            if col in self.ctx.data.columns:
                ax.plot(self.ctx.data["date"], self.ctx.data[col], label=col, alpha=0.8)
        ax.set_title(f"Time Series Plot{'' if len(cols) == 1 else ' (Multi-Series)'}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return _safe_json({"status": "ok", "columns": cols, "save_path": save_path})

    @as_tool(
        name="plot_diagnostics",
        desc="Plot diagnostics: histogram, ACF, PACF (for single column or first).",
    )
    def plot_diagnostics(
        self,
        column: Optional[str] = None,
        nlags: Union[int, str] = 40,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot diagnostics (requires statsmodels)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        if not STATSM_OK:
            return _safe_json(
                {
                    "status": "error",
                    "error": "statsmodels not available for diagnostics",
                }
            )
        os.makedirs("plots", exist_ok=True)
        if save_path is None:
            save_path = "plots/diagnostics.pdf"
        col = column or self.ctx.value_columns[0]
        if col not in self.ctx.data.columns:
            return _safe_json({"status": "error", "error": f"Column '{col}' not found"})
        y = self.ctx.data[col].dropna().values
        if len(y) < 10:
            return _safe_json(
                {"status": "error", "error": "Insufficient data for diagnostics"}
            )
        n_lags = int(nlags)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))
        # Histogram
        axes[0].hist(y, bins=min(30, len(y) // 5), alpha=0.7, edgecolor="black")
        axes[0].set_title(f"Histogram - {col}")
        axes[0].set_ylabel("Frequency")
        # ACF
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(y, lags=n_lags, ax=axes[1], title=f"ACF - {col}")
        axes[1].set_ylabel("ACF")
        # PACF
        from statsmodels.graphics.tsaplots import plot_pacf

        plot_pacf(y, lags=n_lags, ax=axes[2], title=f"PACF - {col}")
        axes[2].set_ylabel("PACF")
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return _safe_json(
            {"status": "ok", "column": col, "nlags": n_lags, "save_path": save_path}
        )

    @as_tool(
        name="plot_decomposition",
        desc="Plot STL decomposition: observed, trend, seasonal, residual.",
    )
    def plot_decomposition(
        self,
        column: Optional[str] = None,
        period: Union[int, str] = 7,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot STL decomposition (requires statsmodels)."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        if not STATSM_OK:
            return _safe_json(
                {"status": "error", "error": "statsmodels not available for STL"}
            )
        os.makedirs("plots", exist_ok=True)
        if save_path is None:
            col = column or self.ctx.value_columns[0]
            save_path = f"plots/decomposition_{col}.pdf"
        col = column or self.ctx.value_columns[0]
        if col not in self.ctx.data.columns:
            return _safe_json({"status": "error", "error": f"Column '{col}' not found"})
        y = self.ctx.data[col].dropna().values
        if len(y) < 2 * int(period):
            return _safe_json(
                {"status": "error", "error": "Insufficient data for STL decomposition"}
            )
        per = int(period)
        from statsmodels.tsa.seasonal import STL as _STL

        stl = _STL(y, period=per, robust=True).fit()
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(stl.observed)
        axes[0].set_title(f"Observed - {col}")
        axes[1].plot(stl.trend)
        axes[1].set_title("Trend")
        axes[2].plot(stl.seasonal)
        axes[2].set_title("Seasonal")
        axes[3].plot(stl.resid)
        axes[3].set_title("Residual")
        axes[3].set_xlabel("Time")
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return _safe_json(
            {"status": "ok", "column": col, "period": per, "save_path": save_path}
        )

    @as_tool(
        name="plot_time_series_cv",
        desc="Overlay y vs prediction from a NeuralForecast cross_validation DataFrame (multi-series).",
    )
    def plot_time_series_cv(
        self,
        model_name: Optional[str] = None,
        series_ids: Optional[List[str]] = None,
        max_series: int = 4,
        save_path: Optional[str] = None,
        cv_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Plot y (ground truth) and predictions from a NeuralForecast cross_validation output (handles multi unique_id).
        """
        os.makedirs("plots", exist_ok=True)
        if save_path is None:
            save_path = "plots/plot_time_series_cv.pdf"
        sub = cv_df if cv_df is not None else self.ctx.nf.cv_full
        pred_col = model_name if model_name is not None else self.ctx.nf.pred_col
        if sub is None or pred_col is None:
            return _safe_json(
                {
                    "status": "error",
                    "error": "cv_df or cached NF cv_full/pred_col not available",
                }
            )
        need = {"unique_id", "ds", "y", pred_col}
        miss = need - set(sub.columns)
        if miss:
            return _safe_json(
                {"status": "error", "error": f"cv_df missing columns: {sorted(miss)}"}
            )
        all_ids = sub["unique_id"].unique().tolist()
        if series_ids is None:
            series_ids = all_ids[:max_series]
        else:
            series_ids = [sid for sid in series_ids if sid in all_ids]
            if not series_ids:
                return _safe_json(
                    {"status": "error", "error": "Provided series_ids not found in cv_df"}
                )
        fig, axes = plt.subplots(
            len(series_ids), 1, figsize=(10.5, 3.0 * len(series_ids)), sharex=True
        )
        if len(series_ids) == 1:
            axes = [axes]
        for i, sid in enumerate(series_ids):
            df_s = (
                sub.loc[sub["unique_id"] == sid, ["ds", "y", pred_col]]
                .sort_values("ds")
                .copy()
            )
            df_s["ds"] = pd.to_datetime(df_s["ds"])
            ax = axes[i]
            ax.plot(df_s["ds"], df_s["y"], lw=1.8, label="y (true)")
            ax.plot(df_s["ds"], df_s[pred_col], lw=1.8, label=f"ŷ ({pred_col})")
            ax.set_title(f"{sid} — Ground truth vs Prediction")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            if i == len(series_ids) - 1:
                ax.set_xlabel("Date")
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return _safe_json(
            {
                "status": "ok",
                "message": f"Plotted {len(series_ids)} series",
                "model": pred_col,
                "series_ids": series_ids,
                "save_path": save_path,
            }
        )

    @as_tool(
        name="plot_cv_sliding_windows",
        desc="Plot aggregated CV predictions with val/test splits (for NF results).",
    )
    def plot_cv_sliding_windows(
        self,
        model_name: Optional[str] = None,
        cv_df: Optional[pd.DataFrame] = None,
        date_val_start: Optional[pd.Timestamp] = None,
        date_test_start: Optional[pd.Timestamp] = None,
        date_end: Optional[pd.Timestamp] = None,
        title: str = "",
        save_path: Optional[str] = None,
    ) -> str:
        """Plot aggregated CV results over time with split lines."""
        os.makedirs("plots", exist_ok=True)
        if save_path is None:
            save_path = "plots/cv_sliding_windows.pdf"
        sub = cv_df if cv_df is not None else self.ctx.nf.cv_full
        pred_col = model_name if model_name is not None else self.ctx.nf.pred_col
        if sub is None or pred_col is None:
            return _safe_json(
                {"status": "error", "error": "CV data or pred_col not available"}
            )
        cv_agg = sub.groupby("ds")[["y", pred_col]].mean().reset_index()
        cv_agg["ds"] = pd.to_datetime(cv_agg["ds"])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cv_agg["ds"], cv_agg["y"], lw=1.8, label="y (true)", alpha=0.8)
        ax.plot(
            cv_agg["ds"], cv_agg[pred_col], lw=1.8, label=f"ŷ ({pred_col})", alpha=0.8
        )
        if date_val_start:
            ax.axvline(
                date_val_start, color="green", linestyle="--", alpha=0.7, label="Val Start"
            )
        if date_test_start:
            ax.axvline(
                date_test_start, color="red", linestyle="--", alpha=0.7, label="Test Start"
            )
        if date_end:
            ax.axvline(date_end, color="orange", linestyle="--", alpha=0.7, label="End")
        ax.set_title(title or f"CV Sliding Windows — {pred_col}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value (Aggregated)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        self._save_or_show(fig, save_path)
        return _safe_json({"status": "ok", "model": pred_col, "save_path": save_path})


# ========== NeuralForecast model selection ==========
class NeuralForecastTools(ToolBase):
    """Neural forecasting with model selection (multivariate/multi-series support)."""

    def _default_nf_models(self) -> List[Tuple[type, str]]:
        """Default list of NF models (modern selection)."""
        if not NF_OK:
            return []
        base = [
            (LSTM, "LSTM"),
            (PatchTST, "PatchTST"),
            (NHITS, "NHITS"),
        ]
        for cls in _NEW_MODELS:
            if cls not in [c[0] for c in base]:
                base.append((cls, cls.__name__))
        return base

    @as_tool(
        name="forecast_neuralforecast",
        desc="Select best NF model on validation, refit, forecast with intervals (multi-series).",
    )
    def forecast_neuralforecast(
        self,
        horizon: Union[int, str],
        input_size: Union[int, str],
        val_frac: Union[float, str] = 0.2,
        test_frac: Union[float, str] = 0.2,
        max_steps: Union[int, str] = 100,
        metric: str = "MSE",
        fallback_baseline: str = "holt_winters",
        with_intervals: bool = True,
        level: Union[int, str] = 80,
    ) -> str:
        """Run NF model selection and CV forecasting; supports multi-series; intervals via predict on refit."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        if not NF_OK:
            fb = ForecastBaselineTools(self.ctx)
            return fb.forecast_baselines(
                fallback_baseline, horizon, with_intervals=with_intervals
            )
        try:
            h = int(float(horizon))
            ctxw = int(float(input_size))
            vfrac = float(val_frac)
            tfrac = float(test_frac)
            steps = int(float(max_steps))
            lvl = int(float(level))
        except Exception:
            return _safe_json(
                {"status": "error", "error": "invalid numeric parameters"}
            )
        metric = str(metric).upper()
        if metric not in {"MSE", "RMSE", "MAE", "MAPE"}:
            metric = "MSE"

        df_wide = self.ctx.data.sort_values("date").copy()
        value_cols = self.ctx.value_columns
        if self.ctx.is_multivariate:
            df_long = pd.melt(
                df_wide, id_vars=["date"], var_name="unique_id", value_name="y"
            )
            df_long = df_long.rename(columns={"date": "ds"}).sort_values(
                ["unique_id", "ds"]
            )
            self.ctx.nf.unique_ids = value_cols
        else:
            df_long = (
                df_wide.rename(columns={"value": "y"})
                .assign(unique_id="series")
                .rename(columns={"date": "ds"})
            )
            self.ctx.nf.unique_ids = ["series"]

        y_all_per_series = {
            uid: df_long[df_long["unique_id"] == uid]["y"].values
            for uid in df_long["unique_id"].unique()
        }
        T_per_series = {uid: len(y) for uid, y in y_all_per_series.items()}
        min_T = min(T_per_series.values())
        if min_T < (3 * h + ctxw):
            return _safe_json(
                {
                    "status": "error",
                    "error": f"Shortest series too short (min_T={min_T}) for h={h}, input_size={ctxw}.",
                }
            )
        norm_params = {}
        for uid in df_long["unique_id"].unique():
            y_ser = y_all_per_series[uid]
            if SKLEARN_OK:
                scaler = RobustScaler()
                y_norm_ser = scaler.fit_transform(y_ser.reshape(-1, 1)).flatten()
                norm_params[uid] = {"scaler_type": "robust"}
            else:
                mu, sd = np.mean(y_ser), np.std(y_ser) or 1.0
                y_norm_ser = (y_ser - mu) / sd
                norm_params[uid] = {"mu": mu, "sd": sd, "scaler_type": "standard"}
            df_long.loc[df_long["unique_id"] == uid, "y"] = y_norm_ser

        freq = self.ctx.freq_cache or _infer_freq_safe(df_wide["date"]) or "D"
        avg_T = np.mean(list(T_per_series.values()))
        test_len = max(h, int(round(avg_T * tfrac)))
        val_len = max(h, int(round(avg_T * vfrac)))
        if test_len + val_len >= avg_T - h:
            val_len = max(h, min(val_len, (avg_T - test_len) // 2))
        cut_test = int(avg_T - test_len)
        cut_val = max(ctxw + h, cut_test - val_len)
        date_val_start = df_wide["date"].iloc[cut_val]
        date_test_start = df_wide["date"].iloc[cut_test]
        date_end = df_wide["date"].iloc[-1]

        models = self._default_nf_models()
        scores = []
        n_windows_val = max(1, (cut_test - cut_val) // h)
        step_size = h
        for ModelClass, name in models:
            try:
                kwargs = {
                    "h": h,
                    "input_size": ctxw,
                    "max_steps": steps,
                    "val_check_steps": 10,
                    "scaler_type": "robust",
                    "alias": name,
                }
                try:
                    model = ModelClass(**kwargs)
                except TypeError:
                    model = ModelClass(h=h, input_size=ctxw, max_steps=steps, alias=name)
                nf = NeuralForecast(models=[model], freq=str(freq))
                cv = nf.cross_validation(
                    df=df_long[df_long["ds"] < date_test_start],
                    n_windows=n_windows_val,
                    step_size=step_size,
                )
                pred_candidates = [
                    c for c in cv.columns if c not in ("unique_id", "ds", "y", "cutoff")
                ]
                pred_col = (
                    name
                    if name in pred_candidates
                    else (pred_candidates[0] if pred_candidates else None)
                )
                if pred_col is None:
                    continue
                cv_val = cv[(cv["ds"] >= date_val_start) & (cv["ds"] < date_test_start)]
                all_true, all_pred = [], []
                for uid in cv_val["unique_id"].unique():
                    ser = cv_val[cv_val["unique_id"] == uid]
                    all_true.extend(ser["y"].values)
                    all_pred.extend(ser[pred_col].values)
                m = _metrics(np.array(all_true), np.array(all_pred))
                pick = m[metric]
                scores.append((pick, name, m))
            except Exception:
                continue
        if not scores:
            fb = ForecastBaselineTools(self.ctx)
            return fb.forecast_baselines(
                fallback_baseline, horizon, with_intervals=with_intervals
            )
        scores.sort(key=lambda x: x[0])
        best_score, best_name, best_val_metrics = scores[0]
        best_cls = {nm: cls for cls, nm in models if nm == best_name}[best_name]

        try:
            kwargs = {
                "h": h,
                "input_size": ctxw,
                "max_steps": steps,
                "val_check_steps": 10,
                "scaler_type": "robust",
                "alias": best_name,
            }
            try:
                best_model = best_cls(**kwargs)
            except TypeError:
                best_model = best_cls(h=h, input_size=ctxw, max_steps=steps, alias=best_name)
            nf_best = NeuralForecast(models=[best_model], freq=str(freq))
        except Exception:
            fb = ForecastBaselineTools(self.ctx)
            return fb.forecast_baselines(
                fallback_baseline, horizon, with_intervals=with_intervals
            )

        n_windows_test = max(1, int(np.ceil((avg_T - cut_test) / float(h))))
        try:
            cv_full_norm = nf_best.cross_validation(
                df=df_long, n_windows=n_windows_test, step_size=h
            )
        except Exception:
            fb = ForecastBaselineTools(self.ctx)
            return fb.forecast_baselines(
                fallback_baseline, horizon, with_intervals=with_intervals
            )
        pred_candidates = [
            c
            for c in cv_full_norm.columns
            if c not in ("unique_id", "ds", "y", "cutoff")
        ]
        pred_col = (
            best_name
            if best_name in pred_candidates
            else (pred_candidates[0] if pred_candidates else None)
        )
        if pred_col is None:
            fb = ForecastBaselineTools(self.ctx)
            return fb.forecast_baselines(
                fallback_baseline, horizon, with_intervals=with_intervals
            )

        cv_full_orig = cv_full_norm.copy()
        for uid in cv_full_orig["unique_id"].unique():
            params = norm_params[uid]
            if params["scaler_type"] == "standard":
                mu, sd = params["mu"], params["sd"]
                cv_full_orig.loc[cv_full_orig["unique_id"] == uid, "y"] *= sd
                cv_full_orig.loc[cv_full_orig["unique_id"] == uid, "y"] += mu
                cv_full_orig.loc[cv_full_orig["unique_id"] == uid, pred_col] *= sd
                cv_full_orig.loc[cv_full_orig["unique_id"] == uid, pred_col] += mu
            # If robust, keep in normalized scale (or inverse with stored scalers if available)

        cv_test = cv_full_orig[
            (cv_full_orig["ds"] >= date_test_start) & (cv_full_orig["ds"] <= date_end)
        ].copy()
        all_true, all_pred = [], []
        for uid in cv_test["unique_id"].unique():
            ser = cv_test[cv_test["unique_id"] == uid]
            all_true.extend(ser["y"].values)
            all_pred.extend(ser[pred_col].values)
        test_orig_metrics = _metrics(np.array(all_true), np.array(all_pred))

        self.ctx.nf = NFState(
            best_model=best_name,
            cv_full=cv_full_orig,
            pred_col=pred_col,
            date_val_start=date_val_start,
            date_test_start=date_test_start,
            date_end=date_end,
            y_norm=None,
            dates=df_wide["date"],
            title=f"NeuralForecast Sliding Windows — Best on VAL: {best_name} ({metric}={best_score:.4g})",
            unique_ids=value_cols,
        )

        PlotTools(self.ctx).plot_time_series_cv(
            model_name=pred_col, cv_df=cv_full_orig, max_series=4
        )

        intervals = None
        if with_intervals:
            last_train_end = date_test_start
            train_df = df_long[df_long["ds"] < last_train_end]
            nf_best.fit(train_df)
            fut_df = pd.DataFrame(
                {
                    "unique_id": self.ctx.nf.unique_ids * h,
                    "ds": pd.date_range(
                        start=date_end + timedelta(days=1), periods=h, freq=freq
                    ),
                }
            )
            pred_df = nf_best.predict(fut_df, level=[lvl], insample=False)
            intervals = pred_df.to_dict("records")

        return _safe_json(
            {
                "status": "ok",
                "best_model": best_name,
                "val_metrics": best_val_metrics,
                "test_metrics": test_orig_metrics,
                "splits": {
                    "train_len": int(cut_val),
                    "val_len": int(cut_test - cut_val),
                    "test_len": int(avg_T - cut_test),
                    "horizon": int(h),
                    "input_size": int(ctxw),
                    "n_windows_val": int(n_windows_val),
                    "n_windows_test": int(n_windows_test),
                },
                "freq": str(freq),
                "normalization": norm_params,
                "intervals": intervals if with_intervals else None,
                "level": lvl if with_intervals else None,
            }
        )


# ========== One-shot "comprehensive analysis" + Intelligent Suggestions ==========
class AnalysisTools(ToolBase):
    """High-level analysis workflows (with suggestions)."""

    @as_tool(
        name="comprehensive_analysis",
        desc="Full analysis: stats, stationarity, seasonality, anomalies, plots (multi-aware).",
    )
    def comprehensive_analysis(
        self, include_plots: bool = True, save_dir: Optional[str] = None
    ) -> str:
        """Run full analysis pipeline (updated for multi)."""
        print("Running comprehensive analysis...")
        if not self.ctx.loaded:
            print("No data loaded.")
            return _safe_json({"status": "error", "error": "No data loaded"})
        results = {"steps": []}
        stats = _try_parse_json(
            StatsTools(self.ctx).compute_statistics(
                aggregate=not self.ctx.is_multivariate
            )
        )
        stationarity = _try_parse_json(StatsTools(self.ctx).stationarity_tests())
        trend = _try_parse_json(SeasonalityTools(self.ctx).detect_trend())
        results["steps"].append({"step": "statistics", "data": stats})
        results["steps"].append({"step": "stationarity", "data": stationarity})
        results["steps"].append({"step": "trend", "data": trend})
        print("Basic stats, stationarity, trend analysis done.")
        if STATSM_OK:
            seasonality = _try_parse_json(SeasonalityTools(self.ctx).stl_seasonality())
            acf_pacf = _try_parse_json(StatsTools(self.ctx).acf_pacf_peaks())
            results["steps"].append({"step": "seasonality", "data": seasonality})
            results["steps"].append({"step": "acf_pacf", "data": acf_pacf})
        if self.ctx.is_multivariate:
            multi_stats = _try_parse_json(
                StatsTools(self.ctx).compute_multivariate_stats()
            )
            results["steps"].append({"step": "multivariate_stats", "data": multi_stats})

        # Now returns compact counts; indices cached
        anomalies = _try_parse_json(
            AnomalyTools(self.ctx).detect_anomalies(method="zscore")
        )
        results["steps"].append({"step": "anomalies", "data": anomalies})
        print("Analysis steps completed.")

        if include_plots:
            print("Generating plots...")
            plot_dir = save_dir or "./plots"
            os.makedirs(plot_dir, exist_ok=True)
            print("Plot directory:", plot_dir)
            PlotTools(self.ctx).plot_diagnostics(
                save_path=f"{plot_dir}/diagnostics.pdf"
            )
            print("Diagnostics plot done.")
            if STATSM_OK:
                for col in self.ctx.value_columns[:3]:
                    PlotTools(self.ctx).plot_decomposition(
                        column=col, save_path=f"{plot_dir}/decomposition_{col}.pdf"
                    )
            PlotTools(self.ctx).plot_series(save_path=f"{plot_dir}/series.pdf")
            results["plots"] = {
                "dir": plot_dir,
                "generated": ["diagnostics.pdf", "series.pdf"]
                + (
                    [f"decomposition_{c}.pdf" for c in self.ctx.value_columns[:3]]
                    if STATSM_OK
                    else []
                ),
            }

        # Compact summary
        anomaly_rates = []
        if isinstance(anomalies, dict) and "summary" in anomalies:
            for col, s in anomalies["summary"].items():
                if isinstance(s, dict):
                    anomaly_rates.append(s.get("anomaly_rate", 0.0))
        results["summary"] = {
            "stationary": any(
                st.get("adf", {}).get("stationary", False)
                for st in stationarity.values()
            )
            if isinstance(stationarity, dict)
            else "unknown",
            "trend": (list(trend.values())[0].get("trend", "unknown") if trend else "unknown")
            if isinstance(trend, dict) and len(trend) > 0
            else "unknown",
            "avg_anomaly_rate": float(np.mean(anomaly_rates)) if anomaly_rates else 0.0,
            "n_records": len(self.ctx.data),
            "is_multivariate": self.ctx.is_multivariate,
        }
        return _safe_json(results)

    @as_tool(
        name="suggest_transformations",
        desc="Intelligent suggestions: Recommend transforms/differencing based on stats/skew/stationarity.",
    )
    def suggest_transformations(self) -> str:
        """LLM-agent friendly suggestions based on diagnostics."""
        if not self.ctx.loaded:
            return _safe_json({"status": "error", "error": "No data loaded"})
        suggestions = {
            "recommended": [],
            "reasons": {},
            "columns": self.ctx.value_columns,
        }
        stats = _try_parse_json(StatsTools(self.ctx).compute_statistics())
        stationarity = _try_parse_json(StatsTools(self.ctx).stationarity_tests())
        for col in self.ctx.value_columns:
            if col not in stats:
                continue
            s = stats[col]
            stat = stationarity.get(col, {})
            skew = s.get("skew", 0)
            adf_p = stat.get("adf", {}).get("pvalue", 1.0)
            recs = []
            if abs(skew) > 1.0:
                recs.append("log" if skew > 0 else "boxcox")
            if adf_p > 0.05:
                recs.append("detrend (order=3)")
            suggestions["recommended"].append({"column": col, "actions": recs})
            suggestions["reasons"][col] = {
                "skew": skew,
                "adf_pvalue": adf_p,
                "stationary": adf_p < 0.05,
            }
        return _safe_json(suggestions)
