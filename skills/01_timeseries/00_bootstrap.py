from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# =============================================================================
# SKILLS BOOTSTRAP — helpers + Context
# Executed once before Tool sections are loaded
# =============================================================================
import numpy as np
import pandas as pd

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


def _regularize_series(
    df: pd.DataFrame, freq: Optional[str], method: str
) -> pd.DataFrame:
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
        "MAPE": float(
            np.mean(np.abs(err) / np.maximum(np.abs(y_true[ok]), 1e-8)) * 100.0
        ),
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
