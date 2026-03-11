"""SOTA structural mining and causality tools for timeseries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import json as _json_mod
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "Advanced Mining",
    "description": "Use for motifs, regimes, advanced pattern mining, and deeper sequence discovery tasks.",
    "workflow": [
        "Use advanced mining only after data is loaded and basic diagnostics are understood.",
        "Prefer targeted structural mining over broad shotgun analysis.",
    ],
}

if "_safe_json" not in globals():
    def _safe_json(obj):
        try:
            return _json_mod.dumps(obj)
        except Exception as e:
            return _json_mod.dumps({"status": "error", "error": str(e)})

@tool
def discover_motifs(window_size: int, column: str = None, top_k: int = 3):
    """Use when: You need to find repeating patterns without knowing their shape, or find structural anomalies/discords in time series context.

    Triggers: matrix profile, motifs, discords, find repeating pattern, find strange shape.
    Avoid when: You only want pointwise statistical anomalies (use detect_anomalies instead).
    Inputs: window_size (int, required): Length of pattern to look for; column (str, optional): Target series; top_k (int, optional): Number of motifs/discords to return.
    Returns: JSON with motifs (similar pairs indices) and discords (most unique subsequences).
    """
    if "ctx" not in globals() or getattr(ctx, "loaded", False) is False:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        import stumpy
    except ImportError:
        return _safe_json({"status": "error", "error": "stumpy library not available for matrix profiles."})

    col = column if column else ctx.value_columns[0]
    if col not in ctx.data.columns:
        return _safe_json({"status": "error", "error": f"Column {col} not found"})

    timeseries = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
    if len(timeseries) <= window_size * 2:
        return _safe_json({"status": "error", "error": "Time series is too short for this window size."})

    valid_mask = np.isfinite(timeseries)
    if not valid_mask.all():
         s = pd.Series(timeseries)
         timeseries = s.ffill().bfill().values

    try:
        mp = stumpy.stump(timeseries, m=int(window_size))
        
        motifs_idx = np.argsort(mp[:, 0])[:top_k]
        discords_idx = np.argsort(mp[:, 0])[::-1][:top_k]
        
        out_motifs = []
        for idx in motifs_idx:
            nn_idx = mp[idx, 1]
            out_motifs.append({
                "index1": int(idx),
                "index2": int(nn_idx),
                "distance": float(mp[idx, 0])
            })
            
        out_discords = []
        for idx in discords_idx:
             out_discords.append({
                "index": int(idx),
                "distance": float(mp[idx, 0])
             })
             
        return _safe_json({
            "status": "ok",
            "motifs": out_motifs,
            "discords": out_discords,
            "window_size": int(window_size)
        })
    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})

@tool
def causal_analysis(maxlag: int = 5):
    """Use when: You need to determine if one time series is predictive of another (Granger causality).

    Triggers: granger causality, causal impact, what causes what, leading indicators.
    Avoid when: The dataset only has a single time series.
    Inputs: maxlag (int, optional): Maximum number of lags to test (default 5).
    Returns: JSON with causation pairs and their lowest p-values across lags.
    """
    if "ctx" not in globals() or getattr(ctx, "loaded", False) is False:
        return _safe_json({"status": "error", "error": "No data loaded"})

    if not getattr(ctx, "is_multivariate", False):
        return _safe_json({"status": "error", "error": "Granger Causality requires multivariate data."})

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return _safe_json({"status": "error", "error": "statsmodels library not available."})

    df = ctx.data[ctx.value_columns].dropna()
    if len(df) <= maxlag * 3:
        return _safe_json({"status": "error", "error": "Not enough data points for the given maxlag."})

    cols = ctx.value_columns
    results = []

    for i, col_y in enumerate(cols):
        for j, col_x in enumerate(cols):
            if i == j:
                continue
            
            data = df[[col_y, col_x]]
            
            try:
                gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                min_p_val = 1.0
                best_lag = -1
                for lag, test_res in gc_res.items():
                    p_val = test_res[0]['ssr_ftest'][1]
                    if p_val < min_p_val:
                        min_p_val = p_val
                        best_lag = lag
                
                if min_p_val < 0.05:
                    results.append({
                        "causing": col_x,
                        "caused": col_y,
                        "min_p_value": float(min_p_val),
                        "best_lag": int(best_lag)
                    })
            except Exception:
                pass

    return _safe_json({
        "status": "ok",
        "causal_relations": sorted(results, key=lambda x: x["min_p_value"])
    })


__all__ = ["discover_motifs", "causal_analysis"]


__tools__ = [discover_motifs, causal_analysis]
