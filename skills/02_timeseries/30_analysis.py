from __future__ import annotations

from typing import Literal

if "llm" not in globals():

    class _NoOpLLM:  # no-op fallback for standalone use; registry injects real llm
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()


@llm.tool(description="Compute descriptive statistics.")
def compute_statistics(aggregate=False):
    """Use when: Compute descriptive statistics.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: aggregate (bool, optional): If True, aggregate all columns (default False).
    Returns: JSON with statistics.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    value_cols = ctx.value_columns

    if aggregate:
        values = pd.concat([ctx.data[col] for col in value_cols], axis=0).dropna()
        stats = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        }
        return _safe_json(stats)

    stats = {}
    for col in value_cols:
        values = pd.to_numeric(ctx.data[col], errors="coerce")
        stats[col] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        }

    return _safe_json(stats)


@llm.tool(description="ADF and Ljung-Box tests (requires statsmodels).")
def stationarity_tests(lags=0, column=None):
    """Use when: ADF and Ljung-Box tests (requires statsmodels).

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: lags (int, optional): Lags for Ljung-Box (default auto); column (str, optional): Specific column (default all).
    Returns: JSON with test results.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        import statsmodels.api as sm
    except ImportError:
        return _safe_json({"status": "error", "error": "statsmodels not available"})

    cols = [column] if column else ctx.value_columns
    out = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
        y = y[np.isfinite(y)]
        col_out = {"adf": None, "ljung_box": None, "notes": []}

        if len(y) < 12:
            col_out["notes"].append("insufficient finite observations (<12)")
            out[col] = col_out
            continue

        try:
            adf = sm.tsa.stattools.adfuller(y, autolag="AIC")
            col_out["adf"] = {
                "stat": float(adf[0]),
                "pvalue": float(adf[1]),
                "stationary": adf[1] < 0.05,
            }
        except Exception as exc:
            col_out["notes"].append(f"ADF failed: {exc}")

        try:
            lb_lags = int(lags) if lags else min(40, max(10, len(y) // 10))
            lb = sm.stats.acorr_ljungbox(y, lags=[lb_lags], return_df=True)
            col_out["ljung_box"] = {
                "lags": lb_lags,
                "stat": float(lb["lb_stat"].iloc[0]),
                "pvalue": float(lb["lb_pvalue"].iloc[0]),
            }
        except Exception as exc:
            col_out["notes"].append(f"Ljung-Box failed: {exc}")

        out[col] = col_out

    return _safe_json(out)


@llm.tool(description="Top ACF/PACF lags by magnitude.")
def acf_pacf_peaks(nlags=40, topk=5, column=None):
    """Use when: Top ACF/PACF lags by magnitude.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: nlags (int, optional): Number of lags (default 40); topk (int, optional): Top K peaks (default 5); column (str, optional): Specific column (default all).
    Returns: JSON with top peaks.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        import statsmodels.tsa.api as smt
    except ImportError:
        return _safe_json({"status": "error", "error": "statsmodels not available"})

    n_lags = int(nlags)
    top_k = int(topk)
    cols = [column] if column else ctx.value_columns
    out = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
        y = y[np.isfinite(y)]
        col_out = {"acf_top": [], "pacf_top": [], "notes": []}

        if len(y) < 8:
            col_out["notes"].append("insufficient finite observations (<8)")
            out[col] = col_out
            continue

        n_eff = min(n_lags, max(2, len(y) // 2))

        try:
            acf = smt.stattools.acf(y, nlags=n_eff, fft=True)
            pacf = smt.stattools.pacf(y, nlags=n_eff, method="yw")

            lag_acf = [
                (lag, float(abs(val))) for lag, val in enumerate(acf[1:], start=1)
            ]
            lag_pacf = [
                (lag, float(abs(val))) for lag, val in enumerate(pacf[1:], start=1)
            ]

            col_out["acf_top"] = sorted(lag_acf, key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            col_out["pacf_top"] = sorted(lag_pacf, key=lambda x: x[1], reverse=True)[
                :top_k
            ]
        except Exception as exc:
            col_out["notes"].append(f"ACF/PACF failed: {exc}")

        out[col] = col_out

    return _safe_json(out)


@llm.tool(description="Correlations and per-series stats for multivariate data.")
def compute_multivariate_stats():
    """Use when: Correlations and per-series stats for multivariate data.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: none.
    Returns: JSON with correlations and stats.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.is_multivariate:
        return _safe_json(
            {
                "status": "error",
                "error": "Not multivariate data (need >=2 value columns)",
            }
        )

    df_num = ctx.data[ctx.value_columns].dropna()
    corr = df_num.corr().to_dict()

    per_series = {}
    for col in ctx.value_columns:
        values = pd.to_numeric(ctx.data[col], errors="coerce")
        per_series[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

        try:
            import statsmodels.api as sm

            per_series[col]["adf_pvalue"] = float(
                sm.tsa.stattools.adfuller(values.dropna())[1]
            )
        except Exception:
            per_series[col]["adf_pvalue"] = None

    return _safe_json({"correlations": corr, "per_series": per_series})


@llm.tool(description="Detect overall trend direction.")
def detect_trend(column=None):
    """Use when: Detect overall trend direction.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: column (str, optional): Specific column (default all).
    Returns: JSON with trend info.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns
    out = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
        valid = np.isfinite(y)
        if int(valid.sum()) < 2:
            out[col] = {"status": "error", "error": "insufficient finite observations"}
            continue

        x = np.arange(len(y))[valid]
        y = y[valid]
        slope = float(np.polyfit(x, y, 1)[0])

        start_val, end_val = float(y[0]), float(y[-1])
        pct_change = float(((end_val - start_val) / (abs(start_val) + 1e-12)) * 100.0)

        trend = "flat" if abs(slope) < 1e-3 else ("upward" if slope > 0 else "downward")

        out[col] = {
            "trend": trend,
            "slope": slope,
            "percent_change": pct_change,
            "start_value": start_val,
            "end_value": end_val,
        }

    return _safe_json(out)


@llm.tool(description="STL-based seasonality strength.")
def stl_seasonality(period=0, column=None):
    """Use when: STL-based seasonality strength.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: period (int, optional): Seasonal period (default auto); column (str, optional): Specific column (default all).
    Returns: JSON with seasonality info.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        return _safe_json({"status": "error", "error": "statsmodels not available"})

    cols = [column] if column else ctx.value_columns
    out = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)
        period_used = int(period) if period else _guess_season_length(y)

        try:
            stl = STL(y, period=max(2, period_used), robust=True).fit()
            var = np.var
            seas_strength = 1.0 - (
                var(stl.resid) / (var(stl.resid + stl.seasonal) + 1e-12)
            )
            strength = float(max(0.0, min(1.0, seas_strength)))

            out[col] = {
                "period_used": int(period_used),
                "seasonal_strength": strength,
                "interpretation": "Strong"
                if strength > 0.64
                else "Moderate"
                if strength > 0.36
                else "Weak",
            }
        except Exception as exc:
            out[col] = {
                "status": "error",
                "error": str(exc),
                "period_attempted": period_used,
            }

    return _safe_json(out)


@llm.tool(description="Detect anomalies - returns counts only (indices cached).")
def detect_anomalies(
    method: Literal["zscore", "iqr", "hampel", "stl_resid", "iforest"],
    threshold=3.0,
    period=0,
    column=None,
):
    """Use when: Detect anomalies - returns counts only (indices cached).

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: method (str, required): 'zscore', 'iqr', 'hampel', 'stl_resid', 'iforest'; threshold (float, optional): Threshold value (default 3.0); period (int, optional): Seasonal period; column (str, optional): Specific column (default all).
    Returns: JSON with anomaly counts/rates.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    mth = str(method).strip().lower()
    cols = [column] if column else ctx.value_columns
    summary = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
        mask = np.zeros_like(y, dtype=bool)
        notes = []

        try:
            if mth == "zscore":
                thr = float(threshold)
                mu = np.nanmean(y)
                sd = np.nanstd(y) + 1e-12
                mask = np.abs((y - mu) / sd) > thr
            elif mth == "iqr":
                thr = float(threshold)
                q1, q3 = np.nanpercentile(y, 25), np.nanpercentile(y, 75)
                iqr = q3 - q1
                lb, ub = q1 - thr * iqr, q3 + thr * iqr
                mask = (y < lb) | (y > ub)
            elif mth == "hampel":
                w = 15
                ns = 3.0
                w = w if w % 2 == 1 else w + 1
                s = pd.Series(y)
                med = s.rolling(w, center=True).median()
                mad = (s - med).abs().rolling(w, center=True).median()
                denom = (1.4826 * mad.replace(0, np.nan)).values
                mask = np.abs(s.values - med.values) > ns * np.nan_to_num(
                    denom, nan=np.inf
                )
            elif mth == "stl_resid":
                try:
                    from statsmodels.tsa.seasonal import STL

                    p = int(period) if period else _guess_season_length(y)
                    resid = STL(y, period=max(2, p), robust=True).fit().resid
                    q1, q3 = np.nanpercentile(resid, 25), np.nanpercentile(resid, 75)
                    iqr = q3 - q1
                    lb, ub = q1 - 3 * iqr, q3 + 3 * iqr
                    mask = (resid < lb) | (resid > ub)
                except ImportError:
                    notes.append("statsmodels not available")
            elif mth == "iforest":
                try:
                    from sklearn.ensemble import IsolationForest

                    x = y.reshape(-1, 1)
                    labels = IsolationForest(
                        n_estimators=200, contamination="auto", random_state=42
                    ).fit_predict(x)
                    mask = labels == -1
                except ImportError:
                    notes.append("sklearn not available")
        except Exception as exc:
            notes.append(f"error: {exc}")

        idxs = np.where(mask)[0].astype(int)
        n_anomalies = int(mask.sum())
        rate = float(100.0 * np.mean(mask))

        ctx.anomaly_store[col] = idxs.tolist()
        ctx.anomaly_meta[col] = {
            "method": mth,
            "n": n_anomalies,
            "rate": rate,
            "notes": notes,
        }

        summary[col] = {"n_anomalies": n_anomalies, "anomaly_rate": rate}

    return _safe_json({"status": "ok", "summary": summary})


@llm.tool(description="Retrieve cached anomaly indices.")
def get_cached_anomalies(column=None):
    """Use when: Retrieve cached anomaly indices.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: column (str, optional): Specific column (default all).
    Returns: JSON with indices and metadata.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if column:
        return _safe_json(
            {
                "indices": ctx.anomaly_store.get(column, []),
                "meta": ctx.anomaly_meta.get(column, {}),
            }
        )

    return _safe_json({"indices": ctx.anomaly_store, "meta": ctx.anomaly_meta})


@llm.tool(description="Detect structural breaks or regimes using ruptures or HMM.")
def change_points(
    algorithm: Literal["ruptures", "hmm"] = "ruptures",
    penalty=0,
    n_regimes=2,
    column=None,
):
    """Use when: Detect structural breaks/regimes.

    Triggers: analyze the series, stationarity test, detect anomalies, check trend, seasonality diagnostics, acf pacf.
    Avoid when: The user only wants data ingestion or a final forecast without diagnostic detail..
    Inputs: algorithm (str, optional): 'ruptures' or 'hmm'; penalty (float, optional): Penalty value for ruptures; n_regimes (int, optional): Number of states for HMM; column (str, optional): Specific column.
    Returns: JSON with change points or regimes.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns
    out = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)

        try:
            if algorithm == "hmm":
                try:
                    from hmmlearn.hmm import GaussianHMM
                except ImportError:
                    return _safe_json({"status": "error", "error": "hmmlearn not available"})
                
                # Reshape for hmmlearn
                y_hmm = y.reshape(-1, 1)
                model = GaussianHMM(n_components=int(n_regimes), covariance_type="diag", n_iter=100)
                model.fit(y_hmm)
                hidden_states = model.predict(y_hmm)
                
                # find indices where state changes
                changes = np.where(hidden_states[:-1] != hidden_states[1:])[0] + 1
                cp_rows = [{"index": int(c), "date": str(ctx.data["date"].iloc[c]), "regime": int(hidden_states[c])} for c in changes]
                out[col] = {"status": "ok", "algorithm": "hmm", "n_regimes": int(n_regimes), "change_points": cp_rows}
            else:
                try:
                    import ruptures as rpt
                except ImportError:
                    return _safe_json({"status": "error", "error": "ruptures not available"})
                    
                pen = float(penalty) if penalty else None
                algo = rpt.Pelt(model="l2").fit(y)
                bks = algo.predict(pen=pen)
                n_points = len(y)
                cps = [int(b) for b in bks if b < n_points]
                cp_rows = [{"index": c, "date": str(ctx.data["date"].iloc[c])} for c in cps]
                out[col] = {"status": "ok", "algorithm": "ruptures", "change_points": cp_rows, "penalty": pen}
        except Exception as exc:
            out[col] = {"status": "error", "error": str(exc)}

    return _safe_json(out)


__all__ = [
    "compute_statistics",
    "stationarity_tests",
    "acf_pacf_peaks",
    "compute_multivariate_stats",
    "detect_trend",
    "stl_seasonality",
    "detect_anomalies",
    "get_cached_anomalies",
    "change_points",
]
