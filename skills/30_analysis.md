## Tool: compute_statistics

**Description:** Compute descriptive statistics.

**Parameters:**
- aggregate (bool, optional): If True, aggregate all columns (default False)

**Returns:** JSON with statistics

**Implementation:**
```python
def compute_statistics(aggregate=False):
    """Compute descriptive statistics."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    value_cols = ctx.value_columns

    if aggregate:
        v = pd.concat([ctx.data[col] for col in value_cols], axis=0).dropna()
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
        v = pd.to_numeric(ctx.data[col], errors="coerce")
        stats[col] = {
            "mean": float(v.mean()),
            "median": float(v.median()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75)),
        }

    return _safe_json(stats)
```

---

## Tool: stationarity_tests

**Description:** ADF and Ljung-Box tests (requires statsmodels).

**Parameters:**
- lags (int, optional): Lags for Ljung-Box (default auto)
- column (str, optional): Specific column (default all)

**Returns:** JSON with test results

**Implementation:**
```python
def stationarity_tests(lags=0, column=None):
    """ADF and Ljung-Box tests."""
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

        out[col] = col_out

    return _safe_json(out)
```

---

## Tool: acf_pacf_peaks

**Description:** Top ACF/PACF lags by magnitude.

**Parameters:**
- nlags (int, optional): Number of lags (default 40)
- topk (int, optional): Top K peaks (default 5)
- column (str, optional): Specific column (default all)

**Returns:** JSON with top peaks

**Implementation:**
```python
def acf_pacf_peaks(nlags=40, topk=5, column=None):
    """Top ACF/PACF peaks."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        import statsmodels.tsa.api as smt
    except ImportError:
        return _safe_json({"status": "error", "error": "statsmodels not available"})

    N = int(nlags)
    K = int(topk)
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

        n_eff = min(N, max(2, len(y) // 2))

        try:
            acf = smt.stattools.acf(y, nlags=n_eff, fft=True)
            pacf = smt.stattools.pacf(y, nlags=n_eff, method="yw")

            lag_acf = [(lag, float(abs(val))) for lag, val in enumerate(acf[1:], start=1)]
            lag_pacf = [(lag, float(abs(val))) for lag, val in enumerate(pacf[1:], start=1)]

            col_out["acf_top"] = sorted(lag_acf, key=lambda x: x[1], reverse=True)[:K]
            col_out["pacf_top"] = sorted(lag_pacf, key=lambda x: x[1], reverse=True)[:K]
        except Exception as e:
            col_out["notes"].append(f"ACF/PACF failed: {e}")

        out[col] = col_out

    return _safe_json(out)
```

---

## Tool: compute_multivariate_stats

**Description:** Correlations and per-series stats for multivariate data.

**Parameters:** None

**Returns:** JSON with correlations and stats

**Implementation:**
```python
def compute_multivariate_stats():
    """Correlations + per-series stats."""
    if not ctx.is_multivariate:
        return _safe_json({
            "status": "error",
            "error": "Not multivariate data (need >=2 value columns)"
        })

    df_num = ctx.data[ctx.value_columns].dropna()
    corr = df_num.corr().to_dict()

    per_series = {}
    for col in ctx.value_columns:
        v = pd.to_numeric(ctx.data[col], errors="coerce")
        per_series[col] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
        }

        try:
            import statsmodels.api as sm
            per_series[col]["adf_pvalue"] = float(sm.tsa.stattools.adfuller(v.dropna())[1])
        except Exception:
            per_series[col]["adf_pvalue"] = None

    return _safe_json({"correlations": corr, "per_series": per_series})
```

---

## Tool: detect_trend

**Description:** Detect overall trend direction.

**Parameters:**
- column (str, optional): Specific column (default all)

**Returns:** JSON with trend info

**Implementation:**
```python
def detect_trend(column=None):
    """Detect overall trend direction."""
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
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])

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
```

---

## Tool: stl_seasonality

**Description:** STL-based seasonality strength.

**Parameters:**
- period (int, optional): Seasonal period (default auto)
- column (str, optional): Specific column (default all)

**Returns:** JSON with seasonality info

**Implementation:**
```python
def stl_seasonality(period=0, column=None):
    """STL-based seasonality strength."""
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
        per = int(period) if period else _guess_season_length(y)

        try:
            stl = STL(y, period=max(2, per), robust=True).fit()
            var = np.var
            seas_strength = 1.0 - (var(stl.resid) / (var(stl.resid + stl.seasonal) + 1e-12))
            s = float(max(0.0, min(1.0, seas_strength)))

            out[col] = {
                "period_used": int(per),
                "seasonal_strength": s,
                "interpretation": "Strong" if s > 0.64 else "Moderate" if s > 0.36 else "Weak",
            }
        except Exception as e:
            out[col] = {"status": "error", "error": str(e), "period_attempted": per}

    return _safe_json(out)
```

---

## Tool: detect_anomalies

**Description:** Detect anomalies - returns counts only (indices cached).

**Parameters:**
- method (str, required): 'zscore', 'iqr', 'hampel', 'stl_resid', 'iforest'
- threshold (float, optional): Threshold value (default 3.0)
- period (int, optional): Seasonal period
- column (str, optional): Specific column (default all)

**Returns:** JSON with anomaly counts/rates

**Implementation:**
```python
def detect_anomalies(method, threshold=3.0, period=0, column=None):
    """Detect anomalies - returns counts/rates, caches indices."""
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
                k = 1.4826
                denom = (k * mad.replace(0, np.nan)).values
                mask = np.abs(s.values - med.values) > ns * np.nan_to_num(denom, nan=np.inf)

            elif mth == "stl_resid":
                try:
                    from statsmodels.tsa.seasonal import STL
                    p = int(period) if period else _guess_season_length(y)
                    res = STL(y, period=max(2, p), robust=True).fit().resid
                    q1, q3 = np.nanpercentile(res, 25), np.nanpercentile(res, 75)
                    iqr = q3 - q1
                    lb, ub = q1 - 3 * iqr, q3 + 3 * iqr
                    mask = (res < lb) | (res > ub)
                except ImportError:
                    notes.append("statsmodels not available")

            elif mth == "iforest":
                try:
                    from sklearn.ensemble import IsolationForest
                    X = y.reshape(-1, 1)
                    labels = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit_predict(X)
                    mask = labels == -1
                except ImportError:
                    notes.append("sklearn not available")

        except Exception as e:
            notes.append(f"error: {e}")

        idxs = np.where(mask)[0].astype(int)
        n = int(mask.sum())
        rate = float(100.0 * np.mean(mask))

        # Cache
        ctx.anomaly_store[col] = idxs.tolist()
        ctx.anomaly_meta[col] = {"method": mth, "n": n, "rate": rate, "notes": notes}

        summary[col] = {"n_anomalies": n, "anomaly_rate": rate}

    return _safe_json({"status": "ok", "summary": summary})
```

---

## Tool: get_cached_anomalies

**Description:** Retrieve cached anomaly indices.

**Parameters:**
- column (str, optional): Specific column (default all)

**Returns:** JSON with indices and metadata

**Implementation:**
```python
def get_cached_anomalies(column=None):
    """Retrieve cached anomaly indices."""
    if column:
        return _safe_json({
            "indices": ctx.anomaly_store.get(column, []),
            "meta": ctx.anomaly_meta.get(column, {}),
        })

    return _safe_json({
        "indices": ctx.anomaly_store,
        "meta": ctx.anomaly_meta
    })
```

---

## Tool: change_points

**Description:** Detect structural breaks using ruptures.

**Parameters:**
- penalty (float, optional): Penalty value (default auto)
- column (str, optional): Specific column (default all)

**Returns:** JSON with change points

**Implementation:**
```python
def change_points(penalty=0, column=None):
    """Detect change points using PELT."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    try:
        import ruptures as rpt
    except ImportError:
        return _safe_json({"status": "error", "error": "ruptures not available"})

    cols = [column] if column else ctx.value_columns
    out = {}
    pen = float(penalty) if penalty else None

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)

        try:
            algo = rpt.Pelt(model="l2").fit(y)
            bks = algo.predict(pen=pen)
            n = len(y)
            cp = [int(b) for b in bks if b < n]
            cps = [{"index": c, "date": str(ctx.data["date"].iloc[c])} for c in cp]
            out[col] = {"status": "ok", "change_points": cps, "penalty": pen}
        except Exception as e:
            out[col] = {"status": "error", "error": str(e)}

    return _safe_json(out)
```

---

