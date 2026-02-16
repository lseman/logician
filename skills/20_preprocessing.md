## Tool: regularize_series

**Description:** Resample to regular frequency and interpolate gaps.

**Parameters:**
- freq (str, optional): Target frequency ('D', 'H', 'W', etc.)
- method (str, optional): 'linear', 'ffill', 'pad' (default 'linear')

**Returns:** JSON with regularization status

**Implementation:**
```python
def regularize_series(freq="", method="linear"):
    """Regularize to full frequency grid."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    f = freq or ctx.freq_cache
    if not f:
        return _safe_json({"status": "error", "error": "Frequency required or inferable"})

    ctx.data = _regularize_series(ctx.data, f, method)
    ctx.freq_cache = f or _infer_freq_safe(ctx.data["date"])

    return _safe_json({
        "status": "ok",
        "freq": ctx.freq_cache,
        "filled_method": method,
        "n_cols": len(ctx.value_columns),
    })
```

---

## Tool: fill_missing

**Description:** Fill missing values via interpolation.

**Parameters:**
- method (str, optional): 'time' (interpolation) or 'ffill'

**Returns:** JSON with fill status

**Implementation:**
```python
def fill_missing(method="time"):
    """Fill missing values per column."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    df = ctx.data.sort_values("date").copy()
    value_cols = ctx.value_columns

    if not value_cols:
        return _safe_json({"status": "error", "error": "No value columns"})

    remaining_nans = {}
    for col in value_cols:
        if method == "ffill":
            df[col] = df[col].ffill().bfill()
        else:
            df_temp = df.set_index("date")
            df_temp[col] = (df_temp[col]
                           .interpolate(method="time", limit_direction="both")
                           .ffill()
                           .bfill())
            df[col] = df_temp[col].reset_index(drop=True)

        remaining_nans[col] = int(df[col].isna().sum())

    ctx.data = df
    return _safe_json({"status": "ok", "method": method, "remaining_nans": remaining_nans})
```

---

## Tool: hampel_filter

**Description:** Apply Hampel filter for robust outlier suppression.

**Parameters:**
- window (int, optional): Window size (default 15)
- n_sigmas (float, optional): Number of sigmas (default 3.0)

**Returns:** JSON with replacement counts

**Implementation:**
```python
def hampel_filter(window=15, n_sigmas=3.0):
    """Apply Hampel filter per column."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    w = int(window)
    w = w if w % 2 == 1 else w + 1
    ns = float(n_sigmas)

    replaced = {}
    for col in ctx.value_columns:
        s = ctx.data[col].copy()
        med = s.rolling(w, center=True).median()
        mad = (s - med).abs().rolling(w, center=True).median()
        k = 1.4826
        mask = (s - med).abs() > ns * k * mad.replace(0, np.nan)
        s_filt = s.copy()
        s_filt[mask] = med[mask]
        ctx.data[col] = s_filt
        replaced[col] = int(mask.sum())

    return _safe_json({
        "status": "ok",
        "replaced": replaced,
        "window": w,
        "n_sigmas": ns
    })
```

---

## Tool: detrend

**Description:** Remove polynomial trend via fitting.

**Parameters:**
- method (str, optional): 'linear' or 'polynomial'
- degree (int, optional): Polynomial degree (default 2)
- column (str, optional): Specific column (default all)

**Returns:** JSON with detrending status

**Implementation:**
```python
def detrend(method="linear", degree=2, column=None):
    """Remove trend via polynomial fitting."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns
    deg_out = 1

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)
        y_clean = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values

        x = np.arange(len(y_clean))

        if method == "linear":
            coeffs = np.polyfit(x, y_clean, 1)
            trend = np.polyval(coeffs, x)
            deg_out = 1
        else:
            deg = max(2, int(degree))
            coeffs = np.polyfit(x, y_clean, deg)
            trend = np.polyval(coeffs, x)
            deg_out = deg

        ctx.data[col] = y_clean - trend

    return _safe_json({"status": "ok", "method": "polyfit", "degree": deg_out, "columns": cols})
```

---

## Tool: transform_series

**Description:** Apply log, sqrt, or Box-Cox transformation.

**Parameters:**
- method (str, required): 'log', 'sqrt', or 'boxcox'
- column (str, optional): Specific column (default all)

**Returns:** JSON with transformation status

**Implementation:**
```python
def transform_series(method, column=None):
    """Apply log, sqrt, or boxcox transform."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns
    results = {}

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)

        if np.any(y <= 0) and method in ["log", "boxcox"]:
            results[col] = "error: requires positive values"
            continue

        try:
            if method == "log":
                ctx.data[col] = np.log(y)
                results[col] = "ok"
            elif method == "sqrt":
                if np.any(y < 0):
                    results[col] = "error: requires non-negative values"
                    continue
                ctx.data[col] = np.sqrt(y)
                results[col] = "ok"
            elif method == "boxcox":
                try:
                    from scipy.stats import boxcox
                    transformed, lmbda = boxcox(y)
                    ctx.data[col] = transformed
                    results[col] = {"ok": True, "lambda": float(lmbda)}
                except ImportError:
                    results[col] = "error: scipy not available"
            else:
                results[col] = f"error: unknown method {method}"
        except Exception as e:
            results[col] = f"error: {str(e)}"

    return _safe_json({"status": "ok", "method": method, "results": results})
```

---

## Tool: scale_series

**Description:** Standardize or normalize data.

**Parameters:**
- method (str, optional): 'standard', 'minmax', or 'robust' (default)
- column (str, optional): Specific column (default all)

**Returns:** JSON with scaling status

**Implementation:**
```python
def scale_series(method="robust", column=None):
    """Scale series using various methods."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)
        y_clean = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values

        if method == "standard":
            scaled = (y_clean - np.mean(y_clean)) / (np.std(y_clean) + 1e-12)
        elif method == "minmax":
            vmin, vmax = np.min(y_clean), np.max(y_clean)
            scaled = (y_clean - vmin) / (vmax - vmin + 1e-12)
        elif method == "robust":
            q25, q75 = np.percentile(y_clean, 25), np.percentile(y_clean, 75)
            iqr = q75 - q25
            median = np.median(y_clean)
            scaled = (y_clean - median) / (iqr + 1e-12)
        else:
            return _safe_json({"status": "error", "error": f"Unknown method: {method}"})

        ctx.data[col] = scaled

    return _safe_json({"status": "ok", "method": method, "columns": cols})
```

---

