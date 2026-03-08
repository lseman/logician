from __future__ import annotations

from typing import Literal

if "llm" not in globals():

    class _NoOpLLM:  # no-op fallback for standalone use; registry injects real llm
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()


@llm.tool(description="Resample to regular frequency and interpolate gaps.")
def regularize_series(
    freq: str = "",
    method: Literal["linear", "ffill", "pad"] = "linear",
):
    """Use when: Resample to regular frequency and interpolate gaps.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: freq (str, optional): Target frequency ('D', 'H', 'W', etc.); method (str, optional): 'linear', 'ffill', 'pad' (default 'linear').
    Returns: JSON with regularization status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    target_freq = freq or ctx.freq_cache
    if not target_freq:
        return _safe_json(
            {"status": "error", "error": "Frequency required or inferable"}
        )

    ctx.data = _regularize_series(ctx.data, target_freq, method)
    ctx.freq_cache = target_freq or _infer_freq_safe(ctx.data["date"])

    return _safe_json(
        {
            "status": "ok",
            "freq": ctx.freq_cache,
            "filled_method": method,
            "n_cols": len(ctx.value_columns),
        }
    )


@llm.tool(description="Fill missing values via interpolation.")
def fill_missing(method: Literal["time", "ffill"] = "time"):
    """Use when: Fill missing values via interpolation.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: method (str, optional): 'time' (interpolation) or 'ffill'.
    Returns: JSON with fill status.
    Side effects: May read/update shared tool context depending on implementation.
    """
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
            df_temp[col] = (
                df_temp[col]
                .interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
            )
            df[col] = df_temp[col].reset_index(drop=True)

        remaining_nans[col] = int(df[col].isna().sum())

    ctx.data = df
    return _safe_json(
        {"status": "ok", "method": method, "remaining_nans": remaining_nans}
    )


@llm.tool(description="Apply Hampel filter for robust outlier suppression.")
def hampel_filter(window=15, n_sigmas=3.0):
    """Use when: Apply Hampel filter for robust outlier suppression.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: window (int, optional): Window size (default 15); n_sigmas (float, optional): Number of sigmas (default 3.0).
    Returns: JSON with replacement counts.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    window_size = int(window)
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    sigma_count = float(n_sigmas)

    replaced = {}
    for col in ctx.value_columns:
        series = ctx.data[col].copy()
        med = series.rolling(window_size, center=True).median()
        mad = (series - med).abs().rolling(window_size, center=True).median()
        k = 1.4826
        mask = (series - med).abs() > sigma_count * k * mad.replace(0, np.nan)
        filtered = series.copy()
        filtered[mask] = med[mask]
        ctx.data[col] = filtered
        replaced[col] = int(mask.sum())

    return _safe_json(
        {
            "status": "ok",
            "replaced": replaced,
            "window": window_size,
            "n_sigmas": sigma_count,
        }
    )


@llm.tool(description="Remove polynomial trend via fitting.")
def detrend(
    method: Literal["linear", "polynomial"] = "linear",
    degree=2,
    column=None,
):
    """Use when: Remove polynomial trend via fitting.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: method (str, optional): 'linear' or 'polynomial'; degree (int, optional): Polynomial degree (default 2); column (str, optional): Specific column (default all).
    Returns: JSON with detrending status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns
    degree_out = 1

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)
        y_clean = pd.Series(y).ffill().bfill().values
        x = np.arange(len(y_clean))

        if method == "linear":
            coeffs = np.polyfit(x, y_clean, 1)
            trend = np.polyval(coeffs, x)
            degree_out = 1
        else:
            degree_val = max(2, int(degree))
            coeffs = np.polyfit(x, y_clean, degree_val)
            trend = np.polyval(coeffs, x)
            degree_out = degree_val

        ctx.data[col] = y_clean - trend

    return _safe_json(
        {"status": "ok", "method": "polyfit", "degree": degree_out, "columns": cols}
    )


@llm.tool(description="Apply log, sqrt, or Box-Cox transformation.")
def transform_series(method: Literal["log", "sqrt", "boxcox"], column=None):
    """Use when: Apply log, sqrt, or Box-Cox transformation.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: method (str, required): 'log', 'sqrt', or 'boxcox'; column (str, optional): Specific column (default all).
    Returns: JSON with transformation status.
    Side effects: May read/update shared tool context depending on implementation.
    """
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
        except Exception as exc:
            results[col] = f"error: {str(exc)}"

    return _safe_json({"status": "ok", "method": method, "results": results})


@llm.tool(description="Standardize or normalize data.")
def scale_series(
    method: Literal["standard", "minmax", "robust"] = "robust",
    column=None,
):
    """Use when: Standardize or normalize data.

    Triggers: clean the data, fill missing values, resample series, regularize frequency, remove outliers, detrend.
    Avoid when: The main user request is pure diagnostics, visualization, or model selection with no data cleanup step..
    Inputs: method (str, optional): 'standard', 'minmax', or 'robust' (default); column (str, optional): Specific column (default all).
    Returns: JSON with scaling status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = [column] if column else ctx.value_columns

    for col in cols:
        if col not in ctx.data.columns:
            continue

        y = ctx.data[col].values.astype(float)
        y_clean = pd.Series(y).ffill().bfill().values

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


__all__ = [
    "regularize_series",
    "fill_missing",
    "hampel_filter",
    "detrend",
    "transform_series",
    "scale_series",
]
