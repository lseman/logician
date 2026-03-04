from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:  # no-op fallback for standalone use; registry injects real llm
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()


@llm.tool(description="Load 1D/2D NumPy array as time series (univariate only).")
def set_numpy(arr, start_date="2018-01-01", freq="D", name="numpy_series"):
    """Use when: Load 1D/2D NumPy array as time series (univariate only).

    Triggers: load csv, read file, import dataset, inspect dataset columns, create sample data.
    Avoid when: Data is already loaded and the user is asking to transform, analyze, plot, or forecast it..
    Inputs: arr (ndarray, required): NumPy array (1D or 2D); start_date (str, optional): Starting date (default "2018-01-01"); freq (str, optional): Frequency (default "D"); name (str, optional): Series name (default "numpy_series").
    Returns: JSON with load status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        y = a.astype(float)
    elif a.ndim == 2:
        y = np.nanmean(a, axis=1).astype(float)
    else:
        return _safe_json({"status": "error", "error": "arr must be 1D or 2D"})

    n_points = len(y)
    if n_points == 0:
        return _safe_json({"status": "error", "error": "Empty array"})

    idx = pd.date_range(start=start_date, periods=n_points, freq=freq)
    ctx.data = pd.DataFrame({"date": idx, "value": y})
    ctx.original_data = ctx.data.copy()
    ctx.data_name = name
    ctx.freq_cache = freq

    return _safe_json(
        {
            "status": "ok",
            "message": "numpy data loaded",
            "n": n_points,
            "freq": freq,
            "name": name,
        }
    )


@llm.tool(description="Load time series from CSV (supports multivariate).")
def load_csv_data(filepath, date_column, value_column, sep=","):
    """Use when: Load time series from CSV (supports multivariate).

    Triggers: load csv, read file, import dataset, inspect dataset columns, create sample data.
    Avoid when: Data is already loaded and the user is asking to transform, analyze, plot, or forecast it..
    Inputs: filepath (str, required): Path to CSV file; date_column (str, required): Name of date column; value_column (str, required): Name(s) of value column(s) - can be string or list; sep (str, optional): CSV delimiter (default ",").
    Returns: JSON with status and data info.
    Side effects: May read/update shared tool context depending on implementation.
    """
    try:
        df = pd.read_csv(filepath, sep=sep)
        if date_column not in df.columns:
            return _safe_json(
                {"status": "error", "error": f"Date column '{date_column}' not found"}
            )

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column])
        df = df.sort_values(date_column).rename(columns={date_column: "date"})

        value_cols = [value_column] if isinstance(value_column, str) else value_column
        if not value_cols:
            return _safe_json(
                {
                    "status": "error",
                    "error": "value_column must be a non-empty string or list",
                }
            )

        for col in value_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                return _safe_json(
                    {"status": "error", "error": f"Column '{col}' not found"}
                )

        ctx.data = df
        ctx.original_data = df.copy()
        ctx.data_name = filepath
        ctx.freq_cache = _infer_freq_safe(pd.DatetimeIndex(df["date"]))

        info = {
            "loaded": True,
            "name": filepath,
            "n_records": int(len(df)),
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "value_columns": value_cols,
            "freq_inferred": ctx.freq_cache,
        }

        return _safe_json({"status": "ok", "message": "data loaded", "info": info})

    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(description="Create synthetic time series for testing.")
def create_sample_data(pattern, n_points=200, noise_level=1.0):
    """Use when: Create synthetic time series for testing.

    Triggers: load csv, read file, import dataset, inspect dataset columns, create sample data.
    Avoid when: Data is already loaded and the user is asking to transform, analyze, plot, or forecast it..
    Inputs: pattern (str, required): 'trend', 'seasonal', 'random', 'anomaly', 'stationary', 'cyclic_trend'; n_points (int, optional): Number of points (default 200); noise_level (float, optional): Noise std (default 1.0).
    Returns: JSON with creation status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    try:
        n = int(n_points)
        if n < 10:
            return _safe_json({"status": "error", "error": "n_points must be >= 10"})

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

        ctx.data = pd.DataFrame({"date": dates, "value": values})
        ctx.original_data = ctx.data.copy()
        ctx.data_name = f"sample_{pattern}"
        ctx.freq_cache = "D"

        return _safe_json(
            {
                "status": "ok",
                "message": "sample created",
                "n": n,
                "pattern": pattern,
            }
        )

    except Exception as exc:
        return _safe_json({"status": "error", "error": str(exc)})


@llm.tool(description="Get metadata about currently loaded data.")
def get_data_info():
    """Use when: Get metadata about currently loaded data.

    Triggers: load csv, read file, import dataset, inspect dataset columns, create sample data.
    Avoid when: Data is already loaded and the user is asking to transform, analyze, plot, or forecast it..
    Inputs: none.
    Returns: JSON with comprehensive data info.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"loaded": False})

    df = ctx.data
    value_cols = ctx.value_columns

    info = {
        "loaded": True,
        "name": ctx.data_name,
        "n_records": int(len(df)),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "value_columns": value_cols,
        "is_multivariate": ctx.is_multivariate,
        "freq_inferred": ctx.freq_cache or _infer_freq_safe(df["date"]),
    }

    for col in value_cols:
        v = df[col]
        info[f"{col}_missing"] = int(v.isna().sum())
        info[f"{col}_min"] = float(v.min())
        info[f"{col}_max"] = float(v.max())

    return _safe_json(info)


@llm.tool(description="Restore original data before any transformations.")
def restore_original():
    """Use when: Restore original data before any transformations.

    Triggers: load csv, read file, import dataset, inspect dataset columns, create sample data.
    Avoid when: Data is already loaded and the user is asking to transform, analyze, plot, or forecast it..
    Inputs: none.
    Returns: JSON with status.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if ctx.original_data is None:
        return _safe_json({"status": "error", "error": "No original data to restore"})

    ctx.data = ctx.original_data.copy()
    return _safe_json({"status": "ok", "message": "Original data restored"})


__all__ = [
    "set_numpy",
    "load_csv_data",
    "create_sample_data",
    "get_data_info",
    "restore_original",
]
