## Tool: plot_series

**Description:** Plot a series with optional anomaly markers and save to file.

**Parameters:**
- column (str, optional): Column to plot (default first value column)
- save_path (str, optional): Output image path (default `plots/series.png`)
- show (bool, optional): Display figure interactively (default True)
- mark_anomalies (bool, optional): Overlay cached anomaly points if available (default True)

**Returns:** JSON with plot metadata and saved path

**Implementation:**
```python
def plot_series(column=None, save_path="plots/series.png", show=True, mark_anomalies=True):
    """Plot a single time series with optional anomaly overlay."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = ctx.value_columns
    if not cols:
        return _safe_json({"status": "error", "error": "No value columns"})
    col = column if (column in cols) else cols[0]

    try:
        import os
        import matplotlib.pyplot as plt
    except Exception as e:
        return _safe_json({"status": "error", "error": f"matplotlib required: {e}"})

    df = ctx.data[["date", col]].copy()
    df = df.dropna(subset=[col]).sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df[col], color="#2563EB", linewidth=1.7, label=col)
    ax.set_title(f"Time Series - {col}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, linestyle="--")

    used_anomalies = 0
    if bool(mark_anomalies):
        idxs = ctx.anomaly_store.get(col, [])
        if idxs:
            idxs = [int(i) for i in idxs if 0 <= int(i) < len(ctx.data)]
            if idxs:
                pts = ctx.data.iloc[idxs][["date", col]].dropna()
                if len(pts) > 0:
                    ax.scatter(
                        pts["date"],
                        pts[col],
                        s=28,
                        color="#DC2626",
                        alpha=0.85,
                        label="anomaly",
                        zorder=3,
                    )
                    used_anomalies = int(len(pts))

    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = str(save_path or "plots/series.png")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    return _safe_json({
        "status": "ok",
        "column": col,
        "n_points": int(len(df)),
        "marked_anomalies": used_anomalies,
        "path": out
    })
```

---

## Tool: plot_diagnostics

**Description:** Plot diagnostic panels (series, histogram, ACF, PACF) for one column.

**Parameters:**
- column (str, optional): Column to diagnose (default first value column)
- lags (int, optional): Number of ACF/PACF lags (default 40)
- save_path (str, optional): Output image path (default `plots/diagnostics.png`)
- show (bool, optional): Display figure interactively (default False)

**Returns:** JSON with diagnostics plot path

**Implementation:**
```python
def plot_diagnostics(column=None, lags=40, save_path="plots/diagnostics.png", show=False):
    """Plot series diagnostics including histogram, ACF, and PACF."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    cols = ctx.value_columns
    if not cols:
        return _safe_json({"status": "error", "error": "No value columns"})
    col = column if (column in cols) else cols[0]

    try:
        import os
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except Exception as e:
        return _safe_json({"status": "error", "error": f"diagnostics dependencies missing: {e}"})

    s = ctx.data[col].dropna()
    if len(s) < 10:
        return _safe_json({"status": "error", "error": "Not enough points for diagnostics"})

    lag_n = int(max(5, min(int(lags), max(5, len(s) // 3))))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    axes[0].plot(ctx.data["date"], ctx.data[col], color="#1D4ED8", linewidth=1.5)
    axes[0].set_title(f"Series - {col}")
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].hist(s.values, bins=30, color="#60A5FA", edgecolor="white", alpha=0.95)
    axes[1].set_title("Distribution")
    axes[1].grid(alpha=0.2, linestyle="--")

    plot_acf(s.values, lags=lag_n, ax=axes[2])
    axes[2].set_title(f"ACF ({lag_n} lags)")

    plot_pacf(s.values, lags=lag_n, ax=axes[3], method="ywm")
    axes[3].set_title(f"PACF ({lag_n} lags)")

    fig.tight_layout()
    out = str(save_path or "plots/diagnostics.png")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    return _safe_json({
        "status": "ok",
        "column": col,
        "lags": lag_n,
        "path": out
    })
```

---

## Tool: plot_forecast

**Description:** Generate a baseline forecast and plot history + forecast with optional intervals.

**Parameters:**
- method (str, optional): Baseline method (`naive`, `snaive`, `moving_avg`, `holt_winters`, `sarimax`)
- periods (int, optional): Forecast horizon (default 30)
- season_length (int, optional): Seasonal period hint (default auto)
- with_intervals (bool, optional): Include forecast intervals if available (default True)
- save_path (str, optional): Output image path (default `plots/forecast.png`)
- show (bool, optional): Display figure interactively (default False)

**Returns:** JSON with forecast plot path and forecast payload

**Implementation:**
```python
def plot_forecast(method="holt_winters", periods=30, season_length=0, with_intervals=True, save_path="plots/forecast.png", show=False):
    """Run forecast_baselines and render a chart with optional intervals."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})
    if ctx.is_multivariate:
        return _safe_json({"status": "error", "error": "plot_forecast currently supports univariate data"})

    fc_json = call_tool(
        "forecast_baselines",
        method=method,
        periods=int(periods),
        season_length=int(season_length),
        with_intervals=bool(with_intervals),
    )
    fc = _try_parse_json(fc_json)
    if not isinstance(fc, dict) or fc.get("status") != "ok":
        return _safe_json({"status": "error", "error": "forecast_baselines failed", "details": fc})

    try:
        import os
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        return _safe_json({"status": "error", "error": f"matplotlib/pandas required: {e}"})

    hist = ctx.data[["date", "value"]].copy().dropna(subset=["value"]).sort_values("date")
    fc_list = fc.get("forecast", []) or []
    if not fc_list:
        return _safe_json({"status": "error", "error": "No forecast points returned", "details": fc})

    fc_df = pd.DataFrame(fc_list)
    if "date" in fc_df.columns:
        fc_df["date"] = pd.to_datetime(fc_df["date"])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(hist["date"], hist["value"], color="#1E3A8A", linewidth=1.5, label="history")
    ax.plot(fc_df["date"], fc_df["value"], color="#EA580C", linewidth=1.8, label=f"forecast:{method}")

    if bool(with_intervals) and {"lower", "upper"}.issubset(set(fc_df.columns)):
        ax.fill_between(
            fc_df["date"],
            fc_df["lower"].astype(float),
            fc_df["upper"].astype(float),
            color="#FB923C",
            alpha=0.22,
            label="interval",
        )

    ax.set_title(f"Forecast ({method}) - horizon {int(periods)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.28, linestyle="--")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = str(save_path or "plots/forecast.png")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    if bool(show):
        plt.show()
    else:
        plt.close(fig)

    return _safe_json({
        "status": "ok",
        "method": method,
        "periods": int(periods),
        "path": out,
        "forecast": fc_list,
    })
```

---

