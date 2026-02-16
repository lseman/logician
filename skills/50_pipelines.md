## Tool: comprehensive_analysis

**Description:** Full analysis pipeline with tool-to-tool calling.

**Parameters:**
- include_plots (bool, optional): Generate plots (default True)
- save_dir (str, optional): Directory for plots

**Returns:** JSON with complete analysis

**Implementation:**
```python
def comprehensive_analysis(include_plots=True, save_dir=None):
    """Run full analysis using tool composition."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    results = {"steps": []}

    # Statistics
    stats_json = call_tool("compute_statistics")
    stats = _try_parse_json(stats_json)
    results["steps"].append({"step": "statistics", "data": stats})

    # Stationarity
    stationarity_json = call_tool("stationarity_tests")
    stationarity = _try_parse_json(stationarity_json)
    results["steps"].append({"step": "stationarity", "data": stationarity})

    # Trend
    trend_json = call_tool("detect_trend")
    trend = _try_parse_json(trend_json)
    results["steps"].append({"step": "trend", "data": trend})

    # Seasonality
    try:
        seasonality_json = call_tool("stl_seasonality")
        seasonality = _try_parse_json(seasonality_json)
        results["steps"].append({"step": "seasonality", "data": seasonality})
    except Exception:
        pass

    # ACF/PACF
    try:
        acf_json = call_tool("acf_pacf_peaks")
        acf = _try_parse_json(acf_json)
        results["steps"].append({"step": "acf_pacf", "data": acf})
    except Exception:
        pass

    # Multivariate stats
    if ctx.is_multivariate:
        multi_json = call_tool("compute_multivariate_stats")
        multi = _try_parse_json(multi_json)
        results["steps"].append({"step": "multivariate_stats", "data": multi})

    # Anomalies
    anomalies_json = call_tool("detect_anomalies", method="zscore")
    anomalies = _try_parse_json(anomalies_json)
    results["steps"].append({"step": "anomalies", "data": anomalies})

    # Summary
    value_cols = ctx.value_columns
    anomaly_rates = []
    if isinstance(anomalies, dict) and "summary" in anomalies:
        for col, s in anomalies["summary"].items():
            if isinstance(s, dict):
                anomaly_rates.append(s.get("anomaly_rate", 0.0))

    first_trend = "unknown"
    if isinstance(trend, dict) and value_cols and value_cols[0] in trend:
        first_trend = trend[value_cols[0]].get("trend", "unknown")

    results["summary"] = {
        "stationary": any(
            st.get("adf", {}).get("stationary", False)
            for st in stationarity.values()
        ) if isinstance(stationarity, dict) else "unknown",
        "trend": first_trend,
        "avg_anomaly_rate": float(np.mean(anomaly_rates)) if anomaly_rates else 0.0,
        "n_records": len(ctx.data),
        "is_multivariate": ctx.is_multivariate,
    }

    if include_plots:
        plots = {}
        series_plot_json = call_tool(
            "plot_series",
            column=value_cols[0] if value_cols else None,
            save_path=f"{save_dir.rstrip('/') + '/' if save_dir else ''}series.png",
            show=False,
            mark_anomalies=True,
        )
        plots["series"] = _try_parse_json(series_plot_json)

        diag_plot_json = call_tool(
            "plot_diagnostics",
            column=value_cols[0] if value_cols else None,
            save_path=f"{save_dir.rstrip('/') + '/' if save_dir else ''}diagnostics.png",
            show=False,
            lags=40,
        )
        plots["diagnostics"] = _try_parse_json(diag_plot_json)
        results["plots"] = plots

    return _safe_json(results)
```

---

