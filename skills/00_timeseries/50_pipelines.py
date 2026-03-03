from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:  # no-op fallback for standalone use; registry injects real llm
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()


@llm.tool(description="Full analysis pipeline with tool-to-tool calling.")
def comprehensive_analysis(include_plots=True, save_dir=None):
    """Use when: Full analysis pipeline with tool-to-tool calling.

    Triggers: end to end analysis, full pipeline, comprehensive analysis, run the whole workflow.
    Avoid when: The user wants one focused tool step instead of an orchestrated multi-step workflow..
    Inputs: include_plots (bool, optional): Generate plots (default True); save_dir (str, optional): Directory for plots.
    Returns: JSON with complete analysis.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    results = {"steps": []}

    stats_json = call_tool("compute_statistics")
    stats = _try_parse_json(stats_json)
    results["steps"].append({"step": "statistics", "data": stats})

    stationarity_json = call_tool("stationarity_tests")
    stationarity = _try_parse_json(stationarity_json)
    results["steps"].append({"step": "stationarity", "data": stationarity})

    trend_json = call_tool("detect_trend")
    trend = _try_parse_json(trend_json)
    results["steps"].append({"step": "trend", "data": trend})

    try:
        seasonality_json = call_tool("stl_seasonality")
        seasonality = _try_parse_json(seasonality_json)
        results["steps"].append({"step": "seasonality", "data": seasonality})
    except Exception:
        pass

    try:
        acf_json = call_tool("acf_pacf_peaks")
        acf = _try_parse_json(acf_json)
        results["steps"].append({"step": "acf_pacf", "data": acf})
    except Exception:
        pass

    if ctx.is_multivariate:
        multi_json = call_tool("compute_multivariate_stats")
        multi = _try_parse_json(multi_json)
        results["steps"].append({"step": "multivariate_stats", "data": multi})

    anomalies_json = call_tool("detect_anomalies", method="zscore")
    anomalies = _try_parse_json(anomalies_json)
    results["steps"].append({"step": "anomalies", "data": anomalies})

    value_cols = ctx.value_columns
    anomaly_rates = []
    if isinstance(anomalies, dict) and "summary" in anomalies:
        for _, summary in anomalies["summary"].items():
            if isinstance(summary, dict):
                anomaly_rates.append(summary.get("anomaly_rate", 0.0))

    first_trend = "unknown"
    if isinstance(trend, dict) and value_cols and value_cols[0] in trend:
        first_trend = trend[value_cols[0]].get("trend", "unknown")

    results["summary"] = {
        "stationary": any(
            st.get("adf", {}).get("stationary", False) for st in stationarity.values()
        )
        if isinstance(stationarity, dict)
        else "unknown",
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


__all__ = ["comprehensive_analysis"]
