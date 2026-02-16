## Tool: suggest_transformations

**Description:** Intelligent transformation suggestions.

**Parameters:** None

**Returns:** JSON with recommendations

**Implementation:**
```python
def suggest_transformations():
    """Suggest transformations based on diagnostics."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    suggestions = {"recommended": [], "reasons": {}, "columns": ctx.value_columns}

    # Get stats and stationarity
    stats_json = call_tool("compute_statistics")
    stats = _try_parse_json(stats_json)

    stationarity_json = call_tool("stationarity_tests")
    stationarity = _try_parse_json(stationarity_json)

    for col in ctx.value_columns:
        if col not in stats:
            continue

        stat = stationarity.get(col, {})

        # Simple skewness estimate
        v = ctx.data[col].dropna()
        if len(v) > 0:
            mean_val = v.mean()
            std_val = v.std()
            skew = float(((v - mean_val) ** 3).mean() / (std_val ** 3)) if std_val > 0 else 0
        else:
            skew = 0

        adf_p = stat.get("adf", {}).get("pvalue", 1.0)

        recs = []
        if abs(skew) > 1.0:
            recs.append("log" if skew > 0 else "boxcox")
        if adf_p > 0.05:
            recs.append("detrend (degree=3)")

        suggestions["recommended"].append({"column": col, "actions": recs})
        suggestions["reasons"][col] = {
            "skew": skew,
            "adf_pvalue": adf_p,
            "stationary": adf_p < 0.05,
        }

    return _safe_json(suggestions)
```

---

**Note:** This SKILLS.md contains analysis, preprocessing, diagnostics, anomaly, plotting, and baseline forecasting tools for end-to-end time-series work.
