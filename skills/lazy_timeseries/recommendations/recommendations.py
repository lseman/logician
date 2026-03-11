from __future__ import annotations
from skills.coding.bootstrap.runtime_access import tool

__skill__ = {
    "name": "Recommendations",
    "description": "Use for suggesting models, workflow choices, and next steps for time-series tasks.",
    "workflow": [
        "Use recommendations when the user wants guidance rather than immediate execution.",
        "Base suggestions on already computed diagnostics where possible.",
    ],
}

@tool
def suggest_transformations():
    """Use when: Intelligent transformation suggestions.

    Triggers: what should i do next, recommend transformations, suggest preprocessing, choose the next step.
    Avoid when: The user already specified the exact transformation or tool they want to run..
    Inputs: none.
    Returns: JSON with recommendations.
    Side effects: May read/update shared tool context depending on implementation.
    """
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    suggestions = {"recommended": [], "reasons": {}, "columns": ctx.value_columns}

    stats_json = call_tool("compute_statistics")
    stats = _try_parse_json(stats_json)

    stationarity_json = call_tool("stationarity_tests")
    stationarity = _try_parse_json(stationarity_json)

    for col in ctx.value_columns:
        if col not in stats:
            continue

        stat = stationarity.get(col, {})

        values = ctx.data[col].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            skew = (
                float(((values - mean_val) ** 3).mean() / (std_val**3))
                if std_val > 0
                else 0.0
            )
        else:
            skew = 0.0

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


__all__ = ["suggest_transformations"]


__tools__ = [suggest_transformations]
