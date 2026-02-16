# Time Series Agent Skills

Skills are now organized under `skills/` for maintainability.

## Layout
- `skills/00_overview.md`: setup, playbooks, bootstrap helpers, and context reference
- `skills/10_data_loading.md`: loading and data setup tools
- `skills/20_preprocessing.md`: cleaning and transformation tools
- `skills/30_analysis.md`: statistical analysis, trend, seasonality, anomaly, and break detection
- `skills/40_forecasting.md`: baseline and ensemble forecasting tools
- `skills/50_pipelines.md`: end-to-end orchestration tools
- `skills/60_plotting.md`: visualization tools
- `skills/70_recommendations.md`: suggestion helpers

## Loader behavior
`ToolRegistry` now prefers `skills/*.md` when present, while still supporting a single-file `SKILLS.md` for backward compatibility.
