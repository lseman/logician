# Time Series Agent Skills

Skills are organised under `skills/` in category subdirectories.

## Layout

### `skills/00_timeseries/` — time-series analysis tools
- `00_overview.md`: setup, playbooks, bootstrap helpers, and context reference
- `10_data_loading.md`: loading and data setup tools
- `20_preprocessing.md`: cleaning and transformation tools
- `30_analysis.md`: statistical analysis, trend, seasonality, anomaly, and break detection
- `40_forecasting.md`: baseline and ensemble forecasting tools
- `50_pipelines.md`: end-to-end orchestration tools
- `60_plotting.md`: visualization tools
- `70_recommendations.md`: suggestion helpers

### `skills/99_qol/` — general quality-of-life tools
- `80_websearch.md`: web search and scraping via self-hosted Firecrawl

## Loader behavior
`ToolRegistry` scans all `*.md` files under `skills/` **recursively** (subdirectories
are discovered automatically). Files are loaded in alphabetical path order, so
subdirectory prefix numbers control load sequence across categories.
