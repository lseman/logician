# Time Series Agent Skills

Skills are organised under `skills/` in category subdirectories.

## Skill Formats

- `SKILL.md`: instruction-only guidance skills (no executable tools).
- `*.py`: executable tools exposed as plain Python functions decorated with `@llm.tool`.
- `skills/_legacy_tool_metadata.py`: migrated metadata snapshot (description/parameters/triggers/returns) used to compose rich runtime tool docstrings.

## Layout

### `skills/00_timeseries/` — time-series analysis tools
- Canonical executable modules with `@llm.tool` functions:
	- `00_overview.py` (helpers/bootstrap)
	- `10_data_loading.py`
	- `20_preprocessing.py`
	- `30_analysis.py`
	- `40_forecasting.py`
	- `50_pipelines.py`
	- `60_plotting.py`
	- `70_recommendations.py`

### `skills/99_qol/` — general quality-of-life tools
- `tools.py`: canonical executable tool module with `@llm.tool` functions.

## Loader behavior
`ToolRegistry` loads in this order:

1. Python skill modules: all `skills/**/*.py` files (except private/init modules), registering any `@llm.tool` functions.
2. Metadata enrichment: migrated tool metadata from `skills/_legacy_tool_metadata.py` is applied to runtime docstrings.
3. Guidance skills: `SKILL.md` files are loaded as instruction-only cards.

Executable legacy `## Tool:` markdown sections have been fully migrated and removed.
