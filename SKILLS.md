# Skills Index

This repository follows the modern leaf-skill layout:

```text
skills/<group>/<nn_skill_name>/
  SKILL.md
  scripts/
  references/   # optional
  assets/       # optional
```

Conventions:

- `SKILL.md` is singular and belongs to one leaf skill.
- Group folders such as `coding` and `timeseries` are containers, not skills.
- `bootstrap` folders may omit `SKILL.md` because they only provide shared runtime helpers.
- `scripts/` is the canonical location for executable tool modules.
- Python tool modules should export their public tools through `__tools__ = [...]`.
- Decorator-based `@llm.tool(...)` registration is retained only as a compatibility path for older modules.

Active groups:

- `skills/10_superpowers`: 14 core agent meta-skills — brainstorming, systematic-debugging, TDD, writing-plans, executing-plans, subagent-driven-development, dispatching-parallel-agents, finishing-a-development-branch, git-worktrees, writing-skills, requesting/receiving code review, and verification-before-completion.
- `skills/20_ralph`: Ralph PRD format and autonomous execution workflow (`ralph`, `prd`).
- `skills/30_anthropics`: Anthropic-sourced patterns — frontend-design, MCP builder, webapp-testing, document coauthoring, canvas/theme design, and more.
- `skills/global`: session utilities — `think` (deliberate reasoning), `scratch` (working memory), `todo` (persistent task list), `orchestrator` (multi-skill sequencing), `memory_management` (context budget and checkpoint management).
- `skills/coding`: coding-focused leaf skills — `explore`, `file_ops`, `edit_block`, `multi_edit`, `search_replace`, `patch`, `quality`, `shell`, `git`, `repl`, `web`, `parallel_dispatch` (fan-out → consolidate → serialize).
- `skills/timeseries`: data loading, preprocessing, analysis, forecasting, plotting, advanced mining, and pipeline workflows.
- `skills/academic`: literature search, paper retrieval, citation mining, and systematic review across S2, OpenAlex, IEEE, Unpaywall.
- `skills/svg`: SVG diagram and visual asset generation.
- `skills/rag`: RAG ingestion, retrieval, reranking, and tuning.
- `skills/qol`: context ingestion — `firecrawl` (multi-page web crawl/scrape) and `docling_context` (PDF/DOCX layout-aware extraction).
