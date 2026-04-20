---
name: Academic
description: Use for academic literature discovery and provider selection, then connect provider-specific results into a systematic review workflow.
aliases:
  - academic search
  - literature discovery
  - research search
  - paper discovery
triggers:
  - find academic papers
  - literature review
  - academic search workflow
  - search papers across databases
preferred_tools:
  - s2_search
  - openalex_search
  - ieee_search
  - unpaywall_resolve
example_queries:
  - find papers on explainable AI for smart grids
  - locate recent papers on transformer time series forecasting
  - identify open-access PDFs for these citations
when_not_to_use:
  - the task is general web search rather than academic literature
  - the answer should be based on code or software documentation instead of papers
next_skills:
  - academic/systematic
  - academic/semantic_scholar
  - academic/openalex
  - academic/ieee
  - academic/unpaywall
preferred_sequence:
  - academic/semantic_scholar
  - academic/openalex
  - academic/ieee
  - academic/unpaywall
entry_criteria:
  - the user needs papers, citations, or scholarly evidence
  - the question benefits from cross-database literature coverage
decision_rules:
  - choose the best provider for each search need:
      * arXiv for preprints and fast open-access discovery
      * Semantic Scholar for citation-aware literature exploration
      * OpenAlex for broad metadata, affiliations, and funding information
      * IEEE Xplore for applied engineering and electronics
      * Unpaywall for DOI-to-PDF resolution
  - avoid over-relying on a single database when completeness matters
failure_recovery:
  - if one provider returns poor coverage, rerun the search on another provider
  - if a DOI has no open text, keep the paper for abstract-level evidence and note the access limitation
exit_criteria:
  - the candidate paper list is relevant, deduplicated, and grounded in scholarly sources
  - open-access availability or access strategy is clearly identified for each key citation
anti_patterns:
  - searching only one source without cross-checking other providers
  - citing a paper without checking whether it actually addresses the query
  - assuming citation count equals relevance
---

## Academic Provider Directory

Use provider-specific skills for the best fit:

- `academic/arxiv` — preprints and open-access papers on arXiv
- `academic/semantic_scholar` — citation-aware discovery and abstracts
- `academic/openalex` — broad metadata, institutions, and funding context
- `academic/ieee` — IEEE Xplore coverage for applied engineering and electronics
- `academic/unpaywall` — legal open-access PDFs for known DOIs

For structured literature synthesis, use `academic/systematic`.

## When to use this skill

- You want a high-level academic search strategy and provider selection guide.
- You need to decide which database is most effective for a research question.
- You want to combine provider-specific results into a single systematic review workflow.

## Typical workflow

1. Identify the research need.
2. Choose the first provider that matches the need:
   - start with `academic/semantic_scholar` for broad search and citation context,
   - use `academic/arxiv` for preprints and open versions,
   - use `academic/openalex` for author/institution/funding metadata,
   - use `academic/ieee` for IEEE-focused applied research,
   - use `academic/unpaywall` to resolve PDFs by DOI.
3. Collect candidate papers.
4. Deduplicate and screen by title/abstract.
5. Use `academic/systematic` to formalize inclusion and synthesis.

## Example questions

- "Find the most relevant open-access arXiv and IEEE papers on demand forecasting."
- "Search for citations and abstracts for papers on distribution network optimization."
- "Resolve the open-access PDF for DOI 10.1145/xxxxxxx."

## Related skills

- `academic/systematic`
- `academic/arxiv`
- `academic/semantic_scholar`
- `academic/openalex`
- `academic/ieee`
- `academic/unpaywall`
