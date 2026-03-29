---
name: Academic
description: Use for systematic literature search, paper retrieval, citation mining, and evidence synthesis across academic databases (Semantic Scholar, OpenAlex, IEEE, Unpaywall). Covers both exploratory discovery and structured systematic review.
aliases:
  - literature search
  - paper search
  - citation lookup
  - systematic review
  - research survey
triggers:
  - find papers on this topic
  - search semantic scholar
  - literature review
  - find related work
  - cite papers for this claim
  - get the paper PDF
preferred_tools:
  - s2_search
  - s2_get_paper
  - openalex_search
  - ieee_search
  - unpaywall_resolve
example_queries:
  - find the top 10 cited papers on diffusion models
  - systematic search for SOTA time series forecasting methods
  - find the full text of this paper
  - trace citations of this seminal paper
when_not_to_use:
  - the source is a web page or documentation (use fetch_url or firecrawl)
  - you need code, not papers (use coding/web or explore)
next_skills:
  - rag
  - global/think
  - qol/docling_context
preferred_sequence:
  - s2_search
  - s2_get_paper
  - openalex_search
  - unpaywall_resolve
entry_criteria:
  - the task requires papers, citations, or evidence synthesis rather than code or docs
  - the answer should be grounded in academic literature, not just web search
decision_rules:
  - start broad, then narrow to the papers that actually match the claim or method
  - prefer multiple databases when completeness matters
  - verify relevance from abstracts or methods before citing
failure_recovery:
  - if one database is sparse, cross-check another before concluding there is little literature
  - if full text is unavailable, fall back to abstract plus open-access resolution instead of stopping early
exit_criteria:
  - the shortlist of papers is relevant, deduplicated, and sufficient for the user's question
  - evidence is labeled clearly enough to support a grounded synthesis
anti_patterns:
  - citing papers from titles alone without checking relevance
  - treating citation count as proof of correctness
---

## Database Quick Reference

| Source | Tool | Best for |
|---|---|---|
| Semantic Scholar | `s2_search`, `s2_get_paper` | General CS/ML, citation counts, abstracts, open access PDFs |
| OpenAlex | `openalex_search` | Broad cross-discipline, funding info, institution affiliates |
| IEEE Xplore | `ieee_search` | Engineering, electronics, applied CS (requires API key) |
| Unpaywall | `unpaywall_resolve` | Finding legal open-access PDF for a known DOI |

All four API keys are configured in `agent_config.json` → `env`.

## Workflow

### 1. Exploratory Search (find the space)

Start with a broad Semantic Scholar search:
```
s2_search(
  query="time series anomaly detection transformer 2023",
  limit=20,
  fields=["title", "year", "citationCount", "abstract", "isOpenAccess"]
)
```

Sort by `citationCount` descending to surface seminal work first. Skim abstracts for relevance.

### 2. Narrow and Retrieve

For each paper worth reading:
```
s2_get_paper(
  paper_id="<S2PaperId or DOI>",
  fields=["title", "abstract", "references", "tldr", "openAccessPdf"]
)
```

If `openAccessPdf` is null → try Unpaywall:
```
unpaywall_resolve(doi="10.xxxx/xxxxx")
```

### 3. Citation Mining

Trace backwards from a focal paper:
- Get the paper's references → identify key prior work
- Get papers that cite the focal paper → find follow-up work
- Cross-check with OpenAlex for completeness

```
s2_get_paper(
  paper_id="<id>",
  fields=["references", "citations"]
)
```

### 4. Systematic Review

For a structured literature review:
1. Define inclusion/exclusion criteria before searching
2. Use multiple databases (S2 + OpenAlex + IEEE) and merge by DOI
3. Deduplicate by DOI / title
4. Screen titles + abstracts → shortlist
5. Full text review of shortlist
6. Synthesize: group by method, performance, dataset

See `academic/systematic/` for the full systematic review workflow skill.

### 5. Ingest to RAG

After collecting full-text PDFs:
1. Convert with `docling_context` skill (layout-aware PDF extraction)
2. Ingest into local RAG index via `rag` skill
3. Query during analysis turns for grounded retrieval

## Exit Criteria

- A focused set of relevant papers has been identified.
- Each cited paper has been checked at least at the abstract level, and preferably methods/results when needed.
- The synthesis is grounded in direct or clearly labeled proxy evidence.

## Evidence Quality Signals

When synthesizing findings, distinguish:
- **Direct evidence**: paper explicitly reports the metric you need
- **Proxy evidence**: related metric that supports an inference
- **Negative evidence**: paper that contradicts the claim
- Label these clearly in your synthesis

## Anti-Patterns

| Anti-pattern | Correct behavior |
|---|---|
| Reporting paper titles without checking abstracts | Always verify relevance before citing |
| Using S2 only — missing IEEE for applied engineering | Cross-search at least two databases |
| Treating citation count as quality signal | High citations ≠ correct; read the methodology |
| Dumping 20 abstracts into context | Summarize and select; max 5–8 for deep read |
| Missing open-access version | Always try Unpaywall before saying full text is unavailable |
| Citing papers you have not read | Either read the abstract + methods or flag as "not verified" |
