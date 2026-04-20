---
name: Systematic
description: Use for structured academic review, multi-source literature search, deduplication, and evidence synthesis.
aliases:
  - systematic review
  - literature review workflow
  - multi-source academic search
  - evidence synthesis
triggers:
  - systematic literature review
  - build a review table
  - compare papers across databases
  - deduplicate academic results
preferred_tools:
  - s2_search
  - openalex_search
  - ieee_search
  - unpaywall_resolve
example_queries:
  - perform a systematic search for machine learning energy forecasting papers
  - compare results from Semantic Scholar, OpenAlex, and IEEE
  - deduplicate paper candidates from different providers
when_not_to_use:
  - the task is a narrow code question rather than a literature review
  - there is no need for structured evidence from multiple sources
next_skills:
  - academic/arxiv
  - academic/semantic_scholar
  - academic/openalex
  - academic/ieee
  - academic/unpaywall
entry_criteria:
  - the user needs a reproducible search workflow across scholarly sources
  - the answer should be grounded in academic literature and not just web search results
decision_rules:
  - search broadly first, then narrow by relevance
  - collect metadata from at least two complementary providers
  - deduplicate by DOI/arXiv ID/title before shortlisting
  - mark papers with open-access availability clearly
failure_recovery:
  - if one database returns weak coverage, retry with another provider
  - if a DOI cannot be resolved, keep the record and search for alternate identifiers
exit_criteria:
  - a deduplicated shortlist of relevant papers has been assembled
  - inclusion/exclusion reasoning is documented for the candidate set
  - the workflow can be explained clearly to the user
anti_patterns:
  - merging results without deduplication
  - using only a single source for a systematic review
  - reporting search results without screening for relevance
---

## Systematic Academic Search Workflow

Use the following process when you need a structured, reproducible literature review.

### 1. Define the research question

- Clarify scope: topic, method, dataset, timeframe.
- Set inclusion/exclusion criteria:
  - domain relevance
  - publication type (preprint, journal, conference)
  - access requirements
  - recency and citation thresholds

### 2. Search multiple providers

Use complementary sources:

- `academic/semantic_scholar` for citation-aware discovery and abstracts
- `academic/openalex` for broad metadata, affiliations, and funding links
- `academic/ieee` for applied engineering and electronics when relevant
- `academic/arxiv` for preprints and fast open-access retrieval

### 3. Collect and normalize results

- Capture title, authors, year, DOI, arXiv ID, venue, abstract, and URL.
- Normalize identifiers across providers.
- Track open-access availability and source provenance.

### 4. Deduplicate before screening

- Merge by DOI when available.
- Use arXiv ID as a secondary unique identifier.
- Use normalized title matching if DOI is missing.

### 5. Screen titles and abstracts

- Remove clearly irrelevant papers.
- Keep diverse methodological perspectives if the question is broad.
- Flag papers that require full-text access.

### 6. Retrieve full text and open-access links

- Use `academic/unpaywall` to resolve DOIs to legal PDFs.
- Use arXiv versions when available for preprint access.
- Note papers that remain paywalled and mark them separately.

### 7. Synthesize and summarize

Group shortlisted papers by:

- method or model architecture
- dataset or problem domain
- evaluation metrics and results
- strengths, limitations, and open gaps

Provide a concise summary with clear evidence citations.

## Evidence management

- Label sources by provider and access status.
- Distinguish direct evidence from inferred or proxy evidence.
- Cite the precise paper and indicate if full text was not available.

## Related skills

- `academic/arxiv`
- `academic/semantic_scholar`
- `academic/openalex`
- `academic/ieee`
- `academic/unpaywall`

## Implementation

The executable code lives in `skills/academic/systematic/scripts/systematic.py`.
