---
name: Semantic Scholar
description: Use for exploratory literature search, citation counts, abstracts, and open-access paper discovery.
aliases:
  - semantic scholar
  - s2 search
  - semantic scholar search
triggers:
  - search Semantic Scholar
  - find papers on Semantic Scholar
  - semantic scholar query
preferred_tools:
  - s2_search
  - s2_get_paper
example_queries:
  - search Semantic Scholar for recent transformer-based time series forecasting papers
  - find highly cited papers on reservoir optimization
  - retrieve abstracts and citation counts for ML fairness research
when_not_to_use:
  - you need only DOI metadata or publisher-level bibliography details
  - you need an arXiv preprint that is not indexed yet
next_skills:
  - academic/openalex
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/semantic_scholar/scripts/semantic_scholar.py`.
