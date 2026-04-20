---
name: arXiv
description: Use for preprint search, retrieval, and rapid discovery of open-access research on arXiv.
aliases:
  - arxiv search
  - arxiv papers
  - arxiv preprint
triggers:
  - search arxiv
  - find arxiv paper
  - arxiv preprint
  - open-access arxiv pdf
preferred_tools:
  - arxiv_search
  - arxiv_get
example_queries:
  - find recent arXiv preprints on multimodal transformers
  - get the arXiv abstract for 2301.00001
  - show open-access arXiv papers about traffic forecasting
when_not_to_use:
  - the paper is already published in a journal or conference and you need final published metadata
  - you need citation network or funding data (use OpenAlex or Crossref)
  - you need publisher paywalled content rather than preprints
next_skills:
  - academic/semantic_scholar
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/arxiv/scripts/arxiv.py`.
