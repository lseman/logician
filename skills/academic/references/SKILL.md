---
name: References
description: Use for building and maintaining a reference list from academic search results. Deduplicates papers, normalizes identifiers, and exports structured citations in multiple formats.
aliases:
  - reference manager
  - citation list
  - paper deduplication
  - reference export
  - bibliography builder
triggers:
  - build a reference list
  - deduplicate papers
  - export citations
  - normalize reference list
  - build bibliography
preferred_tools:
  - run_systematic_review
example_queries:
  - build a deduplicated reference list from these search results
  - export these papers as a BibTeX bibliography
  - normalize this reference list by DOI
  - deduplicate papers from multiple sources
when_not_to_use:
  - the task is literature discovery (use semantic scholar or openalex)
  - the task is a systematic review (use academic/systematic directly)
next_skills:
  - academic/systematic
  - academic/semantic_scholar
  - academic/openalex
  - academic/arxiv
---

## References Skill

Use this skill when you need to organize, deduplicate, and export a list of academic references.

## Workflow

1. **Collect** papers from one or more sources using `academic/systematic` or individual provider skills.
2. **Deduplicate** by DOI, arXiv ID, then fuzzy title matching.
3. **Normalize** identifiers (DOIs, arXiv IDs, URLs) using `scripts/common.py` utilities.
4. **Export** in your preferred format:
   - JSON (machine-readable)
   - BibTeX (LaTeX)
   - APA/MLA-style plain text (human-readable)

## Citation Format Templates

### BibTeX

```bibtex
@article{key,
  title     = {Paper Title},
  author    = {Last, F. and First, A.},
  journal   = {Journal Name},
  volume    = {1},
  number    = {2},
  pages     = {100--120},
  year      = {2024},
  doi       = {10.xxxx/xxxxx},
  url       = {https://doi.org/10.xxxx/xxxxx},
  note      = {Accessed: 2024-01-01}
}
```

### APA

```
Author, A. A., & Author, B. B. (Year). Title of the article. *Journal Name, Volume*(Issue), pages. https://doi.org/xxxxx
```

### MLA

```
Author, First A., and Second B. Author. "Title of Article." *Journal Name*, vol. Volume, no. Issue, Year, pp. pages. DOI.
```

## Deduplication Priority

1. **DOI** — exact match (after normalizing `doi.org/` prefixes)
2. **arXiv ID** — exact match (after stripping `arXiv:` prefix and version suffix)
3. **Normalized title** — case-insensitive, punctuation-stripped, whitespace-normalized
4. **Fuzzy title** — token-set ratio ≥ 93% (requires `rapidfuzz`)

## Implementation

The core logic is in `skills/academic/systematic/scripts/systematic.py`:

```python
from systematic import SystematicReview, SearchPlan, ScreeningPlan

sr = SystematicReview()
plan = SearchPlan(
    query="time series forecasting",
    per_source_limit=20,
    screening=ScreeningPlan(),
)
result = sr.run(plan)

# Access deduplicated papers
papers = sr.papers  # Already deduplicated

# Export
sr.save_jsonl("references.jsonl", papers)
sr.save_csv("references.csv", papers)
```

## Related Skills

- `academic/systematic` — Multi-source search and screening
- `academic/semantic_scholar` — Citation-aware discovery
- `academic/openalex` — Metadata and funding context
- `academic/arxiv` — Preprint search
- `academic/unpaywall` — Open-access PDF resolution
