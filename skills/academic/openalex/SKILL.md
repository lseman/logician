---
name: OpenAlex
description: Use for broad cross-disciplinary discovery, institution/author metadata, funding, and open citation graph data.
aliases:
  - openalex search
  - openalex papers
  - openalex metadata
triggers:
  - search OpenAlex
  - find author affiliations
  - discover funding data
preferred_tools:
  - openalex_search
example_queries:
  - search OpenAlex for papers on explainable AI in energy systems
  - find institutions linked to reservoir management research
  - list recent work by an author in OpenAlex
when_not_to_use:
  - you only need a single publisher DOI lookup
  - you need arXiv-specific preprints rather than broad scholarly metadata
next_skills:
  - academic/crossref
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/openalex/scripts/openalex.py`.
