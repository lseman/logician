---
name: Crossref
description: Use for authoritative DOI metadata lookup, publication details, and bibliographic discovery across scholarly publishers.
aliases:
  - crossref search
  - doi lookup
  - publication metadata
triggers:
  - lookup DOI metadata
  - find publication details
  - crossref search
preferred_tools:
  - crossref_search
example_queries:
  - find DOI metadata for a paper on temporal forecasting
  - get publisher and citation entries for DOI 10.1145/1234567.1234568
  - resolve publication metadata for a conference paper
when_not_to_use:
  - you only need open-access full text without metadata
  - you need preprint search or arXiv versions of the same work
next_skills:
  - academic/openalex
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/crossref/scripts/crossref.py`.
