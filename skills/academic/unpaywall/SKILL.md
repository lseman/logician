---
name: Unpaywall
description: Use to find legal open-access PDF versions of known papers using DOI resolution.
aliases:
  - unpaywall lookup
  - unpaywall pdf
  - open access DOI
triggers:
  - resolve DOI with Unpaywall
  - find open-access PDF
  - get open paper from DOI
preferred_tools:
  - unpaywall_resolve
example_queries:
  - find an open-access PDF for DOI 10.1145/1234567.1234568
  - resolve a DOI through Unpaywall
  - get the legal PDF link for a specific publication
when_not_to_use:
  - you need literature discovery rather than open-access retrieval
  - you only have a paper title and not a DOI
next_skills:
  - academic/semantic_scholar
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/unpaywall/scripts/unpaywall.py`.
