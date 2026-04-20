---
name: IEEE Xplore
description: Use for engineering, applied CS, electronics, and electrical systems research when IEEE coverage is required.
aliases:
  - ieee search
  - ieee xplore
  - ieee papers
triggers:
  - search IEEE Xplore
  - find IEEE papers
  - IEEE literature search
preferred_tools:
  - ieee_search
example_queries:
  - search IEEE Xplore for power systems forecasting papers
  - find IEEE conference articles on smart grid optimization
  - retrieve IEEE metadata for a known title
when_not_to_use:
  - you need only academic preprints or open-access abstracts
  - you do not have an IEEE API key available
next_skills:
  - academic/openalex
  - academic/systematic
implementation:
  - The executable code lives in `skills/academic/ieee/scripts/ieee.py`.
