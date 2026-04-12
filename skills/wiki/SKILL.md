---
name: Wiki Knowledge Base
description: Use when the user wants to work with the markdown knowledge-base system itself: its structure, retrieval model, workflow, or overall maintenance strategy.
aliases:
  - wiki knowledge base
  - wiki architecture
  - wiki system
triggers:
  - improve the wiki workflow
  - design the wiki system
  - connect wiki and retrieval
  - explain the wiki architecture
preferred_tools:
  - python
  - filesystem
example_queries:
  - make the wiki visible to the agent at startup
  - improve the knowledge-base workflow around raw and compiled notes
  - connect the wiki to retrieval without embeddings
  - make the wiki more Obsidian-friendly
when_not_to_use:
  - generic repo search unrelated to wiki
  - one-off file edits outside the wiki system
next_skills:
  - wiki/wiki_skills
---

## Role

This is the top-level skill for the wiki-based knowledge system under
`./wiki/`.

Use it when the task is about the system as a whole:

- how raw material, source notes, compiled pages, and outputs fit together
- how the agent should retrieve against the structured markdown wiki
- how the Obsidian-facing workspace should be organized
- how linting, health checks, and generated artifacts should evolve

## Layout

The wiki currently has three conceptual layers plus one generated export:

- Raw artifacts: `./wiki/raw/`
- Maintained wiki source notes: `./wiki/source/`
- Schema and maintainer rules: `./wiki/AGENTS.md`
- Generated export and compiled workspace: `./wiki/wiki.md` and `./wiki/dist/` (also produces `mkdocs.yml` and an HTML site under `./wiki/dist/site/` when MkDocs is available)

## Routing Guidance

When the user asks for concrete wiki operations such as rebuild/search/add/update
on the corpus, defer to the more specific operational playbook in:

- `skills/wiki/wiki_skills/SKILL.md`

Use this top-level card when the request is more architectural, strategic, or
cross-cutting than a single wiki operation.
