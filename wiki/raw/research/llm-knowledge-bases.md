# LLM Knowledge Bases

This raw note captures a workflow for building personal research knowledge bases
with an LLM maintaining the markdown corpus.

## Core Loop

Raw material such as web articles, papers, repositories, datasets, images, and
notes lands in a `raw/` directory. An LLM then incrementally compiles a richer
wiki made of markdown files with summaries, concepts, backlinks, and synthetic
articles that connect the original sources.

## IDE Frontend

Obsidian works well as the browsing frontend because it can render markdown,
images, and derived views while the LLM does most of the writing and upkeep.

## Querying

Once the wiki grows large enough, an LLM agent can answer questions against the
wiki by maintaining strong index files, summaries, and cross-links instead of
depending on a vector store for everything.

## Outputs

The agent should be able to write answers back into the workspace as markdown
notes, slide decks, plots, and other artifacts so that every investigation
improves the knowledge base.

## Maintenance

Health checks and linting passes can look for missing data, inconsistent facts,
or interesting connections that deserve new notes or further research.
