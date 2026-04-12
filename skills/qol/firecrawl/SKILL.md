---
name: Firecrawl
description: Use for deep web crawling, full-site scraping, and structured content ingestion via the Firecrawl API.
aliases:
  - crawl
  - site scrape
  - web extraction
  - firecrawl ingest
triggers:
  - crawl this site
  - scrape all pages under this URL
  - extract structured content from this domain
preferred_tools:
  - firecrawl_search
  - firecrawl_crawl
  - firecrawl_scrape
example_queries:
  - crawl the docs site and ingest all pages
  - scrape this React-rendered docs page into markdown
  - search the docs site before crawling it
when_not_to_use:
  - a single static page that `fetch_url` handles fine
  - content already available in local repo or RAG index
  - the site blocks crawlers or denies automated access
next_skills:
  - rag
  - coding/explore
workflow:
  - Use search first when site scope is unclear.
  - Crawl only the needed URL prefixes.
  - Prefer markdown output for downstream LLM and RAG use.
---

## Role

This skill performs deep site crawling and structured scraping through a self-hosted Firecrawl endpoint. Use it when the task requires ingesting multiple pages or extracting content from a website that is too broad for a single URL fetch.

## Tools

- `firecrawl_search(query, limit, scrape_content)`
- `firecrawl_scrape(url, max_chars, include_links)`
- `firecrawl_crawl(url, limit, include_paths)`

## Implementation

The executable code lives in `skills/qol/firecrawl/scripts/firecrawl.py`.
