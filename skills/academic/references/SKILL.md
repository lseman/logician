---
name: Academic Research Workflows
description: Structured workflows for deep research, comparative analysis, paper-code audits, replication planning, and peer review.
aliases:
  - academic research workflow
  - deep research
  - paper-code audit
  - replication planning
  - peer review workflow
  - provenance tracking
triggers:
  - deep research
  - comparative analysis
  - paper-code audit
  - replication planning
  - peer review
  - systematic review
  - provenance sidecar
  - research plan
preferred_tools:
  - arxiv_search
  - arxiv_get
  - s2_search
  - openalex_search
  - ieee_search
  - unpaywall_resolve
example_queries:
  - conduct deep research on transformer time series
  - compare these papers on time series forecasting
  - audit the code vs claims for this paper
  - plan a replication of this experiment
  - peer review this draft
when_not_to_use:
  - the task is a quick factual lookup (use direct search instead)
  - the user needs a single paper's abstract or PDF
next_skills:
  - academic/semantic_scholar
  - academic/openalex
  - academic/ieee
  - academic/arxiv
  - academic/systematic

## Usage

These workflows guide durable research tasks that produce auditable artifacts. Use them when the user asks for deep research, comparative analysis, paper-code audit, replication planning, peer review, or any literature task needing structured outputs.

## Durable Research Pattern

For substantial research tasks, create a working plan before gathering evidence. Keep it lightweight but auditable.

### Plan Template

```markdown
# Research Plan: <topic>

## Questions
- <primary question>
- <secondary questions>

## Scope
- Sources:
- Date range:
- Inclusion criteria:
- Exclusion criteria:

## Strategy
- Search strings:
- Databases:
- Known seed papers:

## Acceptance Criteria
- <what counts as sufficient evidence>

## Task Ledger
- [ ] Search databases
- [ ] Retrieve full text for key papers
- [ ] Extract claims, methods, datasets, and metrics
- [ ] Compare evidence
- [ ] Verify critical claims

## Verification Log
| Claim | Source | Evidence | Status | Notes |
|---|---|---|---|---|

## Decision Log
| Decision | Rationale |
|---|---|
```

Save the plan under `outputs/.plans/<slug>.md`, `research/<slug>/plan.md`, or the project-local equivalent.

## Deep Research

Use for broad surveys, state-of-the-art briefs, research roadmaps, or technical backgrounders.

1. Define the exact research questions and acceptable source types.
2. Search broadly across Semantic Scholar, OpenAlex, IEEE, arXiv, publisher pages, and citation graphs as appropriate.
3. Build a shortlist before reading deeply. Deduplicate by DOI, arXiv ID, title, and first author/year.
4. Extract for each key paper: problem, method, assumptions, dataset, metric, headline result, limitations, and follow-up citations.
5. Synthesize by theme, not by paper order.
6. Run a claim sweep before finalizing: every important number, ranking, causal statement, and "best/SOTA" claim must map to evidence in the verification log.
7. Label inferences. Do not present extrapolations as source claims.

### Final Output Shape

```markdown
# <Topic>

## Bottom Line

## Scope and Method

## Findings

## Comparison or Taxonomy

## Gaps and Open Questions

## Confidence

## Sources
```

## Comparative Research

Use for comparing papers, methods, datasets, tools, models, benchmarks, or claims.

### Comparison Matrix

```markdown
| Item | Core Claim | Evidence Type | Strengths | Caveats | Confidence |
|---|---|---|---|---|---|
| <source or method> | <claim> | Direct / proxy / negative | <why it matters> | <limits> | High / Medium / Low |
```

Always separate:
- **Agreements:** claims supported by multiple independent sources.
- **Disagreements:** places where findings conflict.
- **Uncertainty:** missing details, incomparable metrics, weak evaluation, or unclear provenance.
- **Applicability:** when a result holds only for certain datasets, domains, scales, hardware, or assumptions.

## Paper-Code Audit

Use when the user wants to compare a paper against its repository or reproduce a method from code.

### Audit Checklist

- **Paper identity:** title, authors, venue/preprint version, DOI/arXiv ID.
- **Code identity:** repository URL, commit/ref if known, license, release date.
- **Claimed method vs. implementation:** architecture, loss, preprocessing, training schedule, decoding/inference, default hyperparameters.
- **Claimed datasets vs. code:** dataset names, splits, filtering, augmentation, leakage risks.
- **Claimed metrics vs. code:** metric definitions, averaging, confidence intervals, seeds, statistical tests.
- **Reproducibility:** environment, dependencies, scripts, checkpoints, configs, hardware assumptions, missing artifacts.
- **Mismatches:** explicit paper/code discrepancies, ambiguous defaults, undocumented steps, or code paths that do not match reported experiments.

Use code tools such as file search, LSP, Python, Rust, or Bash only after identifying the paper claims to verify.

## Replication Planning

Use when the task is to plan or run a replication.

Before running experiments or installing dependencies, confirm the execution environment and budget with the user. Then record:
- Target result and exact metric.
- Dataset and split.
- Code ref or implementation source.
- Hardware/software environment.
- Commands to run.
- Expected runtime and storage.
- Seeds and variance handling.
- What outcome would count as replicated, partially replicated, or failed.

### Replication Report

```markdown
# Replication Report: <paper/result>

## Target
## Environment
## Procedure
## Results
## Deviations
## Failure Modes
## Confidence
## Sources
```

## Peer Review

Use for reviewing papers, drafts, experiments, or research artifacts.

### Review Criteria

- **Novelty:** what is genuinely new relative to prior work.
- **Significance:** whether the result changes practice or understanding.
- **Rigor:** sound methodology, controls, ablations, uncertainty, statistical validity.
- **Baselines:** fair, current, and correctly tuned comparisons.
- **Claims:** claims match evidence and do not overgeneralize.
- **Reproducibility:** enough detail, code/data availability, environment, seeds.
- **Ethics and limitations:** harms, failure cases, data provenance, deployment risks.

### Severity Labels

- **Fatal:** invalidates a main result or makes the artifact unreproducible.
- **Major:** materially weakens the claim but can likely be fixed.
- **Minor:** clarity, presentation, or local technical issue.

## Provenance Sidecar

For durable research outputs, add a provenance section or sidecar file:

```markdown
# Provenance: <topic>

## Search Log
| Date | Query | Source | Results | Notes |
|---|---|---|---|---|

## Included Sources
| Source | Identifier | Why Included | Verification Level |
|---|---|---|---|

## Excluded Sources
| Source | Reason |
|---|---|

## Claim Verification
| Claim | Evidence | Status | Notes |
|---|---|---|---|

## Open Questions
- <question>
```

### Verification Levels

- **Full text checked.**
- **Abstract and metadata checked.**
- **Citation/context checked only.**
- **Unverified, included for discovery only.**
