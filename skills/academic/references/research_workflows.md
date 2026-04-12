# Academic Research Workflows

Use these workflows when the user asks for deep research, comparative analysis,
paper-code audit, replication planning, peer review, or any literature task that
needs durable artifacts rather than a quick answer.

## Durable Research Pattern

For substantial research tasks, create a small working plan before gathering
evidence. Keep it lightweight, but make the work auditable.

Plan template:

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

For long-running work, save the plan under a natural workspace path such as
`outputs/.plans/<slug>.md`, `research/<slug>/plan.md`, or the project-local
equivalent. If the repo already has a research-output convention, use that.

## Deep Research

Use for broad surveys, state-of-the-art briefs, research roadmaps, or technical
backgrounders.

1. Define the exact research questions and the acceptable source types.
2. Search broadly across Semantic Scholar, OpenAlex, IEEE, arXiv, publisher pages,
   and citation graphs as appropriate.
3. Build a shortlist before reading deeply. Deduplicate by DOI, arXiv ID, title,
   and first author/year.
4. Extract for each key paper: problem, method, assumptions, dataset, metric,
   headline result, limitations, and follow-up citations.
5. Synthesize by theme, not by paper order.
6. Run a claim sweep before finalizing: every important number, ranking, causal
   statement, and "best/SOTA" claim must map to evidence in the verification log.
7. Label inferences. Do not present extrapolations as source claims.

Final output shape:

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

Comparison matrix:

```markdown
| Item | Core Claim | Evidence Type | Strengths | Caveats | Confidence |
|---|---|---|---|---|---|
| <source or method> | <claim> | Direct / proxy / negative | <why it matters> | <limits> | High / Medium / Low |
```

Always separate:
- Agreements: claims supported by multiple independent sources.
- Disagreements: places where findings conflict.
- Uncertainty: missing details, incomparable metrics, weak evaluation, or unclear
  provenance.
- Applicability: when a result holds only for certain datasets, domains, scales,
  hardware, or assumptions.

## Paper-Code Audit

Use when the user wants to compare a paper against its repository or reproduce a
method from code.

Audit checklist:
- Paper identity: title, authors, venue/preprint version, DOI/arXiv ID.
- Code identity: repository URL, commit/ref if known, license, release date.
- Claimed method versus implementation: architecture, loss, preprocessing,
  training schedule, decoding/inference, default hyperparameters.
- Claimed datasets versus code: dataset names, splits, filtering, augmentation,
  leakage risks.
- Claimed metrics versus code: metric definitions, averaging, confidence
  intervals, seeds, statistical tests.
- Reproducibility: environment, dependencies, scripts, checkpoints, configs,
  hardware assumptions, missing artifacts.
- Mismatches: explicit paper/code discrepancies, ambiguous defaults, undocumented
  steps, or code paths that do not match reported experiments.

Use code tools such as file search, LSP, Python, Rust, or Bash only after
identifying the paper claims to verify.

## Replication Planning

Use when the task is to plan or run a replication.

Before running experiments or installing dependencies, confirm the execution
environment and budget with the user. Then record:
- Target result and exact metric.
- Dataset and split.
- Code ref or implementation source.
- Hardware/software environment.
- Commands to run.
- Expected runtime and storage.
- Seeds and variance handling.
- What outcome would count as replicated, partially replicated, or failed.

Replication report:

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

Review against:
- Novelty: what is genuinely new relative to prior work.
- Significance: whether the result changes practice or understanding.
- Rigor: sound methodology, controls, ablations, uncertainty, statistical validity.
- Baselines: fair, current, and correctly tuned comparisons.
- Claims: claims match evidence and do not overgeneralize.
- Reproducibility: enough detail, code/data availability, environment, seeds.
- Ethics and limitations: harms, failure cases, data provenance, deployment risks.

Severity labels:
- Fatal: invalidates a main result or makes the artifact unreproducible.
- Major: materially weakens the claim but can likely be fixed.
- Minor: clarity, presentation, or local technical issue.

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

Verification levels:
- Full text checked.
- Abstract and metadata checked.
- Citation/context checked only.
- Unverified, included for discovery only.

