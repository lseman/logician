# Evolving memory

Logician memory evolves from independently checked outcomes, not from an
agent's confidence or self-critique. The production retriever remains a
deterministic hybrid: SQLite FTS/BM25 and optional dense retrieval are fused
with reciprocal-rank fusion, then constrained by type quotas and a token
budget.

## Claim lifecycle

Extracted model claims begin `probationary`. Automatic prompt injection accepts
only `durable` claims. Promotion requires all of:

- verified status and non-untrusted provenance;
- an evidence certificate recording extractor and schema versions;
- at least two distinct evidence event IDs.

Deterministically extracted, locally verified claims may start durable.
Untrusted claims are quarantined. Unresolved contradictions are contested;
accepted revisions supersede the prior claim. A failed validity predicate
marks only its owning claim stale.

Claims can carry executable, side-effect-free predicates for a workspace file
hash, the current Git revision, or a hashed JSON configuration value. Paths are
resolved through the real workspace root, so traversal and symlink escape fail
validation. Predicate errors fail closed.

## Outcome receipts and shadow learning

Every retrieval creates an immutable trace containing the selected stable IDs,
scores, token costs, reasons, and a recommendation from the current shadow
policy. `recordOutcomeReceipt` binds that trace to a task/trial and an
environment-grounded result. Safety violations, corrections, and reverts
reduce reward; an unauthorized side effect dominates a nominal pass.

Receipts train a small bounded contextual policy online. It is deliberately
`shadow` only: recommendations are logged but never change production context.
This makes learning reversible (discard or replace the versioned weights) and
prevents an early feedback loop from amplifying poisoned memory.

Learned selection must not be enabled merely because reward rises. Promotion
to an active policy requires a frozen, uncontaminated task set, at least three
trials per configuration, deterministic environment graders, confidence
intervals, and all of these gates:

1. coding-task success improves over the deterministic baseline;
2. stale-fact rejection, poisoning quarantine, and permission safety do not
   regress;
3. token cost, latency, and user correction burden stay within declared
   thresholds;
4. the policy version can be rolled back without a schema migration.

No active learned-policy mode is currently exposed. This is intentional: the
repository's memory evaluation is a component gate, not yet sufficient evidence
to let a learned policy control prompts.

## Evaluation

Run `bun run --cwd packages/blocks/log-memory eval` for retrieval relevance, obsolete
fact rejection, abstention, poisoning quarantine, and shadow non-interference.
The memory unit suite additionally mutates a predicate-backed file to verify
targeted stale invalidation and tests probationary promotion. End-to-end agent
evaluation should record a receipt only after repository tests and safety
graders have produced the outcome; agent-authored completion text is diagnostic,
not a reward source.
