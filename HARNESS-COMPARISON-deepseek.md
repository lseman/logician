# Logician vs. deepseek-harness: what's worth stealing

Source: `/home/seman/logician/repos/deepseek-harness` (DeepSeek's real, MIT-licensed, open-source
agent harness — `@deepseek-ai/dsh-root`, developer preview, ~13.1k commits / ~468K lines TS / 227
packages, active 2026-06-10 to 2026-08-21). Verified directly: `git remote -v` →
`github.com/deepseek-ai/deepseek-harness`, `package.json`, `LICENSE`.

This is a survey for ideas, not a call to adopt their architecture wholesale — see the KISS-tension
section at the end. DeepSeek built a large product team's harness; logician is deliberately leaner.

## The one big structural bet they made that we didn't

Everything in `dsh` is a plugin registered into a vendored DI/effects kernel (**Cordis**, forked
from the Koishi chatbot framework) — including the agent loop itself. Composability is total: any
behavior can be patched via `cordis.yml` overlays without touching code. This buys them extreme
configurability at the cost of a lot of indirection to reach "just glue" logic. **Not recommending
we adopt this** — it's the opposite of `log-core`'s "lean loop, harness, hooks, types" design goal,
and 227 packages for one product is real overhead we don't want. But a few ideas that fall out of
this bet are worth taking on their own, independent of the framework:

### 1. One universal pre-step interception point (small, worth copying)

Their loop distinguishes a **step** (one model request + its tool calls) from a **turn** (steps
until nothing is owed). Every concern that needs to inspect or rewrite what the model is about to
see — compaction, plan mode, guards — hooks into a single `agent/pre-step` waterfall, instead of
each concern poking special-cased checks into the loop body.

**Where this matters for us:** worth checking whether `agent-loop-runner.ts` has one clean
"about to send this turn" seam, or whether compaction/guards/interventions each have their own
ad-hoc insertion points. If it's the latter, consolidating to one seam is a cheap simplification,
not a feature add.

### 2. "Model-visible ⟺ logged" as a checked invariant

Every event the model sees is required to be a durable, replayable session event — enforced by a
runtime checker, not just convention. This guarantees the transcript can always fully reconstruct
what the model actually saw.

**Relevance:** directly useful for the eval gap we already found (`log-eval`'s `readAgentDeclaration`
bug was exactly a case of the eval runner *not* being able to trust what it thought the model saw).
Doesn't require any of Cordis's plugin machinery — just a lint/assert rule that any content injected
into a turn is traceable to a logged event.

### 3. Deterministic tool-result pruning before paying for LLM summarization

Their compaction path tries a cheap, model-free step first: head/middle/tail truncation of oversized
tool-result content by code point, logged as its own event so a consumer can account for the savings
without re-measuring. Only if that's insufficient does it fall back to an LLM summarization call.

**Relevance:** directly comparable to our `truncateResultMiddle`/`truncate.ts` utilities (which we
just touched in the tool-response-shaping pass) — worth checking whether our compaction engine
already tries deterministic truncation before summarizing, or goes straight to an LLM call.

### 4. Fail-loud capability negotiation

Recurring pattern across their sandbox (`SANDBOX_UNAVAILABLE` fails loud rather than silently running
unconfined) and subagent system (unsupported capability combinations reject at start, never silently
degrade). Enforcement completeness is reported as data (`'full' | 'partial'`), not assumed.

**Relevance:** a cheap, general discipline — "never accept a request for something you can't actually
guarantee, and never claim more confidence than you have" — worth auditing our own sandbox/permission
code against, independent of anything else in this doc.

### 5. Snapshot-transcript regression testing

Every model-visible behavior change requires a recorded-transcript replay test
(`pnpm run test:snapshot`) — deterministic regression testing of agent behavior, not a scored eval.

**Relevance:** this is *not* what we're missing (we already scoped and partly fixed the eval-scoring
gap in `log-eval`). But it's a complementary, cheap addition: pin a few real task transcripts and
assert the loop produces the same tool-call sequence given the same inputs, catching accidental
behavior drift that a pass/fail grader wouldn't.

## Things they do differently that are genuinely worth a closer look

### Parallel/sequential tool execution is more nuanced than a flat batch

Tool calls in one batch are classified by `executionMode` and dispatched through "barriers and a
bounded rolling pool" — some calls in a batch run strictly sequentially while others run concurrently
up to a limit, but results are still returned to the model in original call order.

**Compare:** logician's registry has `executionMode?: ToolExecutionMode` on `Tool` already (seen in
`git.ts`'s `executionMode: "sequential"`) — so we may already have the primitive; worth checking
whether the loop actually respects mixed sequential/parallel batches or just does all-or-nothing.

### Code Mode: tools as generated SDK calls in a script, not JSON tool-call blocks

Under an opt-in mode, ordinary tools disappear from the wire schema entirely and are instead exposed
as generated bindings the model calls from program code running in a worker. This is a genuinely
different tool-calling surface than either "full schema upfront" or our new deferred/search_tools
approach.

**Relevance:** not something to adopt now, but worth knowing about as a future direction if
deferred-tool-loading (which we just shipped) turns out not to be enough for very large MCP fan-out
— this is the next rung up if that ever becomes the bottleneck.

### Subagent delegation is much richer than a simple call/return

Two lifecycles: one-shot (fire, await, dispose) vs. continuable (a durable child session, resumable,
capable of running in the background across turns). A durable `delegationDepth` field that cold
resume cannot lower, with starts rejected above a cap — a concrete safety rail against runaway
recursive delegation. A `report` channel distinct from the return value, letting a continuable child
proactively push content back to steer its parent.

**Relevance:** worth comparing against `log-runtime/capabilities/delegation/` — specifically, does our delegation
mechanism have any cap on recursive depth? That's a cheap, valuable safety property regardless of
whether we adopt the rest of their model (which is heavier than we likely need — see below).

## Confirmed: no RL/eval prior art here

Explicitly searched for reward/trajectory/rollout/RL scaffolding across their docs and packages —
found nothing. This is a serving/product harness, not a training artifact. Their only eval-adjacent
material is the snapshot-regression discipline (§5 above) and a simple thumbs-up/down feedback
sidecar that isn't wired to anything. **Their repo offers no shortcut for the eval-scoring work we
already started in `log-eval`** — that gap has no external prior art to lean on here.

## KISS-tension: what NOT to copy

- **The continuable-subagent Activation model** — three-rung caching for cold-child listing, an
  owner-scope lifecycle distinct from Agent/Session/Task, separate draining/teardown semantics. Real
  machinery for background agents that outlive a single session view; likely overkill unless we
  actually need agents to run unattended across sessions.
- **Everything-is-a-plugin via a vendored DI framework** — the foundational bet, already flagged
  above. Buys patch-any-row composability at a real indirection cost.
- **The compaction event model's full audit trail** (three lock events, shadowed-seq accounting,
  crash-safe replay guarantees) — valuable in a fully event-sourced system, likely more machinery
  than our compaction engine needs unless we specifically need crash-safe replay.
- **227 packages for one product** — even with a consistent Service/Provider/Consumer split pattern,
  this is a lot of workspace/build/typecheck overhead compared to fewer, coarser modules.

## Suggested next step

Of everything above, the highest-value, lowest-cost items to actually act on are (2) the
model-visible-⟺-logged invariant (directly serves the eval work already in progress) and (4) fail-loud
capability negotiation (a cheap audit, not a redesign). Both are small enough to scope as standalone
follow-ups without touching the loop's core structure.
