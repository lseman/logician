# OMP → Logician: engineering priorities

Reviewed on 2026-09-05. OMP checkout: `repos/oh-my-pi`, commit `5964a0f764` (2026-09-04). Logician: current working tree, including ongoing uncommitted implementation work. This is a bounded source comparison with local integration probes, not a benchmark establishing either agent as state of the art. No model calls or paid evaluations were run.

The most valuable lesson from OMP is how shared state and lifecycle handling connect its tools. Logician already has substantial foundations: exact-text editing with stale-read protection, an emerging hashline engine, LSP diagnostics, persistent execution kernels, stable prompt construction, adaptive context, session journals, and independently graded tasks. Their presence alone does not establish that the combined agent works reliably.

**Four reproduced gaps to address first**

| Finding | Evidence | Practical consequence |
| --- | --- | --- |
| Hashline editing does not fulfill the tool's success contract | Calling `edit_file.execute` with `input: '[sample.txt#abcd]\nPUT >1: added'` returned `Applied 0 line change(s) across 0 file(s).`; the existing file remained byte-for-byte unchanged. | The model can continue on the mistaken assumption that an edit happened. This probe establishes a failure for this input, not that every edit path fails. |
| Diagnostics depend on human-readable wording | Against the same malformed JSON file, the diagnostic hook produced diagnostics for `Successfully replaced 1 occurrence.` but none for `Applied 1 line change(s) across 1 file(s).` | New mutation formats silently lose diagnostics. |
| Evaluation fixtures moved without updating the baseline | All three baseline paths under `packages/agent-eval/fixtures/` are absent; the fixtures live under `packages/log-eval/fixtures/`. The runner resolves the supplied path directly. | The documented baseline cannot provision its tasks from the repository root as written. |
| Settings dispatch depends on display labels | Applying the displayed `Budget early-stop` setting yielded `Unknown setting: Budget early-stop`; its handler expects `progress early-stop`. | Renaming a label can break configuration behavior. |

Probes used temporary files only. Their executable record is `/tmp/logician-omp-probes.ts` for this session. These findings concern the current working tree; some affected modules are still under development.

**1. Make file mutation one deep module — highest priority**

OMP keeps an edit store on the tool session, shared by callers: [edit/store.ts](../repos/oh-my-pi/packages/coding-agent/src/edit/store.ts). Its diagnostic lifecycle tracks mutation versions, cancels superseded fetches, rejects stale results, and deduplicates diagnostics: [deferred-diagnostics.ts](../repos/oh-my-pi/packages/coding-agent/src/lsp/deferred-diagnostics.ts), [diagnostics-ledger.ts](../repos/oh-my-pi/packages/coding-agent/src/lsp/diagnostics-ledger.ts).

Logician's [edit-file.ts](../packages/log-runtime/src/capabilities/tools/edit-file.ts) creates a fresh store for each hashline call. It invokes [hashline-engine.ts](../packages/log-runtime/src/capabilities/tools/support/hashline-engine.ts) with its default `dryRun = true`, then formats a success message without checking `applied`. The engine also flushes its current file before attaching a parsed operation, explaining the zero-file probe result. Do not simply flip the dry-run default: the hashline branch precedes the existing path checks and file-mutation queue used by the exact-text path.

Recommended interface: a session-owned mutation module accepts an edit proposal and returns structured facts: `applied`, changed paths, before/after revision identities, diff, and errors. Exact-text, hashline, AST, and write operations should cross this same interface for path policy, serialization, stale reads, checkpoints, atomic writes, and diagnostic events. Keep format parsers internal.

Acceptance criteria: preview never mutates; applied means the filesystem changed as reported; stale reads cannot overwrite newer work; two edits to the same file serialize; diagnostics correspond to the latest file revision; every supported mutation format exercises those contracts. Measure edit retry rate and correct edits per task before choosing a default format.

**2. Make completion depend on structured verification evidence**

Logician's [verified-stop-policy.ts](../packages/log-core/src/control/policy/verified-stop-policy.ts) recognizes selected mutation tool names and infers verification from command/output regexes. It does not include every mutation channel, and textual success/failure inference can misclassify output. This is a Logician-specific opportunity, not a claim that OMP solves task completion perfectly.

Feed the policy structured mutation revisions and verification results: command, exit code, cancellation/timeout, scope, and revision checked. Reuse the existing acceptance-contract machinery where appropriate. A successful check must refer to the current relevant state, and a cancelled check must never count as success. Preserve task-appropriate behavior: a documentation edit should not force an unrelated full application build.

Acceptance criteria: catches edits after the last check, edits via AST and execution tools, failed/cancelled checks, and irrelevant checks. Track independently graded false-completion rate.

**3. Restore and expand the evaluation loop before feature expansion**

[baseline.json](../packages/log-eval/corpus/baseline.json) contains three short, single-source-file tasks. [runner.ts](../packages/log-eval/src/runner.ts) already provisions disposable workspaces and [types.ts](../packages/log-eval/src/types.ts) supports independently specified graders and resource limits. Repair fixture resolution and digest validation first; schema validation alone does not establish that a trial can run.

Then create a proposed 20–30-task corpus covering realistic multi-file fixes, stale edits, long tool output, mid-turn compaction, cancellation, provider failures, and concurrent changes. Run Logician and OMP with the same model endpoint, task revisions, tool access, and resource limits, with repeated trials. Add an OMP output adapter if required; an arbitrary command field alone does not guarantee comparable usage telemetry.

Report environment-graded pass rate, false completion, wall time, tokens per successful task, edit retries, and recovery failures. Mark unsupported cost/cache telemetry as unavailable. Evaluate one change at a time. The old documented 9/9 smoke result is useful history, but it cannot establish broad capability.

**4. Extend provider semantics behind the existing adapter interface**

OMP separates native provider implementations and model policy: [openai-responses.ts](../repos/oh-my-pi/packages/ai/src/providers/openai-responses.ts), [anthropic.ts](../repos/oh-my-pi/packages/ai/src/providers/anthropic.ts), and [catalog compatibility rules](../repos/oh-my-pi/packages/catalog/src/compat/rules/README.md).

Logician's [ProviderAdapter](../packages/log-core/src/capabilities/provider/provider-adapter.ts) currently exposes only `openai-chat-sse`. Keep that useful local-model path, then add native protocols only for providers we intend to support. Capability metadata should govern valid sampling parameters, thinking formats, context limits, and usage interpretation. Do not copy OMP's entire provider catalog before there is a supported runtime path for it.

Acceptance criteria: protocol fixtures cover fragmented streaming, tool arguments, reasoning preservation, cancellation, usage, and retry behavior. Never route an incompatible response through the chat-completions parser.

**5. Combine context selection with measurable cache stability**

OMP's [append-only-context.ts](../repos/oh-my-pi/packages/agent/src/append-only-context.ts) fingerprints its system/tool prefix and preserves unchanged message prefixes; the agent loop invokes it conditionally. Its [tool-protection.ts](../repos/oh-my-pi/packages/agent/src/compaction/tool-protection.ts) protects skill reads and artifact-recovery reads from repeated elision.

Logician already sorts prompt tool snippets and separates a static prefix in [system-prompt.ts](../packages/log-runtime/src/runtime/context/system-prompt.ts). Its [AdaptiveContextController](../packages/log-core/src/system/context/adaptive-context-controller.ts) ranks contributions using lexical relevance and learned utility. Build on these rather than introducing another context manager.

Recommended work: measure the actual provider-bound prefix, record why it changes, keep ephemeral state out of the static prefix, and preserve tool-call/result integrity through compaction. Protect recovered evidence from an endless summarize/re-read cycle. Charge injected system content as well as messages against the relevant budget. Validate any learned context policy against independent outcomes, including forgetting and retrieval regressions.

Acceptance criteria: unchanged request prefixes remain stable, context selection stays within its defined budget, and long tasks retain objectives and unresolved evidence. Compare cache-hit tokens and task success; stable input does not guarantee a provider cache hit.

**6. Turn persistent eval into integrated tool orchestration**

OMP's [JS tool bridge](../repos/oh-my-pi/packages/coding-agent/src/eval/js/tool-bridge.ts) routes session tools into execution, carries cancellation, and emits structured status. Its [isolation runner](../repos/oh-my-pi/packages/coding-agent/src/task/isolation-runner.ts) shares the subagent isolation lifecycle between callers.

Logician's [eval-tool.ts](../packages/log-runtime/src/capabilities/eval/eval-tool.ts) and [kernel-manager.ts](../packages/log-runtime/src/capabilities/eval/kernel-manager.ts) already provide persistent computation. The inspected eval call does not forward its tool context or an abort signal into the kernel request. Add a session-tool bridge through the existing dispatcher so nested calls retain path policy, hooks, mutation tracking, and cancellation. Expose bounded batches and emit only the results needed by the model.

Acceptance criteria: interrupt stops nested work, handles cannot leak between sessions, nested edits trigger the same diagnostics/checkpoints, and one failed batch member has explicit semantics. Measure context reduction and latency before adopting a code-first default. This is a larger follow-up after mutation and verification reliability.

**7. Make the polished settings UI schema-driven**

OMP's [settings-schema.ts](../repos/oh-my-pi/packages/coding-agent/src/config/settings-schema.ts) defines defaults, types, and tab/group metadata together. Logician's [settings controller](../apps/tui/src/app/overlay-controllers/settings.ts) separately constructs rows and dispatches changes by lowercased display name.

Use stable setting IDs and one registry for validation, defaults, persistence mapping, and display metadata. Dedicated selectors can remain specialized handlers. Generate the list from that registry, preserving the new OMP-inspired visual design. Add effective-value provenance—default, user, project, session—only after defining precedence centrally.

Acceptance criteria: changing a label cannot break writes; every visible configurable row has a valid handler; save failures are surfaced; applied values survive reopening. This is the clearest small architectural improvement directly following the menu redesign.

**Recommended implementation order**

First repair the four reproduced gaps and establish a runnable baseline. Next consolidate mutation results and verification evidence. Then use repeated evaluations to choose between provider support, context/cache work, and integrated execution. The settings registry is a small parallel product improvement, but the comparison itself did not require spawning agents or changing implementation code.

The local OMP checkout is the source of implementation findings. Its [current upstream README](https://github.com/can1357/oh-my-pi) was also checked to confirm project identity and direction; its performance marketing is not treated here as independently validated evidence. Graphician's Logician index was stale, and the OMP index build was stopped after failing to become useful during this bounded review; findings were checked directly against source.
