// ── @logician/memory — Hook Factory ──────────────────────────────────────────
// Creates AgentHooks for the Logician agent — observation capture + context injection.
// Returns undefined if memory is disabled.

import type { AgentHooks, ExplicitTaskState } from "@logician/agent-core";
import type { TurnToolEvidence } from "../episodes/episode-synthesizer.js";
import type { MemoryEmbedder } from "../embeddings/local-embedder.js";
import {
	extractSemanticEpisode,
	type SemanticExtractor,
} from "../episodes/semantic-extractor.js";
import type {
	CompressedObservation,
	ContextRetrievalQuery,
	Memory,
	MemoryStore,
	RawObservation,
} from "../types.js";

export interface MemoryHooksConfig {
	/** Whether to capture tool observations. Default: true */
	captureTools?: boolean;
	/** Whether to capture user prompts. Default: true */
	capturePrompts?: boolean;
	/** Whether to inject context into agent messages. Default: true */
	injectContext?: boolean;
	/** Token budget for context injection. Default: 4000 */
	contextBudget?: number;
	/** Called synchronously after an observation has been persisted. */
	onObservationSaved?: (observation: CompressedObservation) => void;
	/** Deduplicate equivalent observations for five minutes. Default: true. */
	deduplicate?: boolean;
	/** Consolidate high-signal observations at turn/compaction boundaries. */
	autoConsolidate?: boolean;
	/** Called after automatic consolidation creates or evolves memories. */
	onMemoriesSaved?: (memories: Memory[]) => void;
	/** Synthesize a grounded semantic episode at each completed turn. Default: true. */
	semanticEpisodes?: boolean;
	/** Optional small-model extractor. Invalid or ungrounded output falls back deterministically. */
	semanticExtractor?: SemanticExtractor;
	/** Receives each background extraction task so hosts can flush it during shutdown. */
	onBackgroundTask?: (task: Promise<void>) => void;
	/** Optional local semantic embedder. Disabled when omitted. */
	embedder?: MemoryEmbedder;
	/**
	 * Aborted when the host is shutting down. Checked between embedding
	 * batches so a large first-run backfill (up to thousands of entries)
	 * doesn't hold up shutdown indefinitely — at most one in-flight batch
	 * completes after abort. Remaining entries are simply re-embedded (or
	 * picked up incrementally) on the next run; nothing is lost.
	 */
	shutdownSignal?: AbortSignal;
}

function compactEvidenceValue(value: unknown, depth = 0): unknown {
	if (depth > 4) return "[truncated]";
	if (typeof value === "string") return value.slice(0, 1_000);
	if (Array.isArray(value))
		return value
			.slice(0, 20)
			.map(item => compactEvidenceValue(item, depth + 1));
	if (!value || typeof value !== "object") return value;
	return Object.fromEntries(
		Object.entries(value as Record<string, unknown>)
			.slice(0, 30)
			.map(([key, item]) => [key, compactEvidenceValue(item, depth + 1)]),
	);
}

function saveObservation(
	store: MemoryStore,
	raw: Omit<RawObservation, "sessionId" | "timestamp" | "workspace"> &
		Partial<Pick<RawObservation, "timestamp" | "workspace">>,
	onSaved?: (observation: CompressedObservation) => void,
	deduplicate: boolean = true,
): CompressedObservation | null {
	const sessionId = store.getCurrentSessionId();
	if (!sessionId) return null;

	const dedupName = raw.toolName || raw.hookType;
	const dedupInput = raw.toolInput ?? raw.userPrompt ?? raw.raw;
	if (deduplicate && store.dedupCheck(sessionId, dedupName, dedupInput)) {
		return null;
	}

	const observation = store.observe({
		...raw,
		sessionId,
		timestamp: raw.timestamp || new Date().toISOString(),
		workspace: raw.workspace ?? store.getCurrentWorkspace(),
	});
	if (observation && deduplicate) {
		store.dedupRecord(sessionId, dedupName, dedupInput);
	}
	if (observation && onSaved) {
		try {
			onSaved(observation);
		} catch {}
	}
	return observation;
}

/**
 * Create memory hooks from a store. Returns an AgentHooks object (or undefined
 * if memory is disabled). The hooks are safe to call repeatedly — the store is
 * read-only during hook execution and never blocks the turn.
 */
export function createMemoryHooks(
	store: MemoryStore,
	config: MemoryHooksConfig = {},
): AgentHooks {
	const captureTools = config.captureTools ?? true;
	const capturePrompts = config.capturePrompts ?? true;
	const injectContext = config.injectContext ?? true;
	const contextBudget = config.contextBudget ?? 4000;
	const onObservationSaved = config.onObservationSaved;
	const deduplicate = config.deduplicate ?? true;
	const autoConsolidate = config.autoConsolidate ?? true;
	const semanticEpisodes = config.semanticEpisodes ?? true;
	let latestPrompt = "";
	let latestAssistantOutcome = "";
	let turnTools: TurnToolEvidence[] = [];

	let extractionWorker: Promise<void> | null = null;
	let retryWakeup: ReturnType<typeof setTimeout> | null = null;

	const consolidate = (targetSessionId?: string): Memory[] => {
		const sessionId = targetSessionId ?? store.getCurrentSessionId();
		if (!sessionId) return [];
		const memories = store.consolidate(sessionId);
		if (memories.length && config.onMemoriesSaved) {
			try {
				config.onMemoriesSaved(memories);
			} catch {}
		}
		return memories;
	};

	const embedEntries = async (
		entries: Array<{
			id: string;
			kind: "observation" | "memory";
			text: string;
			sessionId?: string;
		}>,
	) => {
		if (!config.embedder || !entries.length) return;
		const missing = entries.filter(entry => !store.hasEmbedding(entry.id));
		for (let offset = 0; offset < missing.length; offset += 16) {
			if (config.shutdownSignal?.aborted) return;
			const batch = missing.slice(offset, offset + 16);
			const vectors = await config.embedder.embedBatch(
				batch.map(entry => entry.text),
			);
			batch.forEach((entry, index) => {
				const vector = vectors[index];
				if (vector?.length)
					store.upsertEmbedding(entry.id, entry.kind, vector, entry.sessionId);
			});
		}
	};

	// Backed-off jobs are only reclaimed once next_attempt_at elapses, and the
	// worker only ever runs in response to an explicit trigger (turn end,
	// startup recovery). Without this, a job that fails with no other pending
	// work — and no new turn arriving soon — would sit idle indefinitely. This
	// timer is a pure wakeup: it just re-invokes the normal scheduling path,
	// so it never duplicates the retry logic already enforced in the DB.
	const scheduleRetryWakeup = () => {
		if (retryWakeup) return;
		const pending = store.listExtractionJobs("pending");
		if (!pending.length) return;
		const soonest = Math.min(
			...pending.map(job => new Date(job.nextAttemptAt).getTime()),
		);
		const delay = Math.max(0, Math.min(30_000, soonest - Date.now()));
		retryWakeup = setTimeout(() => {
			retryWakeup = null;
			scheduleExtractionJobs();
		}, delay);
		// Never keep the process (or shutdown) waiting on a retry wakeup —
		// it's a convenience nudge, not durable state. The job itself is
		// safe in the DB and gets reclaimed on next startup regardless.
		(retryWakeup as unknown as { unref?: () => void }).unref?.();
	};

	const runExtractionJobs = async () => {
		while (true) {
			const job = store.claimExtractionJob();
			if (!job) {
				scheduleRetryWakeup();
				return;
			}
			try {
				const evidence = JSON.parse(job.payload) as Parameters<
					typeof extractSemanticEpisode
				>[0];
				const episode = await extractSemanticEpisode(
					evidence,
					config.semanticExtractor,
				);
				if (episode) {
					// Stable IDs make recovery idempotent if a process exits between the
					// observation write and job acknowledgement.
					const observationId = `episode:${job.id}`;
					episode.raw.id = observationId;
					episode.compressed.id = observationId;
					const existing = store.getObservation(observationId, job.sessionId);
					const saved =
						existing || store.observe(episode.raw, episode.compressed);
					if (!existing && saved && onObservationSaved) {
						try {
							onObservationSaved(saved);
						} catch {}
					}
					if (saved) {
						await embedEntries([
							{
								id: saved.id,
								kind: "observation",
								text: `${saved.title}\n${saved.narrative}\n${saved.facts.join("\n")}`,
								sessionId: saved.sessionId,
							},
						]);
					}
				}
				if (autoConsolidate) {
					const memories = consolidate(job.sessionId);
					await embedEntries(
						memories.map(memory => ({
							id: memory.id,
							kind: "memory" as const,
							text: `${memory.title}\n${memory.content}`,
							sessionId: memory.sessionIds[0],
						})),
					);
				}
				store.completeExtractionJob(job.id);
			} catch (error) {
				// Backoff is enforced via next_attempt_at in the job row, not by
				// blocking here — sleeping in-loop would stall every other pending
				// job (including ones from unrelated sessions) behind this one's
				// retry delay, and would keep stop()'s Promise.allSettled waiting
				// on the full delay during shutdown. claimExtractionJob() already
				// skips jobs whose next_attempt_at hasn't elapsed, so looping
				// immediately just lets other ready work proceed.
				const delay = Math.min(
					30_000,
					1_000 * 2 ** Math.max(0, job.attempts - 1),
				);
				store.failExtractionJob(
					job.id,
					error instanceof Error ? error.message : String(error),
					delay,
				);
			}
		}
	};

	const scheduleExtractionJobs = () => {
		if (retryWakeup) {
			clearTimeout(retryWakeup);
			retryWakeup = null;
		}
		if (extractionWorker) return extractionWorker;
		const task = Promise.resolve()
			.then(runExtractionJobs)
			.finally(() => {
				extractionWorker = null;
			});
		extractionWorker = task;
		try {
			config.onBackgroundTask?.(task);
		} catch {}
		return task;
	};

	// Capture the user's request at the start of every turn. AgentMemory treats
	// prompts as first-class observations because they preserve intent even when
	// a turn never reaches a tool call.
	const beforeAgentStart = capturePrompts
		? (ctx: { prompt: string }) => {
				const prompt = ctx.prompt?.trim();
				if (!prompt) return undefined;
				latestPrompt = prompt;
				latestAssistantOutcome = "";
				turnTools = [];
				saveObservation(
					store,
					{
						id: crypto.randomUUID(),
						hookType: "prompt_submit",
						userPrompt: prompt,
						raw: { prompt },
					},
					onObservationSaved,
					deduplicate,
				);
				return undefined;
			}
		: undefined;

	// ── afterToolCall: capture observations ─────────────────────────────

	const afterToolCall = captureTools
		? (ctx: {
				toolCall: { name: string; id?: string; arguments?: string };
				args: Record<string, unknown>;
				result: string;
				isError: boolean;
			}) => {
				const toolName =
					ctx.toolCall.name ||
					(ctx.args.tool_name as string) ||
					(ctx.args.name as string) ||
					"unknown";

				// AgentMemory intentionally ignores interrupted failures: they are
				// user/runtime control flow, not evidence about the project.
				if (
					ctx.isError &&
					/^(?:cancelled|canceled|aborted|interrupted)\b/i.test(
						ctx.result.trim(),
					)
				) {
					return undefined;
				}

				turnTools.push({
					id: ctx.toolCall.id || crypto.randomUUID(),
					name: toolName,
					args: ctx.args,
					result: ctx.result,
					isError: ctx.isError,
				});

				saveObservation(
					store,
					{
						id: `${ctx.toolCall.id || crypto.randomUUID()}:post`,
						hookType: ctx.isError ? "post_tool_failure" : "post_tool_use",
						toolName,
						toolInput: ctx.args,
						toolOutput: ctx.result,
						raw: {
							tool_name: toolName,
							tool_input: ctx.args,
							tool_output: ctx.result,
							...(ctx.isError ? { error: ctx.result } : {}),
						},
					},
					onObservationSaved,
					deduplicate,
				);

				return undefined;
			}
		: undefined;

	// ── transformContext: inject session context into messages ──────────

	const transformContext = injectContext
		? async (ctx: { messages: any[]; taskState?: ExplicitTaskState }) => {
				const sessionId = store.getCurrentSessionId();
				if (!sessionId) return undefined;
				const retrieval: ContextRetrievalQuery = ctx.taskState
					? {
							objective: ctx.taskState.objective || latestPrompt,
							phase: ctx.taskState.phase,
							changedFiles: ctx.taskState.changedFiles,
							recentEvidence: ctx.taskState.evidence
								.slice(-6)
								.map(item => item.summary),
							toolFailures: ctx.taskState.toolFailures,
						}
					: { objective: latestPrompt };
				if (config.embedder?.isReady() && retrieval.objective.trim()) {
					try {
						retrieval.semanticVector = await config.embedder.embed(
							retrieval.objective,
						);
					} catch {}
				}
				const sessionContext = store.getContext(
					sessionId,
					contextBudget,
					retrieval,
				);
				if (!sessionContext) return undefined;

				const messages = [
					...ctx.messages.filter(
						message =>
							!(
								message?.role === "system" &&
								typeof message.content === "string" &&
								message.content.startsWith("# Agent Context\n")
							),
					),
					{
						role: "system" as const,
						content: sessionContext,
					},
				];

				return { messages };
			}
		: undefined;

	// ── Hook composition ───────────────────────────────────────────────

	const hooks: AgentHooks = {};

	if (beforeAgentStart) hooks.beforeAgentStart = beforeAgentStart;
	if (afterToolCall) hooks.afterToolCall = afterToolCall;
	if (transformContext) hooks.transformContext = transformContext;
	if (semanticEpisodes) {
		hooks.afterProviderResponse = ctx => {
			if (ctx.content?.trim()) latestAssistantOutcome = ctx.content.trim();
		};
	}
	if (semanticEpisodes || autoConsolidate) {
		hooks.shouldStopAfterTurn = () => {
			if (semanticEpisodes) {
				const sessionId = store.getCurrentSessionId();
				if (sessionId) {
					// Snapshot mutable turn state before returning control to the caller.
					// Jobs are serialized so episode persistence retains turn order.
					let remainingResultChars = 24_000;
					const boundedTools = turnTools
						.slice(-40)
						.reverse()
						.map(tool => {
							const result = tool.result.slice(
								0,
								Math.min(4_000, remainingResultChars),
							);
							remainingResultChars -= result.length;
							return {
								...tool,
								args: compactEvidenceValue(tool.args) as Record<
									string,
									unknown
								>,
								result,
							};
						})
						.reverse();
					const evidence = {
						sessionId,
						timestamp: new Date().toISOString(),
						workspace: store.getCurrentWorkspace(),
						userIntent: latestPrompt,
						assistantOutcome: latestAssistantOutcome,
						// Bound durable queue size and extractor cost. The semantic
						// extractor needs conclusions and errors, not entire build logs.
						tools: boundedTools,
					};
					store.enqueueExtractionJob(
						sessionId,
						evidence.workspace,
						JSON.stringify(evidence),
					);
					void scheduleExtractionJobs();
				}
				turnTools = [];
				latestAssistantOutcome = "";
			}
			if (autoConsolidate && !semanticEpisodes) consolidate();
			return undefined;
		};
	}
	if (autoConsolidate) {
		hooks.beforeCompact = () => {
			consolidate();
			return undefined;
		};
	}

	// Resume persisted work from a prior process after hooks and the extractor
	// backend are available. This remains entirely off the foreground path.
	if (semanticEpisodes && store.listExtractionJobs("pending").length) {
		void scheduleExtractionJobs();
	}
	if (config.embedder) {
		const warmup = config.embedder
			.warmup()
			.then(async () => {
				const memories = store.list({ limit: 1_000, minStrength: 1 });
				const episodes = store
					.listRecentObservations(1_000)
					.filter(observation => observation.id.startsWith("episode:"));
				await embedEntries([
					...memories.map(memory => ({
						id: memory.id,
						kind: "memory" as const,
						text: `${memory.title}\n${memory.content}`,
						sessionId: memory.sessionIds[0],
					})),
					...episodes.map(observation => ({
						id: observation.id,
						kind: "observation" as const,
						text: `${observation.title}\n${observation.narrative}\n${observation.facts.join("\n")}`,
						sessionId: observation.sessionId,
					})),
				]);
			})
			.catch(() => {});
		try {
			config.onBackgroundTask?.(warmup);
		} catch {}
	}

	return hooks;
}
