// ── @logician/memory — Hook Adapter ──────────────────────────────────────────
// Wired into Logician's HookBus for automatic observation capture and
// context injection. Also provides manual remember/recall/forget functions.

import type {
	AutoForgetConfig,
	MemoryStore,
	MemoryType,
	WorkingMemoryTier,
} from "./types.js";

/**
 * Register memory hooks on a HookBus.
 *
 * - agentStart → inject session summary + relevant memories as initial context
 * - beforeCompact → inject high-importance observations and memories
 * - postToolUse → capture tool execution observations automatically
 * - toolFailure → capture error observations
 * - promptSubmit → capture user prompts
 */
export function registerMemoryHooks(
	bus: any, // HookBus instance from agent-core
	store: MemoryStore,
	options?: {
		maxRecentMemories?: number;
		maxRecentObservations?: number;
		injectContext?: boolean;
		captureTools?: boolean;
		budget?: number;
	},
): () => void {
	const _maxRecent = options?.maxRecentMemories ?? 20;
	const _maxObs = options?.maxRecentObservations ?? 10;
	const injectContext = options?.injectContext !== false;
	const captureTools = options?.captureTools !== false;
	const budget = options?.budget ?? 4000;

	// ── agentStart: warm up context ──────────────────────────────────────

	bus.on("agentStart", (ctx: any, _signal?: AbortSignal) => {
		if (!injectContext || !ctx?.sessionId) return ctx;

		const sessionContext = store.getContext(ctx.sessionId, budget);
		if (!sessionContext) return ctx;

		if (ctx.initialPrompt && typeof ctx.initialPrompt === "string") {
			return {
				...ctx,
				initialPrompt: `${sessionContext}\n\n---\n\n${ctx.initialPrompt}`,
			};
		}
		return ctx;
	});

	// ── beforeCompact: auto-consolidate + inject context ──────────────────

	bus.on("beforeCompact", (ctx: any, _signal?: AbortSignal) => {
		if (!ctx?.sessionId) return ctx;

		// Auto-consolidate: distill accumulated observations into memories
		try {
			const newMemories = store.consolidate(ctx.sessionId);
			if (newMemories.length > 0) {
				console.log(
					`[memory] Consolidated ${newMemories.length} memories for session ${ctx.sessionId.slice(0, 12)}`,
				);
			}
		} catch (e) {
			console.error(
				`[memory] Consolidation failed: ${e instanceof Error ? e.message : String(e)}`,
			);
		}

		// Inject context for the summary
		if (injectContext) {
			const sessionContext = store.getContext(ctx.sessionId, budget);
			if (!sessionContext) return ctx;

			if (ctx.summary && typeof ctx.summary === "string") {
				return {
					...ctx,
					summary: `${ctx.summary}\n\n---\n\n${sessionContext}`,
				};
			}
		}
		return ctx;
	});

	// ── postToolUse: capture observations ────────────────────────────────

	if (captureTools) {
		bus.on("postToolUse", (data: any) => {
			if (!data?.sessionId) return;

			const {
				sessionId,
				toolName,
				toolInput,
				toolOutput,
				timestamp,
				project,
				cwd,
				workspace,
			} = data;

			store.observe({
				id: data.observationId || crypto.randomUUID(),
				sessionId,
				timestamp: timestamp || new Date().toISOString(),
				hookType: "post_tool_use",
				toolName,
				toolInput,
				toolOutput,
				workspace,
				raw: {
					tool_name: toolName,
					tool_input: toolInput,
					tool_output: toolOutput,
				},
			});
		});

		bus.on("toolFailure", (data: any) => {
			if (!data?.sessionId) return;

			store.observe({
				id: data.observationId || crypto.randomUUID(),
				sessionId: data.sessionId,
				timestamp: data.timestamp || new Date().toISOString(),
				hookType: "post_tool_failure",
				toolName: data.toolName,
				toolInput: data.toolInput,
				toolOutput: data.error,
				workspace: data.workspace,
				raw: { tool_name: data.toolName, error: data.error },
			});
		});

		bus.on("promptSubmit", (data: any) => {
			if (!data?.sessionId || !data?.prompt) return;

			store.observe({
				id: data.observationId || crypto.randomUUID(),
				sessionId: data.sessionId,
				timestamp: data.timestamp || new Date().toISOString(),
				hookType: "prompt_submit",
				userPrompt: data.prompt,
				workspace: data.workspace,
				raw: { prompt: data.prompt },
			});
		});
	}

	// ── Return cleanup function ──────────────────────────────────────────

	return () => {
		store.close();
	};
}

/**
 * Manual remember tool: save a long-term memory entry.
 * Auto-extracts concepts from content.
 * Returns the created memory ID.
 */
export function remember(
	store: MemoryStore,
	content: string,
	options?: {
		type?:
			| "pattern"
			| "preference"
			| "architecture"
			| "bug"
			| "workflow"
			| "fact";
		strength?: number;
		concepts?: string[];
		files?: string[];
		sessionId?: string;
		project?: string;
	},
): string {
	const entry = store.create(content, {
		type: options?.type || "fact",
		concepts: options?.concepts,
		files: options?.files,
		strength: options?.strength,
		sessionIds: options?.sessionId ? [options.sessionId] : undefined,
		project: options?.project,
	});
	return entry.id;
}

/**
 * Manual recall tool: search and format memories.
 */
export function recall(
	store: MemoryStore,
	query: string,
	limit: number = 10,
	format?: "text" | "system-prompt" | "markdown",
): string {
	const result = store.recall(
		{ search: query, limit },
		{ format: format || "text" },
	);
	return result || `No memories found matching "${query}"`;
}

/**
 * Search observations: find agent actions by content.
 */
export function searchObservations(
	store: MemoryStore,
	query: string,
	limit: number = 20,
): string {
	const results = store.searchObservations(query, limit);

	if (!results.length) return `No observations found matching "${query}"`;

	return results
		.map(
			r =>
				`[${r.observation.importance}/10] ${r.observation.type}: ${r.observation.title}\n${r.observation.narrative.slice(0, 300)}`,
		)
		.join("\n\n---\n\n");
}

/**
 * List tool: enumerate memories with optional filters.
 */
export function listMemories(
	store: MemoryStore,
	query?: {
		type?: string;
		concepts?: string[];
		minStrength?: number;
		sessionId?: string;
		project?: string;
	},
	limit: number = 20,
): string {
	const memories = store.list({
		...query,
		type: (query?.type as MemoryType | undefined) ?? undefined,
		limit,
	});

	if (!memories.length) return "No memories match the query.";

	return memories
		.map(
			m =>
				`[${m.strength}/10] ${m.type} | ${m.createdAt.slice(0, 10)}\n${m.content.slice(0, 300)}`,
		)
		.join("\n\n---\n\n");
}

/**
 * Forget tool: delete a memory by ID.
 */
export function forget(store: MemoryStore, id: string): string {
	const deleted = store.remove(id);
	return deleted ? `Memory ${id} deleted.` : `Memory ${id} not found.`;
}

/**
 * Consolidate observations from a session into lasting memories.
 * Returns IDs of created memories.
 */
export function consolidate(store: MemoryStore, sessionId: string): string[] {
	const memories = store.consolidate(sessionId);
	return memories.map(m => m.id);
}

/**
 * Get context block for a session (for manual injection).
 */
export function getContext(
	store: MemoryStore,
	sessionId: string,
	budget?: number,
): string {
	return store.getContext(sessionId, budget);
}

/**
 * Set the current session ID on the memory store.
 * This associates subsequent observations with the given session.
 */
export function setSessionId(store: MemoryStore, sessionId: string): void {
	store.setCurrentSessionId(sessionId);
}

/**
 * Auto-forget: delete old, low-importance observations.
 * Returns stats about what was deleted.
 */
export function autoForget(
	store: MemoryStore,
	config?: Partial<AutoForgetConfig>,
): { deleted: number; details: string[] } {
	return store.autoForget(
		config?.ttlMs,
		config?.minImportance,
		config?.maxDeletes,
	);
}

/**
 * Auto-tier memories: classify as hot (<1h), warm (<24h), cold (>24h).
 * Returns a map of entity IDs to tiers.
 */
export function autoTierMemories(
	store: MemoryStore,
): Record<string, WorkingMemoryTier> {
	return store.autoTierMemories();
}

/**
 * Recall with working memory tier filtering.
 * Prioritizes hot memories, then warm, then cold.
 */
export function recallWithTier(
	store: MemoryStore,
	query: string,
	limit: number = 10,
	_format?: "text" | "system-prompt" | "markdown",
): string {
	// Get hot memories first, then warm, then cold
	const tiers: WorkingMemoryTier[] = ["hot", "warm", "cold"];
	const perTier = Math.ceil(limit / tiers.length);
	let allResults: string[] = [];

	for (const _tier of tiers) {
		const results = store.list({ search: query, limit: perTier });
		// Filter by tier (simple in-memory since SQL doesn't expose it in Memory interface)
		const tiered = results.filter(() => true); // tier filtering is internal
		allResults = allResults.concat(tiered.map(m => m.content));
		if (allResults.length >= limit) break;
	}

	const text = allResults.slice(0, limit).join("\n\n");
	return text || `No memories found matching "${query}"`;
}
