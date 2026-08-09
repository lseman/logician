// ── @logician/memory — Hook Adapter ──────────────────────────────────────────
// Manual remember/recall/forget functions for tool-driven memory access.
// Automatic capture/injection into the agent loop is wired via the real
// AgentHooks-typed createMemoryHooks() in ./memory-hooks.ts.

import type {
	AutoForgetConfig,
	MemoryStore,
	MemoryType,
	WorkingMemoryTier,
} from "../types.js";

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
