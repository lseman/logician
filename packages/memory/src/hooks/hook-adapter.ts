// ── @logician/memory — Hook Adapter ──────────────────────────────────────────
// Manual remember/recall/forget functions for tool-driven memory access.
// Automatic capture/injection into an agent loop is exposed through the
// host-neutral structural hooks returned by createMemoryHooks().

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
 * Prioritizes hot memories, then warm, then cold, using each memory's real
 * (retention-scored) working_tier rather than raw search order.
 */
export function recallWithTier(
	store: MemoryStore,
	query: string,
	limit: number = 10,
	_format?: "text" | "system-prompt" | "markdown",
): string {
	const tierRank: Record<WorkingMemoryTier, number> = {
		hot: 0,
		warm: 1,
		cold: 2,
		archived: 3,
	};

	// Over-fetch once, then sort by real tier — avoids the N-tier-queries
	// bug of re-running the same search per tier with no actual filter.
	const results = store.list({
		search: query,
		limit: Math.max(limit * 3, limit),
	});
	const ranked = results
		.map(m => ({ memory: m, tier: store.getWorkingMemoryTier(m.id) }))
		.sort((a, b) => tierRank[a.tier] - tierRank[b.tier]);

	const text = ranked
		.slice(0, limit)
		.map(r => r.memory.content)
		.join("\n\n");
	return text || `No memories found matching "${query}"`;
}
