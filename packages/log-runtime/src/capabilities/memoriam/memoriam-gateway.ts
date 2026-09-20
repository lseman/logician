/** Memoriam gateway — owns enablement, hooks, and worker lifecycle.

Mirrors the LegroomGateway pattern: exposes typed methods over the JSON-lines
worker and injects hooks into the agent config (e.g. auto-observe tool calls,
context retrieval).
*/

import type { AgentConfig, AgentHooks } from "@logician/log-core";
import type {
	CompressedObservation,
	ExportData,
	MemoriamSdkConfig,
	Memory,
	Session,
} from "./worker.ts";
import { MemoriamWorker } from "./worker.ts";

/** Owns Memoriam enablement, hooks, and worker lifecycle. */
export class MemoriamGateway {
	private readonly worker: MemoriamWorker;
	private enabled: boolean;
	/** Bumped whenever the memory store may have changed. Keys the
	 *  injected-context cache so the (relatively expensive) retrieval only
	 *  re-runs when memories actually change, keeping the injected block
	 *  byte-stable across turns for prompt-cache stability. */
	private memoryRevision = 0;
	/** Cached get_context results per (session, query, budget) key, valid
	 *  until the memory revision bumps. */
	private readonly contextCache = new Map<
		string,
		{ revision: number; text: string }
	>();

	constructor(options: MemoriamSdkConfig = {}) {
		this.worker = new MemoriamWorker(options);
		this.enabled = options.mode === "sdk";
	}

	isEnabled(): boolean {
		return this.enabled;
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		if (!enabled) this.worker.close();
	}

	/** Inject Memoriam hooks into the agent config.

	Calls:
	- `beforeProviderPayload`: retrieve memory context and append it to the
	  conversation so the model sees relevant memories every turn.

	The block is appended (not prepended) so the leading system prompt and
	history — the provider's cacheable prefix — are never disturbed; see
	`normalizeProviderMessages` in log-core's backend, which re-roles the
	trailing system message at the transport boundary. The retrieval itself
	is revision-keyed: it only re-runs when the memory store actually
	changed (any mutation through this gateway bumps the revision), so the
	injected text stays byte-stable across turns.
	*/
	createHooks(existingHooks: AgentConfig["hooks"]): AgentHooks {
		return {
			...existingHooks,
			beforeProviderPayload: async context => {
				const existing = await existingHooks?.beforeProviderPayload?.(context);
				const payload = existing?.payload ?? context.payload;
				if (!this.enabled) return { payload };
				// Retrieve memory context and inject it as a system note.
				const sessionIds = context.hookSessionId ? [context.hookSessionId] : [];
				if (!sessionIds.length) return { payload };
				const sessionId = sessionIds[0];
				const query = (payload as { messages?: unknown[] })?.messages?.length
					? "all"
					: "recent";
				const budget = context.payload?.maxTokens
					? (context.payload.maxTokens as number) * 0.4
					: 16_000;
				try {
					const cacheKey = `${sessionId}|${query}|${budget}`;
					const cached = this.contextCache.get(cacheKey);
					const contextText =
						cached && cached.revision === this.memoryRevision
							? cached.text
							: await this.worker.getContext(sessionId, query, budget);
					this.contextCache.set(cacheKey, {
						revision: this.memoryRevision,
						text: contextText,
					});
					if (!contextText) return { payload };
					// Append a trailing system context block.
					const messages = payload.messages as {
						role: string;
						content: string;
					}[];
					if (!Array.isArray(messages)) return { payload };
					const injectMsg = {
						role: "system" as const,
						content: `# Memoriam Memory Context\n${contextText}`,
					};
					return {
						payload: {
							...payload,
							messages: [...messages, injectMsg],
						},
					};
				} catch {
					// Fail open — return payload unchanged if memory retrieval fails.
					// The cache is not updated on failure, so the next call retries.
					return { payload };
				}
			},
		};
	}

	/** Invalidate cached context — called by every store-mutating method. */
	private bumpMemoryRevision(): void {
		this.memoryRevision += 1;
	}

	// ── Session operations ────────────────────────────────────────────────

	async createSession(
		id: string,
		name: string,
		project: string,
		cwd: string,
	): Promise<Session> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.createSession(id, name, project, cwd);
	}

	async getSession(id: string): Promise<Session | null> {
		this.assertEnabled();
		return this.worker.getSession(id);
	}

	async listSessions(query?: Record<string, unknown>): Promise<Session[]> {
		this.assertEnabled();
		return this.worker.listSessions(query ?? null);
	}

	async updateSession(
		id: string,
		updates: Record<string, unknown>,
	): Promise<Session | null> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.updateSession(id, updates);
	}

	async clearSessions(keepSessionId?: string | null): Promise<void> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.clearSessions(keepSessionId ?? null);
	}

	// ── Observation operations ────────────────────────────────────────────

	async observe(
		sessionId: string,
		hookType: string,
		opts?: {
			toolName?: string;
			toolInput?: unknown;
			toolOutput?: unknown;
			userPrompt?: string;
			raw?: unknown;
		},
	): Promise<CompressedObservation | null> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.observe(sessionId, hookType, opts);
	}

	async listObservations(
		sessionId: string,
		limit: number,
	): Promise<CompressedObservation[]> {
		this.assertEnabled();
		return this.worker.listObservations(sessionId, limit);
	}

	async searchObservations(query: string, limit: number): Promise<unknown[]> {
		this.assertEnabled();
		return this.worker.searchObservations(query, limit);
	}

	async clearObservations(): Promise<number> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.clearObservations();
	}

	// ── Memory operations ─────────────────────────────────────────────────

	async createMemory(
		content: string,
		opts?: {
			type?: string;
			concepts?: string[];
			files?: string[];
			strength?: number;
			sessionIds?: string[];
		},
	): Promise<Memory> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.createMemory(content, opts ?? {});
	}

	async getMemory(id: string): Promise<Memory | null> {
		this.assertEnabled();
		return this.worker.getMemory(id);
	}

	async listMemories(query?: Record<string, unknown>): Promise<Memory[]> {
		this.assertEnabled();
		return this.worker.listMemories(query ?? null);
	}

	async removeMemory(id: string): Promise<boolean> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.removeMemory(id);
	}

	async recall(
		query: Record<string, unknown>,
		format: string,
	): Promise<string> {
		this.assertEnabled();
		return this.worker.recall(query, format);
	}

	async consolidate(sessionId: string): Promise<Memory[]> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.consolidate(sessionId);
	}

	// ── Retrieval ─────────────────────────────────────────────────────────

	async retrieve(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<unknown> {
		this.assertEnabled();
		return this.worker.retrieve(sessionId, query, budget);
	}

	async getContext(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<string> {
		this.assertEnabled();
		return this.worker.getContext(sessionId, query, budget);
	}

	// ── Working memory ───────────────────────────────────────────────────

	async autoTier(
		config?: Record<string, unknown>,
	): Promise<Record<string, string>> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.autoTier(config);
	}

	async autoForget(opts?: {
		ttlMs?: number;
		minImportance?: number;
		maxDeletes?: number;
	}): Promise<Record<string, unknown>> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.autoForget(opts);
	}

	// ── Relations ─────────────────────────────────────────────────────────

	async relate(
		sourceId: string,
		targetId: string,
		type: string,
		confidence: number,
	): Promise<unknown> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.relate(sourceId, targetId, type, confidence);
	}

	async getRelations(memoryId: string): Promise<unknown[]> {
		this.assertEnabled();
		return this.worker.getRelations(memoryId);
	}

	// ── Export / Import ───────────────────────────────────────────────────

	async exportData(): Promise<unknown> {
		this.assertEnabled();
		return this.worker.exportData();
	}

	async importData(data: ExportData, onConflict: string): Promise<unknown> {
		this.assertEnabled();
		this.bumpMemoryRevision();
		return this.worker.importData(data, onConflict);
	}

	// ── Temporal reasoning ────────────────────────────────────────────────

	async temporalQuery(
		queryText: string,
		workspace?: string,
		queryTime?: string,
		budget?: number,
		limit?: number,
	): Promise<unknown[]> {
		this.assertEnabled();
		return this.worker.temporalQuery(
			queryText,
			workspace,
			queryTime,
			budget,
			limit,
		);
	}

	// ── Observability ─────────────────────────────────────────────────────

	async workerStats(): Promise<Record<string, unknown>> {
		this.assertEnabled();
		return this.worker.workerStats();
	}

	async workerHistory(
		limit: number,
		offset: number,
	): Promise<Record<string, unknown>> {
		this.assertEnabled();
		return this.worker.workerHistory(limit, offset);
	}

	close(): void {
		this.worker.close();
	}

	private assertEnabled(): void {
		if (!this.enabled) throw new Error("Memoriam SDK is not enabled");
	}
}
