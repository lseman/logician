/** Memoriam gateway — owns enablement, hooks, and worker lifecycle.

Mirrors the LegroomGateway pattern: exposes typed methods over the JSON-lines
worker and injects hooks into the agent config (e.g. auto-observe tool calls,
context retrieval).
*/

import type { AgentConfig } from "@logician/log-core";
import type {
	MemoriamSdkConfig,
	Memory,
	Session,
	CompressedObservation,
} from "./worker.ts";
import { MemoriamWorker } from "./worker.ts";

/** Owns Memoriam enablement, hooks, and worker lifecycle. */
export class MemoriamGateway {
	readonly worker: MemoriamWorker;
	private enabled: boolean;

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
	- `beforeProviderPayload`: retrieve memory context and prepend it to the
	  conversation so the model sees relevant memories every turn.
	*/
	createHooks(
		existingHooks: AgentConfig["hooks"],
	): AgentConfig["hooks"] {
		return {
			...existingHooks,
			beforeProviderPayload: async context => {
				const existing = await existingHooks?.beforeProviderPayload?.(context);
				const payload = existing?.payload ?? context.payload;
				if (!this.enabled) return { payload };
				// Retrieve memory context and inject it as a system note.
				const sessionIds = context.hookSessionId ? [context.hookSessionId] : [];
				if (!sessionIds.length) return { payload };
				const query =
					(payload as { messages?: unknown[] })?.messages?.length
						? "all"
						: "recent";
				try {
					const contextText = await this.worker.getContext(
						sessionIds[0],
						query,
						context.payload?.maxTokens
							? (context.payload.maxTokens as number) * 0.4
							: 16_000,
					);
					if (!contextText) return { payload };
					// Prepend a system-level memory context block.
					const messages = payload.messages as { role: string; content: string }[];
					if (!Array.isArray(messages)) return { payload };
					const injectMsg = {
						role: "system" as const,
						content: `# Memoriam Memory Context\n${contextText}`,
					};
					return {
						payload: {
							...payload,
							messages: [injectMsg, ...messages],
						},
					};
				} catch {
					// Fail open — return payload unchanged if memory retrieval fails.
					return { payload };
				}
			},
		};
	}

	// ── Session operations ────────────────────────────────────────────────

	async createSession(
		id: string,
		name: string,
		project: string,
		cwd: string,
	): Promise<Session> {
		this.assertEnabled();
		return this.worker.createSession(id, name, project, cwd);
	}

	async getSession(id: string): Promise<Session | null> {
		this.assertEnabled();
		return this.worker.getSession(id);
	}

	async listSessions(
		query?: Record<string, unknown>,
	): Promise<Session[]> {
		this.assertEnabled();
		return this.worker.listSessions(query ?? null);
	}

	async updateSession(
		id: string,
		updates: Record<string, unknown>,
	): Promise<Session | null> {
		this.assertEnabled();
		return this.worker.updateSession(id, updates);
	}

	async clearSessions(keepSessionId?: string | null): Promise<void> {
		this.assertEnabled();
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
		return this.worker.observe(sessionId, hookType, opts);
	}

	async listObservations(
		sessionId: string,
		limit: number,
	): Promise<CompressedObservation[]> {
		this.assertEnabled();
		return this.worker.listObservations(sessionId, limit);
	}

	async searchObservations(
		query: string,
		limit: number,
	): Promise<unknown[]> {
		this.assertEnabled();
		return this.worker.searchObservations(query, limit);
	}

	async clearObservations(): Promise<number> {
		this.assertEnabled();
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

	async autoTier(config?: Record<string, unknown>): Promise<Record<string, string>> {
		this.assertEnabled();
		return this.worker.autoTier(config);
	}

	async autoForget(
		opts?: { ttlMs?: number; minImportance?: number; maxDeletes?: number },
	): Promise<Record<string, unknown>> {
		this.assertEnabled();
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

	async importData(
		data: unknown,
		onConflict: string,
	): Promise<unknown> {
		this.assertEnabled();
		return this.worker.importData(data as any, onConflict);
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

	async workerHistory(limit: number, offset: number): Promise<Record<string, unknown>> {
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
