/** JSON-lines SDK worker client for Memoriam — the memory store RPC layer.

Protocol v1 — exposes the Memoriam MemoryStore as a JSON-lines RPC service.
All requests follow the JSON-lines protocol:
  Input:  {"id":"req-1","method":"observe","session_id":"s1",...}
  Output: {"id":"req-1","ok":true,"result":{...}}

Usage:
  import { MemoriamWorker } from "./worker.ts";
  const worker = new MemoriamWorker(config);
  const result = await worker.observe({ session_id: "s1", ... });
*/

import { JsonlWorker } from "../sdk/jsonl-worker.ts";

// ── Config ───────────────────────────────────────────────────────────────────

export interface MemoriamSdkConfig {
	mode?: "off" | "sdk";
	python?: string;
	args?: string[];
	failOpen?: boolean;
	timeoutMs?: number;
	config?: Record<string, unknown>;
}

// ── Types ────────────────────────────────────────────────────────────────────

// ── Type aliases for memory entities (lightweight — no circular deps) ────────

export interface Session {
	id: string;
	name: string;
	project: string;
	cwd: string;
	workspace: string;
	startedAt: string;
	endedAt: string | null;
	status: string;
	observationCount: number;
	model: string | null;
	tags: string[];
	firstPrompt: string | null;
	summary: string | null;
	commitShas: string[];
}

export interface CompressedObservation {
	id: string;
	sessionId: string;
	timestamp: string;
	type: string;
	title: string;
	subtitle: string | null;
	facts: string[];
	narrative: string;
	concepts: string[];
	files: string[];
	importance: number;
	consolidated: boolean;
	workspace: string | null;
	claims: unknown;
	provenance: Record<string, unknown>;
}

export interface Memory {
	id: string;
	createdAt: string;
	updatedAt: string;
	type: string;
	title: string;
	content: string;
	concepts: string[];
	files: string[];
	sessionIds: string[];
	strength: number;
	version: number;
	parentId: string | null;
	supersedes: string[];
	relatedIds: string[];
	sourceObservationIds: string[];
	isLatest: boolean;
	project: string | null;
	workspace: string;
	accessCount: number;
	lastAccessed: string | null;
	workingTier: string;
}

export interface MemoryRetrievalResult {
	context: string;
	trace: RetrievalTrace;
}

export interface RetrievalTrace {
	sessionId: string;
	query: string;
	budget: number;
	steps: Array<{
		phase: string;
		candidates: number;
		kept: number;
	}>;
	totalTokens: number;
}

export interface SearchResult {
	id: string;
	score: number;
	title: string;
	content: string;
	type: string;
	sessionId: string;
}

export interface MemoryRelation {
	sourceId: string;
	targetId: string;
	type: string;
	confidence: number;
}

export interface ExpandedMemoryEntry {
	id: string;
	type: string;
	title: string;
	content: string;
	concepts: string[];
	files: string[];
	sessionIds: string[];
	sourceObservationIds: string[];
	parentId: string | null;
	supersedes: string[];
	relations: MemoryRelation[];
}

export interface ExportData {
	version: string;
	schemaVersion: number;
	workspace: string;
	sessions: Session[];
	observations: CompressedObservation[];
	memories: Memory[];
	relations: MemoryRelation[];
	embeddings: unknown[];
}

export interface ImportResult {
	importedSessions: number;
	importedObservations: number;
	importedMemories: number;
	importedRelations: number;
	ignored: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

// ── Worker ───────────────────────────────────────────────────────────────────

/** A lazy, persistent JSONL client for Memoriam's Python SDK worker. */
export class MemoriamWorker {
	private readonly transport: JsonlWorker;
	constructor(options: MemoriamSdkConfig) {
		this.transport = new JsonlWorker({
			name: "Memoriam",
			python: options.python,
			args: options.args ?? ["-m", "memoriam.integration.sdk_worker"],
			timeoutMs: options.timeoutMs,
			initialize:
				options.config && Object.keys(options.config).length
					? { method: "init", config: options.config }
					: undefined,
		});
	}

	// ── Sessions ───────────────────────────────────────────────────────────

	async createSession(
		sessionId: string,
		name: string,
		project: string,
		cwd: string,
	): Promise<Session> {
		return this.request({
			method: "create_session",
			session_id: sessionId,
			name,
			project,
			cwd,
		});
	}

	async getSession(sessionId: string): Promise<Session | null> {
		return this.request({
			method: "get_session",
			session_id: sessionId,
		});
	}

	async listSessions(
		query: Record<string, unknown> | null,
	): Promise<Session[]> {
		return this.request({
			method: "list_sessions",
			query,
		});
	}

	async updateSession(
		sessionId: string,
		updates: Record<string, unknown>,
	): Promise<Session | null> {
		return this.request({
			method: "update_session",
			session_id: sessionId,
			updates,
		});
	}

	async clearSessions(keepSessionId: string | null): Promise<void> {
		return this.request({
			method: "clear_sessions",
			keep_session_id: keepSessionId,
		});
	}

	// ── Observations ───────────────────────────────────────────────────────

	async observe(
		sessionId: string,
		hookType: string,
		opts: {
			toolName?: string;
			toolInput?: unknown;
			toolOutput?: unknown;
			userPrompt?: string;
			raw?: unknown;
		} = {},
	): Promise<CompressedObservation | null> {
		return this.request({
			method: "observe",
			session_id: sessionId,
			hook_type: hookType,
			tool_name: opts.toolName,
			tool_input: opts.toolInput,
			tool_output: opts.toolOutput,
			user_prompt: opts.userPrompt,
			raw: opts.raw,
		});
	}

	async listObservations(
		sessionId: string,
		limit: number,
	): Promise<CompressedObservation[]> {
		return this.request({
			method: "list_observations",
			session_id: sessionId,
			limit,
		});
	}

	async searchObservations(
		query: string,
		limit: number,
	): Promise<SearchResult[]> {
		return this.request({
			method: "search_observations",
			query,
			limit,
		});
	}

	async expandEntries(ids: string[]): Promise<ExpandedMemoryEntry[]> {
		return this.request({
			method: "expand_entries",
			ids,
		});
	}

	async clearObservations(): Promise<number> {
		return this.request({
			method: "clear_observations",
		});
	}

	// ── Memories ───────────────────────────────────────────────────────────

	async createMemory(
		content: string,
		opts: {
			type?: string;
			concepts?: string[];
			files?: string[];
			strength?: number;
			sessionIds?: string[];
		} = {},
	): Promise<Memory> {
		return this.request({
			method: "create_memory",
			content,
			type: opts.type,
			concepts: opts.concepts,
			files: opts.files,
			strength: opts.strength,
			session_ids: opts.sessionIds,
		});
	}

	async getMemory(memoryId: string): Promise<Memory | null> {
		return this.request({
			method: "get_memory",
			memory_id: memoryId,
		});
	}

	async listMemories(query: Record<string, unknown> | null): Promise<Memory[]> {
		return this.request({
			method: "list_memories",
			query,
		});
	}

	async removeMemory(memoryId: string): Promise<boolean> {
		return this.request({
			method: "remove_memory",
			memory_id: memoryId,
		});
	}

	async recall(
		query: Record<string, unknown>,
		format: string,
	): Promise<string> {
		return this.request({
			method: "recall",
			query,
			format,
		});
	}

	async consolidate(sessionId: string): Promise<Memory[]> {
		return this.request({
			method: "consolidate",
			session_id: sessionId,
		});
	}

	// ── Retrieval ──────────────────────────────────────────────────────────

	async retrieve(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<MemoryRetrievalResult> {
		return this.request({
			method: "retrieve",
			session_id: sessionId,
			query,
			budget,
		});
	}

	async getContext(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<string> {
		return this.request({
			method: "get_context",
			session_id: sessionId,
			query,
			budget,
		});
	}

	async listTraces(limit: number): Promise<RetrievalTrace[]> {
		return this.request({
			method: "list_traces",
			limit,
		});
	}

	// ── Working memory tiers ───────────────────────────────────────────────

	async autoTier(
		config?: Record<string, unknown>,
	): Promise<Record<string, string>> {
		return this.request({
			method: "auto_tier",
			config,
		});
	}

	async autoForget(
		opts: { ttlMs?: number; minImportance?: number; maxDeletes?: number } = {},
	): Promise<Record<string, unknown>> {
		return this.request({
			method: "auto_forget",
			ttl_ms: opts.ttlMs,
			min_importance: opts.minImportance,
			max_deletes: opts.maxDeletes,
		});
	}

	// ── Relations ──────────────────────────────────────────────────────────

	async relate(
		sourceId: string,
		targetId: string,
		type: string,
		confidence: number,
	): Promise<MemoryRelation | null> {
		return this.request({
			method: "relate",
			source_id: sourceId,
			target_id: targetId,
			type,
			confidence,
		});
	}

	async getRelations(memoryId: string): Promise<MemoryRelation[]> {
		return this.request({
			method: "get_relations",
			memory_id: memoryId,
		});
	}

	// ── Export / Import ────────────────────────────────────────────────────

	async exportData(): Promise<ExportData> {
		return this.request({
			method: "export_data",
		});
	}

	async importData(
		data: ExportData,
		onConflict: string,
	): Promise<ImportResult> {
		return this.request({
			method: "import_data",
			data,
			on_conflict: onConflict,
		});
	}

	// ── Temporal reasoning ─────────────────────────────────────────────────

	async temporalQuery(
		queryText: string,
		workspace?: string,
		queryTime?: string,
		budget: number = 4000,
		limit: number = 50,
	): Promise<Record<string, unknown>[]> {
		return this.request({
			method: "temporal_query",
			query_text: queryText,
			workspace,
			query_time: queryTime,
			budget,
			limit,
		});
	}

	// ── Observability ──────────────────────────────────────────────────────

	/** Get aggregate worker statistics. */
	async workerStats(): Promise<Record<string, unknown>> {
		return this.request({
			method: "worker_stats",
		});
	}

	/** Get recent request history. */
	async workerHistory(
		limit: number,
		offset: number,
	): Promise<Record<string, unknown>> {
		return this.request({
			method: "worker_history",
			limit,
			offset,
		});
	}

	// ── Lifecycle ──────────────────────────────────────────────────────────

	private async request<T>(payload: Record<string, unknown>): Promise<T> {
		const response = await this.transport.request(payload);
		return response.result as T;
	}

	close(): void {
		this.transport.close();
	}
}
