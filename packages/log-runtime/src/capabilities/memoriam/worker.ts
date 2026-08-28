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

import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface, type Interface } from "node:readline";

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

interface PendingRequest {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
}

/** Response from the Memoriam SDK worker. */
interface WorkerResponse {
	id: string;
	ok: boolean;
	error?: string;
	result?: unknown;
}

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

function parseResponse(line: string): WorkerResponse | undefined {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch {
		return undefined;
	}
	if (!value || typeof value !== "object") return undefined;
	const response = value as Record<string, unknown>;
	if (typeof response.id !== "string" || typeof response.ok !== "boolean")
		return undefined;
	return response as unknown as WorkerResponse;
}

// ── Worker ───────────────────────────────────────────────────────────────────

/** A lazy, persistent JSONL client for Memoriam's Python SDK worker. */
export class MemoriamWorker {
	private process?: ChildProcessWithoutNullStreams;
	private lines?: Interface;
	private readonly pending = new Map<string, PendingRequest>();
	private nextId = 0;
	private stopping = false;
	private stderrTail = "";

	constructor(private readonly options: MemoriamSdkConfig) {}

	// ── Sessions ───────────────────────────────────────────────────────────

	async createSession(
		sessionId: string,
		name: string,
		project: string,
		cwd: string,
	): Promise<Session> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "create_session",
				session_id: sessionId,
				name,
				project,
				cwd,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async getSession(sessionId: string): Promise<Session | null> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "get_session",
				session_id: sessionId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async listSessions(
		query: Record<string, unknown> | null,
	): Promise<Session[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "list_sessions",
				query,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async updateSession(
		sessionId: string,
		updates: Record<string, unknown>,
	): Promise<Session | null> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "update_session",
				session_id: sessionId,
				updates,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async clearSessions(keepSessionId: string | null): Promise<void> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "clear_sessions",
				keep_session_id: keepSessionId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
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
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "observe",
				session_id: sessionId,
				hook_type: hookType,
				tool_name: opts.toolName,
				tool_input: opts.toolInput,
				tool_output: opts.toolOutput,
				user_prompt: opts.userPrompt,
				raw: opts.raw,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async listObservations(
		sessionId: string,
		limit: number,
	): Promise<CompressedObservation[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "list_observations",
				session_id: sessionId,
				limit,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async searchObservations(
		query: string,
		limit: number,
	): Promise<SearchResult[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "search_observations",
				query,
				limit,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async expandEntries(ids: string[]): Promise<ExpandedMemoryEntry[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "expand_entries",
				ids,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async clearObservations(): Promise<number> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "clear_observations",
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
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
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "create_memory",
				content,
				type: opts.type,
				concepts: opts.concepts,
				files: opts.files,
				strength: opts.strength,
				session_ids: opts.sessionIds,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async getMemory(memoryId: string): Promise<Memory | null> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "get_memory",
				memory_id: memoryId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async listMemories(query: Record<string, unknown> | null): Promise<Memory[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "list_memories",
				query,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async removeMemory(memoryId: string): Promise<boolean> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "remove_memory",
				memory_id: memoryId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async recall(
		query: Record<string, unknown>,
		format: string,
	): Promise<string> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "recall",
				query,
				format,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async consolidate(sessionId: string): Promise<Memory[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "consolidate",
				session_id: sessionId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Retrieval ──────────────────────────────────────────────────────────

	async retrieve(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<MemoryRetrievalResult> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "retrieve",
				session_id: sessionId,
				query,
				budget,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async getContext(
		sessionId: string,
		query: string,
		budget: number,
	): Promise<string> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "get_context",
				session_id: sessionId,
				query,
				budget,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async listTraces(limit: number): Promise<RetrievalTrace[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "list_traces",
				limit,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Working memory tiers ───────────────────────────────────────────────

	async autoTier(
		config?: Record<string, unknown>,
	): Promise<Record<string, string>> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "auto_tier",
				config,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async autoForget(
		opts: { ttlMs?: number; minImportance?: number; maxDeletes?: number } = {},
	): Promise<Record<string, unknown>> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "auto_forget",
				ttl_ms: opts.ttlMs,
				min_importance: opts.minImportance,
				max_deletes: opts.maxDeletes,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Relations ──────────────────────────────────────────────────────────

	async relate(
		sourceId: string,
		targetId: string,
		type: string,
		confidence: number,
	): Promise<MemoryRelation | null> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "relate",
				source_id: sourceId,
				target_id: targetId,
				type,
				confidence,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async getRelations(memoryId: string): Promise<MemoryRelation[]> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "get_relations",
				memory_id: memoryId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Export / Import ────────────────────────────────────────────────────

	async exportData(): Promise<ExportData> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "export_data",
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	async importData(
		data: ExportData,
		onConflict: string,
	): Promise<ImportResult> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "import_data",
				data,
				on_conflict: onConflict,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
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
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "temporal_query",
				query_text: queryText,
				workspace,
				query_time: queryTime,
				budget,
				limit,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Observability ──────────────────────────────────────────────────────

	/** Get aggregate worker statistics. */
	async workerStats(): Promise<Record<string, unknown>> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "worker_stats",
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	/** Get recent request history. */
	async workerHistory(
		limit: number,
		offset: number,
	): Promise<Record<string, unknown>> {
		const child = this.ensureStarted();
		const id = `memoriam-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(
					new Error(`Memoriam SDK request timed out after ${timeoutMs}ms`),
				);
			}, timeoutMs);
			this.pending.set(id, {
				resolve: resolve as PendingRequest["resolve"],
				reject,
				timer,
			});
			const payload = JSON.stringify({
				id,
				method: "worker_history",
				limit,
				offset,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Memoriam SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Lifecycle ──────────────────────────────────────────────────────────

	private ensureStarted(): ChildProcessWithoutNullStreams {
		if (this.process && this.process.exitCode === null) return this.process;
		this.stopping = false;
		this.stderrTail = "";
		const child = spawn(
			this.options.python ?? "python3",
			this.options.args ?? ["-m", "memoriam.integration.sdk_worker"],
			{ stdio: ["pipe", "pipe", "pipe"] },
		);
		this.process = child;
		this.lines = createInterface({ input: child.stdout });
		this.lines.on("line", line => this.handleLine(line));
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk: string) => {
			this.stderrTail = `${this.stderrTail}${chunk}`.slice(-4_096);
		});
		child.on("error", error => this.failAll(error));
		child.on("exit", (code, signal) => {
			if (this.process === child) this.process = undefined;
			if (!this.stopping) {
				const detail = this.stderrTail.trim();
				this.failAll(
					new Error(
						`Memoriam SDK worker exited (${signal ?? code ?? "unknown"})${detail ? `: ${detail}` : ""}`,
					),
				);
			}
		});
		// Hand the worker its configuration (db_path, fail_open) up front. The
		// worker also accepts config on `init`, so fire-and-forget: the response
		// is ignored and later requests queue behind it on the same pipe.
		if (this.options.config && Object.keys(this.options.config).length > 0) {
			const initId = `memoriam-${process.pid}-init-${++this.nextId}`;
			const initPayload = JSON.stringify({
				id: initId,
				method: "init",
				config: this.options.config,
			});
			child.stdin.write(`${initPayload}\n`, () => {});
		}
		return child;
	}

	private handleLine(line: string): void {
		const response = parseResponse(line);
		if (!response) return;
		const pending = this.pending.get(response.id);
		if (!pending) return;
		clearTimeout(pending.timer);
		this.pending.delete(response.id);
		if (!response.ok) {
			pending.reject(
				new Error(response.error ?? "Memoriam SDK request failed"),
			);
			return;
		}
		// All responses use the "result" field.
		pending.resolve(response.result);
	}

	private failAll(error: Error): void {
		for (const pending of this.pending.values()) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}
		this.pending.clear();
	}

	close(): void {
		this.stopping = true;
		this.lines?.close();
		this.lines = undefined;
		this.failAll(new Error("Memoriam SDK worker closed"));
		const child = this.process;
		this.process = undefined;
		if (!child) return;
		child.stdin.end();
		if (child.exitCode === null) child.kill("SIGTERM");
	}
}
