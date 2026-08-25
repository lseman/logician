import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createInterface, type Interface } from "node:readline";

// ── Config ───────────────────────────────────────────────────────────────────

export interface LegroomSdkConfig {
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

/** Response from the SDK worker — v2 format. */
interface WorkerResponse {
	id: string;
	ok: boolean;
	error?: string;
	messages?: unknown[];
	stats?: Record<string, unknown>;
	calibration?: Record<string, unknown>;
	history?: Record<string, unknown>[];
	total?: number;
	hit?: boolean;
	content?: string;
}

export interface CompressResult {
	messages: Record<string, unknown>[];
	tokensBefore: number;
	tokensAfter: number;
	tokensSaved: number;
	transformsApplied: string[];
	warnings: string[];
	metadata?: {
		ccrHashes?: string[];
		phaseReports?: Record<string, unknown>[];
		salienceScoresBefore?: number[];
		salienceScoresAfter?: number[];
		storeStats?: {
			entries: number;
			maxEntries: number;
			totalBytesBefore: number;
			totalBytesAfter: number;
			savings: number;
		};
	};
}

export interface StoreStats {
	entries: number;
	maxEntries: number;
	totalBytesBefore: number;
	totalBytesAfter: number;
	savings: number;
}

export interface CalibrationStatus {
	disabledPhases: string[];
	snapshots: Array<{
		phase: string;
		samples: number;
		successRate: number;
		disabled: boolean;
	}>;
}

export interface WorkerStats {
	totalRequests: number;
	totalTokensBefore: number;
	totalTokensAfter: number;
	totalTokensSaved: number;
	compressionRatio: number;
	strategyCounts: Record<string, number>;
	cacheHits: number;
	cacheMisses: number;
	uptimeSeconds: number;
}

export interface WorkerHistory {
	history: Array<{
		requestId: string;
		timestamp: number;
		model: string;
		messagesBefore: number;
		tokensBefore: number;
		tokensAfter: number;
		tokensSaved: number;
		transformsApplied: string[];
		warnings: string[];
	}>;
	total: number;
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

function buildCompressResult(
	stats: Record<string, unknown>,
): CompressResult {
	const metadata: Record<string, unknown> = {};
	const statsMetadata = stats.metadata as Record<string, unknown> | undefined;
	if (statsMetadata) {
		if (statsMetadata.ccrHashes) metadata.ccrHashes = statsMetadata.ccrHashes as string[];
		if (statsMetadata.phaseReports) metadata.phaseReports = statsMetadata.phaseReports as Record<string, unknown>[];
		if (statsMetadata.salienceScoresBefore) metadata.salienceScoresBefore = statsMetadata.salienceScoresBefore as number[];
		if (statsMetadata.salienceScoresAfter) metadata.salienceScoresAfter = statsMetadata.salienceScoresAfter as number[];
		if (statsMetadata.storeStats) metadata.storeStats = statsMetadata.storeStats;
	}
	return {
		messages: (statsMetadata?.messages as Record<string, unknown>[]) ?? [],
		tokensBefore: stats.tokensBefore as number ?? 0,
		tokensAfter: stats.tokensAfter as number ?? 0,
		tokensSaved: stats.tokensSaved as number ?? 0,
		transformsApplied: stats.transformsApplied as string[] ?? [],
		warnings: stats.warnings as string[] ?? [],
		metadata: Object.keys(metadata).length > 0 ? (metadata as CompressResult["metadata"]) : undefined,
	};
}

// ── Worker ───────────────────────────────────────────────────────────────────

/** A lazy, persistent JSONL client for Legroom's Python SDK worker (v2). */
export class LegroomWorker {
	private process?: ChildProcessWithoutNullStreams;
	private lines?: Interface;
	private readonly pending = new Map<string, PendingRequest>();
	private nextId = 0;
	private stopping = false;
	private stderrTail = "";

	constructor(private readonly options: LegroomSdkConfig) {}

	// ── Compression ────────────────────────────────────────────────────────

	/** Compress messages through the worker. Fail-open by default. */
	async compress(
		messages: Record<string, unknown>[],
		model: string,
	): Promise<Record<string, unknown>[]> {
		try {
			const result = await this._compressFull(messages, model);
			return result.messages;
		} catch (error) {
			if (this.options.failOpen !== false) return messages;
			throw error;
		}
	}

	/** Compress and return full result with stats and metadata. */
	private _compressFull(
		messages: Record<string, unknown>[],
		model: string,
	): Promise<CompressResult> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "compress",
				messages,
				model: model || "gpt-4o",
				config: this.options.config ?? {},
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── CCR Store ──────────────────────────────────────────────────────────

	/** Compress with a named CCR store (enables CCR automatically). */
	async compressWithStore(
		storeId: string,
		messages: Record<string, unknown>[],
		model: string,
	): Promise<CompressResult> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "compress_with_store",
				storeId,
				messages,
				model: model || "gpt-4o",
				config: this.options.config ?? {},
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	/** Retrieve original content from a CCR store by hash. */
	async storeRetrieve(
		storeId: string,
		hash: string,
	): Promise<string> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "store_retrieve",
				storeId,
				hash,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	/** Get CCR store statistics. */
	async storeStats(storeId: string): Promise<StoreStats> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "store_stats",
				storeId,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Cache ──────────────────────────────────────────────────────────────

	/** Query the compression result cache. */
	async cacheGet(key: string): Promise<{ hit: boolean; result?: CompressResult } | null> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "cache_get",
				key,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Calibration ────────────────────────────────────────────────────────

	/** Record quality feedback for phase calibration. */
	async calibrationRecord(
		phaseReports: Record<string, unknown>[],
		quality: number,
	): Promise<CalibrationStatus> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "calibration_record",
				phaseReports,
				quality,
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	/** Query current calibration state. */
	async calibrationStatus(): Promise<CalibrationStatus> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
			const payload = JSON.stringify({
				id,
				method: "calibration_status",
			});
			child.stdin.write(`${payload}\n`, error => {
				if (!error) return;
				const pending = this.pending.get(id);
				if (!pending) return;
				clearTimeout(pending.timer);
				this.pending.delete(id);
				pending.reject(
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	// ── Observability ──────────────────────────────────────────────────────

	/** Get aggregate worker statistics. */
	async workerStats(): Promise<WorkerStats> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
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
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
				);
			});
		});
	}

	/** Get recent request history. */
	async workerHistory(
		limit = 50,
		offset = 0,
	): Promise<WorkerHistory> {
		const child = this.ensureStarted();
		const id = `legroom-${process.pid}-${++this.nextId}`;
		const timeoutMs = this.options.timeoutMs ?? 30_000;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`Legroom SDK request timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.pending.set(id, { resolve: resolve as PendingRequest["resolve"], reject, timer });
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
					new Error(`Unable to write to Legroom SDK worker: ${error.message}`),
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
			this.options.args ?? ["-m", "legroom.sdk_worker"],
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
						`Legroom SDK worker exited (${signal ?? code ?? "unknown"})${detail ? `: ${detail}` : ""}`,
					),
				);
			}
		});
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
			pending.reject(new Error(response.error ?? "Legroom SDK request failed"));
			return;
		}
		// v2: compress returns stats.metadata.messages
		if (response.stats) {
			pending.resolve(buildCompressResult(response.stats));
			return;
		}
		// v2: calibration methods return calibration field
		if (response.calibration !== undefined) {
			pending.resolve(response.calibration);
			return;
		}
		// v2: worker_history returns history + total
		if (response.history !== undefined) {
			pending.resolve({
				history: response.history,
				total: response.total ?? 0,
			});
			return;
		}
		// v2: cache_get returns hit + optional result
		if (response.hit !== undefined) {
			const result = response.stats
				? buildCompressResult(response.stats)
				: undefined;
			pending.resolve({ hit: response.hit as boolean, result });
			return;
		}
		// v2: store_retrieve returns content
		if (response.content !== undefined) {
			pending.resolve(response.content);
			return;
		}
		// Fallback: should not happen with v2 worker
		pending.reject(new Error("Legroom SDK response format unrecognized"));
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
		this.failAll(new Error("Legroom SDK worker closed"));
		const child = this.process;
		this.process = undefined;
		if (!child) return;
		child.stdin.end();
		if (child.exitCode === null) child.kill("SIGTERM");
	}
}
