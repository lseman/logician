import { JsonlWorker } from "../sdk/jsonl-worker.ts";

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

/** Normalize schema field names only; message bodies and keyed maps stay untouched. */
function fields(value: Record<string, unknown>): Record<string, unknown> {
	return Object.fromEntries(
		Object.entries(value).map(([key, item]) => [
			key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase()),
			item,
		]),
	);
}

function buildCompressResult(
	rawStats: Record<string, unknown>,
	messages?: unknown[],
): CompressResult {
	const stats = fields(rawStats);
	const metadata: Record<string, unknown> = {};
	const statsMetadata = stats.metadata
		? fields(stats.metadata as Record<string, unknown>)
		: undefined;
	if (statsMetadata) {
		if (statsMetadata.ccrHashes)
			metadata.ccrHashes = statsMetadata.ccrHashes as string[];
		if (statsMetadata.phaseReports)
			metadata.phaseReports = statsMetadata.phaseReports as Record<
				string,
				unknown
			>[];
		if (statsMetadata.salienceScoresBefore)
			metadata.salienceScoresBefore =
				statsMetadata.salienceScoresBefore as number[];
		if (statsMetadata.salienceScoresAfter)
			metadata.salienceScoresAfter =
				statsMetadata.salienceScoresAfter as number[];
		if (statsMetadata.storeStats)
			metadata.storeStats = fields(
				statsMetadata.storeStats as Record<string, unknown>,
			);
	}
	return {
		messages:
			((statsMetadata?.messages ?? messages) as Record<string, unknown>[]) ??
			[],
		tokensBefore: (stats.tokensBefore as number) ?? 0,
		tokensAfter: (stats.tokensAfter as number) ?? 0,
		tokensSaved: (stats.tokensSaved as number) ?? 0,
		transformsApplied: (stats.transformsApplied as string[]) ?? [],
		warnings: (stats.warnings as string[]) ?? [],
		metadata:
			Object.keys(metadata).length > 0
				? (metadata as CompressResult["metadata"])
				: undefined,
	};
}

// ── Worker ───────────────────────────────────────────────────────────────────

/** A lazy, persistent JSONL client for Legroom's Python SDK worker (v2). */
export class LegroomWorker {
	private readonly transport: JsonlWorker;
	constructor(private readonly options: LegroomSdkConfig) {
		this.transport = new JsonlWorker({
			name: "Legroom",
			python: options.python,
			args: options.args ?? ["-m", "legroom.integration.sdk_worker"],
			timeoutMs: options.timeoutMs,
		});
	}

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
		return this.request({
			method: "compress",
			messages,
			model: model || "gpt-4o",
			config: this.options.config ?? {},
		});
	}

	// ── CCR Store ──────────────────────────────────────────────────────────

	/** Compress with a named CCR store (enables CCR automatically). */
	async compressWithStore(
		storeId: string,
		messages: Record<string, unknown>[],
		model: string,
	): Promise<CompressResult> {
		return this.request({
			method: "compress_with_store",
			store_id: storeId,
			messages,
			model: model || "gpt-4o",
			config: this.options.config ?? {},
		});
	}

	/** Retrieve original content from a CCR store by hash. */
	async storeRetrieve(storeId: string, hash: string): Promise<string> {
		return this.request({
			method: "store_retrieve",
			store_id: storeId,
			hash,
		});
	}

	/** Get CCR store statistics. */
	async storeStats(storeId: string): Promise<StoreStats> {
		return this.request({
			method: "store_stats",
			store_id: storeId,
		});
	}

	// ── Cache ──────────────────────────────────────────────────────────────

	/** Query the compression result cache. */
	async cacheGet(
		key: string,
	): Promise<{ hit: boolean; result?: CompressResult } | null> {
		return this.request({
			method: "cache_get",
			key,
		});
	}

	// ── Calibration ────────────────────────────────────────────────────────

	/** Record quality feedback for phase calibration. */
	async calibrationRecord(
		phaseReports: Record<string, unknown>[],
		quality: number,
	): Promise<CalibrationStatus> {
		return this.request({
			method: "calibration_record",
			phase_reports: phaseReports,
			quality,
		});
	}

	/** Query current calibration state. */
	async calibrationStatus(): Promise<CalibrationStatus> {
		return this.request({
			method: "calibration_status",
		});
	}

	// ── Observability ──────────────────────────────────────────────────────

	/** Get aggregate worker statistics. */
	async workerStats(): Promise<WorkerStats> {
		return this.request({
			method: "worker_stats",
		});
	}

	/** Get recent request history. */
	async workerHistory(limit = 50, offset = 0): Promise<WorkerHistory> {
		return this.request({
			method: "worker_history",
			limit,
			offset,
		});
	}

	// ── Lifecycle ──────────────────────────────────────────────────────────

	private async request<T>(payload: Record<string, unknown>): Promise<T> {
		const response = await this.transport.request(payload);
		return this.decode(
			response as unknown as WorkerResponse,
			payload.method,
		) as T;
	}

	private decode(response: WorkerResponse, method: unknown): unknown {
		switch (method) {
			case "compress":
			case "compress_with_store":
				if (response.stats)
					return buildCompressResult(response.stats, response.messages);
				break;
			case "store_stats":
			case "worker_stats":
				if (response.stats) return fields(response.stats);
				break;
			case "calibration_status":
			case "calibration_record":
				if (response.calibration) {
					const calibration = fields(response.calibration);
					return {
						...calibration,
						snapshots: Array.isArray(calibration.snapshots)
							? calibration.snapshots.map(snapshot => fields(snapshot))
							: [],
					};
				}
				break;
			case "worker_history":
				if (response.history)
					return {
						history: response.history.map(fields),
						total: response.total ?? 0,
					};
				break;
			case "cache_get":
				if (response.hit !== undefined)
					return {
						hit: response.hit,
						result: response.stats
							? buildCompressResult(response.stats, response.messages)
							: undefined,
					};
				break;
			case "store_retrieve":
				if (response.content !== undefined) return response.content;
				break;
		}
		throw new Error("Legroom SDK response format unrecognized");
	}

	close(): void {
		this.transport.close();
	}
}
