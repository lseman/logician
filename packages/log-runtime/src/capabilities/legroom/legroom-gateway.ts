import type { AgentConfig, AgentHooks } from "@logician/log-core";
import type {
	CalibrationStatus,
	CompressResult,
	LegroomSdkConfig,
	StoreStats,
	WorkerHistory,
	WorkerStats,
} from "./worker.ts";
import { LegroomWorker } from "./worker.ts";

/** FNV-1a 32-bit — cheap content fingerprint for prefix memoization. */
function fnv1a(input: string): number {
	let hash = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return hash >>> 0;
}

interface LegroomMemo {
	/** Per-message content hash of the input last compressed. */
	hashes: Uint32Array;
	/** Compressed output corresponding to `hashes`. */
	compressed: Record<string, unknown>[];
	model: string;
	configKey: string;
}

/** Owns Legroom enablement, hooks, CCR operations, and worker lifecycle. */
export class LegroomGateway {
	private readonly worker: LegroomWorker;
	private readonly options: LegroomSdkConfig;
	private enabled: boolean;
	/** Prefix memo over the last compressed payload.
	 *
	 * Provider payloads are append-only, so when the stable prefix (all but
	 * the last `protect_recent` messages) is unchanged, only the new tail is
	 * sent to the worker and spliced onto the cached compressed prefix —
	 * avoiding a full-history Python round-trip on every provider call and
	 * every retry. Rewritten or shrunk histories fall back to a full pass.
	 */
	private memo: LegroomMemo | null = null;

	constructor(options: LegroomSdkConfig = {}) {
		this.worker = new LegroomWorker(options);
		this.options = options;
		this.enabled = options.mode === "sdk";
	}

	isEnabled(): boolean {
		return this.enabled;
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		if (!enabled) this.worker.close();
	}

	createHooks(existingHooks: AgentConfig["hooks"]): AgentHooks {
		return {
			...existingHooks,
			beforeProviderPayload: async context => {
				const existing = await existingHooks?.beforeProviderPayload?.(context);
				const payload = existing?.payload ?? context.payload;
				if (!this.enabled) return { payload };
				const messages = payload.messages;
				if (!Array.isArray(messages)) return { payload };
				const compressible = messages.filter(
					(message): message is Record<string, unknown> =>
						message !== null && typeof message === "object",
				);
				if (compressible.length !== messages.length) return { payload };
				return {
					payload: {
						...payload,
						messages: await this.compressMemoized(compressible, context.model),
					},
				};
			},
		};
	}

	/** Compress with a prefix memo (see `memo`). Fail-open like worker.compress. */
	private async compressMemoized(
		messages: Record<string, unknown>[],
		model: string,
	): Promise<Record<string, unknown>[]> {
		const configKey = JSON.stringify(this.options.config ?? {});
		const memo = this.memo;
		if (
			memo &&
			memo.model === model &&
			memo.configKey === configKey &&
			messages.length >= memo.hashes.length
		) {
			const stableLen = Math.max(0, memo.hashes.length - this.protectRecent());
			// Verify the stable prefix message-by-message. Content-based: the
			// loop rebuilds chat messages on every call, so object identity
			// cannot be used. For an identical-length payload the whole
			// history is checked (retry fast path).
			const checkLen =
				messages.length === memo.hashes.length ? messages.length : stableLen;
			let prefixMatches = true;
			try {
				for (let i = 0; i < checkLen; i++) {
					if (fnv1a(JSON.stringify(messages[i])) !== memo.hashes[i]) {
						prefixMatches = false;
						break;
					}
				}
			} catch {
				prefixMatches = false;
			}
			if (prefixMatches) {
				if (messages.length === memo.hashes.length) {
					// Identical payload (retry path) — serve the cached result.
					return memo.compressed;
				}
				// Cross-message phases make per-prefix splicing unsound.
				if (!this.crossMessageCompression()) {
					// Send only the unstable tail (the previously protected
					// window plus new messages). The worker's protect_recent
					// window is relative to the end of the list it receives,
					// so it aligns exactly with the full history: the tail
					// ends at the history's end.
					const tail = messages.slice(stableLen);
					const tailCompressed = await this.worker.compress(tail, model);
					const result = [
						...memo.compressed.slice(0, stableLen),
						...tailCompressed,
					];
					const hashes = new Uint32Array(messages.length);
					hashes.set(memo.hashes.subarray(0, stableLen), 0);
					hashes.set(this.fingerprint(tail), stableLen);
					this.memo = { hashes, compressed: result, model, configKey };
					return result;
				}
			}
		}
		const result = await this.worker.compress(messages, model);
		this.memo = {
			hashes: this.fingerprint(messages),
			compressed: result,
			model,
			configKey,
		};
		return result;
	}

	private fingerprint(messages: Record<string, unknown>[]): Uint32Array {
		const hashes = new Uint32Array(messages.length);
		for (let i = 0; i < messages.length; i++) {
			hashes[i] = fnv1a(JSON.stringify(messages[i]));
		}
		return hashes;
	}

	/** Mirror of legroom's CompressConfig.protect_recent default (3). */
	private protectRecent(): number {
		const value = this.options.config?.protect_recent;
		return typeof value === "number" && Number.isInteger(value) && value >= 0
			? value
			: 3;
	}

	private crossMessageCompression(): boolean {
		return this.options.config?.semantic_dedup_enabled === true;
	}

	async compressWithStore(
		storeId: string,
		messages: Record<string, unknown>[],
		model: string,
	): Promise<CompressResult> {
		this.assertEnabled();
		return this.worker.compressWithStore(storeId, messages, model);
	}

	async storeRetrieve(storeId: string, hash: string): Promise<string> {
		this.assertEnabled();
		return this.worker.storeRetrieve(storeId, hash);
	}

	async storeStats(storeId: string): Promise<StoreStats> {
		this.assertEnabled();
		return this.worker.storeStats(storeId);
	}

	async workerStats(): Promise<WorkerStats> {
		this.assertEnabled();
		return this.worker.workerStats();
	}

	async workerHistory(limit = 50, offset = 0): Promise<WorkerHistory> {
		this.assertEnabled();
		return this.worker.workerHistory(limit, offset);
	}

	async calibrationStatus(): Promise<CalibrationStatus> {
		this.assertEnabled();
		return this.worker.calibrationStatus();
	}

	async calibrationRecord(
		phaseReports: Record<string, unknown>[],
		quality: number,
	): Promise<CalibrationStatus> {
		this.assertEnabled();
		return this.worker.calibrationRecord(phaseReports, quality);
	}

	close(): void {
		this.worker.close();
	}

	private assertEnabled(): void {
		if (!this.enabled) throw new Error("Legroom SDK is not enabled");
	}
}
