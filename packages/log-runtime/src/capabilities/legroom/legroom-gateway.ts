import type { AgentConfig } from "@logician/log-core";
import type {
	CalibrationStatus,
	CompressResult,
	LegroomSdkConfig,
	StoreStats,
	WorkerHistory,
	WorkerStats,
} from "./worker.ts";
import { LegroomWorker } from "./worker.ts";

/** Owns Legroom enablement, hooks, CCR operations, and worker lifecycle. */
export class LegroomGateway {
	readonly worker: LegroomWorker;
	private enabled: boolean;

	constructor(options: LegroomSdkConfig = {}) {
		this.worker = new LegroomWorker(options);
		this.enabled = options.mode === "sdk";
	}

	isEnabled(): boolean {
		return this.enabled;
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		if (!enabled) this.worker.close();
	}

	createHooks(existingHooks: AgentConfig["hooks"]): AgentConfig["hooks"] {
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
						messages: await this.worker.compress(compressible, context.model),
					},
				};
			},
		};
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
