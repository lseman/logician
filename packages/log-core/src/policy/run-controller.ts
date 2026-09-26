import type { AcceptanceVerificationResult } from "../guards/acceptance-contract.ts";
import { ProgressTracker } from "./progress-tracker.ts";

export interface PermissionEscalation {
	consecutive: number;
	total: number;
}

/** Owns mutable policy state for exactly one agent run. */
export class AgentRunController {
	readonly progress = new ProgressTracker();
	readonly compaction = { lastTurn: -3, consecutiveCompactions: 0 };

	private consecutivePermissionDenials = 0;
	private totalPermissionDenials = 0;
	private verificationRepairs = 0;
	private acceptanceStop = false;

	recordPermissionBatch(input: {
		denials: number;
		executed: number;
	}): PermissionEscalation | undefined {
		if (input.denials > 0) {
			this.totalPermissionDenials += input.denials;
			this.consecutivePermissionDenials =
				input.executed === 0
					? this.consecutivePermissionDenials + input.denials
					: 0;
		} else if (input.executed > 0) {
			this.consecutivePermissionDenials = 0;
		}
		if (
			this.consecutivePermissionDenials < 3 &&
			this.totalPermissionDenials < 20
		)
			return undefined;
		return {
			consecutive: this.consecutivePermissionDenials,
			total: this.totalPermissionDenials,
		};
	}

	requestVerificationRepair(
		results: readonly AcceptanceVerificationResult[],
		canContinue: boolean,
	): boolean {
		const failed = results.some(result => result.result === "failed");
		if (!failed || !canContinue || this.verificationRepairs >= 1) return false;
		this.verificationRepairs++;
		this.acceptanceStop = false;
		return true;
	}

	requestAcceptanceStop(): void {
		this.acceptanceStop = true;
	}

	get acceptanceStopRequested(): boolean {
		return this.acceptanceStop;
	}
}
