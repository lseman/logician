import type { HarnessPhase } from "../run-kernel.ts";

const PHASE_TRANSITIONS: Record<HarnessPhase, readonly HarnessPhase[]> = {
	idle: ["turn", "compaction", "branch_summary"],
	turn: ["idle"],
	compaction: ["idle"],
	branch_summary: ["idle"],
};

export class HarnessBusyError extends Error {
	constructor(op: string, phase: HarnessPhase, required: HarnessPhase) {
		super(
			`AgentHarness cannot ${op}: phase is "${phase}", requires "${required}"`,
		);
		this.name = "HarnessBusyError";
	}
}

export function assertPhaseTransition(
	from: HarnessPhase,
	to: HarnessPhase,
	op: string,
): void {
	if (!PHASE_TRANSITIONS[from].includes(to)) {
		throw new HarnessBusyError(op, from, "idle");
	}
}

export function assertIdlePhase(phase: HarnessPhase, op: string): void {
	if (phase !== "idle") {
		throw new HarnessBusyError(op, phase, "idle");
	}
}
