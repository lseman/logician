// ── SessionManager ─────────────────────────────────────────────────────────────
// Owns queue management, session lifecycle, and continuation logic extracted
// from AgentRuntime.  The harness owns the actual queue state; this manager
// mirrors it to the UI, wires slash commands, and drives the autonomous
// continuation budget policy.

import type { QueueMode } from "@logician/agent-core";
import type { AgentHarness } from "@logician/agent-core/harness";
import type { HarnessPhase } from "@logician/agent-core/runtime";
import type { RuntimeEvent } from "@logician/agent-protocol";
import {
	formatActivatedSkills,
	type SkillActivation,
} from "../../capabilities/skills/activation.ts";

// ── Options ────────────────────────────────────────────────────────────────────

export interface SessionManagerDeps {
	harness: AgentHarness | null;
	emit: (event: RuntimeEvent) => void;
	getSystemPrompt: () => string;
	getSteeringInterrupt: () => boolean;
	setConfigSteeringMode: (mode: QueueMode) => void;
	setConfigSteeringInterrupt: (enabled: boolean) => void;
	setConfigFollowUpMode: (mode: QueueMode) => void;
}

// ── SessionManager class ───────────────────────────────────────────────────────

export class BridgeSessionManager {
	private harness: AgentHarness | null;
	private pendingContinuation = false;
	activeRepositoryQuery?: string;

	constructor(private readonly deps: SessionManagerDeps) {
		this.harness = deps.harness;
	}

	setPendingContinuation(value: boolean): void {
		this.pendingContinuation = value;
	}

	/** Set the harness reference (called once when ensureHarness builds it). */
	setHarness(harness: AgentHarness): void {
		this.harness = harness;
	}

	// ── Queue operations ─────────────────────────────────────────────────────

	/** Inject guidance into the running turn (drained at the next save point). */
	steer(message: string): void {
		this.harness?.steer(message);
	}

	/** Queue a message for after the current turn completes. */
	followUp(message: string): void {
		this.harness?.followUp(message);
	}

	/** Queue a message before the next user prompt; survives abort. */
	nextTurn(message: string): void {
		this.harness?.nextTurn(message);
	}

	/** Controls how queued steering messages are drained. */
	setSteeringMode(mode: QueueMode): void {
		this.deps.setConfigSteeringMode(mode);
		this.harness?.setSteeringMode(mode);
	}

	/** Toggle mid-stream steering interrupt (cut the stream vs. queue). */
	setSteeringInterrupt(enabled: boolean): void {
		this.deps.setConfigSteeringInterrupt(enabled);
		this.harness?.configure({ steeringInterrupt: enabled });
	}

	getSteeringInterrupt(): boolean {
		return this.deps.getSteeringInterrupt();
	}

	/** Controls how queued follow-up messages are drained. */
	setFollowUpMode(mode: QueueMode): void {
		this.deps.setConfigFollowUpMode(mode);
		this.harness?.setFollowUpMode(mode);
	}

	/** Get current steering messages (read-only). */
	getSteeringMessages(): string[] {
		return this.harness?.getQueues().steering ?? [];
	}

	/** Get current follow-up messages (read-only). */
	getFollowUpMessages(): string[] {
		return this.harness?.getQueues().followUp ?? [];
	}

	/** Get current next-turn messages (read-only). */
	getNextTurnMessages(): string[] {
		return this.harness?.getQueues().nextTurn ?? [];
	}

	/** Clear all pending messages, returns the messages that were cleared. */
	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return (
			this.harness?.clearQueues() ?? {
				steering: [],
				followUp: [],
				nextTurn: [],
			}
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.harness?.dropQueuedMessage(displayIndex);
	}

	/** Interrupt the current provider step and process queued steering immediately. */
	flushSteeringNow(): number {
		if (!this.harness) return 0;
		return this.harness.flushSteeringNow();
	}

	/** Emit queue state to the UI (called by harness onQueueChange callback). */
	emitQueueUpdate(): void {
		const q = this.harness?.getQueues() ?? {
			steering: [],
			followUp: [],
			nextTurn: [],
		};
		this.deps.emit({
			type: "queue_update",
			steering: q.steering,
			followUp: q.followUp,
			nextTurn: q.nextTurn,
		});
	}

	// ── Continuation logic ───────────────────────────────────────────────────

	/**
	 * After a turn settles, resume with whichever continuation was queued.
	 */
	checkPendingContinuation(activations: SkillActivation[]): void {
		if (!this.pendingContinuation) return;
		this.pendingContinuation = false;
		void this.runQueuedContinuation(activations).catch(error => {
			const message = error instanceof Error ? error.message : String(error);
			this.deps.emit({
				type: "notice",
				level: "error",
				label: "Continuation",
				text: message,
			});
		});
	}

	async runQueuedContinuation(activations: SkillActivation[]): Promise<void> {
		const harness = this.harness;
		if (!harness) return;

		const turnId = `turn_${Date.now()}`;
		const originalPrompt = this.deps.getSystemPrompt();
		let turnSucceeded = false;
		try {
			const dynamicContext = [
				this.activeRepositoryQuery
					? this.activeRepositoryQuery // repository context handled elsewhere
					: undefined,
				activations.length ? formatActivatedSkills(activations) : undefined,
			].filter((value): value is string => Boolean(value));
			if (dynamicContext.length) {
				harness.configure({
					systemPrompt: `${originalPrompt}\n\n${dynamicContext.join("\n\n")}`,
				});
			}
			this.deps.emit({ type: "turn_start", turnId });
			await harness.continueWithNextTurn();
			turnSucceeded = true;
		} finally {
			if (activations.length || this.activeRepositoryQuery) {
				harness.configure({ systemPrompt: originalPrompt });
			}
			this.deps.emit({ type: "turn_end", turnId });
			this.deps.emit({ type: "phase", state: "ready" });
			if (turnSucceeded) this.checkPendingContinuation(activations);
		}
	}

	// ── Harness phase mapping ────────────────────────────────────────────────

	/** Map harness structural phases to UI phase states. */
	emitHarnessPhase(phase: HarnessPhase): void {
		if (phase === "turn") return;
		const state =
			phase === "compaction"
				? "compacting"
				: phase === "branch_summary"
					? "branching"
					: "ready";
		this.deps.emit({ type: "phase", state });
	}

	// ── Slash command handling ───────────────────────────────────────────────

	/** Handle queue-related slash commands. Returns true if the command was handled. */
	handleSlashCommand(trimmed: string): boolean {
		if (trimmed === "/steer-now") {
			const count = this.flushSteeringNow();
			this.deps.emit({
				type: "notice",
				level: count > 0 ? "info" : "warn",
				label: "Steering",
				text:
					count > 0
						? `Processing ${count} queued steering message${count === 1 ? "" : "s"} now.`
						: "No queued steering messages to process.",
			});
			return true;
		}
		if (trimmed === "/queue") {
			const steering = this.getSteeringMessages();
			const followUp = this.getFollowUpMessages();
			const rows = [
				...steering.map(message => `▸ ${message}`),
				...followUp.map(message => `↳ ${message}`),
			];
			this.deps.emit({
				type: "notice",
				level: "info",
				label: "Queue",
				text: rows.length
					? rows.map((row, index) => `${index + 1}. ${row}`).join("\n")
					: "Queue is empty.",
			});
			return true;
		}
		if (trimmed === "/queue-clear") {
			const cleared = this.clearQueue();
			const count =
				cleared.steering.length +
				cleared.followUp.length +
				cleared.nextTurn.length;
			this.deps.emit({
				type: "notice",
				level: "info",
				label: "Queue",
				text: `Cleared ${count} queued message${count === 1 ? "" : "s"}.`,
			});
			return true;
		}
		if (trimmed === "/queue-drop" || trimmed.startsWith("/queue-drop ")) {
			const value = Number.parseInt(
				trimmed.slice("/queue-drop".length).trim(),
				10,
			);
			const removed =
				Number.isInteger(value) && value > 0
					? this.dropQueuedMessage(value - 1)
					: undefined;
			this.deps.emit({
				type: "notice",
				level: removed ? "info" : "warn",
				label: "Queue",
				text: removed ? `Removed: ${removed}` : "Usage: /queue-drop <number>",
			});
			return true;
		}
		return false;
	}
}
