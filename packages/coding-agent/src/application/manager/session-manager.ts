// ── SessionManager ─────────────────────────────────────────────────────────────
// Owns queue management, session lifecycle, and continuation logic extracted
// from AgentCoreBridge.  The harness owns the actual queue state; this manager
// mirrors it to the UI, wires slash commands, and drives the autonomous
// continuation budget policy.

import type {
	AgentHarness,
	HarnessPhase,
	Message,
	Session,
	QueueMode,
} from "@logician/agent-core";
import {
	formatActivatedSkills,
	formatSkillActivationNotice,
	type SkillActivation,
} from "../../skills/activation.ts";
import type { RuntimeEvent } from "../../runtime/events.ts";

// ── Options ────────────────────────────────────────────────────────────────────

export interface SessionManagerDeps {
	harness: AgentHarness | null;
	emit: (event: RuntimeEvent) => void;
	getSystemPrompt: () => string;
	setSystemPrompt: (prompt: string) => void;
	setSteeringInterrupt: (enabled: boolean) => void;
	getSteeringInterrupt: () => boolean;
	setConfigSteeringMode: (mode: QueueMode) => void;
	setConfigSteeringInterrupt: (enabled: boolean) => void;
	setConfigFollowUpMode: (mode: QueueMode) => void;
}

// ── SessionManager class ───────────────────────────────────────────────────────

export class SessionManager {
	private pendingAutoContinue = false;
	private pendingSteeringContinue = false;
	activeRepositoryQuery?: string;

	constructor(private readonly deps: SessionManagerDeps) {}

	/** Set pending steering continuation flag. */
	setPendingSteeringContinue(v: boolean): void {
		this.pendingSteeringContinue = v;
	}

	/** Set pending auto-continuation flag. */
	setPendingAutoContinue(v: boolean): void {
		this.pendingAutoContinue = v;
	}

	/** Set the harness reference (called once when ensureHarness builds it). */
	setHarness(harness: AgentHarness): void {
		// The harness reference is already in deps; this is a no-op placeholder
		// for any bridge-side wiring that might be needed.
	}

	// ── Queue operations ─────────────────────────────────────────────────────

	/** Inject guidance into the running turn (drained at the next save point). */
	steer(message: string): void {
		this.deps.harness?.steer(message);
	}

	/** Queue a message for after the current turn completes. */
	followUp(message: string): void {
		this.deps.harness?.followUp(message);
	}

	/** Queue a message before the next user prompt; survives abort. */
	nextTurn(message: string): void {
		this.deps.harness?.nextTurn(message);
	}

	/** Controls how queued steering messages are drained. */
	setSteeringMode(mode: QueueMode): void {
		this.deps.setConfigSteeringMode(mode);
		this.deps.harness?.setSteeringMode(mode);
	}

	/** Toggle mid-stream steering interrupt (cut the stream vs. queue). */
	setSteeringInterrupt(enabled: boolean): void {
		this.deps.setConfigSteeringInterrupt(enabled);
		this.deps.harness?.setSteeringInterrupt(enabled);
	}

	getSteeringInterrupt(): boolean {
		return this.deps.getSteeringInterrupt();
	}

	/** Controls how queued follow-up messages are drained. */
	setFollowUpMode(mode: QueueMode): void {
		this.deps.setConfigFollowUpMode(mode);
		this.deps.harness?.setFollowUpMode(mode);
	}

	/** Get current steering messages (read-only). */
	getSteeringMessages(): string[] {
		return this.deps.harness?.getQueues().steering ?? [];
	}

	/** Get current follow-up messages (read-only). */
	getFollowUpMessages(): string[] {
		return this.deps.harness?.getQueues().followUp ?? [];
	}

	/** Get current next-turn messages (read-only). */
	getNextTurnMessages(): string[] {
		return this.deps.harness?.getQueues().nextTurn ?? [];
	}

	/** Clear all pending messages, returns the messages that were cleared. */
	clearQueue(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	} {
		return (
			this.deps.harness?.clearQueues() ??
			{ steering: [], followUp: [], nextTurn: [] }
		);
	}

	dropQueuedMessage(displayIndex: number): string | undefined {
		return this.deps.harness?.dropQueuedMessage(displayIndex);
	}

	/** Interrupt the current provider step and process queued steering immediately. */
	flushSteeringNow(): number {
		if (!this.deps.harness) return 0;
		return this.deps.harness.flushSteeringNow();
	}

	/** Emit queue state to the UI (called by harness onQueueChange callback). */
	emitQueueUpdate(): void {
		const q = this.deps.harness?.getQueues() ?? {
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

	// ── Session lifecycle ────────────────────────────────────────────────────

	/**
	 * Replace the harness conversation with restored session history (resume /
	 * session switch). Pass [] to clear (new session).
	 */
	restoreHistory(messages: Message[]): boolean {
		try {
			this.deps.harness?.setHistory(messages);
			return true;
		} catch {
			return false;
		}
	}

	/**
	 * Use the user-facing conversation session as the hook and memory session.
	 */
	useConversationSession(
		sessionId: string,
		durableSession?: Session,
	): void {
		if (!sessionId.trim()) return;
		this.deps.harness?.setSessionId(sessionId);
		if (durableSession) this.deps.harness?.attachSession(durableSession);
	}

	renameConversationSession(sessionId: string, name: string): void {
		// Session rename is handled by MemoryManager — no-op here
	}

	// ── Continuation logic ───────────────────────────────────────────────────

	/**
	 * After a turn settles, resume with whichever continuation was queued.
	 */
	checkPendingContinuation(activations: SkillActivation[]): void {
		if (this.pendingSteeringContinue) {
			this.pendingSteeringContinue = false;
			this.runQueuedContinuation(activations).catch(error => {
				this.pendingSteeringContinue = false;
				const message = error instanceof Error ? error.message : String(error);
				this.deps.harness?.failRun(message);
			});
			return;
		}
		if (this.pendingAutoContinue) {
			this.pendingAutoContinue = false;
			this.scheduleAutoContinuation(activations);
		}
	}

	scheduleAutoContinuation(activations: SkillActivation[]): void {
		const harness = this.deps.harness;
		if (!harness) return;
		const decision = harness.requestContinuation(
			"next_turn_queue",
			"", // progressFingerprint — simplified
		);
		if (decision.action === "pause") {
			this.deps.emit({
				type: "notice",
				level: "warn",
				label: "Continuation paused",
				text: decision.reason,
			});
			return;
		}
		this.deps.emit({
			type: "notice",
			level: "info",
			label: "Continuation",
			text: `Starting native continuation run ${decision.state.continuationRuns}.`,
		});
		void this.runQueuedContinuation(activations).catch(error => {
			this.pendingAutoContinue = false;
			const message = error instanceof Error ? error.message : String(error);
			harness.failRun(message);
		});
	}

	async runQueuedContinuation(activations: SkillActivation[]): Promise<void> {
		const harness = this.deps.harness;
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
				harness.setSystemPrompt(
					`${originalPrompt}\n\n${dynamicContext.join("\n\n")}`,
				);
			}
			this.deps.emit({ type: "turn_start", turnId });
			await harness.continueWithNextTurn();
			turnSucceeded = true;
		} finally {
			if (activations.length || this.activeRepositoryQuery) {
				harness.setSystemPrompt(originalPrompt);
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
