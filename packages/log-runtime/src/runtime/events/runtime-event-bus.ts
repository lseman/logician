import {
	EventJournal,
	type EventJournalQuery,
} from "@logician/log-core/event-journal";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	type AgentProtocolCorrelation,
	type AgentProtocolNotification,
	createNotification,
} from "@logician/log-core/protocol";
import type { ErrorCallback, ProtocolCallback } from "../bridge/types.ts";

export interface RuntimeEventBusOptions {
	/** Number of protocol notifications retained for reconnect replay. */
	historyCapacity?: number;
	/** Wall-clock now function (for deterministic testing). */
	now?: () => number;
}

export interface RuntimeDiagnosticContext {
	component?: string;
	operation?: string;
	code?: string;
	recoverable?: boolean;
}

export interface RuntimeEventSubscriptionOptions {
	/** Replay retained notifications before attaching to the live stream. */
	replay?: boolean | EventJournalQuery<RuntimeEvent>;
	/** Called before replay when requested events have already been evicted. */
	onReplayGap?: (gap: RuntimeEventReplayGap) => void;
	/**
	 * When true, the bus attaches a gap-aware replay probe that calls
	 * `onReplayGap` before replay begins, giving the subscriber a chance
	 * to reset state or request a full refresh.
	 * @default false
	 */
	preReplayGapCheck?: boolean;
}

export interface RuntimeEventReplayGap {
	/** The cursor the subscriber last knew about. */
	requestedAfterSequence: number;
	/** The first sequence that is missing. */
	missingFromSequence: number;
	/** The last sequence that is missing. */
	missingThroughSequence: number;
	/** Oldest sequence still in the journal, if any. */
	oldestAvailableSequence?: number;
	/**
	 * The session identity at the time this gap was detected.
	 * Useful when the session has rotated between the subscriber's
	 * cursor and the current bus state.
	 */
	sessionId?: string;
	/**
	 * Recommended action hint: "full_refresh" when the gap is larger
	 * than a full run, "partial" when a subset of the gap is available.
	 */
	resolutionHint?: "full_refresh" | "partial";
	/**
	 * Whether the gap falls entirely within a single run group,
	 * suggesting the subscriber can simply restart that run.
	 */
	intraRunGap?: boolean;
}

/** Ordered runtime notifications and asynchronous error delivery. */
export class RuntimeEventBus {
	private subscribers = new Set<ProtocolCallback>();
	private errorCallback: ErrorCallback | null = null;
	private readonly journal: EventJournal<RuntimeEvent>;
	private readonly historyCapacity: number;

	/** Current run/turn correlation. */
	private correlation: AgentProtocolCorrelation = {};
	/** Sequence → correlation map, pruned on eviction. */
	private readonly correlationBySequence = new Map<
		number,
		AgentProtocolCorrelation
	>();

	/**
	 * Run group tracking: each `beginRun` creates a group with a
	 * unique id. Events emitted between `beginRun` and `endRun`
	 * are tagged with that group id. This enables replaying an
	 * entire run atomically.
	 */
	private activeRunGroup: { id: string } | null = null;
	/**
	 * Per-run-group correlation snapshots, keyed by run id. `firstSequence`/
	 * `lastSequence` track the journal id range spanned by events emitted
	 * while this group was active, so gaps can be checked against the
	 * actual range instead of just session identity.
	 */
	private runGroups = new Map<
		string,
		{
			firstSequence?: number;
			lastSequence?: number;
		}
	>();

	constructor(options: RuntimeEventBusOptions = {}) {
		const capacity = options.historyCapacity ?? 1000;
		this.historyCapacity = capacity;
		this.journal = new EventJournal({
			capacity,
			now: options.now,
		});
	}

	subscribe(
		callback: ProtocolCallback,
		options: RuntimeEventSubscriptionOptions = {},
	): () => void {
		if (options.replay) {
			const query = options.replay === true ? {} : options.replay;
			const gap = this.replayGap(query);
			if (gap) {
				options.onReplayGap?.(gap);
			}
			// preReplayGapCheck: the subscriber has already been notified of the
			// gap and is expected to request a full refresh itself, so skip the
			// (necessarily incomplete) partial snapshot replay.
			const skipReplay = gap !== undefined && options.preReplayGapCheck;
			if (!skipReplay) {
				for (const notification of this.snapshot(query)) {
					try {
						callback(notification);
					} catch {
						// Replay has the same client-fault isolation as live delivery.
					}
				}
			}
		}
		this.subscribers.add(callback);
		return () => this.subscribers.delete(callback);
	}

	/** Retained protocol notifications ordered oldest to newest. */
	snapshot(
		query: EventJournalQuery<RuntimeEvent> = {},
	): readonly AgentProtocolNotification[] {
		return this.journal
			.snapshot(query)
			.map(entry =>
				createNotification(
					entry.event,
					entry.id,
					entry.recordedAt,
					this.correlationBySequence.get(entry.id),
				),
			);
	}

	/**
	 * Set correlation for subsequent events in one user-visible run.
	 * Also creates a run group for group-based replay.
	 */
	beginRun(correlation: {
		sessionId: string;
		runId: string;
		turnId: string;
	}): void {
		this.correlation = { ...correlation };
		this.activeRunGroup = { id: correlation.runId };
		this.runGroups.set(correlation.runId, {});
	}

	/** Clear run/turn correlation while retaining the current session identity. */
	endRun(): void {
		if (this.activeRunGroup) {
			const group = this.runGroups.get(this.activeRunGroup.id);
			if (group?.firstSequence === undefined) {
				this.runGroups.delete(this.activeRunGroup.id);
			}
		}
		this.correlation = { sessionId: this.correlation.sessionId };
		this.activeRunGroup = null;
	}

	setSessionId(sessionId: string): void {
		this.correlation = { ...this.correlation, sessionId };
	}

	get latestSequence(): number {
		return this.journal.latestId;
	}

	/**
	 * Describe an evicted prefix for a reconnect cursor, if one exists.
	 * Includes resolution hints based on the gap size and run group context.
	 */
	replayGap(
		query: EventJournalQuery<RuntimeEvent>,
	): RuntimeEventReplayGap | undefined {
		if (query.afterId === undefined || query.afterId >= this.latestSequence) {
			return undefined;
		}
		const oldest = this.journal.oldestId;
		const firstMissing = query.afterId + 1;
		const missingThrough =
			oldest === undefined ? this.latestSequence : oldest - 1;
		if (firstMissing > missingThrough) return undefined;

		const gapSize = missingThrough - firstMissing + 1;

		return {
			requestedAfterSequence: query.afterId,
			missingFromSequence: firstMissing,
			missingThroughSequence: missingThrough,
			oldestAvailableSequence: oldest,
			sessionId: this.correlation.sessionId,
			resolutionHint: gapSize > 500 ? "full_refresh" : "partial",
			intraRunGap: this.isIntraRunGap(query.afterId, missingThrough),
		};
	}

	/**
	 * Check whether the gap `[afterId + 1, throughId]` falls entirely within
	 * a single known run group's event range. Returns false if no run group
	 * fully covers the gap.
	 */
	private isIntraRunGap(afterId: number, throughId: number): boolean {
		const gapStart = afterId + 1;
		for (const [, group] of this.runGroups) {
			if (group.firstSequence === undefined || group.lastSequence === undefined)
				continue;
			if (group.firstSequence <= gapStart && group.lastSequence >= throughId) {
				return true;
			}
		}
		return false;
	}

	clearHistory(): void {
		this.journal.clear();
		this.correlationBySequence.clear();
		this.runGroups.clear();
	}

	onError(callback: ErrorCallback): () => void {
		this.errorCallback = callback;
		return () => {
			if (this.errorCallback === callback) this.errorCallback = null;
		};
	}

	reportError(error: unknown, context: RuntimeDiagnosticContext = {}): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.emit({
			type: "diagnostic",
			severity: "error",
			component: context.component ?? "runtime",
			operation: context.operation ?? "unknown",
			code: context.code ?? (normalized.name || "Error"),
			message: normalized.message,
			recoverable: context.recoverable ?? false,
		});
		this.emit({
			type: "notice",
			level: "error",
			label: "Error",
			text: normalized.message,
		});
		this.notifyError(normalized);
	}

	/** Deliver an error without adding a second transcript notice. */
	notifyError(error: unknown): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.errorCallback?.(normalized);
	}

	emit(event: RuntimeEvent): void {
		const entry = this.journal.append(event);
		const eventCorrelation = this.correlate(event);
		this.correlationBySequence.set(entry.id, eventCorrelation);
		if (this.activeRunGroup) {
			const group = this.runGroups.get(this.activeRunGroup.id);
			if (group) {
				group.firstSequence ??= entry.id;
				group.lastSequence = entry.id;
			}
		}
		this.pruneCorrelationHistory();
		const notification = createNotification(
			event,
			entry.id,
			entry.recordedAt,
			eventCorrelation,
		);
		for (const callback of this.subscribers) {
			try {
				callback(notification);
			} catch {
				// A client subscriber cannot interrupt the runtime.
			}
		}
	}

	private correlate(event: RuntimeEvent): AgentProtocolCorrelation {
		const explicitTurnId =
			"turnId" in event && typeof event.turnId === "string"
				? event.turnId
				: undefined;
		const toolCallId =
			"toolCallId" in event && typeof event.toolCallId === "string"
				? event.toolCallId
				: undefined;
		return {
			...this.correlation,
			...(explicitTurnId && { turnId: explicitTurnId }),
			...(toolCallId && { toolCallId }),
		};
	}

	/**
	 * Prune correlation and run metadata after journal eviction.
	 */
	private pruneCorrelationHistory(): void {
		const oldest = this.journal.oldestId;

		// Prune sequence-level correlation entries.
		if (oldest === undefined) {
			this.correlationBySequence.clear();
		} else {
			for (const sequence of this.correlationBySequence.keys()) {
				if (sequence < oldest) this.correlationBySequence.delete(sequence);
			}
		}

		// Keep one additional journal window behind the eviction frontier. That
		// is enough to classify near-boundary replay gaps without allowing run
		// metadata to grow with process lifetime.
		const groupRetentionFloor =
			oldest === undefined ? undefined : oldest - this.historyCapacity;
		for (const [runId, group] of this.runGroups) {
			if (this.activeRunGroup?.id === runId) continue;
			if (
				groupRetentionFloor === undefined ||
				(group.lastSequence ?? 0) < groupRetentionFloor
			) {
				this.runGroups.delete(runId);
			}
		}
	}
}
