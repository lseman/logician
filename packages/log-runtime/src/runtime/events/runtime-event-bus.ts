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
	historyCapacity?: number;
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
}

export interface RuntimeEventReplayGap {
	requestedAfterSequence: number;
	missingFromSequence: number;
	missingThroughSequence: number;
	oldestAvailableSequence?: number;
}

/** Ordered runtime notifications and asynchronous error delivery. */
export class RuntimeEventBus {
	private subscribers = new Set<ProtocolCallback>();
	private errorCallback: ErrorCallback | null = null;
	private readonly journal: EventJournal<RuntimeEvent>;
	private correlation: AgentProtocolCorrelation = {};
	private readonly correlationBySequence = new Map<
		number,
		AgentProtocolCorrelation
	>();

	constructor(options: RuntimeEventBusOptions = {}) {
		this.journal = new EventJournal({
			capacity: options.historyCapacity,
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
			if (gap) options.onReplayGap?.(gap);
			for (const notification of this.snapshot(query)) {
				try {
					callback(notification);
				} catch {
					// Replay has the same client-fault isolation as live delivery.
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

	/** Set correlation for subsequent events in one user-visible run. */
	beginRun(correlation: {
		sessionId: string;
		runId: string;
		turnId: string;
	}): void {
		this.correlation = { ...correlation };
	}

	/** Clear run/turn correlation while retaining the current session identity. */
	endRun(): void {
		this.correlation = { sessionId: this.correlation.sessionId };
	}

	setSessionId(sessionId: string): void {
		this.correlation = { ...this.correlation, sessionId };
	}

	get latestSequence(): number {
		return this.journal.latestId;
	}

	/** Describe an evicted prefix for a reconnect cursor, if one exists. */
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
		return {
			requestedAfterSequence: query.afterId,
			missingFromSequence: firstMissing,
			missingThroughSequence: missingThrough,
			oldestAvailableSequence: oldest,
		};
	}

	clearHistory(): void {
		this.journal.clear();
		this.correlationBySequence.clear();
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

	private pruneCorrelationHistory(): void {
		const oldest = this.journal.oldestId;
		if (oldest === undefined) {
			this.correlationBySequence.clear();
			return;
		}
		for (const sequence of this.correlationBySequence.keys()) {
			if (sequence < oldest) this.correlationBySequence.delete(sequence);
		}
	}
}
