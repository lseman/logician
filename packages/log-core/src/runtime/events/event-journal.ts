/** A typed, bounded in-memory journal for runtime event replay and diagnostics. */

export interface JournalEvent {
	type: string;
}

export interface EventJournalEntry<E extends JournalEvent> {
	/** Monotonic journal-local cursor. Independent from a run's event sequence. */
	id: number;
	recordedAt: number;
	event: E;
}

export interface EventJournalQuery<E extends JournalEvent> {
	/** Return entries strictly newer than this journal cursor. */
	afterId?: number;
	types?: Iterable<E["type"]>;
	/** Return only the newest matching entries. */
	limit?: number;
}

export interface EventJournalSubscriptionOptions<E extends JournalEvent> {
	/** Replay retained entries before receiving new ones. */
	replay?: boolean | EventJournalQuery<E>;
}

export interface EventJournalOptions<E extends JournalEvent> {
	/** Maximum retained entries. Zero keeps live subscriptions but no history. */
	capacity?: number;
	now?: () => number;
	onSubscriberError?: (error: Error, entry: EventJournalEntry<E>) => void;
}

type Subscriber<E extends JournalEvent> = (entry: EventJournalEntry<E>) => void;

const DEFAULT_CAPACITY = 1_000;

/**
 * Records an event stream in a bounded ring and exposes cursor-based replay.
 * Events are retained by reference; producers should treat emitted events as
 * immutable, as the core runtime already does.
 */
export class EventJournal<E extends JournalEvent = JournalEvent> {
	private readonly capacity: number;
	private readonly now: () => number;
	private readonly onSubscriberError?: EventJournalOptions<E>["onSubscriberError"];
	private entries: Array<EventJournalEntry<E> | undefined>;
	private start = 0;
	private count = 0;
	private nextId = 1;
	private subscribers = new Set<Subscriber<E>>();

	constructor(options: EventJournalOptions<E> = {}) {
		const capacity = options.capacity ?? DEFAULT_CAPACITY;
		if (!Number.isSafeInteger(capacity) || capacity < 0) {
			throw new RangeError(
				"Event journal capacity must be a non-negative safe integer.",
			);
		}
		this.capacity = capacity;
		this.now = options.now ?? Date.now;
		this.onSubscriberError = options.onSubscriberError;
		this.entries = new Array(capacity);
	}

	get size(): number {
		return this.count;
	}

	get latestId(): number {
		return this.nextId - 1;
	}

	/** Oldest retained cursor, or undefined when the retained window is empty. */
	get oldestId(): number | undefined {
		return this.count > 0 ? this.entries[this.start]?.id : undefined;
	}

	append(event: E): EventJournalEntry<E> {
		const entry = Object.freeze({
			id: this.nextId++,
			recordedAt: this.now(),
			event,
		});
		if (this.capacity > 0) {
			const index = (this.start + this.count) % this.capacity;
			this.entries[index] = entry;
			if (this.count < this.capacity) {
				this.count++;
			} else {
				this.start = (this.start + 1) % this.capacity;
			}
		}
		this.publish(entry);
		return entry;
	}

	snapshot(query: EventJournalQuery<E> = {}): readonly EventJournalEntry<E>[] {
		const afterId = query.afterId ?? 0;
		const types = query.types ? new Set(query.types) : undefined;
		const matches: EventJournalEntry<E>[] = [];
		for (let offset = 0; offset < this.count; offset++) {
			const entry = this.entries[(this.start + offset) % this.capacity];
			if (
				!entry ||
				entry.id <= afterId ||
				(types && !types.has(entry.event.type))
			) {
				continue;
			}
			matches.push(entry);
		}
		if (query.limit === undefined) return matches;
		const limit = Math.max(0, Math.trunc(query.limit));
		return limit === 0 ? [] : matches.slice(-limit);
	}

	subscribe(
		handler: Subscriber<E>,
		options: EventJournalSubscriptionOptions<E> = {},
	): () => void {
		if (options.replay) {
			const query = options.replay === true ? {} : options.replay;
			for (const entry of this.snapshot(query)) this.invoke(handler, entry);
		}
		this.subscribers.add(handler);
		return () => this.subscribers.delete(handler);
	}

	clear(): void {
		this.entries = new Array(this.capacity);
		this.start = 0;
		this.count = 0;
	}

	private publish(entry: EventJournalEntry<E>): void {
		for (const subscriber of [...this.subscribers]) {
			this.invoke(subscriber, entry);
		}
	}

	private invoke(subscriber: Subscriber<E>, entry: EventJournalEntry<E>): void {
		try {
			subscriber(entry);
		} catch (error) {
			this.onSubscriberError?.(
				error instanceof Error ? error : new Error(String(error)),
				entry,
			);
		}
	}
}
