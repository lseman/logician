import { describe, expect, test } from "bun:test";
import { EventJournal } from "../../runtime/events/event-journal.ts";

type TestEvent =
	| { type: "start"; run: string }
	| { type: "progress"; value: number }
	| { type: "end"; ok: boolean };

describe("EventJournal", () => {
	test("retains a bounded ordered window with monotonic cursors", () => {
		let now = 100;
		const journal = new EventJournal<TestEvent>({
			capacity: 2,
			now: () => now++,
		});
		journal.append({ type: "start", run: "a" });
		journal.append({ type: "progress", value: 1 });
		journal.append({ type: "end", ok: true });

		expect(journal.size).toBe(2);
		expect(journal.latestId).toBe(3);
		expect(journal.snapshot().map(entry => entry.id)).toEqual([2, 3]);
		expect(journal.snapshot().map(entry => entry.recordedAt)).toEqual([
			101, 102,
		]);
	});

	test("queries by cursor, event type, and newest limit", () => {
		const journal = new EventJournal<TestEvent>();
		journal.append({ type: "start", run: "a" });
		journal.append({ type: "progress", value: 1 });
		journal.append({ type: "progress", value: 2 });
		journal.append({ type: "end", ok: true });

		const result = journal.snapshot({
			afterId: 1,
			types: ["progress"],
			limit: 1,
		});
		expect(result.map(entry => entry.event)).toEqual([
			{ type: "progress", value: 2 },
		]);
	});

	test("replays before live delivery and supports unsubscribe", () => {
		const journal = new EventJournal<TestEvent>();
		journal.append({ type: "start", run: "a" });
		const received: number[] = [];
		const unsubscribe = journal.subscribe(entry => received.push(entry.id), {
			replay: true,
		});
		journal.append({ type: "progress", value: 1 });
		unsubscribe();
		journal.append({ type: "end", ok: true });

		expect(received).toEqual([1, 2]);
	});

	test("isolates subscriber failures and reports them", () => {
		const errors: string[] = [];
		const delivered: number[] = [];
		const journal = new EventJournal<TestEvent>({
			onSubscriberError: error => errors.push(error.message),
		});
		journal.subscribe(() => {
			throw new Error("broken observer");
		});
		journal.subscribe(entry => delivered.push(entry.id));
		journal.append({ type: "start", run: "a" });

		expect(errors).toEqual(["broken observer"]);
		expect(delivered).toEqual([1]);
	});

	test("zero capacity supports live delivery without retaining history", () => {
		const journal = new EventJournal<TestEvent>({ capacity: 0 });
		const received: number[] = [];
		journal.subscribe(entry => received.push(entry.id));
		journal.append({ type: "start", run: "a" });

		expect(received).toEqual([1]);
		expect(journal.size).toBe(0);
		expect(journal.snapshot()).toEqual([]);
	});

	test("rejects invalid capacities and clear preserves cursor monotonicity", () => {
		expect(() => new EventJournal({ capacity: -1 })).toThrow(RangeError);
		const journal = new EventJournal<TestEvent>();
		journal.append({ type: "start", run: "a" });
		journal.clear();
		const entry = journal.append({ type: "end", ok: true });
		expect(entry.id).toBe(2);
		expect(journal.snapshot()).toHaveLength(1);
	});
});
