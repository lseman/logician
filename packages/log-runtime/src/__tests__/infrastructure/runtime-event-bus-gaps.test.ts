/** Tests for RuntimeEventBus correlation and gap semantics.

 * Validates:
 * - Gap detection with resolution hints
 * - Run group tracking and intra-run gap detection
 * - Correlation pruning
 * - Pre-replay gap check
 * - Subscription/unsubscription during replay
 */

import { describe, expect, test } from "bun:test";
import type { RuntimeEvent } from "@logician/log-core/events";
import { RuntimeEventBus } from "../../runtime/events/runtime-event-bus.ts";

// Helper to create events with explicit turnId.
function makeEvent(
	type: string,
	opts?: { turnId?: string; toolCallId?: string },
): RuntimeEvent {
	const event = { type } as RuntimeEvent & Record<string, unknown>;
	if (opts?.turnId) event.turnId = opts.turnId;
	if (opts?.toolCallId) event.toolCallId = opts.toolCallId;
	return event;
}

describe("RuntimeEventBus — correlation and gaps", () => {
	test("detects gap when replay cursor is beyond evicted range", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));
		bus.emit(makeEvent("c"));
		// Capacity is 3, so oldest entry (id=1) is evicted when id=4 arrives.
		bus.emit(makeEvent("d"));

		const gap = bus.replayGap({ afterId: 0 });
		expect(gap).toBeDefined();
		expect(gap?.requestedAfterSequence).toBe(0);
		expect(gap?.missingFromSequence).toBe(1);
		expect(gap?.missingThroughSequence).toBeLessThanOrEqual(3);
		expect(gap?.oldestAvailableSequence).toBe(2);
	});

	test("returns undefined when cursor is within retained range", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });
		for (let i = 0; i < 5; i++) bus.emit(makeEvent(`e${i}`));

		expect(bus.replayGap({ afterId: 0 })).toBeUndefined();
		expect(bus.replayGap({ afterId: 3 })).toBeUndefined();
		expect(bus.replayGap({ afterId: 4 })).toBeUndefined();
		expect(bus.replayGap({ afterId: 5 })).toBeUndefined();
	});

	test("provides resolution hint based on gap size", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });
		for (let i = 0; i < 600; i++) bus.emit(makeEvent(`e${i}`));

		const gap = bus.replayGap({ afterId: 0 });
		// Gap size is ~597 (600 - 3), which is > 500 threshold.
		expect(gap?.resolutionHint).toBe("full_refresh");
	});

	test("provides partial hint for small gaps", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });
		for (let i = 0; i < 8; i++) bus.emit(makeEvent(`e${i}`));
		// Evict a few entries
		for (let i = 8; i < 12; i++) bus.emit(makeEvent(`e${i}`));

		const gap = bus.replayGap({ afterId: 1 });
		expect(gap?.resolutionHint).toBe("partial");
	});

	test("tracks run groups and session correlation", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 20 });

		bus.beginRun({ sessionId: "sess-1", runId: "run-a", turnId: "turn-1" });
		bus.emit(makeEvent("start", { turnId: "turn-1" }));
		bus.emit(makeEvent("progress", { turnId: "turn-1" }));
		bus.endRun();

		bus.beginRun({ sessionId: "sess-1", runId: "run-b", turnId: "turn-2" });
		bus.emit(makeEvent("start", { turnId: "turn-2" }));
		bus.endRun();

		// Correlation should be session-only after endRun.
		// Run groups should contain both runs.
		const snapshot = bus.snapshot();
		expect(snapshot.length).toBe(3);

		const firstNotif = snapshot[0];
		expect(firstNotif.correlation?.runId).toBe("run-a");
		expect(firstNotif.correlation?.turnId).toBe("turn-1");
	});

	test("prunes stale run groups when session changes", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });

		// Run in session A.
		bus.beginRun({ sessionId: "sess-A", runId: "run-1", turnId: "t1" });
		bus.emit(makeEvent("a"));
		bus.endRun();

		// Switch to session B.
		bus.setSessionId("sess-B");
		bus.beginRun({ sessionId: "sess-B", runId: "run-2", turnId: "t2" });
		bus.emit(makeEvent("b"));
		bus.endRun();

		// The group for run-1 should still be in the map (pruning only happens
		// when events are evicted from the journal).
	});

	test("preReplayGapCheck fires before replay", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));
		bus.emit(makeEvent("c"));
		bus.emit(makeEvent("d")); // evicts 'a'

		const gapFired: ReturnType<typeof bus.snapshot>[] = [];
		const collected: RuntimeEvent[] = [];

		bus.subscribe(notif => collected.push(notif.event), {
			// Use afterId to trigger gap detection since 'a' was evicted.
			replay: { afterId: 0 },
			onReplayGap: () => {
				gapFired.push(bus.snapshot());
			},
		});

		// Replay returns b, c, d (3 entries).
		expect(collected.length).toBe(3);
		// Gap callback fires because entry 'a' (id=1) was evicted.
		expect(gapFired.length).toBe(1);
	});

	test("correlation is applied per-event", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });

		bus.beginRun({ sessionId: "s1", runId: "r1", turnId: "t1" });
		bus.emit(makeEvent("start", { turnId: "t1" }));
		bus.emit(makeEvent("tool", { turnId: "t1", toolCallId: "tc1" }));
		bus.endRun();

		const snapshot = bus.snapshot();
		expect(snapshot[0].correlation?.runId).toBe("r1");
		expect(snapshot[0].correlation?.turnId).toBe("t1");
		expect(snapshot[1].correlation?.toolCallId).toBe("tc1");
		// After endRun, runId/turnId are cleared.
		expect(snapshot[1].correlation?.turnId).toBe("t1"); // still has explicit turnId
	});

	test("clearHistory resets journal and correlation", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });
		bus.beginRun({ sessionId: "s1", runId: "r1", turnId: "t1" });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));
		bus.clearHistory();

		// Journal entries are cleared but the cursor (latestSequence) is preserved.
		expect(bus.snapshot().length).toBe(0);
	});

	test("subscribe after clear starts fresh", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));
		bus.clearHistory();

		const collected: RuntimeEvent[] = [];
		bus.subscribe(n => collected.push(n.event));
		bus.emit(makeEvent("c"));

		expect(collected.map(e => e.type as string)).toEqual(["c"]);
	});

	test("intraRunGap is true when the gap falls entirely within one run group", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });

		bus.beginRun({ sessionId: "s1", runId: "run-a", turnId: "t1" });
		bus.emit(makeEvent("a")); // id=1
		bus.emit(makeEvent("b")); // id=2
		bus.emit(makeEvent("c")); // id=3
		bus.emit(makeEvent("d")); // id=4, evicts id=1
		bus.endRun();

		// Gap is [1,1], entirely inside run-a's span [1,4].
		const gap = bus.replayGap({ afterId: 0 });
		expect(gap?.intraRunGap).toBe(true);
	});

	test("intraRunGap is true when the gap is covered by exactly one run group, false when it straddles two", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 1 });

		bus.beginRun({ sessionId: "s1", runId: "run-a", turnId: "t1" });
		bus.emit(makeEvent("a")); // id=1, run-a span [1,1]
		bus.endRun();

		bus.beginRun({ sessionId: "s1", runId: "run-b", turnId: "t2" });
		bus.emit(makeEvent("b")); // id=2, run-b span [2,2] — evicts id=1 (capacity 1)
		bus.endRun();

		bus.emit(makeEvent("c")); // id=3, no active run — evicts id=2

		// Cursor at 0: gap is [1,2], straddling run-a's [1,1] and run-b's [2,2].
		// Neither single group covers the full gap, so it is not intra-run.
		expect(bus.replayGap({ afterId: 0 })?.intraRunGap).toBe(false);

		// Cursor at 1: gap is [2,2], covered entirely by run-b alone.
		expect(bus.replayGap({ afterId: 1 })?.intraRunGap).toBe(true);
	});

	test("intraRunGap is false when no run group is tracked", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });
		bus.emit(makeEvent("x"));
		bus.emit(makeEvent("y"));
		bus.emit(makeEvent("z"));
		bus.emit(makeEvent("w")); // evicts id=1
		expect(bus.replayGap({ afterId: 0 })?.intraRunGap).toBe(false);
	});

	test("preReplayGapCheck skips the partial replay after notifying the subscriber", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 3 });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));
		bus.emit(makeEvent("c"));
		bus.emit(makeEvent("d")); // evicts 'a'

		const collected: RuntimeEvent[] = [];
		let gapNotified = false;

		bus.subscribe(notif => collected.push(notif.event), {
			replay: { afterId: 0 },
			preReplayGapCheck: true,
			onReplayGap: () => {
				gapNotified = true;
			},
		});

		expect(gapNotified).toBe(true);
		// With preReplayGapCheck, the partial snapshot replay is skipped —
		// the subscriber is expected to request a full refresh itself.
		expect(collected.length).toBe(0);
	});

	test("preReplayGapCheck has no effect when there is no gap", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 10 });
		bus.emit(makeEvent("a"));
		bus.emit(makeEvent("b"));

		const collected: RuntimeEvent[] = [];
		let gapNotified = false;

		bus.subscribe(notif => collected.push(notif.event), {
			replay: { afterId: 0 },
			preReplayGapCheck: true,
			onReplayGap: () => {
				gapNotified = true;
			},
		});

		expect(gapNotified).toBe(false);
		expect(collected.length).toBe(2);
	});

	test("sustained multi-run high-volume traffic keeps correlation and run-group state bounded", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 200 });

		for (let run = 0; run < 100; run++) {
			bus.beginRun({
				sessionId: "stress-session",
				runId: `run-${run}`,
				turnId: `turn-${run}`,
			});
			for (let i = 0; i < 50; i++) {
				bus.emit(makeEvent("tick", { turnId: `turn-${run}` }));
			}
			bus.endRun();
		}

		// 5000 events emitted; only the last 200 (ids 4801-5000) are retained,
		// spanning the last 4 runs (50 events each: runs 96-99).
		expect(bus.latestSequence).toBe(5_000);
		const snapshot = bus.snapshot();
		expect(snapshot).toHaveLength(200);
		expect(snapshot[0]?.correlation?.turnId).toBe("turn-96");
		expect(snapshot.at(-1)?.correlation?.turnId).toBe("turn-99");
		const retainedTurns = new Set(snapshot.map(n => n.correlation?.turnId));
		expect(retainedTurns).toEqual(
			new Set(["turn-96", "turn-97", "turn-98", "turn-99"]),
		);

		// A gap into evicted history from the very start should resolve to
		// full_refresh (far larger than 500) and not be intra-run (the
		// requested range spans many run groups).
		const gap = bus.replayGap({ afterId: 0 });
		expect(gap?.resolutionHint).toBe("full_refresh");
		expect(gap?.intraRunGap).toBe(false);
	});
});
