import { describe, expect, test } from "bun:test";
import {
	RuntimeEventBus,
	type RuntimeEventReplayGap,
} from "../../runtime/events/runtime-event-bus.ts";

describe("RuntimeEventBus replay", () => {
	test("keeps live delivery and monotonic protocol metadata", () => {
		let now = 1_000;
		const bus = new RuntimeEventBus({ now: () => now++ });
		const received: Array<{ sequence: number; timestamp: number }> = [];
		const unsubscribe = bus.subscribe(notification =>
			received.push({
				sequence: notification.sequence,
				timestamp: notification.timestamp,
			}),
		);
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "phase", state: "thinking" });
		unsubscribe();
		bus.emit({ type: "phase", state: "ready" });

		expect(received).toEqual([
			{ sequence: 1, timestamp: 1_000 },
			{ sequence: 2, timestamp: 1_001 },
		]);
		expect(bus.latestSequence).toBe(3);
	});

	test("replays retained notifications before live delivery", () => {
		const bus = new RuntimeEventBus();
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "notice", level: "info", label: "MCP", text: "Loaded" });
		const sequences: number[] = [];
		bus.subscribe(notification => sequences.push(notification.sequence), {
			replay: true,
		});
		bus.emit({ type: "phase", state: "thinking" });

		expect(sequences).toEqual([1, 2, 3]);
	});

	test("supports reconnect cursors and event-type filters", () => {
		const bus = new RuntimeEventBus();
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "notice", level: "info", label: "One", text: "First" });
		bus.emit({ type: "phase", state: "thinking" });
		bus.emit({
			type: "notice",
			level: "success",
			label: "Two",
			text: "Second",
		});

		const replay = bus.snapshot({ afterId: 1, types: ["notice"], limit: 1 });
		expect(replay).toHaveLength(1);
		expect(replay[0]?.sequence).toBe(4);
		expect(replay[0]?.event.type).toBe("notice");
	});

	test("preserves session, run, turn, and tool correlation through replay", () => {
		const bus = new RuntimeEventBus();
		bus.beginRun({ sessionId: "session-1", runId: "run-1", turnId: "turn-1" });
		bus.emit({
			type: "tool_execution_start",
			toolName: "read_file",
			args: { path: "a.ts" },
			toolCallId: "call-1",
		});
		bus.endRun();

		expect(bus.snapshot()[0]?.correlation).toEqual({
			sessionId: "session-1",
			runId: "run-1",
			turnId: "turn-1",
			toolCallId: "call-1",
		});
	});

	test("bounds history without resetting sequence across clear", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 2 });
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "phase", state: "thinking" });
		bus.emit({ type: "phase", state: "tool" });
		expect(bus.snapshot().map(item => item.sequence)).toEqual([2, 3]);

		bus.clearHistory();
		expect(bus.snapshot()).toEqual([]);
		bus.emit({ type: "phase", state: "ready" });
		expect(bus.snapshot()[0]?.sequence).toBe(4);
	});

	test("reports replay gaps when a reconnect cursor predates retained history", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 2 });
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "phase", state: "thinking" });
		bus.emit({ type: "phase", state: "tool" });
		let gap: RuntimeEventReplayGap | undefined;
		const replayed: number[] = [];
		bus.subscribe(notification => replayed.push(notification.sequence), {
			replay: { afterId: 0 },
			onReplayGap: detected => {
				gap = detected;
			},
		});

		expect(gap).toEqual({
			requestedAfterSequence: 0,
			missingFromSequence: 1,
			missingThroughSequence: 1,
			oldestAvailableSequence: 2,
			sessionId: undefined,
			resolutionHint: "partial",
			intraRunGap: false,
		});
		expect(replayed).toEqual([2, 3]);
	});

	test("reports a complete replay gap after retained history is cleared", () => {
		const bus = new RuntimeEventBus();
		bus.emit({ type: "phase", state: "ready" });
		bus.emit({ type: "phase", state: "thinking" });
		bus.clearHistory();

		expect(bus.replayGap({ afterId: 0 })).toEqual({
			requestedAfterSequence: 0,
			missingFromSequence: 1,
			missingThroughSequence: 2,
			oldestAvailableSequence: undefined,
			sessionId: undefined,
			resolutionHint: "partial",
			intraRunGap: false,
		});
	});

	test("keeps a bounded correlated window under sustained event volume", () => {
		const bus = new RuntimeEventBus({ historyCapacity: 64 });
		bus.beginRun({
			sessionId: "stress-session",
			runId: "stress-run",
			turnId: "stress-turn",
		});
		for (let index = 0; index < 5_000; index++) {
			bus.emit({ type: "phase", state: index % 2 ? "thinking" : "tool" });
		}

		const retained = bus.snapshot();
		expect(retained).toHaveLength(64);
		expect(retained[0]?.sequence).toBe(4_937);
		expect(retained.at(-1)?.sequence).toBe(5_000);
		expect(
			retained.every(item => item.correlation?.runId === "stress-run"),
		).toBe(true);
		expect(bus.replayGap({ afterId: 100 })?.missingThroughSequence).toBe(4_936);
	});

	test("isolates a throwing live subscriber", () => {
		const bus = new RuntimeEventBus();
		const received: number[] = [];
		bus.subscribe(() => {
			throw new Error("broken client");
		});
		bus.subscribe(notification => received.push(notification.sequence));
		bus.emit({ type: "phase", state: "ready" });

		expect(received).toEqual([1]);
	});

	test("reports structured diagnostics alongside the compatible notice", () => {
		const bus = new RuntimeEventBus();
		const events: string[] = [];
		bus.subscribe(({ event }) => events.push(event.type));
		bus.reportError(new TypeError("invalid payload"), {
			component: "mcp",
			operation: "decode-response",
			recoverable: true,
		});

		const diagnostic = bus.snapshot({ types: ["diagnostic"] })[0]?.event;
		expect(diagnostic).toEqual({
			type: "diagnostic",
			severity: "error",
			component: "mcp",
			operation: "decode-response",
			code: "TypeError",
			message: "invalid payload",
			recoverable: true,
		});
		expect(events).toEqual(["diagnostic", "notice"]);
	});

	test("isolates replay failures and still attaches the subscriber", () => {
		const bus = new RuntimeEventBus();
		bus.emit({ type: "phase", state: "ready" });
		const seen: number[] = [];
		bus.subscribe(
			notification => {
				seen.push(notification.sequence);
				if (notification.sequence === 1) throw new Error("replay failed");
			},
			{ replay: true },
		);
		bus.emit({ type: "phase", state: "thinking" });

		expect(seen).toEqual([1, 2]);
	});
});
