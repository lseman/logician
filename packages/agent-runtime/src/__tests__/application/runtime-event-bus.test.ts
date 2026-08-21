import { expect, test } from "bun:test";
import { RuntimeEventBus } from "../../application/events/runtime-event-bus.ts";

test("runtime event bus publishes ordered notifications and unsubscribes", () => {
	const bus = new RuntimeEventBus();
	const sequences: number[] = [];
	const unsubscribe = bus.subscribe(notification => {
		sequences.push(notification.sequence);
	});
	bus.emit({ type: "notice", level: "info", label: "one", text: "one" });
	bus.emit({ type: "notice", level: "info", label: "two", text: "two" });
	unsubscribe();
	bus.emit({ type: "notice", level: "info", label: "three", text: "three" });
	expect(sequences).toEqual([1, 2]);
});

test("runtime event bus isolates subscribers and normalizes errors", () => {
	const bus = new RuntimeEventBus();
	const events: string[] = [];
	let error: Error | undefined;
	bus.subscribe(() => {
		throw new Error("client failure");
	});
	bus.subscribe(({ event }) => events.push(event.type));
	bus.onError(value => {
		error = value;
	});
	bus.reportError("runtime failure");
	expect(events).toEqual(["notice"]);
	expect(error?.message).toBe("runtime failure");
});
