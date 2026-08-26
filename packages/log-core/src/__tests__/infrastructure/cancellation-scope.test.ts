import { describe, expect, test } from "bun:test";
import {
	CancellationError,
	CancellationScope,
} from "../../runtime/control/cancellation-scope.ts";

describe("CancellationScope", () => {
	test("propagates a parent reason and detaches through close", async () => {
		const parent = new AbortController();
		const scope = new CancellationScope({
			operation: "child tool",
			parent: parent.signal,
		});
		const reason = new CancellationError("steered", "steering", "agent turn");
		parent.abort(reason);

		expect(scope.signal.aborted).toBe(true);
		expect(scope.signal.reason).toBe(reason);
		await scope.close();
	});

	test("uses a typed timeout reason", async () => {
		const scope = new CancellationScope({
			operation: "provider",
			timeoutMs: 1,
		});
		await new Promise(resolve => setTimeout(resolve, 5));

		expect(scope.signal.reason).toBeInstanceOf(CancellationError);
		expect(scope.signal.reason.kind).toBe("timeout");
		await scope.close();
	});

	test("runs cleanup once in reverse registration order", async () => {
		const scope = new CancellationScope({ operation: "run" });
		const order: number[] = [];
		scope.addCleanup(() => {
			order.push(1);
		});
		scope.addCleanup(() => {
			order.push(2);
		});

		await scope.close();
		await scope.close();
		expect(order).toEqual([2, 1]);
	});
});
