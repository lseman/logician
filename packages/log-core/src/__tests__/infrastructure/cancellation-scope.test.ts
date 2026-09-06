import { describe, expect, test } from "bun:test";
import {
	CancellationError,
	CancellationScope,
} from "../../system/lifecycle/cancellation-scope.ts";

describe("CancellationScope", () => {
	test("does not start queued work after synchronous cancellation", async () => {
		const scope = new CancellationScope({ operation: "queued tool" });
		let started = false;
		const pending = scope.run(async () => {
			started = true;
		});
		scope.abort(new Error("cancel before dispatch"));
		await expect(pending).rejects.toThrow("cancel before dispatch");
		expect(started).toBe(false);
	});

	test("concurrent close calls wait for the same cleanup", async () => {
		const scope = new CancellationScope({ operation: "cleanup" });
		let release!: () => void;
		const gate = new Promise<void>(resolve => {
			release = resolve;
		});
		let cleaned = false;
		scope.addCleanup(async () => {
			await gate;
			cleaned = true;
		});
		const first = scope.close();
		let secondFinished = false;
		const second = scope.close().then(() => {
			secondFinished = true;
		});
		await Promise.resolve();
		expect(secondFinished).toBe(false);
		release();
		await Promise.all([first, second]);
		expect(cleaned).toBe(true);
	});
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
