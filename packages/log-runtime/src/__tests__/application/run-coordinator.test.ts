import { describe, expect, test } from "bun:test";
import { RuntimeRunCoordinator } from "../../runtime/bridge/run-coordinator.ts";

describe("RuntimeRunCoordinator", () => {
	test("serializes runs and recovers its queue after a rejection", async () => {
		const coordinator = new RuntimeRunCoordinator();
		const order: string[] = [];
		const first = coordinator.submit({
			message: "first",
			canSteer: () => false,
			steer: () => {},
			execute: async () => {
				order.push("first");
				throw new Error("failed");
			},
		});
		const second = coordinator.submit({
			message: "second",
			canSteer: () => false,
			steer: () => {},
			execute: async () => {
				order.push("second");
			},
		});

		await expect(first).rejects.toThrow("failed");
		await second;
		expect(order).toEqual(["first", "second"]);
		expect(coordinator.isActive()).toBe(false);
	});

	test("routes an in-flight submission to steering", async () => {
		const coordinator = new RuntimeRunCoordinator();
		let release!: () => void;
		const running = coordinator.submit({
			message: "first",
			canSteer: () => true,
			steer: () => {},
			execute: () =>
				new Promise<void>(resolve => {
					release = resolve;
				}),
		});
		await Promise.resolve();
		const steered: string[] = [];
		await coordinator.submit({
			message: "change direction",
			canSteer: () => true,
			steer: message => steered.push(message),
			execute: async () => {
				throw new Error("must not execute");
			},
		});

		expect(steered).toEqual(["change direction"]);
		release();
		await running;
	});
});
