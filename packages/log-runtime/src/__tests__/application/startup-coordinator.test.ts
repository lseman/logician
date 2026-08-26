import { describe, expect, test } from "bun:test";
import { RuntimeStartupCoordinator } from "../../runtime/bridge/application/startup-coordinator.ts";

describe("RuntimeStartupCoordinator", () => {
	test("joins concurrent callers onto one initialization", async () => {
		let calls = 0;
		let release!: () => void;
		const startup = new RuntimeStartupCoordinator(async () => {
			calls++;
			await new Promise<void>(resolve => {
				release = resolve;
			});
		});
		const first = startup.ensure("first");
		const second = startup.ensure("second");
		await Promise.resolve();
		expect(calls).toBe(1);
		release();
		await Promise.all([first, second]);
		await startup.ensure("third");
		expect(calls).toBe(1);
	});

	test("allows retry after failed initialization", async () => {
		let calls = 0;
		const startup = new RuntimeStartupCoordinator(async () => {
			if (++calls === 1) throw new Error("not ready");
		});

		await expect(startup.ensure()).rejects.toThrow("not ready");
		await startup.ensure();
		expect(calls).toBe(2);
	});

	test("reset prevents stale in-flight completion from marking startup ready", async () => {
		let calls = 0;
		let release!: () => void;
		const startup = new RuntimeStartupCoordinator(async () => {
			calls++;
			await new Promise<void>(resolve => {
				release = resolve;
			});
		});
		const stale = startup.ensure();
		await Promise.resolve();
		startup.reset();
		release();
		await stale;

		const current = startup.ensure();
		await Promise.resolve();
		expect(calls).toBe(2);
		release();
		await current;
	});
});
