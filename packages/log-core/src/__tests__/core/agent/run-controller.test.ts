import { describe, expect, test } from "bun:test";
import { AgentRunController } from "../../../core/policy/run-controller.ts";

describe("AgentRunController", () => {
	test("escalates consecutive permission denials but resets the streak after work", () => {
		const run = new AgentRunController();
		expect(
			run.recordPermissionBatch({ denials: 2, executed: 0 }),
		).toBeUndefined();
		expect(
			run.recordPermissionBatch({ denials: 0, executed: 1 }),
		).toBeUndefined();
		expect(
			run.recordPermissionBatch({ denials: 2, executed: 0 }),
		).toBeUndefined();
		expect(run.recordPermissionBatch({ denials: 1, executed: 0 })).toEqual({
			consecutive: 3,
			total: 5,
		});
	});

	test("allows exactly one bounded verification repair", () => {
		const failed = [
			{ id: "test", command: "bun test", result: "failed" as const },
		];
		const run = new AgentRunController();
		run.requestAcceptanceStop();

		expect(run.requestVerificationRepair(failed, true)).toBe(true);
		expect(run.acceptanceStopRequested).toBe(false);
		expect(run.requestVerificationRepair(failed, true)).toBe(false);
	});
});
