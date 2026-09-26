import { describe, expect, test } from "bun:test";
import { executeProcessCommand } from "../../agent/application/process-command.ts";

describe("executeProcessCommand", () => {
	test("captures combined command output and exit status", async () => {
		const result = await executeProcessCommand(
			process.cwd(),
			"printf command-ok",
		);
		expect(result).toEqual({ output: "command-ok", exitCode: 0 });
	});
});
