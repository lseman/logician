import { expect, test } from "bun:test";
import { runProcess } from "../process.ts";

test("retains bounded tails from noisy subprocesses", async () => {
	const result = await runProcess(
		process.execPath,
		[
			"-e",
			`
		process.stdout.write('x'.repeat(250000) + 'stdout-tail');
		process.stderr.write('y'.repeat(250000) + 'stderr-tail');
	`,
		],
		{ cwd: process.cwd(), timeoutMs: 5000 },
	);
	expect(result.exitCode).toBe(0);
	expect(result.stdout.length).toBe(100_000);
	expect(result.stderr.length).toBe(100_000);
	expect(result.stdout.endsWith("stdout-tail")).toBe(true);
	expect(result.stderr.endsWith("stderr-tail")).toBe(true);
});

test("preserves UTF-8 characters split between subprocess writes", async () => {
	const result = await runProcess(
		process.execPath,
		[
			"-e",
			`
		const bytes = Buffer.from('🌱');
		process.stdout.write(bytes.subarray(0, 2));
		setTimeout(() => process.stdout.write(bytes.subarray(2)), 30);
	`,
		],
		{ cwd: process.cwd(), timeoutMs: 5000 },
	);
	expect(result.exitCode).toBe(0);
	expect(result.stdout).toBe("🌱");
});
