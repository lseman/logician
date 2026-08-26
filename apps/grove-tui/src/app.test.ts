import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { sessionLaunchCommand } from "./app.ts";

describe("session launcher", () => {
	test("runs the existing TypeScript TUI through tsx", () => {
		const command = sessionLaunchCommand("session-123");
		expect(command.executable).toBe(process.execPath);
		expect(command.args[0]).toContain("tsx");
		expect(existsSync(command.args[0] ?? "")).toBe(true);
		expect(command.args.at(-2)).toBe("--session");
		expect(command.args.at(-1)).toBe("session-123");
	});
});
