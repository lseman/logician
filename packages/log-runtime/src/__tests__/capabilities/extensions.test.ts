import { describe, expect, test } from "bun:test";
import { ExtensionRegistry } from "../../capabilities/extensions/extensions.ts";

describe("ExtensionRegistry lifecycle", () => {
	test("initializes and reloads behind one stable interface", async () => {
		const extensions = new ExtensionRegistry({
			sessionId: "test-session",
			cwd: process.cwd(),
			projectTrusted: false,
			extensionDirs: { paths: [] },
		});
		expect(extensions.isInitialized()).toBe(false);

		await extensions.initialize();
		await extensions.getLoadPromise();
		expect(extensions.isInitialized()).toBe(true);
		expect(extensions.getCommands()).toEqual([]);
		expect(await extensions.executeCommand("missing", "")).toBeUndefined();

		await extensions.reload();
		expect(extensions.isInitialized()).toBe(true);
	});
});
