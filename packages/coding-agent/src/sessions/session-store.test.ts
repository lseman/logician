import { test } from "bun:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionStore } from "./session-store.ts";

test("stores the session database inside the project .logician directory", () => {
	const projectDir = mkdtempSync(join(tmpdir(), "logician-session-store-"));
	try {
		const store = new SessionStore(projectDir);
		store.close();
		assert.equal(
			existsSync(
				join(projectDir, ".logician", "tui", "sessions", "history.db"),
			),
			true,
		);
	} finally {
		rmSync(projectDir, { recursive: true, force: true });
	}
});
