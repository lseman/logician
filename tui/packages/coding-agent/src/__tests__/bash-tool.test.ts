import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { bash } from "../tools/bash.ts";

void test("bash includes output and exit code for non-zero commands", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-bash-"));
	const result = await bash.execute(
		{ command: "printf 'before failure'; exit 7" },
		{ cwd },
	);
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /before failure/);
	assert.match(content, /Command exited with code 7/);
});
