import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { RecoveryMemory } from "../agent/guards/recovery-memory.ts";

describe("RecoveryMemory", () => {
	const makeMemory = (overrides = {}) =>
		new RecoveryMemory({ maxEntries: 50, ...overrides });

	// ── recordFailure / similarity matching ──────────────────────────────────

	it("records a failure and returns no similar entries the first time", () => {
		const memory = makeMemory();
		const { entryId, similarEntries } = memory.recordFailure(
			"file not found",
			"read_file /src/foo.ts",
			"Error: ENOENT",
		);
		assert.ok(entryId);
		assert.deepStrictEqual(similarEntries, []);
	});

	it("flags a repeated failure with the same tool and similar failure type", () => {
		const memory = makeMemory();
		memory.recordFailure("file not found", "read_file /src/foo.ts", "Error: ENOENT");
		const { similarEntries } = memory.recordFailure(
			"file not found",
			"read_file /src/foo.ts",
			"Error: ENOENT again",
		);
		assert.strictEqual(similarEntries.length, 1);
		assert.strictEqual(similarEntries[0].repeatCount, 2);
	});

	it("matches failure-type synonyms (missing vs not found)", () => {
		const memory = makeMemory();
		memory.recordFailure("missing", "read_file /a.ts", "not there");
		const { similarEntries } = memory.recordFailure(
			"not found",
			"read_file /a.ts",
			"still not there",
		);
		assert.strictEqual(similarEntries.length, 1);
	});

	it("does not match unrelated tools and failure types", () => {
		const memory = makeMemory();
		memory.recordFailure("timeout", "bash npm test", "timed out");
		const { similarEntries } = memory.recordFailure(
			"permission denied",
			"write_file /etc/passwd",
			"access denied",
		);
		assert.deepStrictEqual(similarEntries, []);
	});

	it("matches on same file + same tool even with a different failure type", () => {
		const memory = makeMemory();
		memory.recordFailure("parse error", "edit_file /src/foo.ts", "malformed JSON");
		const { similarEntries } = memory.recordFailure(
			"parse error",
			"edit_file /src/foo.ts",
			"malformed JSON again",
		);
		assert.strictEqual(similarEntries.length, 1);
	});

	// ── getWarnings ──────────────────────────────────────────────────────────

	it("getWarnings() is empty when no similar failures exist", () => {
		const memory = makeMemory();
		assert.deepStrictEqual(memory.getWarnings("bash npm build", "unknown"), []);
	});

	it("getWarnings() warns with the repeat count and last outcome once repeated", () => {
		const memory = makeMemory();
		memory.recordFailure("timeout", "bash npm test", "timed out after 30s");
		memory.recordFailure("timeout", "bash npm test", "timed out after 30s again");
		const warnings = memory.getWarnings("bash npm test", "timeout");
		assert.ok(warnings.length > 0);
		assert.match(warnings[0], /tried this approach/);
	});

	it("getWarnings() surfaces a suggested alternative when one was recorded", () => {
		const memory = makeMemory();
		memory.recordFailure(
			"timeout",
			"bash npm test",
			"timed out",
			"try running with --maxWorkers=1",
		);
		memory.recordFailure("timeout", "bash npm test", "timed out again");
		const warnings = memory.getWarnings("bash npm test", "timeout");
		assert.ok(warnings.some(w => w.includes("--maxWorkers=1")));
	});

	// ── recordSuccess ────────────────────────────────────────────────────────

	it("recordSuccess() marks matching failures as ultimately successful", () => {
		const memory = makeMemory();
		memory.recordFailure("timeout", "bash npm test", "timed out");
		memory.recordSuccess("bash npm test", "success");
		const warnings = memory.getWarnings("bash npm test", "timeout");
		assert.ok(warnings.some(w => w.includes("eventually successful")));
	});

	it("recordSuccess() does not clear entries for a genuinely different failure type", () => {
		// Regression: recordSuccess used to call approachesMatch(entry, approach, "")
		// with an empty failure type, and because "anything".includes("") is
		// always true in JS, failureTypesSimilar("", anything) always matched —
		// so a success could over-aggressively clear unrelated failures that
		// merely shared a tool name.
		const memory = makeMemory();
		memory.recordFailure("permission denied", "bash rm -rf /protected", "access denied");
		memory.recordSuccess("bash npm test", "success");
		const entries = memory.getEntries();
		assert.strictEqual(entries.length, 1);
		assert.strictEqual(entries[0].ultimatelySuccessful, undefined);
	});

	it("recordSuccess() still clears entries that share tool+file (no failure type needed)", () => {
		const memory = makeMemory();
		memory.recordFailure("parse error", "edit_file /src/foo.ts", "malformed");
		memory.recordSuccess("edit_file /src/foo.ts", "success");
		const entries = memory.getEntries();
		assert.strictEqual(entries[0].ultimatelySuccessful, true);
	});

	// ── clear / getFailureSummary ────────────────────────────────────────────

	it("clear() removes all entries", () => {
		const memory = makeMemory();
		memory.recordFailure("timeout", "bash npm test", "timed out");
		memory.clear();
		assert.deepStrictEqual(memory.getEntries(), []);
	});

	it("getFailureSummary() ranks failure types by repeat count", () => {
		const memory = makeMemory();
		memory.recordFailure("timeout", "bash npm test", "timed out");
		memory.recordFailure("timeout", "bash npm test", "timed out again");
		memory.recordFailure("parse error", "edit_file /a.ts", "bad json");
		const summary = memory.getFailureSummary();
		assert.strictEqual(summary[0].type, "timeout");
		assert.strictEqual(summary[0].count, 2);
	});

	it("trims entries beyond maxEntries", () => {
		const memory = makeMemory({ maxEntries: 3 });
		for (let i = 0; i < 5; i++) {
			memory.recordFailure(`type-${i}`, `bash cmd-${i}`, `failed-${i}`);
		}
		assert.strictEqual(memory.getEntries().length, 3);
	});
});
