import { describe, test } from "bun:test";
import assert from "node:assert/strict";
import {
	inferSessionTitle,
	isGeneratedSessionTitle,
} from "../runtime/sessions/tui-session-service.ts";

describe("session title inference", () => {
	test("extracts the topic from request phrasing", () => {
		assert.equal(
			inferSessionTitle("we need to improve our memory observation policy"),
			"Improve our memory observation policy",
		);
		assert.equal(
			inferSessionTitle("Could you fix the authentication timeout in the API?"),
			"Fix the authentication timeout in the API?",
		);
	});

	test("skips greetings so a later meaningful prompt can name the session", () => {
		assert.equal(inferSessionTitle("hi"), null);
		assert.equal(inferSessionTitle("Thank you!"), null);
	});

	test("uses the agent response to resolve a vague first request", () => {
		assert.equal(
			inferSessionTitle(
				"fix this",
				"Implemented folder-scoped FTS5 memory retrieval and indexing.",
			),
			"Folder-scoped FTS5 memory retrieval and indexing.",
		);
	});

	test("ignores attachment headings and truncates long topics", () => {
		const title = inferSessionTitle(
			`# Files mentioned by the user:\n- /tmp/output.txt\n\n# My request for Codex:\nPlease investigate why the current session browser includes conversations from unrelated working directories and correct the folder isolation`,
		);
		assert.match(title || "", /^Investigate why the current session browser/);
		assert.ok((title?.length || 0) <= 60);
	});

	test("only generated placeholders are eligible for inference", () => {
		assert.equal(isGeneratedSessionTitle("New Session"), true);
		assert.equal(isGeneratedSessionTitle("Untitled Session"), true);
		assert.equal(isGeneratedSessionTitle("Authentication timeout"), false);
	});

	test("returns null for null/undefined content without crashing", () => {
		assert.equal(inferSessionTitle(null as unknown as string), null);
		assert.equal(inferSessionTitle(undefined as unknown as string), null);
		assert.equal(inferSessionTitle(""), null);
		assert.equal(inferSessionTitle("   "), null);
		assert.equal(inferSessionTitle("\n\n"), null);
	});

	test("handles null/undefined agentResponse gracefully", () => {
		assert.equal(
			inferSessionTitle("fix the login bug", null as unknown as string),
			"Fix the login bug",
		);
		assert.equal(
			inferSessionTitle("fix the login bug", undefined as unknown as string),
			"Fix the login bug",
		);
	});
});
