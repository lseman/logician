import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
	inferSessionTitle,
	isGeneratedSessionTitle,
} from "../sessions/session-store.ts";

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

	test("ignores attachment headings and truncates long topics", () => {
		const title = inferSessionTitle(`# Files mentioned by the user:\n- /tmp/output.txt\n\n# My request for Codex:\nPlease investigate why the current session browser includes conversations from unrelated working directories and correct the folder isolation`);
		assert.match(title || "", /^Investigate why the current session browser/);
		assert.ok((title?.length || 0) <= 60);
	});

	test("only generated placeholders are eligible for inference", () => {
		assert.equal(isGeneratedSessionTitle("New Session"), true);
		assert.equal(isGeneratedSessionTitle("Untitled Session"), true);
		assert.equal(isGeneratedSessionTitle("Authentication timeout"), false);
	});
});
