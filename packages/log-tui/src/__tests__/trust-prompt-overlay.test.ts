import { test } from "bun:test";
import assert from "node:assert/strict";
import { TrustPromptOverlay } from "../overlays/trust-prompt-overlay.ts";
import { visibleWidth } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

function prompt(): TrustPromptOverlay {
	const overlay = new TrustPromptOverlay();
	overlay.setOptions({
		cwd: "/workspace/example",
		paths: [".logician.json", ".logician/skills/review/SKILL.md"],
	});
	overlay.show();
	return overlay;
}

void test("trust prompt explains scope and renders within terminal width", () => {
	const overlay = prompt();
	for (const width of [36, 80, 120]) {
		const lines = overlay.render(width);
		const text = lines.join("\n");
		assert.match(text, /TRUST THIS WORKSPACE/);
		assert.match(text, /Trust this folder/);
		assert.match(text, /Trust for this session/);
		assert.match(text, /Exit without saving/);
		assert.match(text, /\.logician\.json/);
		assert.ok(lines.every(line => visibleWidth(line) <= width));
	}
});

void test("trust prompt supports navigation and direct shortcuts", () => {
	const navigated = prompt();
	navigated.handleInput("\x1b[B");
	assert.deepEqual(navigated.handleInput("\r"), {
		type: "trust-choice",
		choice: "trust-parent",
	});

	assert.deepEqual(prompt().handleInput("s"), {
		type: "trust-choice",
		choice: "session-only",
	});
	assert.deepEqual(prompt().handleInput("n"), {
		type: "trust-choice",
		choice: "deny",
	});
	assert.deepEqual(prompt().handleInput("\x1b"), {
		type: "trust-choice",
		choice: "deny-session",
	});
});
