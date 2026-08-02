import assert from "node:assert/strict";
import { test } from "node:test";
import { TrustPromptOverlay } from "../overlays/trust-prompt-overlay.ts";
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

void test("trust prompt exposes workspace scope to Ink", () => {
	const overlay = prompt();
	const model = overlay.getInkOverlayModel();
	assert.match(model.title, /Trust this workspace/);
	assert.match(model.items.map((item) => item.label).join("\n"), /Trust this folder/);
	assert.match(model.items.map((item) => item.label).join("\n"), /Trust for this session/);
	assert.match(model.items.map((item) => item.label).join("\n"), /Exit without saving/);
	assert.match(model.headerLines?.join("\n") ?? "", /\.logician\.json/);
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
