import { test } from "bun:test";
import assert from "node:assert/strict";
import type { SessionStore } from "@logician/coding-agent/sessions";
import { SessionBrowserOverlay } from "../overlays/session-manager.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

const plain = (value: string): string =>
	value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

test("session browser shows the live filter typed in list mode", () => {
	const overlay = new SessionBrowserOverlay();
	overlay.setStore({
		listSessions: () => [
			{
				id: "alpha",
				title: "Alpha session",
				name: null,
				preview: "first task",
				lastUpdated: "2026-08-09T00:00:00.000Z",
				messageCount: 2,
			},
			{
				id: "beta",
				title: "Beta session",
				name: null,
				preview: "second task",
				lastUpdated: "2026-08-09T00:00:00.000Z",
				messageCount: 3,
			},
		],
	} as unknown as SessionStore);
	overlay.show();

	overlay.handleInput("b");
	overlay.handleInput("e");

	const output = plain(overlay.render(100).join("\n"));
	assert.match(output, /\/filter: be/);
	assert.match(output, /Beta session/);
	assert.doesNotMatch(output, /Alpha session/);
});

test("session browser scrolls the window as selection moves past the visible rows", () => {
	const overlay = new SessionBrowserOverlay();
	const sessions = Array.from({ length: 20 }, (_, i) => ({
		id: `s${i}`,
		title: `Session ${i}`,
		name: null,
		preview: "task",
		lastUpdated: "2026-08-09T00:00:00.000Z",
		messageCount: 1,
	}));
	overlay.setStore({
		listSessions: () => sessions,
	} as unknown as SessionStore);
	overlay.show();

	const initial = plain(overlay.render(100).join("\n"));
	assert.match(initial, /Session 0/);
	assert.doesNotMatch(initial, /↑ \d+ more above/);
	assert.match(initial, /↓ \d+ more below/);

	for (let i = 0; i < 19; i++) overlay.handleInput("\x1b[B");

	const scrolled = plain(overlay.render(100).join("\n"));
	assert.match(scrolled, /Session 19/);
	assert.match(scrolled, /↑ \d+ more above/);
});
