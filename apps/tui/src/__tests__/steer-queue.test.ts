import { test } from "bun:test";
import assert from "node:assert/strict";
import { SteerQueue } from "../status/steer-queue.ts";
import { visibleWidth } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

const plain = (value: string): string =>
	value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

void test("SteerQueue renders nothing when every queue is empty", () => {
	const queue = new SteerQueue();
	assert.deepEqual(queue.render(80), []);
});

void test("SteerQueue shows queued, follow-up, and next-turn rows", () => {
	const queue = new SteerQueue();
	queue.setItems(["fix the bug"], ["run the tests"], ["ship the release"]);
	const rendered = plain(queue.render(80).join("\n"));

	assert.match(rendered, /QUEUE\s+.*queued.*follow-up.*next/);
	assert.match(rendered, /fix the bug/);
	assert.match(rendered, /run the tests/);
	assert.match(rendered, /ship the release/);
});

void test("SteerQueue renders clean lines (no borders)", () => {
	const queue = new SteerQueue();
	queue.setItems(["fix the bug"]);
	const rows = queue.render(80);

	for (const row of rows) {
		assert.ok(!row.includes("┌"));
		assert.ok(!row.includes("├"));
		assert.ok(!row.includes("│"));
		assert.ok(!row.includes("─"));
	}
});

void test("SteerQueue clamps every line to the render width", () => {
	const queue = new SteerQueue();
	queue.setItems(["a".repeat(200)], ["b".repeat(200)], ["c".repeat(200)]);
	const rows = queue.render(60);
	assert.ok(rows.every(row => visibleWidth(row) <= 60));
});

void test("SteerQueue shows a hidden count past MAX_ROWS", () => {
	const queue = new SteerQueue();
	queue.setItems(["a", "b", "c", "d", "e", "f", "g"]);
	const rendered = plain(queue.render(80).join("\n"));
	assert.match(rendered, /1 more/);
});

void test("SteerQueue shows steer-now affordance on first steering row", () => {
	const queue = new SteerQueue();
	queue.setItems(["first message"]);
	const rendered = plain(queue.render(80).join("\n"));
	assert.match(rendered, /first message/);
	assert.match(rendered, /steer now/);
	// First steering row should have the ▶ clickable indicator
	assert.match(rendered, /▶.*first message/);
});

void test("SteerQueue shows clickable indicators on steering rows", () => {
	const queue = new SteerQueue();
	queue.setItems(["steer this"], ["follow later"]);
	const rendered = plain(queue.render(80).join("\n"));
	// Steering rows get ▶, follow-up rows get ·
	assert.match(rendered, /▶.*steer this/);
	assert.match(rendered, /·.*follow later/);
});

void test("SteerQueue footer has action hints", () => {
	const queue = new SteerQueue();
	queue.setItems(["msg"]);
	const rendered = plain(queue.render(80).join("\n"));
	assert.match(rendered, /click to steer/);
	assert.match(rendered, /Ctrl\+Enter/);
	assert.match(rendered, /Ctrl\+Q/);
});
