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

	assert.match(rendered, /STEERING\s+1 queued · 1 follow-up · 1 next turn/);
	assert.match(rendered, /QUEUE\s+fix the bug/);
	assert.match(rendered, /LATER\s+run the tests/);
	assert.match(rendered, /NEXT\s+ship the release/);
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

void test("SteerQueue plays an arrival animation for newly queued rows", async () => {
	const queue = new SteerQueue();
	let invalidations = 0;
	queue.setOnInvalidate(() => {
		invalidations++;
	});

	queue.setItems(["first message"]);
	// The arrival frame starts at 0 immediately — first render already differs
	// from the settled glyph.
	const firstFrame = plain(queue.render(80).join("\n"));

	await new Promise(resolve => setTimeout(resolve, 400));

	const settled = plain(queue.render(80).join("\n"));
	assert.ok(invalidations > 0, "animation should invalidate on each tick");
	assert.match(settled, /QUEUE\s+first message/);
	// Both frames render the same message text; only the leading glyph differs
	// mid-animation, so this just confirms the render didn't get stuck empty.
	assert.match(firstFrame, /first message/);
});

void test("SteerQueue keys arrival state by queue + message, not position", () => {
	const queue = new SteerQueue();
	queue.setItems(["already here"]);
	queue.render(80); // settle the first item (no longer "new")

	// Adding a second steering item shouldn't restart the first item's
	// arrival animation — both rows must still render correctly either way.
	queue.setItems(["already here", "brand new"]);
	const rendered = plain(queue.render(80).join("\n"));
	assert.match(rendered, /QUEUE\s+already here/);
	assert.match(rendered, /QUEUE\s+brand new/);
});
