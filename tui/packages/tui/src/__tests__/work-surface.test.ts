import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";
import { WorkSurface } from "../status/work-surface.ts";
import { initTheme } from "../terminal/theme.ts";

void test("work surface renders working set and turn evidence", () => {
	initTheme("dark");
	const surface = new WorkSurface();
	surface.startRun();
	surface.startTurn();
	surface.recordToolStart("1", "edit_file", { path: "src/main.ts" });
	surface.recordToolEnd("1", "edit_file", "ok\n<post_edit_diagnostics>", false);
	surface.setContext(1200, 8000);
	const text = surface.render(100).join("\n");
	assert.match(text, /Working set/);
	assert.match(text, /Activity/);
	assert.match(text, /● running/);
	assert.match(text, /src\/main\.ts/);
	assert.match(text, /1 changed/);
	assert.match(text, /1 diagnostics/);
	assert.match(text, /1,200\/8,000/);

	surface.endTurn();
	const settled = surface.render(100).join("\n");
	assert.doesNotMatch(settled, /Activity|● running/);
	assert.match(settled, /Run summary/);
	assert.match(settled, /1 turn( |$)/);
	assert.match(settled, /✓/);
});

void test("work surface resets turn count across loop iterations", () => {
	initTheme("dark");
	const surface = new WorkSurface();

	// Simulate loop iteration 1 with 3 turns
	surface.setLoopIteration(1);
	surface.startTurn();
	surface.recordToolStart("1", "read_file", { path: "a.ts" });
	surface.endTurn();

	surface.startTurn();
	surface.recordToolStart("2", "read_file", { path: "b.ts" });
	surface.endTurn();

	surface.startTurn();
	surface.recordToolStart("3", "write_file", { path: "c.ts" });
	surface.setContext(1000, 8000);
	surface.endTurn();

	let settled = surface.render(100).join("\n");
	assert.match(settled, /3 turns/);
	assert.match(settled, /loop 1/);

	// Advance to loop iteration 2 — turn count should reset
	surface.setLoopIteration(2);

	// Start two new turns in the new loop
	surface.startTurn();
	surface.recordToolStart("4", "read_file", { path: "d.ts" });
	surface.endTurn();

	surface.startTurn();
	surface.recordToolStart("5", "write_file", { path: "e.ts" });
	surface.setContext(2000, 8000);
	surface.endTurn();

	settled = surface.render(100).join("\n");
	assert.match(settled, /2 turns/); // 2 turns in loop 2, not 5 total
	assert.match(settled, /loop 2/);

	// Transition out of loop back to regular prompt — turn count resets again
	surface.setLoopIteration(0);
	surface.startTurn();
	surface.endTurn();

	settled = surface.render(100).join("\n");
	assert.match(settled, /1 turn( |$)/); // 1 turn after leaving loop
});

void test("a new agent run resets its internal turn count and evidence", () => {
	initTheme("dark");
	const surface = new WorkSurface();
	surface.startRun();
	surface.startTurn();
	surface.recordToolStart("1", "read_file", { path: "old.ts" });
	surface.startTurn();
	assert.match(surface.render(100).join("\n"), /2 turns/);

	surface.startRun();
	surface.startTurn();
	surface.recordToolStart("2", "read_file", { path: "new.ts" });
	const nextRun = surface.render(100).join("\n");
	assert.match(nextRun, /1 turn( |$)/);
	assert.match(nextRun, /1 tools/);
});
