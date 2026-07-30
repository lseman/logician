import assert from "node:assert/strict";
import { test } from "node:test";
import { WorkSurface } from "../status/work-surface.ts";
import { initTheme } from "../terminal/theme.ts";

void test("work surface renders working set and turn evidence", () => {
	initTheme("dark");
	const surface = new WorkSurface();
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
	assert.match(settled, /Turn summary/);
	assert.match(settled, /✓/);
});
