import assert from "node:assert/strict";
import { test } from "node:test";
import { SelectorController } from "../components/selector-controller.ts";

void test("selector controller shares wrapping and viewport behavior", () => {
	const selector = new SelectorController();
	selector.move(-1, 12);
	assert.equal(selector.index, 11);
	assert.deepEqual(selector.window(12, 5), { start: 7, end: 12 });
	selector.set(4, 12);
	assert.deepEqual(selector.window(12, 5), { start: 2, end: 7 });
});
