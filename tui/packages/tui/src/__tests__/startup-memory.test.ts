import assert from "node:assert/strict";
import { test } from "node:test";
import { formatStartupMemory } from "../app/startup/memory.ts";

void test("formatStartupMemory returns empty array (legacy feature removed)", () => {
	assert.deepEqual(formatStartupMemory({}), []);
	assert.deepEqual(formatStartupMemory({ legacy_state: { count: 1 } }), []);
});
