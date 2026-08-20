import { test } from "bun:test";
import assert from "node:assert/strict";
import { resetToRunCheckpoint } from "../../../core/compaction/run-checkpoint.ts";

void test("structured checkpoint keeps system, objective, tasks, and recent evidence", () => {
	const reset = resetToRunCheckpoint(
		[
			{ role: "system", content: "system" },
			{ role: "user", content: "fix the bug" },
			{ role: "tool", content: "test failed on line 4", tool_call_id: "1" },
			{ role: "assistant", content: "I found the cause" },
		],
		[{ id: 2, subject: "verify fix", status: "in_progress" }],
	);
	assert.equal(reset.length, 2);
	assert.equal(reset[0].role, "system");
	assert.match(String(reset[1].content), /fix the bug/);
	assert.match(String(reset[1].content), /#2 \[in_progress\] verify fix/);
	assert.match(String(reset[1].content), /test failed on line 4/);
});
