import { test } from "bun:test";
import assert from "node:assert/strict";
import { sanitizeToolCallArguments } from "../../core/provider/messages.ts";

void test("sanitizeToolCallArguments leaves valid JSON untouched", () => {
	const calls = [{ id: "a", name: "read_file", arguments: '{"path":"x"}' }];
	const result = sanitizeToolCallArguments(calls);
	assert.deepEqual(result, calls);
	assert.equal(
		result,
		calls,
		"returns the same array reference when nothing changed",
	);
});

void test("sanitizeToolCallArguments replaces unparseable arguments with '{}'", () => {
	const calls = [
		{
			id: "a",
			name: "write_file",
			arguments: '{"path":"x","content":"cut off',
		},
	];
	const result = sanitizeToolCallArguments(calls);
	assert.equal(result[0].arguments, "{}");
	assert.equal(result[0].id, "a");
	assert.equal(result[0].name, "write_file");
});

void test("sanitizeToolCallArguments only repairs the broken call among several", () => {
	const calls = [
		{ id: "a", name: "read_file", arguments: '{"path":"x"}' },
		{ id: "b", name: "write_file", arguments: '{"path":"y","content":"trunc' },
	];
	const result = sanitizeToolCallArguments(calls);
	assert.equal(result[0].arguments, '{"path":"x"}');
	assert.equal(result[1].arguments, "{}");
});
