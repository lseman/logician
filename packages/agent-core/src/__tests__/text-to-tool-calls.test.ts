import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../tools/shared/text-to-tool-calls.ts";

const textualGrepCall = `I will inspect the notices.
<tool_call>
<function=grep>
<parameter=pattern>
notice|NoticeEvent
</parameter>
<parameter=path>
/data/dev/logician/tui/packages/coding-agent/src/application/agent-bridge.ts
</parameter>
<parameter=limit>
50
</parameter>
</function>
</tool_call>`;

void test("converts XML parameter tool text into structured JSON arguments", () => {
	const calls = parseTextToolCalls(textualGrepCall);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].name, "grep");
	assert.deepEqual(JSON.parse(calls[0].arguments), {
		pattern: "notice|NoticeEvent",
		path: "/data/dev/logician/tui/packages/coding-agent/src/application/agent-bridge.ts",
		limit: 50,
	});
});

void test("removes promoted tool markup while preserving surrounding prose", () => {
	assert.equal(
		stripTextToolCalls(textualGrepCall),
		"I will inspect the notices.",
	);
});

void test("accepts markdown-escaped tool_call wrapper names", () => {
	const content = textualGrepCall.replaceAll("tool_call", "tool\\_call");
	assert.equal(stripTextToolCalls(content), "I will inspect the notices.");
	assert.equal(parseTextToolCalls(content)[0]?.name, "grep");
});

void test("accepts bold escaped XML tool markup without leaking markdown", () => {
	const content = `**<tool\\_call>**
    <function=read_file>
    <parameter=limit>
    50
    **</parameter>**
    <parameter=offset>
    2088
    **</parameter>**
    <parameter=path>
    /workspace/transcript-display.ts
    **</parameter>**
    **</function>**
    **</tool\\_call>**`;
	const calls = parseTextToolCalls(content);
	assert.equal(calls.length, 1);
	assert.equal(calls[0].name, "read_file");
	assert.deepEqual(JSON.parse(calls[0].arguments), {
		limit: 50,
		offset: 2088,
		path: "/workspace/transcript-display.ts",
	});
	assert.equal(stripTextToolCalls(content), "");
});

// Regression test for the exact two-call format agents emit with stray `</tool_call>` markers.
const textualTwoReadFileCalls = `<tool_call>
    <function=read_file>
    <parameter=path>
    /data/dev/solvers/tests/test_amd_debug.cpp
    </parameter>
    </function>
</tool_call>
    <function=read_file>
    <parameter=path>
    /data/dev/solvers/CMakeLists.txt
    </parameter>
    <parameter=offset>
    1
    </parameter>
    <parameter=limit>
    30
    </parameter>
    </function>
</tool_call>`;

void test("parses multiple XML-parameter tool calls with stray markers", () => {
	const calls = parseTextToolCalls(textualTwoReadFileCalls);
	assert.equal(calls.length, 2);

	assert.equal(calls[0].name, "read_file");
	assert.deepEqual(JSON.parse(calls[0].arguments), {
		path: "/data/dev/solvers/tests/test_amd_debug.cpp",
	});

	assert.equal(calls[1].name, "read_file");
	const secondArgs = JSON.parse(calls[1].arguments);
	assert.equal(secondArgs.path, "/data/dev/solvers/CMakeLists.txt");
	assert.equal(secondArgs.offset, 1);
	assert.equal(secondArgs.limit, 30);
});

void test("strips stray markers when no surrounding prose exists", () => {
	assert.equal(stripTextToolCalls(textualTwoReadFileCalls), "");
});

void test("does not promote source code that resembles an unknown tool call", () => {
	const content = 'assert.match(catalog, new RegExp(`name="${skill.name}"`));';
	const calls = parseTextToolCalls(content, name => name === "read_file");

	assert.deepEqual(calls, []);
});
