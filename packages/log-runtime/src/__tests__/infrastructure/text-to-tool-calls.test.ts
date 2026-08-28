import { test } from "bun:test";
import assert from "node:assert/strict";
import { parseTextToolCalls, stripTextToolCalls } from "@logician/log-core";

const textualGrepCall = `I will inspect the notices.
<tool_call>
<function=grep>
<parameter=pattern>
notice|NoticeEvent
</parameter>
<parameter=path>
/data/dev/logician/packages/coding-agent/src/application/agent-bridge.ts
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
		path: "/data/dev/logician/packages/coding-agent/src/application/agent-bridge.ts",
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
	// biome-ignore lint/suspicious/noTemplateCurlyInString: fixture contains literal source code.
	const content = 'assert.match(catalog, new RegExp(`name="${skill.name}"`));';
	const calls = parseTextToolCalls(content, name => name === "read_file");

	assert.deepEqual(calls, []);
});

void test("preserves source order across mixed tool-call syntaxes", () => {
	const content = [
		'first(path="/a,b")',
		'<function=second>{"nested":{"values":[1,2]}}</function>',
		'[{"name":"third","arguments":{"query":{"must":[{"term":"x"}]}}}]',
		'[[tool_call(id=fourth, expression="fn(a,b)", flags={"deep":true})]]',
	].join("\n");

	const calls = parseTextToolCalls(content);

	assert.deepEqual(
		calls.map(call => call.name),
		["first", "second", "third", "fourth"],
	);
	assert.deepEqual(JSON.parse(calls[0].arguments), { path: "/a,b" });
	assert.deepEqual(JSON.parse(calls[1].arguments), {
		nested: { values: [1, 2] },
	});
	assert.deepEqual(JSON.parse(calls[2].arguments), {
		query: { must: [{ term: "x" }] },
	});
	assert.deepEqual(JSON.parse(calls[3].arguments), {
		expression: "fn(a,b)",
		flags: { deep: true },
	});
});

void test("strips every promoted syntax without leaving nested fragments", () => {
	const content = [
		"before",
		'first(expression="fn(a,b)")',
		'[[tool_call(id=second, payload={"items":[1,2]})]]',
		'[{"name":"third","arguments":{"nested":{"ok":true}}}]',
		"after",
	].join("\n");

	assert.equal(stripTextToolCalls(content), "before\n\n\n\nafter");
});

void test("uses collision-resistant ids and ignores malformed partial calls", () => {
	const first = parseTextToolCalls("read_file(path=/tmp/a)")[0];
	const second = parseTextToolCalls("read_file(path=/tmp/a)")[0];
	assert.ok(first.id.startsWith("tc_"));
	assert.ok(second.id.startsWith("tc_"));
	assert.notEqual(first.id, second.id);
	assert.deepEqual(
		parseTextToolCalls(
			'<function=broken>{"x":1}\n[[tool_call(id=also_broken, value=(1,2)]',
		),
		[],
	);
});
