import assert from "node:assert";
import { test } from "node:test";
import {
	extractFunctionName,
	parseHeuristicOutput,
	validateCode,
} from "../src/evaluator.ts";

test("parseHeuristicOutput parses XML tags", () => {
	const raw = `<thought>Best fit strategy</thought>\n<code>\n\`\`\`python\ndef select_bin(item_size, bins):\n    pass\n\`\`\`\n</code>`;
	const result = parseHeuristicOutput(raw);
	assert.ok(result);
	assert.equal(result.thought, "Best fit strategy");
	assert.ok(result.code.includes("def select_bin"));
});

test("parseHeuristicOutput parses code fence without XML", () => {
	const raw = `<thought>Simple approach</thought>\n\`\`\`python\ndef heuristic(x):\n    return x\n\`\`\``;
	const result = parseHeuristicOutput(raw);
	assert.ok(result);
	assert.equal(result.thought, "Simple approach");
	assert.ok(result.code.includes("def heuristic"));
});

test("parseHeuristicOutput returns null for missing thought", () => {
	assert.equal(parseHeuristicOutput("no thought tags"), null);
	assert.equal(parseHeuristicOutput("<thought></thought>"), null);
});

test("parseHeuristicOutput returns null for missing code", () => {
	const raw = `<thought>Some thought</thought>`;
	assert.equal(parseHeuristicOutput(raw), null);
});

test("validateCode accepts valid Python function", () => {
	const code = "def heuristic(x):\n    return x * 2";
	assert.equal(validateCode(code, "heuristic"), null);
});

test("validateCode rejects code without function definition", () => {
	const code = "x = 5";
	assert.ok(validateCode(code, "heuristic"));
});

test("validateCode rejects unbalanced parentheses", () => {
	const code = "def heuristic(x):\n    return (x";
	assert.ok(validateCode(code, "heuristic"));
});

test("extractFunctionName extracts function name", () => {
	assert.equal(extractFunctionName("def heuristic(x): pass"), "heuristic");
	assert.equal(extractFunctionName("def select_bin(a, b): pass"), "select_bin");
	assert.equal(extractFunctionName("no function here"), null);
});
