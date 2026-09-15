import assert from "node:assert";
import { test } from "node:test";
import {
	promptE1Diversity,
	promptE2Convergence,
	promptInit,
	promptM1Improve,
	promptM2Tune,
	promptM3Simplify,
} from "../src/prompts.ts";

const PROBLEM = {
	name: "Test Problem",
	description: "A test problem",
	functionSignature: "def heuristic(x):",
	instances: [],
	evaluateInstance: async () => 0.5,
};

function makeHeuristic(id, thought = `thought ${id}`, code = `code ${id}`) {
	return {
		id,
		thought,
		code,
		fitness: 0.5,
		generation: 0,
		createdBy: "init",
		parentIds: [],
	};
}

test("promptInit returns system + user messages", () => {
	const msgs = promptInit(PROBLEM, []);
	assert.equal(msgs.length, 2);
	assert.equal(msgs[0].role, "system");
	assert.equal(msgs[1].role, "user");
	assert.ok(msgs[0].content.includes("Test Problem"));
});

test("promptInit includes existing thoughts to avoid", () => {
	const existing = ["approach A", "approach B"];
	const msgs = promptInit(PROBLEM, existing);
	assert.ok(msgs[1].content.includes("approach A"));
	assert.ok(msgs[1].content.includes("approach B"));
});

test("promptE1Diversity includes parents", () => {
	const parents = [makeHeuristic(1), makeHeuristic(2)];
	const msgs = promptE1Diversity(PROBLEM, parents);
	assert.equal(msgs.length, 2);
	assert.ok(msgs[1].content.includes("Heuristic 1"));
	assert.ok(msgs[1].content.includes("DIFFERENT"));
});

test("promptE2Convergence includes parents", () => {
	const parents = [makeHeuristic(1), makeHeuristic(2)];
	const msgs = promptE2Convergence(PROBLEM, parents);
	assert.ok(msgs[1].content.includes("Heuristic 1"));
	assert.ok(msgs[1].content.includes("common algorithmic principles"));
});

test("promptM1Improve includes parent", () => {
	const parent = makeHeuristic(1);
	const msgs = promptM1Improve(PROBLEM, parent);
	assert.ok(msgs[1].content.includes("Heuristic 1"));
	assert.ok(msgs[1].content.includes("IMPROVED"));
});

test("promptM2Tune includes parent", () => {
	const parent = makeHeuristic(1);
	const msgs = promptM2Tune(PROBLEM, parent);
	assert.ok(msgs[1].content.includes("TUNE"));
});

test("promptM3Simplify includes parent", () => {
	const parent = makeHeuristic(1);
	const msgs = promptM3Simplify(PROBLEM, parent);
	assert.ok(msgs[1].content.includes("SIMPLIFIED"));
});
