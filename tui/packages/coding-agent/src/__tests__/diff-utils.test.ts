import assert from "node:assert/strict";
import { test } from "node:test";
import {
	generateDiffString,
	syntheticUnifiedDiff,
} from "../tools/diff-utils.ts";

void test("disjoint edits produce separate hunks, untouched lines stay context", () => {
	const before = [
		"line 1",
		"line 2 old",
		...Array.from({ length: 30 }, (_, i) => `middle ${i}`),
		"line 33 old",
		"line 34",
	].join("\n");
	const after = before
		.replace("line 2 old", "line 2 new")
		.replace("line 33 old", "line 33 new");

	const { diff, firstChangedLine } = generateDiffString(before, after);

	assert.equal(firstChangedLine, 2);
	// Two hunks — the untouched middle must not appear at all.
	assert.equal((diff.match(/^@@/gm) ?? []).length, 2);
	assert.ok(!diff.includes("middle 15"), "untouched middle must be elided");
	// Only the actually-changed lines are -/+.
	assert.deepEqual(
		diff.split("\n").filter(l => l.startsWith("-") && !l.startsWith("---")),
		["-line 2 old", "-line 33 old"],
	);
	assert.deepEqual(
		diff.split("\n").filter(l => l.startsWith("+") && !l.startsWith("+++")),
		["+line 2 new", "+line 33 new"],
	);
});

void test("replaceAll-style scattered changes never mark untouched lines removed", () => {
	const before = Array.from({ length: 50 }, (_, i) =>
		i % 10 === 0 ? `call foo(${i})` : `unchanged ${i}`,
	).join("\n");
	const after = before.replaceAll("foo", "bar");

	const { diff } = generateDiffString(before, after);
	const removed = diff
		.split("\n")
		.filter(l => l.startsWith("-") && !l.startsWith("---"));
	assert.equal(removed.length, 5);
	assert.ok(removed.every(l => l.includes("call foo(")));
});

void test("hunk headers carry correct line numbers", () => {
	const before = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj";
	const after = "a\nb\nc\nd\nE\nf\ng\nh\ni\nj";
	const { diff } = generateDiffString(before, after);
	assert.match(diff, /@@ -2,7 \+2,7 @@/);
});

void test("new-file diff is all additions from /dev/null", () => {
	const diff = syntheticUnifiedDiff("x/new.txt", null, "one\ntwo");
	assert.match(diff, /--- \/dev\/null/);
	assert.match(diff, /\+one\n\+two/);
	assert.ok(!diff.includes("\n-"));
});

void test("identical content yields empty diff", () => {
	const { diff, firstChangedLine } = generateDiffString("same\n", "same\n");
	assert.equal(diff, "");
	assert.equal(firstChangedLine, undefined);
});
