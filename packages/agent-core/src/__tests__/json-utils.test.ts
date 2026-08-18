import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	parseJsonWithComments,
	parseJsonWithCommentsSafe,
	stripJsonComments,
} from "../tools/json-utils.ts";

function describe(_name: string, fn: () => void) {
	fn();
}
function it(name: string, fn: () => void | Promise<void>) {
	test(name, fn);
}
function expect<T>(actual: T) {
	return {
		toBe(expected: unknown) {
			assert.equal(actual, expected);
		},
		toThrow() {
			let threw = false;
			try {
				(actual as () => void)();
			} catch {
				threw = true;
			}
			assert.ok(threw, "expected function to throw");
		},
	};
}

void describe("json-utils", () => {
	void it("strips single-line comments", () => {
		const input = `{
      "key": "value" // this is a comment
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value");
	});

	void it("strips multi-line comments", () => {
		const input = `{
      /* this is a block comment */
      "key": "value"
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value");
	});

	void it("preserves comments inside strings", () => {
		const input = `{
      "key": "value with // comment inside"
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value with // comment inside");
	});

	void it("handles escaped quotes in strings", () => {
		const input = `{
      "key": "value with \\"escaped\\" quotes" // comment
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe('value with "escaped" quotes');
	});

	void it("parseJsonWithComments works", () => {
		const input = `{
      "name": "test", // name field
      "value": 42 /* numeric */
    }`;
		const result = parseJsonWithComments<{ name: string; value: number }>(
			input,
		);
		expect(result.name).toBe("test");
		expect(result.value).toBe(42);
	});

	void it("parseJsonWithComments throws on invalid JSON", () => {
		const input = "{ invalid }";
		expect(() => parseJsonWithComments(input)).toThrow();
	});

	void it("parseJsonWithCommentsSafe returns default on error", () => {
		const input = "{ invalid }";
		const result = parseJsonWithCommentsSafe(input, { fallback: true });
		expect(result.fallback).toBe(true);
	});

	void it("parseJsonWithCommentsSafe parses valid JSON", () => {
		const input = '{ "key": "value" // comment }';
		const result = parseJsonWithCommentsSafe<{ key: string }>(input, {
			key: "default",
		});
		expect(result.key).toBe("value");
	});
});
