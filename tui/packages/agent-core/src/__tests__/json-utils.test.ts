import { describe, it, expect } from "bun:test";
import {
	stripJsonComments,
	parseJsonWithComments,
	parseJsonWithCommentsSafe,
} from "../tools/shared/json-utils.ts";

describe("json-utils", () => {
	it("strips single-line comments", () => {
		const input = `{
      "key": "value" // this is a comment
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value");
	});

	it("strips multi-line comments", () => {
		const input = `{
      /* this is a block comment */
      "key": "value"
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value");
	});

	it("preserves comments inside strings", () => {
		const input = `{
      "key": "value with // comment inside"
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe("value with // comment inside");
	});

	it("handles escaped quotes in strings", () => {
		const input = `{
      "key": "value with \\"escaped\\" quotes" // comment
    }`;
		const stripped = stripJsonComments(input);
		const parsed = JSON.parse(stripped);
		expect(parsed.key).toBe('value with "escaped" quotes');
	});

	it("parseJsonWithComments works", () => {
		const input = `{
      "name": "test", // name field
      "value": 42 /* numeric */
    }`;
		const result = parseJsonWithComments<{ name: string; value: number }>(input);
		expect(result.name).toBe("test");
		expect(result.value).toBe(42);
	});

	it("parseJsonWithComments throws on invalid JSON", () => {
		const input = "{ invalid }";
		expect(() => parseJsonWithComments(input)).toThrow();
	});

	it("parseJsonWithCommentsSafe returns default on error", () => {
		const input = "{ invalid }";
		const result = parseJsonWithCommentsSafe(input, { fallback: true });
		expect(result.fallback).toBe(true);
	});

	it("parseJsonWithCommentsSafe parses valid JSON", () => {
		const input = `{ "key": "value" // comment }`;
		const result = parseJsonWithCommentsSafe<{ key: string }>(input, { key: "default" });
		expect(result.key).toBe("value");
	});
});
