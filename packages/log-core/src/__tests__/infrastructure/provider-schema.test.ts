import { expect, test } from "bun:test";
import { normalizeProviderToolSchema } from "../../capabilities/tools/provider-schema.ts";

test("flattens requirement-only alternatives without losing fields or mutating source", () => {
	const source = {
		type: "object",
		properties: {
			command: { type: "string" },
			commands: { type: "array", items: { type: "string" } },
		},
		anyOf: [{ required: ["command"] }, { required: ["commands"] }],
	};
	const result = normalizeProviderToolSchema(source);
	expect(result.anyOf).toBeUndefined();
	expect(result.properties).toEqual(source.properties);
	expect(result.description).toContain("command OR commands");
	expect(result.required).toBeUndefined();
	expect(source.anyOf).toHaveLength(2);
});

test("keeps common and explicit requirements when flattening alternatives", () => {
	const result = normalizeProviderToolSchema({
		type: "object",
		properties: {
			path: { type: "string" },
			a: { type: "string" },
			b: { type: "string" },
		},
		required: ["path"],
		oneOf: [{ required: ["path", "a"] }, { required: ["path", "b"] }],
	});
	expect(result.required).toEqual(["path"]);
	expect(result.oneOf).toBeUndefined();
	expect(result.description).toContain("exactly one");
});

test("preserves value unions and alternatives with additional constraints", () => {
	const result = normalizeProviderToolSchema({
		type: "object",
		properties: {
			value: { anyOf: [{ type: "string" }, { type: "number" }] },
		},
		anyOf: [{ required: ["value"], properties: { value: { const: "x" } } }],
	});
	expect(result.anyOf).toBeDefined();
	expect(
		(result.properties as Record<string, Record<string, unknown>>).value?.anyOf,
	).toEqual([{ type: "string" }, { type: "number" }]);
});
