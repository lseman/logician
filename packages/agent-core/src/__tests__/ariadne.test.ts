// ── Ariadne tool tests ────────────────────────────────────────────────────────

import { describe, expect, it } from "bun:test";
import { ariadne } from "../infrastructure/tools/ariadne.ts";

describe("ariadne tool", () => {
	it("should have correct metadata", () => {
		expect(ariadne.name).toBe("ariadne");
		expect(ariadne.label).toBe("Ariadne Code Graph");
		expect(ariadne.readOnly).toBe(true);
		expect(ariadne.description).toContain("Ariadne code graph");
	});

	it("should have required parameters", () => {
		const props = ariadne.parameters as Record<string, unknown>;
		expect(props.required).toContain("operation");
	});

	it("should parse string arguments as operation", () => {
		const result = ariadne.prepareArguments?.("minimal_context") ?? {};
		expect(result.operation).toBe("minimal_context");
	});

	it("should parse object arguments with operation", () => {
		const result =
			ariadne.prepareArguments?.({
				operation: "search",
				target: "login",
			}) ?? {};
		expect(result.operation).toBe("search");
		expect(result.target).toBe("login");
	});

	it("should handle argument aliases", () => {
		const result =
			ariadne.prepareArguments?.({
				op: "impact",
				symbol: "Graph::add_node",
			}) ?? {};
		expect(result.operation).toBe("impact");
		expect(result.target).toBe("Graph::add_node");
	});

	it("should merge params from JSON string", () => {
		const result =
			ariadne.prepareArguments?.({
				operation: "minimal_context",
				params: '{"mode":"review"}',
			}) ?? {};
		expect(result.operation).toBe("minimal_context");
		expect(result.mode).toBe("review");
	});

	it("should merge params from object", () => {
		const result =
			ariadne.prepareArguments?.({
				operation: "search",
				params: { limit: 10, offset: 5 },
			}) ?? {};
		expect(result.operation).toBe("search");
		expect(result.limit).toBe(10);
		expect(result.offset).toBe(5);
	});

	it("should handle max_hops alias", () => {
		const result =
			ariadne.prepareArguments?.({
				operation: "impact",
				maxHops: 4,
			}) ?? {};
		expect(result.max_hops).toBe(4);
	});

	it("should handle limit alias", () => {
		const result =
			ariadne.prepareArguments?.({
				operation: "search",
				response_limit: 25,
			}) ?? {};
		expect(result.limit).toBe(25);
	});
});
