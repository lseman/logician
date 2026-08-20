// ── Tool argument validation ─────────────────────────────────────────────
// Validates and coerces a tool call's arguments against its TypeBox schema.
// Scoped-down equivalent of pi-ai's utils/validation.ts: covers the
// convert-then-check path (numeric string coercion, defaults) without pi's
// full legacy-JSON-Schema compatibility shim.

import { Value } from "@sinclair/typebox/value";
import type { Tool, ToolCall } from "./types.ts";

/**
 * Validates tool call arguments against the tool's TypeBox schema.
 * Converts values (e.g. numeric strings to numbers) before checking so models that emit
 * loosely-typed JSON still validate. Throws with a formatted message when validation fails.
 */
export function validateToolArguments(
	tool: Tool,
	toolCall: ToolCall,
): Record<string, unknown> {
	const args = structuredClone(toolCall.arguments) as Record<string, unknown>;
	Value.Default(tool.parameters, args);
	const converted = Value.Convert(tool.parameters, args) as Record<
		string,
		unknown
	>;

	if (Value.Check(tool.parameters, converted)) {
		return converted;
	}

	const errors = [...Value.Errors(tool.parameters, converted)].slice(0, 5);
	const formatted = errors
		.map(e => `${e.path || "root"}: ${e.message}`)
		.join("; ");
	throw new Error(
		`Invalid arguments for tool "${tool.name}": ${formatted || "schema validation failed"}`,
	);
}

/** Finds a tool by name and validates the tool call arguments against its TypeBox schema. */
export function validateToolCall(
	tools: Tool[],
	toolCall: ToolCall,
): Record<string, unknown> {
	const tool = tools.find(t => t.name === toolCall.name);
	if (!tool) throw new Error(`Tool "${toolCall.name}" not found`);
	return validateToolArguments(tool, toolCall);
}
