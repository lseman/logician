// ── Text-to-Tool-Call Parser ──────────────────────────────────────────────
// Extracts tool calls from LLM response content when the LLM emits them as
// text (e.g. <function=tool_name>...args...</function>) instead of structured
// tool_calls array.

import type { ToolCall } from "../../../core/types/index.ts";

function normalizeTextToolMarkup(content: string): string {
	return content.replace(
		/\*\*(\s*<\/?(?:tool\\?_call|function|parameter)\b[^>]*>)\*\*/gi,
		"$1",
	);
}

function parseParameterValue(raw: string): unknown {
	const value = raw.trim();
	if (!value) return "";
	try {
		return JSON.parse(value) as unknown;
	} catch {
		return value;
	}
}

function parseFunctionArguments(body: string): string {
	const args: Record<string, unknown> = {};
	const parameterRegex =
		/<parameter\s*=\s*["']?([a-zA-Z_][\w.-]*)["']?\s*>([\s\S]*?)<\/parameter\s*>/gi;
	let parameter: RegExpExecArray | null;
	while ((parameter = parameterRegex.exec(body)) !== null) {
		args[parameter[1]] = parseParameterValue(parameter[2]);
	}
	if (Object.keys(args).length > 0) return JSON.stringify(args);

	const trimmed = body.trim();
	if (!trimmed) return "{}";
	try {
		const parsed = JSON.parse(trimmed) as unknown;
		return JSON.stringify(parsed);
	} catch {
		return trimmed;
	}
}

/**
 * Parse tool calls from plain text content.
 * Handles formats:
 *   <function=name>args</function>
 *   <function name="name">args</function>
 *   [[tool_call(id=tool_name, arg1=val1, arg2=val2)]]
 *   tool_name(arg1=value1, arg2=value2)
 */
export function parseTextToolCalls(
	content: string,
	/** When provided, discard candidates that are not registered tools. */
	isKnownTool?: (name: string) => boolean,
): ToolCall[] {
	if (!content || typeof content !== "string") return [];
	content = normalizeTextToolMarkup(content);

	const calls: ToolCall[] = [];
	let idCounter = 0;

	// Strategy 1: XML-style <function=...>...</function>
	const xmlRegex =
		/<function\s*=?\s*["']?([a-zA-Z_][\w.-]*)["']?\s*>([\s\S]*?)<\/function>/gi;
	let match: RegExpExecArray | null;
	while ((match = xmlRegex.exec(content)) !== null) {
		const name = match[1];
		if (name) {
			calls.push({
				id: `tc_${Date.now()}_${idCounter++}`,
				name,
				arguments: parseFunctionArguments(match[2]),
			});
		}
	}

	// Strategy 2: ReAct-style [[tool_call(...)]]
	const reactRegex = /\[\[tool_call\(([^)]+)\)\]\]/g;
	while ((match = reactRegex.exec(content)) !== null) {
		const inner = match[1];
		const parts = inner.split(",").map(p => p.trim());
		if (parts.length >= 1) {
			const name = parts[0].replace(/["']/g, "").trim();
			const args = parts.slice(1).join(", ").trim();
			if (name) {
				calls.push({
					id: `tc_${Date.now()}_${idCounter++}`,
					name,
					arguments: args || "{}",
				});
			}
		}
	}

	// Strategy 3: JSON-style [{"name": "...", "arguments": "..."}]
	const jsonRegex = /\[\s*\{[^}]*"name"\s*:[^}]*\}\s*\]/gs;
	while ((match = jsonRegex.exec(content)) !== null) {
		try {
			const parsed = JSON.parse(match[0]);
			if (Array.isArray(parsed)) {
				for (const item of parsed) {
					if (item.name && item.arguments) {
						calls.push({
							id: `tc_${Date.now()}_${idCounter++}`,
							name: item.name,
							arguments:
								typeof item.arguments === "string"
									? item.arguments
									: JSON.stringify(item.arguments),
						});
					}
				}
			} else if (parsed.name && parsed.arguments) {
				calls.push({
					id: `tc_${Date.now()}_${idCounter++}`,
					name: parsed.name,
					arguments:
						typeof parsed.arguments === "string"
							? parsed.arguments
							: JSON.stringify(parsed.arguments),
				});
			}
		} catch {
			// Not valid JSON, skip
		}
	}

	// Strategy 4: Function-call style tool_name(arg1=value1, arg2=value2)
	const funcCallRegex = /([a-zA-Z_][\w.-]*)\s*\(([^)]+)\)/g;
	while ((match = funcCallRegex.exec(content)) !== null) {
		const name = match[1];
		const argsStr = match[2];

		// Check if this looks like a tool call (has key=value pattern)
		if (argsStr.match(/[\w._-]+\s*=/)) {
			// Parse key=value arguments
			const args: Record<string, unknown> = {};
			// Match key="value" or key='value' or key=value
			// Group 1: key for quoted value, Group 2: quoted value
			// Group 3: key for unquoted value, Group 4: unquoted value
			const argRegex =
				/([\w._-]+)\s*=\s*["']([^"']*)["']|([\w._-]+)\s*=\s*([^\s,)\n]+)/g;
			let argMatch: RegExpExecArray | null;
			while ((argMatch = argRegex.exec(argsStr)) !== null) {
				let key: string | undefined;
				let value: string | undefined;
				if (argMatch[1] !== undefined) {
					key = argMatch[1];
					value = argMatch[2];
				} else if (argMatch[3] !== undefined) {
					key = argMatch[3];
					value = argMatch[4];
				}
				if (key) {
					args[key] = parseParameterValue(value || "");
				}
			}
			if (Object.keys(args).length > 0) {
				calls.push({
					id: `tc_${Date.now()}_${idCounter++}`,
					name,
					arguments: JSON.stringify(args),
				});
			}
		}
	}

	// Deduplicate by name+arguments
	const seen = new Set<string>();
	return calls.filter(call => {
		if (isKnownTool && !isKnownTool(call.name)) return false;
		const key = `${call.name}:${call.arguments}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});
}

/** Remove textual tool markup after it has been promoted to structured calls. */
export function stripTextToolCalls(content: string): string {
	if (!content) return content;
	content = normalizeTextToolMarkup(content);
	// 1. Remove the outer tool_call wrapper block (ny...art).
	// Use greedy match to handle multiple closing markers inside one block.
	let stripped = content.replace(
		/<tool\\?_call\s*>[\s\S]*?<\/tool\\?_call\s*>/gi,
		"",
	);
	// 2. Some models omit the outer tool_call wrapper — strip <function>...</function> blocks.
	stripped = stripped.replace(
		/<function\s*=?\s*["']?[a-zA-Z_][\w.-]*["']?\s*>[\s\S]*?<\/function\s*>/gi,
		"",
	);
	// 3. Remove function-call style tool calls: tool_name(arg1=value1, ...)
	stripped = stripped.replace(/[a-zA-Z_][\w.-]*\s*\([^)]+\)/g, "");
	// 4. Strip stray single-line markers left after steps 1-3:
	//    - Angle-bracket tags like <ny>, </ny>, <tool_call>, </tool_call>
	//    - Garbled closing tags rendered as standalone lines
	//    - Literal "art" on its own line (garbled </ny> from LLM)
	//    - Unicode N-with-caron (U+0147) agents sometimes emit as opening marker
	stripped = stripped.replace(/^\s*(<\/?[a-zA-Z_]\w*>|art|\u0147)\s*$/gm, "");
	return stripped.trim();
}
