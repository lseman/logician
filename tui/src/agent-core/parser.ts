// ── Tool call parsing ────────────────────────────────────────────────────────────
// Parses tool calls from LLM response text.
// Mirrors Python parse_tool_calls with JSON/YAML support.

import type { ToolCall } from "./types.ts";

export function parseToolCalls(text: string): ToolCall[] {
	const calls: ToolCall[] = [];
	let index = 0;

	const push = (name: unknown, args: unknown, id?: unknown) => {
		if (typeof name !== "string" || !name.trim()) return;
		calls.push({
			id: typeof id === "string" && id.trim() ? id : `tool_${index++}`,
			name: name.trim(),
			arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
		});
	};

	const readObject = (value: unknown) => {
		if (Array.isArray(value)) {
			for (const item of value) readObject(item);
			return;
		}
		if (!value || typeof value !== "object") return;
		const obj = value as Record<string, unknown>;
		const fn =
			obj.function && typeof obj.function === "object"
				? (obj.function as Record<string, unknown>)
				: undefined;
		push(
			obj.name ?? obj.tool_name ?? fn?.name,
			obj.arguments ?? obj.args ?? obj.input ?? fn?.arguments,
			obj.id ?? obj.tool_call_id,
		);
	};

	// Strategy 1: JSON in fenced blocks. This catches the most common textual
	// fallback from OpenAI-compatible models when native tool calls wobble.
	const fencePattern = /```(?:json|tool|tool_call)?\s*([\s\S]*?)```/gi;
	let match: RegExpExecArray | null;
	while ((match = fencePattern.exec(text)) !== null) {
		const parsed = parseLooseJson(match[1].trim());
		if (parsed.ok) readObject(parsed.value);
	}

	// Strategy 2: direct JSON objects/arrays containing tool-call fields.
	for (const candidate of extractJsonCandidates(text)) {
		const parsed = parseLooseJson(candidate);
		if (parsed.ok) readObject(parsed.value);
	}

	if (calls.length > 0) {
		return dedupeCalls(calls);
	}

	// Strategy 3: JSON objects { "name": "...", "arguments": "..." }
	const jsonPattern =
		/\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}/g;
		while ((match = jsonPattern.exec(text)) !== null) {
		push(match[1], match[2]);
	}

	if (calls.length > 0) {
		return calls;
	}

	// Strategy 4: YAML-style tool_call blocks
	const yamlPattern =
		/tool_call:\s*\n\s*name:\s*([^\n]+)\s*\n\s*arguments:\s*"((?:[^"\\]|\\.)*)"/g;
		while ((match = yamlPattern.exec(text)) !== null) {
		push(match[1].trim(), match[2]);
	}

	if (calls.length > 0) {
		return calls;
	}

	// Strategy 5: Anthropic/OpenClaude-ish XML fallback.
	const xmlPattern =
		/<tool_use\b[^>]*>\s*<name>\s*([^<]+?)\s*<\/name>\s*<(?:arguments|input)>\s*([\s\S]*?)\s*<\/(?:arguments|input)>\s*<\/tool_use>/gi;
		while ((match = xmlPattern.exec(text)) !== null) {
		push(match[1].trim(), match[2].trim());
	}

	return calls;
}

export function parseToolInput(raw: string): Record<string, unknown> {
	const parsed = parseLooseJson(raw || "{}");
	if (parsed.ok && parsed.value && typeof parsed.value === "object") {
		return Array.isArray(parsed.value)
			? { items: parsed.value }
			: (parsed.value as Record<string, unknown>);
	}

	const args: Record<string, unknown> = {};
	const lines = (raw || "").split("\n");
	for (const line of lines) {
		const match = line.match(/^\s*([\w.-]+)\s*:\s*(.+)\s*$/);
		if (match) {
			args[match[1]] = stripQuotes(match[2].trim());
		}
	}
	return args;
}

function parseLooseJson(
	raw: string,
): { ok: true; value: unknown } | { ok: false } {
	const text = raw.trim();
	if (!text) return { ok: false };
	try {
		return { ok: true, value: JSON.parse(text) };
	} catch {
		// Fall through to a conservative repair pass for common model slips.
	}
	try {
		const repaired = text
			.replace(/,\s*([}\]])/g, "$1")
			.replace(/([{,]\s*)([A-Za-z_][\w-]*)(\s*:)/g, "$1\"$2\"$3")
			.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, (_, body: string) =>
				JSON.stringify(body.replace(/\\'/g, "'")),
			);
		return { ok: true, value: JSON.parse(repaired) };
	} catch {
		return { ok: false };
	}
}

function extractJsonCandidates(text: string): string[] {
	const out: string[] = [];
	for (let i = 0; i < text.length; i++) {
		const start = text[i];
		if (start !== "{" && start !== "[") continue;
		const end = matchingJsonEnd(text, i, start, start === "{" ? "}" : "]");
		if (end > i) out.push(text.slice(i, end + 1));
	}
	return out;
}

function matchingJsonEnd(
	text: string,
	start: number,
	open: string,
	close: string,
): number {
	let depth = 0;
	let inString = false;
	let quote = "";
	let escaped = false;
	for (let i = start; i < text.length; i++) {
		const ch = text[i];
		if (inString) {
			if (escaped) {
				escaped = false;
			} else if (ch === "\\") {
				escaped = true;
			} else if (ch === quote) {
				inString = false;
			}
			continue;
		}
		if (ch === "\"" || ch === "'") {
			inString = true;
			quote = ch;
			continue;
		}
		if (ch === open) depth++;
		if (ch === close) {
			depth--;
			if (depth === 0) return i;
		}
	}
	return -1;
}

function dedupeCalls(calls: ToolCall[]): ToolCall[] {
	const seen = new Set<string>();
	return calls.filter((call) => {
		const key = `${call.name}\0${call.arguments}`;
		if (seen.has(key)) return false;
		seen.add(key);
		return true;
	});
}

function stripQuotes(value: string): string {
	return value.replace(/^["']|["']$/g, "");
}
