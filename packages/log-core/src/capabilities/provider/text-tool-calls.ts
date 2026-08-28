// ── Text-to-Tool-Call Parser ──────────────────────────────────────────────
// Promotes textual provider markup to structured tool calls while keeping
// syntax discovery isolated behind one internal scanner interface.

import { randomUUID } from "node:crypto";
import type { ToolCall } from "../../system/types/types-messages.ts";
import {
	removeTextToolMarkupRanges,
	scanTextToolMarkup,
} from "./text-tool-markup-scanner.ts";

function normalizeTextToolMarkup(content: string): string {
	return content.replace(
		/\*\*(\s*<\/?(?:tool\\?_call|function|parameter)\b[^>]*>)\*\*/gi,
		"$1",
	);
}

/**
 * Parse tool calls from plain text content.
 * Handles XML, ReAct, JSON-array, and function-call forms.
 */
export function parseTextToolCalls(
	content: string,
	/** When provided, discard candidates that are not registered tools. */
	isKnownTool?: (name: string) => boolean,
): ToolCall[] {
	if (!content || typeof content !== "string") return [];
	const normalized = normalizeTextToolMarkup(content);
	const { candidates } = scanTextToolMarkup(normalized, isKnownTool);
	const seen = new Set<string>();
	const calls: ToolCall[] = [];

	for (const candidate of candidates.sort(
		(left, right) => left.index - right.index,
	)) {
		if (isKnownTool && !isKnownTool(candidate.name)) continue;
		const key = `${candidate.name}:${candidate.arguments}`;
		if (seen.has(key)) continue;
		seen.add(key);
		calls.push({
			id: `tc_${randomUUID()}`,
			name: candidate.name,
			arguments: candidate.arguments,
		});
	}
	return calls;
}

/** Remove textual tool markup after it has been promoted to structured calls. */
export function stripTextToolCalls(content: string): string {
	if (!content) return content;
	const normalized = normalizeTextToolMarkup(content);

	// Outer wrappers can contain malformed extra closing markers, so remove them
	// before scanning standalone calls that models emit without the wrapper.
	let stripped = normalized.replace(
		/<tool\\?_call\s*>[\s\S]*?<\/tool\\?_call\s*>/gi,
		"",
	);
	stripped = removeTextToolMarkupRanges(
		stripped,
		scanTextToolMarkup(stripped).ranges,
	);

	// Remove standalone garbled markers left by malformed provider markup.
	stripped = stripped.replace(/^\s*(<\/?[a-zA-Z_]\w*>|art|\u0147)\s*$/gm, "");
	return stripped.trim();
}
