// ── Tool call parsing ────────────────────────────────────────────────────────────
// Parses tool calls from LLM response text.
// Mirrors Python parse_tool_calls with JSON/YAML support.

import type { ToolCall } from "./types.ts";

export function parseToolCalls(text: string): ToolCall[] {
    const calls: ToolCall[] = [];

    // Strategy 1: JSON objects { "name": "...", "arguments": "..." }
    const jsonPattern =
        /\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}/g;
    let match;
    let index = 0;
    while ((match = jsonPattern.exec(text)) !== null) {
        calls.push({
            id: `tool_${index++}`,
            name: match[1],
            arguments: match[2],
        });
    }

    if (calls.length > 0) {
        return calls;
    }

    // Strategy 2: YAML-style tool_call blocks
    const yamlPattern =
        /tool_call:\s*\n\s*name:\s*([^\n]+)\s*\n\s*arguments:\s*"((?:[^"\\]|\\.)*)"/g;
    while ((match = yamlPattern.exec(text)) !== null) {
        calls.push({
            id: `tool_${index++}`,
            name: match[1].trim(),
            arguments: match[2],
        });
    }

    return calls;
}
