// ── Tool call parsing ────────────────────────────────────────────────────────────
// Parses tool call arguments from LLM response text.

export function parseToolInput(raw: string): Record<string, unknown> {
	const text = raw?.trim() ?? "";
	if (!text) return {};

	const parsed = parseLooseJson(text);
	if (parsed.ok && parsed.value && typeof parsed.value === "object") {
		return Array.isArray(parsed.value)
			? { items: parsed.value }
			: (parsed.value as Record<string, unknown>);
	}

	// Line-by-line key: value parser
	const args: Record<string, unknown> = {};
	const lines = text.split("\n");
	for (const line of lines) {
		const match = line.match(/^\s*([\w.-]+)\s*:\s*(.+)\s*$/);
		const key = match?.[1];
		if (match && key) {
			args[key] = stripQuotes((match[2] ?? "").trim());
		}
	}

	// If nothing parsed and input is non-empty, treat as a bare command
	if (Object.keys(args).length === 0 && text.length > 0) {
		return { command: text };
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
	} catch (_e: unknown) {
		// Fall through to a conservative repair pass for common model slips.
	}
	try {
		const repaired = text
			.replace(/,\s*([}\]])/g, "$1")
			.replace(/([{,]\s*)([A-Za-z_][\w-]*)(\s*:)/g, '$1"$2"$3')
			.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, (_, body: string) =>
				JSON.stringify(body.replace(/\\'/g, "'")),
			);
		return { ok: true, value: JSON.parse(repaired) };
	} catch (_e: unknown) {
		return { ok: false };
	}
}

function stripQuotes(value: string): string {
	return value.replace(/^["']|["']$/g, "");
}
