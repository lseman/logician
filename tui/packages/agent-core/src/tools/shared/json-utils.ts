// ── JSON utilities ────────────────────────────────────────────────────────────

/**
 * Strip single-line (//) and multi-line (/* *\/) comments from a JSON string.
 * Handles comments inside strings correctly (does not strip them).
 * Ported from pi packages/coding-agent/src/utils/json.ts.
 */
export function stripJsonComments(input: string): string {
	let output = "";
	let i = 0;
	let inString = false;

	while (i < input.length) {
		const ch = input[i];

		if (inString) {
			if (ch === "\\" && i + 1 < input.length) {
				// Escaped character — pass both chars through unchanged.
				output += ch + input[i + 1];
				i += 2;
				continue;
			}
			if (ch === '"') inString = false;
			output += ch;
			i++;
			continue;
		}

		if (ch === '"') {
			inString = true;
			output += ch;
			i++;
			continue;
		}

		// Single-line comment
		if (ch === "/" && input[i + 1] === "/") {
			while (i < input.length && input[i] !== "\n") i++;
			continue;
		}

		// Multi-line comment
		if (ch === "/" && input[i + 1] === "*") {
			i += 2;
			while (i < input.length && !(input[i] === "*" && input[i + 1] === "/")) i++;
			i += 2;
			continue;
		}

		output += ch;
		i++;
	}

	return output;
}

/** Parse JSON with comment stripping. Throws SyntaxError on invalid JSON. */
export function parseJsonWithComments<T = unknown>(input: string): T {
	return JSON.parse(stripJsonComments(input)) as T;
}
