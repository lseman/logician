const STRING_TERMINATORS = new Set(["\x07"]);

function consumeStringControl(value: string, start: number): number {
	let index = start;
	while (index < value.length) {
		if (STRING_TERMINATORS.has(value[index])) return index + 1;
		if (value[index] === "\x1b" && value[index + 1] === "\\") {
			return index + 2;
		}
		index++;
	}
	return value.length;
}

function consumeCsi(value: string, start: number): number {
	let index = start;
	while (index < value.length) {
		const code = value.charCodeAt(index);
		if (code >= 0x40 && code <= 0x7e) return index + 1;
		index++;
	}
	return value.length;
}

/** Remove terminal commands from untrusted text while preserving readable data. */
export function sanitizeTerminalText(value: string): string {
	let output = "";
	let index = 0;
	while (index < value.length) {
		const code = value.charCodeAt(index);
		if (code === 0x1b) {
			const kind = value[index + 1];
			if (kind === "[") {
				index = consumeCsi(value, index + 2);
			} else if (
				kind === "]" ||
				kind === "P" ||
				kind === "_" ||
				kind === "^" ||
				kind === "X"
			) {
				index = consumeStringControl(value, index + 2);
			} else {
				// Generic two-byte ESC command, including charset selection.
				index = Math.min(value.length, index + 2);
			}
			continue;
		}
		if (code === 0x9b) {
			index = consumeCsi(value, index + 1);
			continue;
		}
		if ([0x90, 0x9d, 0x9e, 0x9f].includes(code)) {
			index = consumeStringControl(value, index + 1);
			continue;
		}
		if (code === 0x0d) {
			if (value[index + 1] === "\n") index++;
			output += "\n";
			index++;
			continue;
		}
		if (
			(code < 0x20 && code !== 0x09 && code !== 0x0a) ||
			code === 0x7f ||
			(code >= 0x80 && code <= 0x9f)
		) {
			index++;
			continue;
		}
		const codePoint = value.codePointAt(index);
		if (codePoint === undefined) break;
		const char = String.fromCodePoint(codePoint);
		output += char;
		index += char.length;
	}
	return output;
}

/** Deeply sanitize JSON-like tool data without mutating transcript state. */
export function sanitizeTerminalValue<T>(
	value: T,
	seen = new WeakMap<object, unknown>(),
): T {
	if (typeof value === "string") return sanitizeTerminalText(value) as T;
	if (value === null || typeof value !== "object") return value;
	const existing = seen.get(value);
	if (existing !== undefined) return existing as T;
	if (Array.isArray(value)) {
		const result: unknown[] = [];
		seen.set(value, result);
		for (const item of value) result.push(sanitizeTerminalValue(item, seen));
		return result as T;
	}
	const result: Record<string, unknown> = {};
	seen.set(value, result);
	for (const [key, item] of Object.entries(value)) {
		result[key] = sanitizeTerminalValue(item, seen);
	}
	return result as T;
}
