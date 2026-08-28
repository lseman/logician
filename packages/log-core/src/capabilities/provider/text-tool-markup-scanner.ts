export interface TextToolCandidate {
	index: number;
	name: string;
	arguments: string;
}

export interface SourceRange {
	start: number;
	end: number;
}

export interface TextToolMarkupScan {
	candidates: TextToolCandidate[];
	ranges: SourceRange[];
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
	for (const parameter of body.matchAll(parameterRegex)) {
		args[parameter[1]] = parseParameterValue(parameter[2]);
	}
	if (Object.keys(args).length > 0) return JSON.stringify(args);

	const trimmed = body.trim();
	if (!trimmed) return "{}";
	try {
		return JSON.stringify(JSON.parse(trimmed) as unknown);
	} catch {
		return trimmed;
	}
}

function matchingDelimiter(
	content: string,
	openingIndex: number,
	open: string,
	close: string,
): number | undefined {
	let depth = 0;
	let quote: string | undefined;
	let escaped = false;
	for (let index = openingIndex; index < content.length; index++) {
		const character = content[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (character === "\\") escaped = true;
			else if (character === quote) quote = undefined;
			continue;
		}
		if (character === '"' || character === "'") {
			quote = character;
			continue;
		}
		if (character === open) depth++;
		else if (character === close) {
			depth--;
			if (depth === 0) return index;
		}
	}
	return undefined;
}

function splitTopLevel(input: string): string[] {
	const parts: string[] = [];
	let start = 0;
	let quote: string | undefined;
	let escaped = false;
	const depths = { parentheses: 0, brackets: 0, braces: 0 };
	for (let index = 0; index < input.length; index++) {
		const character = input[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (character === "\\") escaped = true;
			else if (character === quote) quote = undefined;
			continue;
		}
		if (character === '"' || character === "'") {
			quote = character;
			continue;
		}
		if (character === "(") depths.parentheses++;
		else if (character === ")") depths.parentheses--;
		else if (character === "[") depths.brackets++;
		else if (character === "]") depths.brackets--;
		else if (character === "{") depths.braces++;
		else if (character === "}") depths.braces--;
		else if (
			character === "," &&
			depths.parentheses === 0 &&
			depths.brackets === 0 &&
			depths.braces === 0
		) {
			parts.push(input.slice(start, index).trim());
			start = index + 1;
		}
	}
	parts.push(input.slice(start).trim());
	return parts.filter(Boolean);
}

function topLevelEquals(input: string): number | undefined {
	let quote: string | undefined;
	let escaped = false;
	let depth = 0;
	for (let index = 0; index < input.length; index++) {
		const character = input[index];
		if (quote) {
			if (escaped) escaped = false;
			else if (character === "\\") escaped = true;
			else if (character === quote) quote = undefined;
			continue;
		}
		if (character === '"' || character === "'") quote = character;
		else if (character === "(" || character === "[" || character === "{")
			depth++;
		else if (character === ")" || character === "]" || character === "}")
			depth--;
		else if (character === "=" && depth === 0) return index;
	}
	return undefined;
}

function parseKeyValueArguments(input: string): string | undefined {
	const args: Record<string, unknown> = {};
	for (const part of splitTopLevel(input)) {
		const equals = topLevelEquals(part);
		if (equals === undefined) continue;
		const key = part.slice(0, equals).trim();
		if (!/^[\w.-]+$/.test(key)) continue;
		let rawValue = part.slice(equals + 1).trim();
		if (
			rawValue.length >= 2 &&
			((rawValue.startsWith('"') && rawValue.endsWith('"')) ||
				(rawValue.startsWith("'") && rawValue.endsWith("'")))
		) {
			rawValue = rawValue.slice(1, -1);
		}
		args[key] = parseParameterValue(rawValue);
	}
	return Object.keys(args).length > 0 ? JSON.stringify(args) : undefined;
}

function inRanges(index: number, ranges: SourceRange[]): boolean {
	return ranges.some(range => index >= range.start && index < range.end);
}

function serializedArguments(value: unknown): string | undefined {
	if (typeof value === "string") return value;
	const serialized = JSON.stringify(value);
	return typeof serialized === "string" ? serialized : undefined;
}

function scanJsonToolCalls(content: string): TextToolMarkupScan {
	const candidates: TextToolCandidate[] = [];
	const ranges: SourceRange[] = [];
	let cursor = 0;
	while (cursor < content.length) {
		const start = content.indexOf("[", cursor);
		if (start < 0) break;
		const end = matchingDelimiter(content, start, "[", "]");
		cursor = start + 1;
		if (end === undefined) continue;
		try {
			const value = JSON.parse(content.slice(start, end + 1)) as unknown;
			if (!Array.isArray(value)) continue;
			let matched = false;
			for (const item of value) {
				if (!item || typeof item !== "object") continue;
				const record = item as Record<string, unknown>;
				if (typeof record.name !== "string" || !("arguments" in record))
					continue;
				const argumentsValue = serializedArguments(record.arguments);
				if (argumentsValue === undefined) continue;
				candidates.push({
					index: start,
					name: record.name,
					arguments: argumentsValue,
				});
				matched = true;
			}
			if (matched) ranges.push({ start, end: end + 1 });
		} catch {
			// Non-JSON brackets are ordinary prose.
		}
	}
	return { candidates, ranges };
}

/** Discover supported textual tool-call forms and the source ranges they own. */
export function scanTextToolMarkup(
	content: string,
	isKnownTool?: (name: string) => boolean,
): TextToolMarkupScan {
	const candidates: TextToolCandidate[] = [];
	const ranges: SourceRange[] = [];

	const xmlRegex =
		/<function\s*=?\s*["']?([a-zA-Z_][\w.-]*)["']?\s*>([\s\S]*?)<\/function>/gi;
	for (const match of content.matchAll(xmlRegex)) {
		if (!match[1] || match.index === undefined) continue;
		candidates.push({
			index: match.index,
			name: match[1],
			arguments: parseFunctionArguments(match[2]),
		});
		ranges.push({ start: match.index, end: match.index + match[0].length });
	}

	const reactPrefix = "[[tool_call(";
	let reactCursor = 0;
	while (reactCursor < content.length) {
		const start = content.indexOf(reactPrefix, reactCursor);
		if (start < 0) break;
		const openingParen = start + reactPrefix.length - 1;
		const closingParen = matchingDelimiter(content, openingParen, "(", ")");
		reactCursor = start + reactPrefix.length;
		if (
			closingParen === undefined ||
			content.slice(closingParen + 1, closingParen + 3) !== "]]"
		) {
			continue;
		}
		const parts = splitTopLevel(content.slice(openingParen + 1, closingParen));
		const first = parts.shift();
		if (!first) continue;
		const firstEquals = topLevelEquals(first);
		const name = (
			firstEquals !== undefined &&
			["id", "name", "tool"].includes(
				first.slice(0, firstEquals).trim().toLowerCase(),
			)
				? first.slice(firstEquals + 1)
				: first
		)
			.replace(/^['"]|['"]$/g, "")
			.trim();
		if (!name) continue;
		candidates.push({
			index: start,
			name,
			arguments: parseKeyValueArguments(parts.join(",")) ?? "{}",
		});
		ranges.push({ start, end: closingParen + 3 });
	}

	const json = scanJsonToolCalls(content);
	candidates.push(...json.candidates);
	ranges.push(...json.ranges);

	const functionRegex = /([a-zA-Z_][\w.-]*)\s*\(/g;
	for (const match of content.matchAll(functionRegex)) {
		if (match.index === undefined || inRanges(match.index, ranges)) continue;
		const name = match[1];
		if (isKnownTool && !isKnownTool(name)) continue;
		const openingParen = match.index + match[0].lastIndexOf("(");
		const closingParen = matchingDelimiter(content, openingParen, "(", ")");
		if (closingParen === undefined) continue;
		const argumentsValue = parseKeyValueArguments(
			content.slice(openingParen + 1, closingParen),
		);
		if (!argumentsValue) continue;
		candidates.push({ index: match.index, name, arguments: argumentsValue });
		ranges.push({ start: match.index, end: closingParen + 1 });
	}

	return { candidates, ranges };
}

export function removeTextToolMarkupRanges(
	content: string,
	ranges: SourceRange[],
): string {
	let result = content;
	for (const range of [...ranges].sort(
		(left, right) => right.start - left.start,
	)) {
		result = `${result.slice(0, range.start)}${result.slice(range.end)}`;
	}
	return result;
}
