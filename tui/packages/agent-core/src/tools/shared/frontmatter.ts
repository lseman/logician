// ── YAML frontmatter parsing ─────────────────────────────────────────────────
// Shared by skills and agent-definition loaders: splits a `---`-delimited
// YAML header from a markdown body, with a lenient fallback parser for
// frontmatter that isn't strictly valid YAML.

import { parse } from "yaml";

export function parseFrontmatter<T extends Record<string, unknown>>(
	content: string,
):
	| { ok: true; value: { frontmatter: T; body: string } }
	| { ok: false; error: Error } {
	try {
		const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
		if (!normalized.startsWith("---"))
			return {
				ok: true,
				value: { frontmatter: {} as T, body: normalized },
			};
		const endIndex = normalized.indexOf("\n---", 3);
		const yamlString =
			endIndex === -1 ? normalized.slice(4) : normalized.slice(4, endIndex);
		const body = endIndex === -1 ? "" : normalized.slice(endIndex + 4).trim();
		let frontmatter: T;
		try {
			frontmatter = (parse(yamlString) ?? {}) as T;
		} catch (e: unknown) {
			frontmatter = parseLenientFrontmatter(yamlString) as T;
		}
		return {
			ok: true,
			value: {
				frontmatter,
				body,
			},
		};
	} catch (error) {
		return { ok: false, error: error as Error };
	}
}

function parseLenientFrontmatter(source: string): Record<string, unknown> {
	const result: Record<string, unknown> = {};
	const lines = source.replace(/\t/g, "  ").split("\n");
	let currentKey: string | null = null;
	let currentList: string[] | null = null;

	for (const rawLine of lines) {
		if (!rawLine.trim() || rawLine.trim().startsWith("#")) continue;
		const topLevel = /^([A-Za-z0-9_-]+):(?:\s*(.*))?$/.exec(rawLine);
		if (topLevel && !rawLine.startsWith(" ")) {
			currentKey = topLevel[1];
			const rawValue = topLevel[2] ?? "";
			if (rawValue.trim()) {
				result[currentKey] = unquoteScalar(rawValue.trim());
				currentList = null;
			} else {
				currentList = [];
				result[currentKey] = currentList;
			}
			continue;
		}

		const listItem = /^\s*-\s*(.*)$/.exec(rawLine);
		if (listItem && currentKey) {
			if (!currentList) {
				currentList = [];
				result[currentKey] = currentList;
			}
			currentList.push(unquoteScalar(listItem[1].trim()));
			continue;
		}

		if (currentList?.length) {
			currentList[currentList.length - 1] += `\n${rawLine.trim()}`;
		}
	}

	for (const [key, value] of Object.entries(result)) {
		if (Array.isArray(value) && value.length === 0) result[key] = "";
	}
	return result;
}

function unquoteScalar(value: string): string {
	if (
		(value.startsWith('"') && value.endsWith('"')) ||
		(value.startsWith("'") && value.endsWith("'"))
	) {
		return value.slice(1, -1);
	}
	return value;
}
