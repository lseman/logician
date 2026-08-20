// ── Frontmatter parsing ──────────────────────────────────────────────────
// Shared `---`-delimited YAML frontmatter parser used by both skills.ts and
// prompt-templates.ts (pi keeps a byte-identical copy in each file; this port
// hoists it into one shared module instead).

import { parse } from "yaml";
import { err, ok, type Result, toError } from "../core/result.ts";

export function parseFrontmatter<T extends Record<string, unknown>>(
	content: string,
): Result<{ frontmatter: T; body: string }, Error> {
	try {
		const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
		if (!normalized.startsWith("---"))
			return ok({ frontmatter: {} as T, body: normalized });
		const endIndex = normalized.indexOf("\n---", 3);
		if (endIndex === -1) return ok({ frontmatter: {} as T, body: normalized });
		const yamlString = normalized.slice(4, endIndex);
		const body = normalized.slice(endIndex + 4).trim();
		return ok({ frontmatter: (parse(yamlString) ?? {}) as T, body });
	} catch (error) {
		return err(toError(error));
	}
}
