/** Pure ID/time/path/sanitization helpers with no dependency on the store's
 * db connection or workspace state — usable from any store submodule. */

import { normalize, resolve } from "node:path";

export function generateId(): string {
	return crypto.randomUUID();
}

export function now(): string {
	return new Date().toISOString();
}

export function normalizeWorkspacePath(workspace: string): string {
	const value = workspace.trim();
	return normalize(resolve(value || process.cwd()));
}

const REDACTED = "[REDACTED]";
const MAX_RAW_STRING = 8_000;

export function sanitizeString(value: string): string {
	return value
		.replace(
			/-----BEGIN[^-]*PRIVATE KEY-----[\s\S]*?-----END[^-]*PRIVATE KEY-----/gi,
			REDACTED,
		)
		.replace(/\bBearer\s+[A-Za-z0-9._~+/-]{12,}/gi, `Bearer ${REDACTED}`)
		.replace(
			/\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,})\b/g,
			REDACTED,
		)
		.replace(
			/\b(api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|secret)\s*[:=]\s*([^\s,;]+)/gi,
			`$1=${REDACTED}`,
		)
		.slice(0, MAX_RAW_STRING);
}

export function sanitizePayload(value: unknown, depth = 0): unknown {
	if (depth > 8) return "[TRUNCATED]";
	if (typeof value === "string") return sanitizeString(value);
	if (typeof value !== "object" || value === null) return value;
	if (Array.isArray(value))
		return value.slice(0, 100).map(item => sanitizePayload(item, depth + 1));
	const result: Record<string, unknown> = {};
	for (const [key, item] of Object.entries(value).slice(0, 100)) {
		result[key] =
			/(?:authorization|cookie|api[_-]?key|token|secret|password|passwd|private[_-]?key)/i.test(
				key,
			)
				? REDACTED
				: sanitizePayload(item, depth + 1);
	}
	return result;
}

export function toFtsQuery(query: string): string {
	const terms = query.normalize("NFKC").match(/[\p{L}\p{N}_]+/gu) || [];
	return [...new Set(terms.slice(0, 12).map(term => term.toLowerCase()))]
		.map(term => `"${term.replace(/"/g, '""')}"${term.length > 1 ? "*" : ""}`)
		.join(" AND ");
}

export function toFtsAnyQuery(query: string): string {
	const stop = new Set([
		"a",
		"an",
		"and",
		"are",
		"as",
		"at",
		"be",
		"by",
		"for",
		"from",
		"in",
		"is",
		"it",
		"of",
		"on",
		"or",
		"the",
		"this",
		"to",
		"with",
	]);
	const terms = query.normalize("NFKC").match(/[\p{L}\p{N}_]+/gu) || [];
	return [
		...new Set(
			terms
				.map(term => term.toLowerCase())
				.filter(term => term.length > 1 && !stop.has(term))
				.slice(0, 12),
		),
	]
		.map(term => `"${term.replace(/"/g, '""')}"*`)
		.join(" OR ");
}
