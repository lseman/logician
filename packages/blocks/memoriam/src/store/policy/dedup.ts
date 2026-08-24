/** Hash-based tool-call deduplication within a rolling time window. */

import type { Database } from "bun:sqlite";
import { now } from "../module-helpers.js";

const DEDUP_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

function computeDedupHash(
	sessionId: string,
	toolName: string,
	toolInput: unknown,
): string {
	const inputStr =
		typeof toolInput === "string"
			? toolInput
			: JSON.stringify(toolInput ?? "").slice(0, 500);
	const raw = `${sessionId}:${toolName}:${inputStr}`;
	// Simple hash: use node:crypto if available, otherwise fallback
	try {
		const { createHash } =
			require("node:crypto") as typeof import("node:crypto");
		return createHash("sha256").update(raw).digest("hex").slice(0, 16);
	} catch {
		// Fallback: simple string hash
		let hash = 0;
		for (let i = 0; i < raw.length; i++) {
			const char = raw.charCodeAt(i);
			hash = (hash << 5) - hash + char;
			hash = hash & hash;
		}
		return Math.abs(hash).toString(36);
	}
}

export function dedupCheck(
	db: Database,
	sessionId: string,
	toolName: string,
	toolInput: unknown,
): boolean {
	const hash = computeDedupHash(sessionId, toolName, toolInput);
	const row = db
		.prepare("SELECT created_at FROM dedup WHERE hash = ?")
		.get(hash) as { created_at: string } | undefined;
	if (!row) return false;
	const age = Date.now() - new Date(row.created_at).getTime();
	return age < DEDUP_WINDOW_MS;
}

export function dedupRecord(
	db: Database,
	sessionId: string,
	toolName: string,
	toolInput: unknown,
): void {
	const hash = computeDedupHash(sessionId, toolName, toolInput);
	db.prepare(
		`
      INSERT INTO dedup (hash, created_at) VALUES (?, ?)
      ON CONFLICT(hash) DO UPDATE SET created_at = excluded.created_at
    `,
	).run(hash, now());
	// Clean up old entries
	db.prepare("DELETE FROM dedup WHERE created_at < ?").run(
		new Date(Date.now() - DEDUP_WINDOW_MS * 2).toISOString(),
	);
}
