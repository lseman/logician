/** Tracks per-memory access count/recency for working-memory tiering. */

import type { Database } from "bun:sqlite";
import { now } from "./module-helpers.js";

export function trackAccess(
	db: Database,
	getWorkspace: () => string,
	entityId: string,
): void {
	db.prepare(
		`
	      UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ? AND workspace = ?
	    `,
	).run(now(), entityId, getWorkspace());
}

export function getAccessStats(
	db: Database,
	getWorkspace: () => string,
	entityId: string,
): { lastAccessed: string; accessCount: number } | null {
	const row = db
		.prepare(
			"SELECT last_accessed, access_count FROM memories WHERE id = ? AND workspace = ?",
		)
		.get(entityId, getWorkspace()) as
		| { last_accessed: string; access_count: number }
		| undefined;
	if (!row) return null;
	return {
		lastAccessed: row.last_accessed || "",
		accessCount: row.access_count || 0,
	};
}
