/** Prunes old, low-importance observations past a TTL. */

import type { Database } from "bun:sqlite";

interface AutoForgetResult {
	deleted: number;
	details: string[];
}

export function autoForget(
	db: Database,
	getWorkspace: () => string,
	ttlMs: number = 30 * 24 * 60 * 60 * 1000,
	minImportance: number = 3,
	maxDeletes: number = 100,
): AutoForgetResult {
	const cutoff = new Date(Date.now() - ttlMs).toISOString();
	const result: AutoForgetResult = { deleted: 0, details: [] };

	// Find old, low-importance observations
	const oldObs = db
		.prepare(
			"SELECT id, session_id, importance, timestamp FROM observations WHERE workspace = ? AND timestamp < ? AND importance < ? LIMIT ?",
		)
		.all(getWorkspace(), cutoff, minImportance, maxDeletes) as {
		id: string;
		session_id: string;
		importance: number;
		timestamp: string;
	}[];

	for (const obs of oldObs) {
		db.prepare("DELETE FROM observations WHERE id = ?").run(obs.id);
		result.deleted++;
		result.details.push(
			`Deleted obs ${obs.id.slice(0, 8)} from session ${obs.session_id.slice(0, 8)} (${obs.importance}/10)`,
		);
	}

	return result;
}
