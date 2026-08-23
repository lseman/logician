/** Caps the number of observations retained per session, deleting the
 * oldest excess once a session's workspace has drifted from the store's
 * current workspace scope. */

import type { Database } from "bun:sqlite";
import { normalizeWorkspacePath } from "./module-helpers.ts";
import { getSession } from "./sessions.ts";

export function slidingWindowCap(
	db: Database,
	getWorkspace: () => string,
	sessionId: string,
	cap: number = 200,
): number {
	const session = getSession(db, sessionId);
	if (!session || normalizeWorkspacePath(session.workspace) !== getWorkspace())
		return 0;
	const excess = db
		.prepare(
			"SELECT COUNT(*) as cnt FROM observations WHERE session_id = ? AND id NOT IN (SELECT id FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?)",
		)
		.get(sessionId, sessionId, cap) as { cnt: number };
	if (!excess || excess.cnt <= 0) return 0;

	db.prepare(
		`
      DELETE FROM observations WHERE session_id = ? AND id NOT IN (
        SELECT id FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
      )
    `,
	).run(sessionId, sessionId, cap);

	return excess.cnt;
}
