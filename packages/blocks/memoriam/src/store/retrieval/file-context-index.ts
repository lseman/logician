/** Looks up observations mentioning a given file, for file-scoped context injection. */

import type { Database } from "bun:sqlite";
import type { FileContextEntry, ObservationType } from "../../types.ts";

export function getFileContext(
	db: Database,
	getWorkspace: () => string,
	file: string,
	sessionId?: string,
): FileContextEntry | null {
	const pattern = `%${file}%`;
	const rows = sessionId
		? (db
				.prepare(
					`
          SELECT id, session_id, type, title, narrative, importance, timestamp
          FROM observations
          WHERE workspace = ? AND (title LIKE ? OR narrative LIKE ? OR files LIKE ?) AND session_id = ?
          ORDER BY timestamp DESC
        `,
				)
				.all(getWorkspace(), pattern, pattern, pattern, sessionId) as any[])
		: (db
				.prepare(
					`
          SELECT id, session_id, type, title, narrative, importance, timestamp
          FROM observations
          WHERE workspace = ? AND (title LIKE ? OR narrative LIKE ? OR files LIKE ?)
          ORDER BY timestamp DESC
        `,
				)
				.all(getWorkspace(), pattern, pattern, pattern) as any[]);

	if (rows.length === 0) return null;

	return {
		file,
		observations: rows.map(r => ({
			sessionId: r.session_id,
			obsId: r.id,
			type: r.type as ObservationType,
			title: r.title || "",
			narrative: r.narrative || "",
			importance: r.importance ?? 5,
			timestamp: r.timestamp,
		})),
	};
}

export function getFilesContext(
	db: Database,
	getWorkspace: () => string,
	files: string[],
	sessionId?: string,
): FileContextEntry[] {
	return files
		.map(f => getFileContext(db, getWorkspace, f, sessionId))
		.filter((e): e is FileContextEntry => e !== null);
}

export function rebuildFileIndex(
	db: Database,
	getWorkspace: () => string,
): number {
	// Count observations with non-empty files array
	// Since JSON arrays like ["a"] are > 4 chars while [] is exactly 2 chars
	const count = db
		.prepare(
			`SELECT COUNT(*) as cnt FROM observations WHERE workspace = ? AND LENGTH(files) > 2`,
		)
		.get(getWorkspace()) as { cnt: number };
	return count.cnt || 0;
}
