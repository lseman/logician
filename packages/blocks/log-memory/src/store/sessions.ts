/** CRUD over the sessions table: one row per conversation/run, tracking
 * lifecycle, tags, and observation count. */

import type { Database } from "bun:sqlite";
import type { Session } from "../types.js";
import { safeParseJsonArray } from "./db-helpers.ts";
import { normalizeWorkspacePath, now } from "./module-helpers.ts";

export function createSession(
	db: Database,
	getWorkspace: () => string,
	id: string,
	data: Partial<Session>,
): Session {
	const ts = now();
	const sessionCwd = data.cwd
		? normalizeWorkspacePath(data.cwd)
		: getWorkspace();
	const sessionWorkspace = normalizeWorkspacePath(data.workspace || sessionCwd);
	db.prepare(
		`
      INSERT OR IGNORE INTO sessions (id, name, project, cwd, workspace, started_at, status, observation_count,
                                      model, tags, first_prompt, summary, commit_shas)
      VALUES (?, ?, COALESCE(?, ''), COALESCE(?, ''), COALESCE(?, ''), ?, 'active', 0, ?, ?, ?, ?, ?)
    `,
	).run(
		id,
		data.name || null,
		data.project || "",
		sessionCwd,
		sessionWorkspace,
		ts,
		data.model || null,
		JSON.stringify(data.tags || []),
		data.firstPrompt || null,
		data.summary || null,
		JSON.stringify(data.commitShas || []),
	);
	return {
		id,
		name: data.name,
		project: data.project || "",
		cwd: sessionCwd,
		workspace: sessionWorkspace,
		startedAt: ts,
		status: "active",
		observationCount: 0,
		model: data.model,
		tags: data.tags || [],
		firstPrompt: data.firstPrompt,
		summary: data.summary,
		commitShas: data.commitShas || [],
	};
}

export function getSession(db: Database, id: string): Session | null {
	const row = db.prepare(`SELECT * FROM sessions WHERE id = ?`).get(id) as any;
	if (!row) return null;
	return {
		id: row.id,
		name: row.name || undefined,
		project: row.project || "",
		cwd: row.cwd || "",
		workspace: row.workspace || "",
		startedAt: row.started_at,
		endedAt: row.ended_at,
		status: row.status || "active",
		observationCount: row.observation_count || 0,
		model: row.model,
		tags: safeParseJsonArray(row.tags),
		firstPrompt: row.first_prompt,
		summary: row.summary,
		commitShas: safeParseJsonArray(row.commit_shas),
	};
}

export function listSessions(
	db: Database,
	getWorkspace: () => string,
	query?: {
		status?: string;
		project?: string;
		workspace?: string;
	},
): Session[] {
	const conditions: string[] = [];
	const params: any[] = [];

	if (query?.status) {
		conditions.push("status = ?");
		params.push(query.status);
	}
	if (query?.project) {
		conditions.push("project = ?");
		params.push(query.project);
	}
	if (query?.workspace) {
		conditions.push("workspace = ?");
		params.push(normalizeWorkspacePath(query.workspace));
	} else {
		conditions.push("workspace = ?");
		params.push(getWorkspace());
	}

	const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
	const rows = db
		.prepare(`SELECT * FROM sessions ${where} ORDER BY started_at DESC`)
		.all(...params) as any[];
	return rows.map(r => ({
		id: r.id,
		name: r.name || undefined,
		project: r.project || "",
		cwd: r.cwd || "",
		workspace: r.workspace || "",
		startedAt: r.started_at,
		endedAt: r.ended_at,
		status: r.status || "active",
		observationCount: r.observation_count || 0,
		model: r.model,
		tags: safeParseJsonArray(r.tags),
		firstPrompt: r.first_prompt,
		summary: r.summary,
		commitShas: safeParseJsonArray(r.commit_shas),
	}));
}

export function updateSession(
	db: Database,
	id: string,
	updates: Partial<Session>,
): Session | null {
	const sets: string[] = [];
	const params: any[] = [];

	if (updates.name !== undefined) {
		sets.push("name = ?");
		params.push(updates.name || null);
	}
	if (updates.project !== undefined) {
		sets.push("project = ?");
		params.push(updates.project);
	}
	if (updates.cwd !== undefined) {
		sets.push("cwd = ?");
		params.push(updates.cwd);
	}
	if (updates.status !== undefined) {
		sets.push("status = ?");
		params.push(updates.status);
	}
	if (updates.endedAt !== undefined) {
		sets.push("ended_at = ?");
		params.push(updates.endedAt);
	}
	if (updates.observationCount !== undefined) {
		sets.push("observation_count = ?");
		params.push(updates.observationCount);
	}
	if (updates.model !== undefined) {
		sets.push("model = ?");
		params.push(updates.model);
	}
	if (updates.tags !== undefined) {
		sets.push("tags = ?");
		params.push(JSON.stringify(updates.tags));
	}
	if (updates.firstPrompt !== undefined) {
		sets.push("first_prompt = ?");
		params.push(updates.firstPrompt);
	}
	if (updates.summary !== undefined) {
		sets.push("summary = ?");
		params.push(updates.summary);
	}

	if (!sets.length) return getSession(db, id);

	params.push(id);
	db.prepare(`UPDATE sessions SET ${sets.join(", ")} WHERE id = ?`).run(
		...params,
	);
	return getSession(db, id);
}

export function clearSessions(
	db: Database,
	getWorkspace: () => string,
	keepSessionId?: string,
): {
	sessions: number;
	observations: number;
} {
	// Completely unscoped rows predate workspace support and can never be
	// shown or attributed safely. Treat them as legacy garbage when the user
	// explicitly asks to clean sessions.
	const rows = db
		.prepare(
			`
      SELECT id FROM sessions
      WHERE (workspace = ? OR (workspace = '' AND cwd = ''))
        AND (? IS NULL OR id != ?)
    `,
		)
		.all(
			getWorkspace(),
			keepSessionId || null,
			keepSessionId || null,
		) as Array<{ id: string }>;
	if (!rows.length) return { sessions: 0, observations: 0 };
	const sessionIds = new Set(rows.map(session => session.id));
	let observations = 0;
	const countObservations = db.prepare(
		"SELECT COUNT(*) AS count FROM observations WHERE session_id = ?",
	);
	const deleteObservations = db.prepare(
		"DELETE FROM observations WHERE session_id = ?",
	);
	const deleteSession = db.prepare("DELETE FROM sessions WHERE id = ?");
	for (const session of rows) {
		observations += (countObservations.get(session.id) as { count: number })
			.count;
		deleteObservations.run(session.id);
		deleteSession.run(session.id);
	}
	const memories = db
		.prepare(
			"SELECT id, session_ids FROM memories WHERE json_valid(session_ids)",
		)
		.all() as Array<{ id: string; session_ids: string }>;
	const updateSources = db.prepare(
		"UPDATE memories SET session_ids = ? WHERE id = ?",
	);
	for (const memory of memories) {
		const retained = safeParseJsonArray(memory.session_ids).filter(
			id => !sessionIds.has(id),
		);
		updateSources.run(JSON.stringify(retained), memory.id);
	}
	return { sessions: rows.length, observations };
}

export function discardEmptySession(db: Database, id: string): boolean {
	const session = db
		.prepare(
			`
      SELECT id FROM sessions
      WHERE id = ? AND observation_count = 0
        AND NOT EXISTS (SELECT 1 FROM observations WHERE session_id = sessions.id)
        AND NOT EXISTS (
          SELECT 1 FROM memories m, json_each(m.session_ids) je
          WHERE json_valid(m.session_ids) AND je.value = sessions.id
        )
    `,
		)
		.get(id) as { id: string } | undefined;
	if (!session) return false;
	return db.prepare("DELETE FROM sessions WHERE id = ?").run(id).changes > 0;
}
