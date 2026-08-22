/** CRUD over the memories table: create/get/list/update/delete plus a
 * recall() formatter for prompt injection. */

import type { Database } from "bun:sqlite";
import type {
	CreateMemoryOptions,
	Memory,
	MemoryQuery,
	MemoryType,
	RecallOptions,
	WorkingMemoryTier,
} from "../types.js";
import { safeParseJsonArray } from "./db-helpers.ts";
import { generateId, normalizeWorkspacePath, now, toFtsQuery } from "./module-helpers.ts";
import { assignStrength, extractConcepts, extractFiles } from "./text-helpers.ts";

export function rowToMemory(row: any): Memory {
	return {
		id: row.id,
		createdAt: row.created_at,
		updatedAt: row.updated_at,
		type: row.type as MemoryType,
		title: row.title || "",
		content: row.content,
		concepts: safeParseJsonArray(row.concepts),
		files: safeParseJsonArray(row.files),
		sessionIds: safeParseJsonArray(row.session_ids),
		strength: row.strength ?? 5,
		version: row.version ?? 1,
		parentId: row.parent_id,
		supersedes: safeParseJsonArray(row.supersedes),
		relatedIds: safeParseJsonArray(row.related_ids),
		sourceObservationIds: safeParseJsonArray(row.source_observation_ids),
		isLatest: row.is_latest === 1 || row.is_latest === true,
		project: row.project,
		workspace: row.workspace || "",
		accessCount: row.access_count ?? 0,
		lastAccessed: row.last_accessed || undefined,
		workingTier: (row.working_tier || "cold") as WorkingMemoryTier,
	};
}

export function create(
	db: Database,
	getWorkspace: () => string,
	content: string,
	options: CreateMemoryOptions = {},
): Memory {
	const id = options.id?.trim() || generateId();
	const ts = now();
	const memoryWorkspace = normalizeWorkspacePath(
		options.workspace || getWorkspace(),
	);
	const existingRow = db
		.prepare(
			"SELECT * FROM memories WHERE id = ? AND is_latest = 1 AND workspace = ?",
		)
		.get(id, memoryWorkspace) as any;
	if (existingRow) return rowToMemory(existingRow);

	// Auto-extract concepts from content
	const concepts = options.concepts || extractConcepts(content);
	const files = options.files || extractFiles(content);

	// Auto-assign strength
	const strength = options.strength ?? assignStrength(content);

	// Derive workspace from options or current workspace
	db.prepare(
		`
      INSERT INTO memories (id, created_at, updated_at, type, title, content,
                            concepts, files, session_ids, strength, version,
                            parent_id, related_ids, source_observation_ids, is_latest, project, workspace,
                            access_count, last_accessed, working_tier)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, '[]', '[]', 1, ?, ?, 0, NULL, 'cold')
    `,
	).run(
		id,
		ts,
		ts,
		options.type || "fact",
		content.slice(0, 200),
		content,
		JSON.stringify(concepts),
		JSON.stringify(files),
		JSON.stringify(options.sessionIds || []),
		strength,
		options.parentId || null,
		options.project || null,
		memoryWorkspace,
	);

	return {
		id,
		createdAt: ts,
		updatedAt: ts,
		type: options.type || "fact",
		title: content.slice(0, 200),
		content,
		concepts,
		files,
		sessionIds: options.sessionIds || [],
		strength,
		version: 1,
		parentId: options.parentId,
		relatedIds: [],
		sourceObservationIds: [],
		isLatest: true,
		project: options.project,
		workspace: memoryWorkspace,
		accessCount: 0,
		workingTier: "cold",
	};
}

export function get(
	db: Database,
	getWorkspace: () => string,
	id: string,
): Memory | null {
	const row = db
		.prepare(
			"SELECT * FROM memories WHERE id = ? AND is_latest = 1 AND workspace = ?",
		)
		.get(id, getWorkspace()) as any;
	if (!row) return null;
	return rowToMemory(row);
}

export function getAny(
	db: Database,
	getWorkspace: () => string,
	id: string,
): Memory | null {
	const row = db
		.prepare("SELECT * FROM memories WHERE id = ? AND workspace = ?")
		.get(id, getWorkspace()) as any;
	if (!row) return null;
	return rowToMemory(row);
}

export function list(
	db: Database,
	getWorkspace: () => string,
	query: MemoryQuery = {},
): Memory[] {
	const workspace = normalizeWorkspacePath(
		query.workspace || getWorkspace(),
	);
	const conditions: string[] = [];
	const params: any[] = [];
	const ftsQuery = query.search ? toFtsQuery(query.search) : "";
	const from = ftsQuery
		? "memories_fts JOIN memories m ON m.id = memories_fts.id"
		: "memories m";
	if (ftsQuery) {
		conditions.push("memories_fts MATCH ?");
		params.push(ftsQuery);
	}
	conditions.push("m.workspace = ?", "m.is_latest = 1");
	params.push(workspace);

	if (query.type) {
		conditions.push("m.type = ?");
		params.push(query.type);
	}

	if (query.project) {
		conditions.push("m.project = ?");
		params.push(query.project);
	}

	if (query.minStrength !== undefined) {
		conditions.push("m.strength >= ?");
		params.push(query.minStrength);
	}

	const baseWhere = conditions.join(" AND ");

	// Subquery for concept AND filtering: all concepts must match
	const conceptSub = query.concepts?.length
		? `(SELECT COUNT(DISTINCT je.value) FROM json_each(m.concepts) je WHERE je.value IN (${Array(query.concepts?.length).fill("?").join(", ")})) >= ${query.concepts?.length}`
		: "1=1";

	// Subquery for session matching
	const sessionSub = query.sessionId
		? `(SELECT COUNT(*) FROM json_each(m.session_ids) je WHERE je.value = ?) >= 1`
		: "1=1";

	// Add params for subqueries
	if (query.concepts?.length) {
		for (const c of query.concepts!) params.push(c);
	}
	if (query.sessionId) {
		params.push(query.sessionId);
	}

	const limit = query.limit ?? 10;

	const sql = `
      SELECT m.*${ftsQuery ? ", bm25(memories_fts, 0, 8, 4, 2, 2) AS lexical_rank" : ""}
      FROM ${from}
      WHERE ${baseWhere}
        AND ${conceptSub}
        AND ${sessionSub}
      ORDER BY ${ftsQuery ? "lexical_rank ASC," : ""} m.strength DESC, m.updated_at DESC
      LIMIT ?
    `;
	params.push(limit);

	return db
		.prepare(sql)
		.all(...params)
		.map(rowToMemory);
}

export function deleteEntry(
	db: Database,
	getWorkspace: () => string,
	id: string,
): boolean {
	// Only delete if still latest (prevent double-delete)
	const remove = db.transaction(() => {
		const workspace = getWorkspace();
		const result = db
			.prepare(
				"UPDATE memories SET is_latest = 0 WHERE id = ? AND workspace = ? AND is_latest = 1",
			)
			.run(id, workspace);
		if (result.changes > 0) {
			db.prepare(
				"DELETE FROM memory_embeddings WHERE entity_id = ? AND workspace = ?",
			).run(id, workspace);
		}
		return result;
	});
	const result = remove();
	return result.changes > 0;
}

export function clearMemories(db: Database, getWorkspace: () => string): number {
	const workspace = getWorkspace();
	const ids = db
		.prepare("SELECT id FROM memories WHERE workspace = ?")
		.all(workspace) as Array<{ id: string }>;
	if (!ids.length) return 0;
	const removeRelations = db.prepare(
		"DELETE FROM relations WHERE source_id = ? OR target_id = ?",
	);
	for (const { id } of ids) removeRelations.run(id, id);
	db.prepare(
		"DELETE FROM memory_embeddings WHERE workspace = ? AND entity_kind = 'memory'",
	).run(workspace);
	db.prepare("DELETE FROM memories WHERE workspace = ?").run(workspace);
	return ids.length;
}

export function update(
	db: Database,
	getWorkspace: () => string,
	id: string,
	updates: Partial<Pick<Memory, "content" | "concepts" | "strength" | "title">>,
): Memory | null {
	const sets: string[] = [];
	const params: any[] = [];

	if (updates.content !== undefined) {
		sets.push("content = ?");
		params.push(updates.content);
	}
	if (updates.title !== undefined) {
		sets.push("title = ?");
		params.push(updates.title);
	}
	if (updates.concepts !== undefined) {
		sets.push("concepts = ?");
		params.push(JSON.stringify(updates.concepts));
	}
	if (updates.strength !== undefined) {
		sets.push("strength = ?");
		params.push(updates.strength);
	}

	if (!sets.length) return get(db, getWorkspace, id);

	sets.push("updated_at = ?");
	params.push(now());
	const workspace = getWorkspace();
	params.push(id, workspace);

	const updateMemory = db.transaction(() => {
		const result = db
			.prepare(
				`UPDATE memories SET ${sets.join(", ")} WHERE id = ? AND workspace = ?`,
			)
			.run(...params);
		// Content-derived vectors cannot remain valid after semantic fields change.
		if (
			result.changes > 0 &&
			(updates.content !== undefined ||
				updates.title !== undefined ||
				updates.concepts !== undefined)
		) {
			db.prepare(
				"DELETE FROM memory_embeddings WHERE entity_id = ? AND workspace = ?",
			).run(id, workspace);
		}
		return result;
	});
	updateMemory();
	return get(db, getWorkspace, id);
}

export function recall(
	db: Database,
	getWorkspace: () => string,
	query: MemoryQuery,
	options: RecallOptions = {},
): string {
	const memories = list(db, getWorkspace, query);
	if (!memories.length) return "";

	const format = options.format || "text";
	const template = options.template || "{{title}}: {{content}}";

	if (format === "markdown") {
		return memories
			.map(
				m =>
					`## ${m.title} [${m.strength}/10]\n\n${m.content}\n\n${m.concepts.length ? ` Concepts: ${m.concepts.join(", ")}` : ""}`,
			)
			.join("\n\n---\n\n");
	}

	if (format === "system-prompt") {
		return memories
			.map(m => `## ${m.type} [${m.strength}/10]\n\n${m.content}`)
			.join("\n\n");
	}

	// text
	return memories
		.map(m =>
			template
				.replace("{{content}}", m.content)
				.replace("{{title}}", m.title)
				.replace("{{strength}}", String(m.strength)),
		)
		.join("\n\n");
}
