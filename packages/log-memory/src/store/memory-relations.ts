/** Typed relations between memories (supersedes/contradicts/related_to/
 * supports/extends), contradiction detection, and memory evolution
 * (a superseding revision that keeps the old row for history). */

import type { Database } from "bun:sqlite";
import type { Memory, MemoryRelation, MemoryRelationType } from "../types.js";
import { generateId, now } from "./module-helpers.ts";
import { get, rowToMemory } from "./memories.ts";

export function relate(
	db: Database,
	getWorkspace: () => string,
	sourceId: string,
	targetId: string,
	type: MemoryRelationType,
	confidence: number = 0.5,
): MemoryRelation | null {
	// Validate both memories exist
	const source = get(db, getWorkspace, sourceId);
	const target = get(db, getWorkspace, targetId);
	if (!source || !target) return null;

	const relationId = generateId();
	const ts = now();
	const clampedConf = Math.max(
		0,
		Math.min(
			1,
			confidence || computeRelationConfidence(source, target, type),
		),
	);

	db.prepare(
		`
      INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `,
	).run(relationId, type, sourceId, targetId, clampedConf, ts);

	// Update related_ids on both memories
	db.prepare(
		`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id IN (?, ?)`,
	).run(targetId, sourceId, sourceId);
	db.prepare(
		`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id IN (?, ?)`,
	).run(sourceId, targetId, targetId);

	return {
		id: relationId,
		type,
		sourceId,
		targetId,
		confidence: clampedConf,
		createdAt: ts,
	};
}

function computeRelationConfidence(
	source: Memory,
	target: Memory,
	relationType: MemoryRelationType,
): number {
	let score = 0.5;

	// Shared sessions boost confidence
	const sharedSessions = source.sessionIds.filter(sid =>
		target.sessionIds.includes(sid),
	);
	score += Math.min(sharedSessions.length * 0.1, 0.3);

	// Recency boost
	const now = Date.now();
	const sourceAge = now - new Date(source.updatedAt).getTime();
	const targetAge = now - new Date(target.updatedAt).getTime();
	const sevenDays = 7 * 24 * 60 * 60 * 1000;
	const ninetyDays = 90 * 24 * 60 * 60 * 1000;

	if (sourceAge < sevenDays && targetAge < sevenDays) score += 0.1;
	else if (sourceAge > ninetyDays && targetAge > ninetyDays) score -= 0.1;

	// Relation-type adjustments
	if (relationType === "supersedes") score += 0.1;
	if (relationType === "contradicts") score -= 0.05;

	return Math.max(0, Math.min(1, score));
}

// Negation/polarity terms used by detectContradictions below. A candidate
// pair is flagged only when they share a subject (file or concept) AND
// disagree in polarity — this is a cheap synchronous heuristic (no LLM
// round-trip), so it stays conservative: same-title collisions already go
// through consolidate()'s supersession path, this only catches
// cross-title memories about the same file/concept that assert opposite
// things (e.g. "auth uses JWT" vs "auth does not use JWT").
const NEGATION_TERMS =
	/\b(not|never|no longer|isn't|doesn't|don't|can't|cannot|fails?|failing|failed|broken|deprecated|removed|disabled|incorrect|wrong|instead of)\b/i;

function hasNegation(text: string): boolean {
	return NEGATION_TERMS.test(text);
}

/**
 * Find existing memories that share a file/concept subject with the given
 * candidate but disagree in polarity, and record a `contradicts` relation
 * for each. Returns the memory IDs flagged. Synchronous, transaction-safe
 * — intended to run inside consolidate()'s transaction right after a new
 * (non-superseding) memory is written.
 */
export function detectContradictions(
	db: Database,
	getWorkspace: () => string,
	candidate: Memory,
): string[] {
	const subjects = [...candidate.concepts, ...candidate.files];
	if (!subjects.length) return [];
	const candidatePolarity = hasNegation(candidate.content);
	const workspace = getWorkspace();

	const rows = db
		.prepare(
			`SELECT * FROM memories WHERE workspace = ? AND is_latest = 1 AND id != ? LIMIT 200`,
		)
		.all(workspace, candidate.id) as any[];

	const flagged: string[] = [];
	for (const row of rows) {
		const other = rowToMemory(row);
		const sharesSubject =
			other.concepts.some(c => subjects.includes(c)) ||
			other.files.some(f => subjects.includes(f));
		if (!sharesSubject) continue;
		if (hasNegation(other.content) === candidatePolarity) continue;

		relate(db, getWorkspace, candidate.id, other.id, "contradicts");
		flagged.push(other.id);
	}
	return flagged;
}

export function getRelations(
	db: Database,
	getWorkspace: () => string,
	memoryId: string,
): MemoryRelation[] {
	const workspace = getWorkspace();
	const rows = db
		.prepare(
			`SELECT r.* FROM relations r
			 JOIN memories source ON source.id = r.source_id
			 JOIN memories target ON target.id = r.target_id
			 WHERE (r.source_id = ? OR r.target_id = ?)
			   AND source.workspace = ? AND target.workspace = ?
			 ORDER BY r.created_at DESC`,
		)
		.all(memoryId, memoryId, workspace, workspace) as any[];

	return rows.map(r => ({
		id: r.id,
		type: r.type as MemoryRelationType,
		sourceId: r.source_id,
		targetId: r.target_id,
		confidence: r.confidence ?? 0.5,
		createdAt: r.created_at,
	}));
}

export function getRelatedMemories(
	db: Database,
	getWorkspace: () => string,
	memoryId: string,
	maxHops: number = 2,
	minConfidence: number = 0,
): Array<{ memory: Memory; hop: number; confidence: number }> {
	if (!get(db, getWorkspace, memoryId)) return [];
	const workspace = getWorkspace();
	const allRelations = db
		.prepare(
			`SELECT r.* FROM relations r
		  JOIN memories source ON source.id = r.source_id
		  JOIN memories target ON target.id = r.target_id
		  WHERE source.workspace = ? AND target.workspace = ?`,
		)
		.all(workspace, workspace) as any[];

	const visited = new Set<string>([memoryId]);
	const result: Array<{ memory: Memory; hop: number; confidence: number }> =
		[];
	const queue: Array<{ id: string; hop: number }> = [
		{ id: memoryId, hop: 0 },
	];
	const MAX_VISITED = 500;

	while (queue.length > 0 && visited.size < MAX_VISITED) {
		const current = queue.shift()!;
		if (current.hop >= maxHops) continue;
		visited.add(current.id);

		const memory = get(db, getWorkspace, current.id);
		if (!memory) continue;

		// Find relations involving this memory
		const relatedRelations = allRelations.filter(
			r => r.source_id === current.id || r.target_id === current.id,
		);

		// Get the target memory IDs from these relations
		for (const rel of relatedRelations) {
			const targetId =
				rel.source_id === current.id ? rel.target_id : rel.source_id;
			if (visited.has(targetId)) continue;

			const targetMemory = get(db, getWorkspace, targetId);
			if (!targetMemory) continue;

			visited.add(targetId);
			const confidence = rel.confidence ?? 0.5;

			if (current.hop >= 0 && confidence >= minConfidence) {
				result.push({
					memory: targetMemory,
					hop: current.hop + 1,
					confidence,
				});
			}

			queue.push({ id: targetId, hop: current.hop + 1 });
		}
	}

	result.sort((a, b) => b.confidence - a.confidence);
	return result;
}

export function evolve(
	db: Database,
	getWorkspace: () => string,
	memoryId: string,
	newContent: string,
	newTitle?: string,
): { memory: Memory; previousId: string } | null {
	// First get the existing memory (must be latest)
	const existing = get(db, getWorkspace, memoryId);
	if (!existing) return null;

	const ts = now();
	const evolved: Memory = {
		...existing,
		id: generateId(),
		createdAt: ts,
		updatedAt: ts,
		title: newTitle || existing.title,
		content: newContent,
		version: (existing.version || 1) + 1,
		parentId: existing.id,
		supersedes: [existing.id, ...(existing.supersedes || [])],
		isLatest: true,
	};

	// Retire the previous latest row and insert its replacement atomically.
	// The partial unique index permits only one latest row per title, so the
	// old row must be retired before inserting a same-title revision.
	const applyEvolution = db.transaction(() => {
		db.prepare("UPDATE memories SET is_latest = 0 WHERE id = ?").run(
			memoryId,
		);
		db.prepare(
			`
      INSERT INTO memories (id, created_at, updated_at, type, title, content,
                            concepts, files, session_ids, strength, version,
                            parent_id, related_ids, source_observation_ids, is_latest, project,
                            workspace, access_count, last_accessed, working_tier, supersedes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', '[]', 1, ?, ?, 0, NULL, 'cold', ?)
    `,
		).run(
			evolved.id,
			ts,
			ts,
			existing.type,
			evolved.title,
			newContent,
			JSON.stringify(existing.concepts),
			JSON.stringify(existing.files),
			JSON.stringify(existing.sessionIds),
			existing.strength,
			evolved.version,
			memoryId,
			existing.project || null,
			existing.workspace || getWorkspace(),
			JSON.stringify([existing.id]),
		);

		const relationId = generateId();
		db.prepare(
			`
      INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
      VALUES (?, 'supersedes', ?, ?, 1.0, ?)
    `,
		).run(relationId, evolved.id, memoryId, ts);

		db.prepare(
			`UPDATE memories SET related_ids = json_insert(related_ids, '$', ?) WHERE id = ?`,
		).run(memoryId, evolved.id);
	});
	applyEvolution();

	return { memory: evolved, previousId: memoryId };
}

export function removeRelation(
	db: Database,
	getWorkspace: () => string,
	relationId: string,
): boolean {
	const workspace = getWorkspace();
	const result = db
		.prepare(
			`DELETE FROM relations WHERE id = ? AND EXISTS (
		  SELECT 1 FROM memories source JOIN memories target
		  WHERE source.id = relations.source_id AND target.id = relations.target_id
		    AND source.workspace = ? AND target.workspace = ?
		)`,
		)
		.run(relationId, workspace, workspace);
	return result.changes > 0;
}
