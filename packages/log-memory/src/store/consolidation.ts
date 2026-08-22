/** Merges a session's pending high-importance observations into durable
 * memories, grouped by topic (file/concept/type), superseding an existing
 * same-title memory or creating a fresh one. */

import type { Database } from "bun:sqlite";
import type { Memory, MemoryType, ObservationType } from "../types.js";
import {
	parseObservationClaims,
	parseObservationProvenance,
	safeParseJsonArray,
} from "./db-helpers.ts";
import { detectContradictions } from "./memory-relations.ts";
import { rowToMemory } from "./memories.ts";
import { generateId, now } from "./module-helpers.ts";

export function consolidate(
	db: Database,
	getWorkspace: () => string,
	sessionId: string,
): Memory[] {
	const workspace = getWorkspace();
	const pendingRows = db
		.prepare(
			`SELECT * FROM observations
       WHERE session_id = ? AND workspace = ? AND consolidated = 0
         AND importance >= 5
       ORDER BY timestamp ASC LIMIT 100`,
		)
		.all(sessionId, workspace) as any[];

	// Semantic episodes are complete grounded units. When present, consolidate
	// those rather than mechanically merging their underlying tool telemetry.
	const semanticRows = pendingRows.filter(row => row.hook_type === "stop");
	const rows = semanticRows.length ? semanticRows : pendingRows;

	if (rows.length < 1 || (!semanticRows.length && rows.length < 2)) return [];

	const observations = rows.map(r => ({
		id: r.id,
		sessionId: r.session_id,
		timestamp: r.timestamp,
		type: r.type as ObservationType,
		title: r.title || "",
		subtitle: r.subtitle,
		facts: safeParseJsonArray(r.facts),
		narrative: r.narrative || "",
		concepts: safeParseJsonArray(r.concepts),
		files: safeParseJsonArray(r.files),
		importance: r.importance ?? 5,
		consolidated: r.consolidated === 1,
		workspace: r.workspace || workspace,
		claims: parseObservationClaims(r.claims),
		provenance: parseObservationProvenance(r.provenance),
	}));

	// Group by the most concrete topic available. A file or concept is much
	// more useful than broad buckets such as "command_run".
	const groups: Record<string, typeof observations> = {};
	for (const obs of observations) {
		const key = obs.files[0]
			? `file:${obs.files[0]}`
			: obs.concepts[0]
				? `concept:${obs.concepts[0].toLowerCase()}`
				: `type:${obs.type}`;
		if (!groups[key]) groups[key] = [];
		groups[key].push(obs);
	}

	const memories: Memory[] = [];
	const usedObservationIds: string[] = [];

	// A single transaction for the whole consolidation pass: it makes each
	// group's read-existing/supersede/insert sequence atomic with respect to
	// other writers on this connection (the extraction worker also calls
	// consolidate() after every job), and turns what was one commit per
	// group/observation into a single commit for the entire pass.
	const applyConsolidation = db.transaction(() => {
		for (const [topic, group] of Object.entries(groups)) {
			if (group.length < 2 && !semanticRows.length) continue;
			const dominantType = group
				.map(item => item.type)
				.sort(
					(a, b) =>
						group.filter(item => item.type === b).length -
						group.filter(item => item.type === a).length,
				)[0];
			const allFacts = [
				...new Set(
					group.flatMap(observation => {
						// Structured claims are the durable semantic unit. Never promote
						// invalidated/untrusted claims, and do not mix transient user intent
						// back in through the legacy facts fallback.
						if (observation.claims.length > 0) {
							if (observation.provenance?.trust === "untrusted") return [];
							return observation.claims
								.filter(claim => claim.status !== "invalidated")
								.sort((a, b) => {
									const status =
										(b.status === "verified" ? 1 : 0) -
										(a.status === "verified" ? 1 : 0);
									return status || b.confidence - a.confidence;
								})
								.map(claim => claim.text);
						}
						return (
							observation.facts.length
								? observation.facts
								: [observation.narrative]
						).filter(fact => !/^user intent\s*:/i.test(fact.trim()));
					}),
				),
			].slice(0, 8);
			if (allFacts.length === 0) continue;
			const allConcepts = [...new Set(group.flatMap(o => o.concepts))].slice(
				0,
				10,
			);
			const allFiles = [...new Set(group.flatMap(o => o.files))].slice(0, 10);
			const avgStrength = Math.round(
				group.reduce((s, o) => s + o.importance, 0) / group.length,
			);

			const typeNames: Record<string, MemoryType> = {
				file_read: "fact",
				file_write: "pattern",
				file_edit: "pattern",
				command_run: "workflow",
				search: "fact",
				web_fetch: "fact",
				conversation: "pattern",
				error: "bug",
				decision: "pattern",
				discovery: "fact",
				implementation: "architecture",
				bugfix: "bug",
				notification: "fact",
				other: "fact",
			};

			const label = topic.replace(/^(?:file|concept|type):/, "");
			const title = `${label} — ${dominantType.replace(/_/g, " ")}`.slice(
				0,
				200,
			);
			const content = allFacts.join("\n");
			const sourceIds = group.map(o => o.id);
			const strength = Math.min(10, Math.max(1, avgStrength + 1));

			// Writes the group as either a fresh memory or a superseding version
			// of `existingRow`. idx_memories_latest_title enforces at most one
			// is_latest=1 row per (workspace, title); the caller retries this
			// once against a fresh read if that constraint fires, which only
			// happens when a separate process consolidates the same title
			// between the SELECT and this INSERT (writes on this connection are
			// already serialized by the enclosing transaction).
			const writeMemory = (existingRow: any) => {
				const ts = now();
				if (existingRow) {
					const existing = rowToMemory(existingRow);
					const id = generateId();
					const mergedContent = [
						...new Set([...existing.content.split("\n"), ...allFacts]),
					]
						.filter(Boolean)
						.slice(-12)
						.join("\n");
					const mergedConcepts = [
						...new Set([...existing.concepts, ...allConcepts]),
					].slice(0, 15);
					const mergedFiles = [
						...new Set([...existing.files, ...allFiles]),
					].slice(0, 15);
					const mergedSessions = [
						...new Set([...existing.sessionIds, sessionId]),
					];
					const mergedSources = [
						...new Set([
							...(existing.sourceObservationIds || []),
							...sourceIds,
						]),
					].slice(-100);
					db.prepare("UPDATE memories SET is_latest = 0 WHERE id = ?").run(
						existing.id,
					);
					db.prepare(
						`INSERT INTO memories
                (id, created_at, updated_at, type, title, content, concepts, files, session_ids,
                 strength, version, parent_id, related_ids, source_observation_ids, is_latest,
                 project, workspace, supersedes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)`,
					).run(
						id,
						existing.createdAt,
						ts,
						existing.type,
						title,
						mergedContent,
						JSON.stringify(mergedConcepts),
						JSON.stringify(mergedFiles),
						JSON.stringify(mergedSessions),
						Math.min(10, Math.max(existing.strength, strength)),
						existing.version + 1,
						existing.id,
						JSON.stringify(existing.relatedIds || []),
						JSON.stringify(mergedSources),
						existing.project || null,
						workspace,
						JSON.stringify([existing.id]),
					);
					db.prepare(
						`INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
              VALUES (?, 'supersedes', ?, ?, 1, ?)`,
					).run(generateId(), id, existing.id, ts);
					return id;
				}
				const id = generateId();
				db.prepare(
					`INSERT INTO memories
            (id, created_at, updated_at, type, title, content, concepts, files, session_ids,
             strength, version, parent_id, related_ids, source_observation_ids, is_latest, project, workspace)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, '[]', ?, 1, NULL, ?)`,
				).run(
					id,
					ts,
					ts,
					typeNames[dominantType] || "fact",
					title,
					content,
					JSON.stringify(allConcepts),
					JSON.stringify(allFiles),
					JSON.stringify([sessionId]),
					strength,
					JSON.stringify(sourceIds),
					workspace,
				);
				return id;
			};

			const existingRow = db
				.prepare(
					"SELECT * FROM memories WHERE workspace = ? AND is_latest = 1 AND title = ? LIMIT 1",
				)
				.get(workspace, title) as any;

			let newId: string;
			let wasNewTopic = !existingRow;
			try {
				newId = writeMemory(existingRow);
			} catch (error) {
				if (
					!(error instanceof Error) ||
					!/UNIQUE constraint failed/.test(error.message)
				)
					throw error;
				const retryRow = db
					.prepare(
						"SELECT * FROM memories WHERE workspace = ? AND is_latest = 1 AND title = ? LIMIT 1",
					)
					.get(workspace, title) as any;
				wasNewTopic = !retryRow;
				newId = writeMemory(retryRow);
			}
			const newMemory = rowToMemory(
				db.prepare("SELECT * FROM memories WHERE id = ?").get(newId),
			);
			memories.push(newMemory);
			usedObservationIds.push(...sourceIds);

			// Same-title collisions already went through the supersession path
			// above (an intentional update to the same subject). Only check a
			// genuinely new topic against the rest of the workspace's memory —
			// that is the case a title-keyed lookup can't catch.
			if (wasNewTopic) detectContradictions(db, getWorkspace, newMemory);
		}

		if (usedObservationIds.length) {
			const mark = db.prepare(
				"UPDATE observations SET consolidated = 1 WHERE id = ?",
			);
			usedObservationIds.forEach(id => mark.run(id));
		}
	});
	applyConsolidation();

	return memories;
}
