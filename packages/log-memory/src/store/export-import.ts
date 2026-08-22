/** Full-workspace export/import of sessions, observations, claims,
 * memories, and relations — the backup/restore and cross-machine transfer
 * path. */

import type { Database } from "bun:sqlite";
import type {
	ExportData,
	ImportData,
	ImportResult,
	MemoryRelationType,
	ObservationType,
} from "../types.js";
import {
	parseObservationClaims,
	parseObservationProvenance,
	safeParseJson,
	safeParseJsonArray,
} from "./db-helpers.ts";
import { rowToMemory } from "./memories.ts";
import { normalizeWorkspacePath, now } from "./module-helpers.ts";
import { persistObservationClaims, rowToClaim } from "./observations.ts";
import { listSessions } from "./sessions.ts";

export function exportData(db: Database, getWorkspace: () => string): ExportData {
	const workspace = getWorkspace();
	const sessions = listSessions(db, getWorkspace);
	const memories = db
		.prepare(
			`SELECT * FROM memories WHERE workspace = ? ORDER BY created_at, version, id`,
		)
		.all(workspace)
		.map(rowToMemory);
	const observations = db
		.prepare(
			`SELECT * FROM observations WHERE workspace = ? ORDER BY timestamp DESC`,
		)
		.all(workspace) as any[];
	const claims = (
		db
			.prepare(
				"SELECT * FROM claims WHERE workspace = ? ORDER BY transaction_time, id",
			)
			.all(workspace) as any[]
	).map(row => rowToClaim(db, row));
	const relations = db
		.prepare(
			`SELECT r.* FROM relations r
          JOIN memories source ON source.id = r.source_id
          JOIN memories target ON target.id = r.target_id
          WHERE source.workspace = ? AND target.workspace = ?
          ORDER BY r.created_at DESC`,
		)
		.all(workspace, workspace) as any[];

	return {
		version: 4,
		exportedAt: now(),
		sessions,
		observations: observations.map(r => ({
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
			consolidated: r.consolidated === 1 || r.consolidated === true,
			workspace: r.workspace || workspace,
			claims: parseObservationClaims(r.claims),
			provenance: parseObservationProvenance(r.provenance),
			hookType: r.hook_type || "import",
			rawData: safeParseJson(r.raw_data || "null"),
		})),
		claims,
		memories,
		relations: relations.map(r => ({
			id: r.id,
			type: r.type as MemoryRelationType,
			sourceId: r.source_id,
			targetId: r.target_id,
			confidence: r.confidence ?? 0.5,
			createdAt: r.created_at,
		})),
	};
}

export function importData(
	db: Database,
	getWorkspace: () => string,
	data: ImportData,
): ImportResult {
	const result: ImportResult = { imported: 0, skipped: 0, errors: [] };
	const mode = data.onConflict || "skip";
	if (![1, 2, 3, 4].includes(data.version)) {
		return {
			imported: 0,
			skipped: 0,
			errors: [`Unsupported memory export version: ${data.version}`],
		};
	}
	const currentWorkspace = getWorkspace();
	const normalizedScope = (workspace?: string): string =>
		normalizeWorkspacePath(workspace || currentWorkspace);
	const collisionIsOutsideScope = (
		table: "sessions" | "observations" | "memories",
		id: string,
		workspace: string,
	): boolean => {
		const row = db
			.prepare(`SELECT workspace FROM ${table} WHERE id = ?`)
			.get(id) as { workspace: string } | undefined;
		return Boolean(row && normalizedScope(row.workspace) !== workspace);
	};

	// Import sessions
	for (const session of data.sessions) {
		try {
			const workspace = normalizedScope(session.workspace || session.cwd);
			if (collisionIsOutsideScope("sessions", session.id, workspace)) {
				throw new Error("ID belongs to a different workspace");
			}
			const existing = db
				.prepare("SELECT id FROM sessions WHERE id = ?")
				.get(session.id);
			if (existing && mode === "skip") {
				result.skipped++;
				continue;
			}
			db.prepare(
				`INSERT INTO sessions
            (id, name, project, cwd, workspace, started_at, ended_at, status,
             observation_count, model, tags, first_prompt, summary, commit_shas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name, project=excluded.project, cwd=excluded.cwd,
              workspace=excluded.workspace, started_at=excluded.started_at,
              ended_at=excluded.ended_at, status=excluded.status,
              observation_count=excluded.observation_count, model=excluded.model,
              tags=excluded.tags, first_prompt=excluded.first_prompt,
              summary=excluded.summary, commit_shas=excluded.commit_shas`,
			).run(
				session.id,
				session.name || null,
				session.project || "",
				session.cwd || workspace,
				workspace,
				session.startedAt,
				session.endedAt || null,
				session.status,
				session.observationCount,
				session.model || null,
				JSON.stringify(session.tags || []),
				session.firstPrompt || null,
				session.summary || null,
				JSON.stringify(session.commitShas || []),
			);
			result.imported++;
		} catch (e) {
			result.errors.push(`Session ${session.id}: ${(e as Error).message}`);
		}
	}

	// Import observations
	for (const obs of data.observations) {
		try {
			const workspace = normalizedScope(obs.workspace);
			if (collisionIsOutsideScope("observations", obs.id, workspace)) {
				throw new Error("ID belongs to a different workspace");
			}
			const existing = db
				.prepare("SELECT id FROM observations WHERE id = ?")
				.get(obs.id) as { id: string } | undefined;
			if (existing && mode === "skip") {
				result.skipped++;
				continue;
			}
			if (existing) {
				db.prepare("DELETE FROM claims WHERE observation_id = ?").run(obs.id);
			}
			const exported = obs as typeof obs & {
				hookType?: string;
				rawData?: unknown;
			};
			db.prepare(
				`INSERT INTO observations
            (id, session_id, timestamp, hook_type, type, title, subtitle,
             narrative, facts, concepts, files, importance, workspace,
             consolidated, claims, provenance, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              session_id=excluded.session_id, timestamp=excluded.timestamp,
              hook_type=excluded.hook_type, type=excluded.type, title=excluded.title,
              subtitle=excluded.subtitle, narrative=excluded.narrative,
              facts=excluded.facts, concepts=excluded.concepts, files=excluded.files,
              importance=excluded.importance, workspace=excluded.workspace,
              consolidated=excluded.consolidated, claims=excluded.claims,
              provenance=excluded.provenance, raw_data=excluded.raw_data`,
			).run(
				obs.id,
				obs.sessionId,
				obs.timestamp,
				exported.hookType || "import",
				obs.type,
				obs.title,
				obs.subtitle || null,
				obs.narrative,
				JSON.stringify(obs.facts),
				JSON.stringify(obs.concepts),
				JSON.stringify(obs.files),
				obs.importance,
				workspace,
				obs.consolidated ? 1 : 0,
				JSON.stringify(obs.claims || []),
				obs.provenance ? JSON.stringify(obs.provenance) : null,
				JSON.stringify(exported.rawData ?? null),
			);
			if (!data.claims) {
				persistObservationClaims(
					db,
					{
						...obs,
						workspace,
						claims: parseObservationClaims(obs.claims),
						provenance: parseObservationProvenance(obs.provenance),
					},
					workspace,
					obs.timestamp,
				);
			}
			result.imported++;
		} catch (e) {
			result.errors.push(`Observation ${obs.id}: ${(e as Error).message}`);
		}
	}

	// Version 3+ carries the temporal truth layer explicitly. Import it in two
	// passes so forward superseded-by links never depend on array ordering.
	if (data.claims) {
		for (const claim of data.claims) {
			try {
				const workspace = normalizedScope(claim.workspace);
				if (workspace !== currentWorkspace)
					throw new Error("claim belongs to a different workspace");
				db.prepare(
					`INSERT INTO claims
				  (id, workspace, observation_id, session_id, text, status, confidence,
				   operation, valid_from, valid_to, transaction_time, source, trust,
				   extractor_version, schema_version, supersedes_claim_id,
				   superseded_by_claim_id, tombstoned_at, lifecycle,
				   validity_predicates, evidence_certificate)
				  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
				  ON CONFLICT(id) DO UPDATE SET text=excluded.text, status=excluded.status,
				    confidence=excluded.confidence, operation=excluded.operation,
				    valid_from=excluded.valid_from, valid_to=excluded.valid_to,
				    transaction_time=excluded.transaction_time, source=excluded.source,
				    trust=excluded.trust, extractor_version=excluded.extractor_version,
				    schema_version=excluded.schema_version,
				    lifecycle=excluded.lifecycle,
				    validity_predicates=excluded.validity_predicates,
				    evidence_certificate=excluded.evidence_certificate,
				    supersedes_claim_id=NULL,
				    superseded_by_claim_id=NULL, tombstoned_at=excluded.tombstoned_at`,
				).run(
					claim.id,
					workspace,
					claim.observationId,
					claim.sessionId,
					claim.text,
					claim.status,
					claim.confidence,
					claim.operation,
					claim.validFrom,
					claim.validTo || null,
					claim.transactionTime,
					claim.source,
					claim.trust,
					claim.extractorVersion,
					claim.schemaVersion,
					claim.tombstonedAt || null,
					claim.lifecycle || "probationary",
					JSON.stringify(claim.validityPredicates || []),
					JSON.stringify(claim.evidenceCertificate || {}),
				);
				db.prepare("DELETE FROM claim_evidence WHERE claim_id = ?").run(
					claim.id,
				);
				for (const evidenceId of claim.evidenceEventIds)
					db.prepare(
						`INSERT INTO claim_evidence
					  (claim_id, observation_id, evidence_event_id) VALUES (?, ?, ?)`,
					).run(claim.id, claim.observationId, evidenceId);
				result.imported++;
			} catch (e) {
				result.errors.push(`Claim ${claim.id}: ${(e as Error).message}`);
			}
		}
		for (const claim of data.claims) {
			if (!claim.supersededByClaimId && !claim.supersedesClaimId) continue;
			try {
				db.prepare(
					`UPDATE claims
				  SET supersedes_claim_id = ?, superseded_by_claim_id = ? WHERE id = ?`,
				).run(
					claim.supersedesClaimId || null,
					claim.supersededByClaimId || null,
					claim.id,
				);
			} catch (e) {
				result.errors.push(`Claim link ${claim.id}: ${(e as Error).message}`);
			}
		}
	}

	// Import memories
	for (const mem of data.memories) {
		try {
			const workspace = normalizedScope(mem.workspace);
			if (collisionIsOutsideScope("memories", mem.id, workspace)) {
				throw new Error("ID belongs to a different workspace");
			}
			const existing = db
				.prepare("SELECT id FROM memories WHERE id = ?")
				.get(mem.id) as { id: string } | undefined;
			if (existing && mode === "skip") {
				result.skipped++;
				continue;
			}
			db.prepare(
				`INSERT INTO memories
            (id, created_at, updated_at, type, title, content, concepts, files,
             session_ids, strength, version, parent_id, related_ids,
             source_observation_ids, is_latest, project, workspace,
             access_count, last_accessed, working_tier, supersedes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              created_at=excluded.created_at, updated_at=excluded.updated_at,
              type=excluded.type, title=excluded.title, content=excluded.content,
              concepts=excluded.concepts, files=excluded.files,
              session_ids=excluded.session_ids, strength=excluded.strength,
              version=excluded.version, parent_id=excluded.parent_id,
              related_ids=excluded.related_ids,
              source_observation_ids=excluded.source_observation_ids,
              is_latest=excluded.is_latest, project=excluded.project,
              workspace=excluded.workspace, access_count=excluded.access_count,
              last_accessed=excluded.last_accessed,
              working_tier=excluded.working_tier,
              supersedes=excluded.supersedes`,
			).run(
				mem.id,
				mem.createdAt,
				mem.updatedAt,
				mem.type,
				mem.title,
				mem.content,
				JSON.stringify(mem.concepts),
				JSON.stringify(mem.files),
				JSON.stringify(mem.sessionIds),
				mem.strength,
				mem.version,
				mem.parentId || null,
				JSON.stringify(mem.relatedIds || []),
				JSON.stringify(mem.sourceObservationIds || []),
				mem.isLatest ? 1 : 0,
				mem.project || null,
				workspace,
				mem.accessCount || 0,
				mem.lastAccessed || null,
				mem.workingTier || "cold",
				JSON.stringify(mem.supersedes || []),
			);
			result.imported++;
		} catch (e) {
			result.errors.push(`Memory ${mem.id}: ${(e as Error).message}`);
		}
	}

	// Import relations
	if (data.relations) {
		for (const rel of data.relations) {
			try {
				const endpoints = db
					.prepare(
						`SELECT source.workspace AS source_workspace,
                           target.workspace AS target_workspace
                    FROM memories source JOIN memories target
                    WHERE source.id = ? AND target.id = ?`,
					)
					.get(rel.sourceId, rel.targetId) as
					| { source_workspace: string; target_workspace: string }
					| undefined;
				if (
					!endpoints ||
					normalizedScope(endpoints.source_workspace) !==
						normalizedScope(endpoints.target_workspace)
				) {
					throw new Error(
						"relation endpoints are missing or cross-workspace",
					);
				}
				const existing = db
					.prepare("SELECT id FROM relations WHERE id = ?")
					.get(rel.id) as { id: string } | undefined;
				if (existing && mode === "skip") {
					result.skipped++;
					continue;
				}
				db.prepare(
					`
            INSERT INTO relations (id, type, source_id, target_id, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET type=excluded.type,
              source_id=excluded.source_id, target_id=excluded.target_id,
              confidence=excluded.confidence, created_at=excluded.created_at
          `,
				).run(
					rel.id,
					rel.type,
					rel.sourceId,
					rel.targetId,
					rel.confidence,
					rel.createdAt,
				);
				result.imported++;
			} catch (e) {
				result.errors.push(`Relation ${rel.id}: ${(e as Error).message}`);
			}
		}
	}

	return result;
}
