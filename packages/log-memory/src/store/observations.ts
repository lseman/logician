/** CRUD over the observations table (raw hook events compressed into a
 * title/narrative/facts/concepts summary) plus the append-only claims
 * truth layer derived from them. */

import type { Database } from "bun:sqlite";
import type {
	CompressedObservation,
	ExpandedMemoryEntry,
	MemoryClaim,
	ObservationType,
	RawObservation,
	SearchResult,
} from "../types.js";
import { trackAccess } from "./access-tracker.ts";
import {
	parseObservationClaims,
	parseObservationProvenance,
	safeParseJson,
	safeParseJsonArray,
} from "./db-helpers.ts";
import { rowToMemory } from "./memories.ts";
import {
	normalizeWorkspacePath,
	now,
	sanitizePayload,
	sanitizeString,
	toFtsQuery,
} from "./module-helpers.ts";
import { createSession, getSession } from "./sessions.ts";
import { slidingWindowCap } from "./sliding-window.ts";
import { buildSyntheticCompression } from "./synthetic-compression.ts";

const CLAIM_NEGATION =
	/\b(?:not|never|no longer|isn't|doesn't|don't|can't|cannot|without|disabled|removed|deprecated|incorrect|wrong)\b/gi;

const claimTerms = (text: string, ignorePolarity = false): Set<string> => {
	const normalized = ignorePolarity
		? text.replace(CLAIM_NEGATION, " ").replace(/\b(?:do|does|did)\b/gi, " ")
		: text;
	const stop = new Set([
		"a",
		"an",
		"and",
		"are",
		"as",
		"at",
		"be",
		"by",
		"for",
		"from",
		"in",
		"is",
		"it",
		"of",
		"on",
		"or",
		"the",
		"this",
		"to",
		"with",
	]);
	return new Set(
		(
			normalized
				.normalize("NFKC")
				.toLowerCase()
				.match(/[\p{L}\p{N}_-]{2,}/gu) || []
		)
			.map(token =>
				token === "uses"
					? "use"
					: token.endsWith("ies") && token.length > 4
						? `${token.slice(0, -3)}y`
						: token.endsWith("s") && token.length > 4
							? token.slice(0, -1)
							: token,
			)
			.filter(token => !stop.has(token)),
	);
};

const claimSimilarity = (left: string, right: string): number => {
	const a = claimTerms(left, true);
	const b = claimTerms(right, true);
	if (!a.size || !b.size) return 0;
	let overlap = 0;
	for (const token of a) if (b.has(token)) overlap++;
	return overlap / (a.size + b.size - overlap);
};

const claimHasNegation = (text: string): boolean => {
	CLAIM_NEGATION.lastIndex = 0;
	return CLAIM_NEGATION.test(text);
};

export function persistObservationClaims(
	db: Database,
	observation: CompressedObservation,
	workspace: string,
	transactionTime: string,
): void {
	const provenance = observation.provenance || {
		source: "deterministic" as const,
		trust: "trusted_local" as const,
		extractorVersion: "legacy-observation/1",
		schemaVersion: 1,
	};
	const insertClaim = db.prepare(`INSERT OR IGNORE INTO claims
      (id, workspace, observation_id, session_id, text, status, confidence,
       operation, valid_from, transaction_time, source, trust,
	       extractor_version, schema_version, supersedes_claim_id, tombstoned_at,
	       lifecycle, validity_predicates, evidence_certificate)
	      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
	const insertEvidence = db.prepare(`INSERT OR IGNORE INTO claim_evidence
      (claim_id, observation_id, evidence_event_id) VALUES (?, ?, ?)`);
	const activeRows = db.prepare(`SELECT id, text, status, confidence, trust
	      FROM claims
	      WHERE workspace = ? AND valid_to IS NULL
	        AND superseded_by_claim_id IS NULL AND tombstoned_at IS NULL
	      ORDER BY transaction_time DESC LIMIT 500`);
	const closeClaim = db.prepare(`UPDATE claims
	      SET valid_to = ?, superseded_by_claim_id = ?, lifecycle = 'superseded'
	      WHERE id = ? AND valid_to IS NULL AND superseded_by_claim_id IS NULL`);
	const contestClaim = db.prepare(
		"UPDATE claims SET lifecycle = 'contested' WHERE id = ? AND lifecycle != 'superseded'",
	);
	for (const [index, claim] of (observation.claims || []).entries()) {
		const claimId = `${observation.id}:claim:${index}`;
		const active = activeRows.all(workspace) as Array<{
			id: string;
			text: string;
			status: MemoryClaim["status"];
			confidence: number;
			trust: MemoryClaim["trust"];
		}>;
		const exact = active.find(candidate => {
			const left = [...claimTerms(candidate.text)].sort().join(" ");
			const right = [...claimTerms(claim.text)].sort().join(" ");
			return left.length > 0 && left === right;
		});
		const contradiction = active.find(
			candidate =>
				claimSimilarity(candidate.text, claim.text) >= 0.8 &&
				claimHasNegation(candidate.text) !== claimHasNegation(claim.text),
		);
		let operation: MemoryClaim["operation"] = "ADD";
		let supersedesClaimId: string | undefined;
		let tombstonedAt: string | undefined;
		let claimToClose: string | undefined;

		if (exact) {
			operation = claim.status === "invalidated" ? "INVALIDATE" : "NOOP";
			supersedesClaimId = exact.id;
			if (operation === "INVALIDATE") claimToClose = exact.id;
			else tombstonedAt = transactionTime;
		} else if (contradiction) {
			const trustedRevision =
				provenance.trust !== "untrusted" &&
				claim.status === "verified" &&
				claim.confidence >= contradiction.confidence;
			if (claim.status === "invalidated" || trustedRevision) {
				operation =
					claim.status === "invalidated" ? "INVALIDATE" : "SUPERSEDE";
				supersedesClaimId = contradiction.id;
				claimToClose = contradiction.id;
			} else contestClaim.run(contradiction.id);
		}
		const lifecycle: MemoryClaim["lifecycle"] =
			provenance.trust === "untrusted"
				? "quarantined"
				: claim.status === "invalidated"
					? "stale"
					: provenance.source === "deterministic" &&
							claim.status === "verified"
						? "durable"
						: contradiction && !claimToClose
							? "contested"
							: "probationary";
		const certificate = {
			extractorVersion: provenance.extractorVersion,
			schemaVersion: provenance.schemaVersion,
			evidenceEventIds: claim.evidenceEventIds,
			issuedAt: transactionTime,
		};
		insertClaim.run(
			claimId,
			workspace,
			observation.id,
			observation.sessionId,
			claim.text,
			claim.status,
			claim.confidence,
			operation,
			observation.timestamp,
			transactionTime,
			provenance.source,
			provenance.trust,
			provenance.extractorVersion,
			provenance.schemaVersion,
			supersedesClaimId || null,
			tombstonedAt || null,
			lifecycle,
			JSON.stringify(claim.validityPredicates || []),
			JSON.stringify(certificate),
		);
		if (claimToClose)
			closeClaim.run(observation.timestamp, claimId, claimToClose);
		for (const evidenceId of claim.evidenceEventIds) {
			insertEvidence.run(claimId, observation.id, evidenceId);
		}
	}
}

export function observe(
	db: Database,
	getWorkspace: () => string,
	raw: RawObservation,
	compressed?: CompressedObservation,
): CompressedObservation | null {
	const ts = now();
	const safeRaw = {
		...raw,
		toolInput: sanitizePayload(raw.toolInput),
		toolOutput: sanitizePayload(raw.toolOutput),
		userPrompt: raw.userPrompt ? sanitizeString(raw.userPrompt) : undefined,
		raw: sanitizePayload(raw.raw),
	};
	const generated = compressed || buildSyntheticCompression(safeRaw);
	const comp: CompressedObservation = {
		...generated,
		title: sanitizeString(generated.title).slice(0, 200),
		subtitle: generated.subtitle
			? sanitizeString(generated.subtitle).slice(0, 300)
			: undefined,
		narrative: sanitizeString(generated.narrative).slice(0, 2000),
		facts: generated.facts.map(sanitizeString).slice(0, 20),
		concepts: generated.concepts.map(sanitizeString).slice(0, 20),
		files: generated.files.map(sanitizeString).slice(0, 20),
		claims: parseObservationClaims(generated.claims).map(claim => ({
			...claim,
			text: sanitizeString(claim.text).slice(0, 1000),
			evidenceEventIds: claim.evidenceEventIds
				.map(sanitizeString)
				.slice(0, 12),
		})),
		provenance: parseObservationProvenance(generated.provenance),
	};
	// Derive workspace from observation data or current workspace
	const obsWorkspace = normalizeWorkspacePath(
		raw.workspace || getWorkspace(),
	);
	// Direct callers may capture evidence before explicitly registering the
	// session. Materialize the owning session so foreign-key enforcement does
	// not turn robust capture into a startup-order dependency.
	if (!getSession(db, raw.sessionId)) {
		createSession(db, getWorkspace, raw.sessionId, {
			cwd: obsWorkspace,
			workspace: obsWorkspace,
		});
	}

	const persistObservation = db.transaction(() => {
		db.prepare(
			`
      INSERT INTO observations (id, session_id, timestamp, hook_type, type, title, subtitle,
                                narrative, facts, concepts, files, importance, workspace, claims,
                                provenance, raw_data)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
		).run(
			comp.id,
			raw.sessionId,
			raw.timestamp || ts,
			raw.hookType,
			comp.type,
			comp.title,
			comp.subtitle || null,
			comp.narrative,
			JSON.stringify(comp.facts),
			JSON.stringify(comp.concepts),
			JSON.stringify(comp.files),
			comp.importance,
			obsWorkspace,
			JSON.stringify(comp.claims || []),
			comp.provenance ? JSON.stringify(comp.provenance) : null,
			JSON.stringify(safeRaw.raw).slice(0, 32_000),
		);
		persistObservationClaims(db, comp, obsWorkspace, ts);
		db.prepare(
			`
      UPDATE sessions SET observation_count = observation_count + 1 WHERE id = ?
			`,
		).run(raw.sessionId);
	});
	persistObservation();

	// Apply sliding window cap (enforce max observations per session)
	slidingWindowCap(db, getWorkspace, raw.sessionId, 200);

	return comp;
}

export function getObservation(
	db: Database,
	id: string,
	sessionId: string,
): CompressedObservation | null {
	const row = db
		.prepare(`SELECT * FROM observations WHERE id = ? AND session_id = ?`)
		.get(id, sessionId) as any;
	if (!row) return null;
	return {
		id: row.id,
		sessionId: row.session_id,
		timestamp: row.timestamp,
		type: row.type as ObservationType,
		title: row.title || "",
		subtitle: row.subtitle,
		facts: safeParseJsonArray(row.facts),
		narrative: row.narrative || "",
		concepts: safeParseJsonArray(row.concepts),
		files: safeParseJsonArray(row.files),
		importance: row.importance ?? 5,
		consolidated: row.consolidated === 1 || row.consolidated === true,
		workspace: row.workspace || "",
		claims: parseObservationClaims(row.claims),
		provenance: parseObservationProvenance(row.provenance),
	};
}

export function rowToObservation(row: any): CompressedObservation {
	return {
		id: row.id,
		sessionId: row.session_id,
		timestamp: row.timestamp,
		type: row.type as ObservationType,
		title: row.title || "",
		subtitle: row.subtitle,
		facts: safeParseJsonArray(row.facts),
		narrative: row.narrative || "",
		concepts: safeParseJsonArray(row.concepts),
		files: safeParseJsonArray(row.files),
		importance: row.importance ?? 5,
		consolidated: row.consolidated === 1 || row.consolidated === true,
		workspace: row.workspace || "",
		claims: parseObservationClaims(row.claims),
		provenance: parseObservationProvenance(row.provenance),
	};
}

export function listObservations(
	db: Database,
	sessionId: string,
	limit: number = 50,
): CompressedObservation[] {
	const rows = db
		.prepare(
			`SELECT * FROM observations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?`,
		)
		.all(sessionId, limit) as any[];
	return rows.map(rowToObservation);
}

export function listRecentObservations(
	db: Database,
	getWorkspace: () => string,
	limit: number = 50,
	type?: ObservationType,
): CompressedObservation[] {
	const conditions: string[] = [];
	const params: Array<string | number> = [];
	conditions.push("workspace = ?");
	params.push(getWorkspace());
	if (type) {
		conditions.push("type = ?");
		params.push(type);
	}
	const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
	params.push(Math.max(1, Math.min(limit, 1000)));
	const rows = db
		.prepare(
			`SELECT * FROM observations ${where} ORDER BY timestamp DESC, rowid DESC LIMIT ?`,
		)
		.all(...params) as any[];
	return rows.map(rowToObservation);
}

export function listClaims(
	db: Database,
	getWorkspace: () => string,
	options: {
		observationId?: string;
		status?: MemoryClaim["status"];
		includeSuperseded?: boolean;
		limit?: number;
	} = {},
): MemoryClaim[] {
	const conditions = ["workspace = ?"];
	const params: Array<string | number> = [getWorkspace()];
	if (options.observationId) {
		conditions.push("observation_id = ?");
		params.push(options.observationId);
	}
	if (options.status) {
		conditions.push("status = ?");
		params.push(options.status);
	}
	if (!options.includeSuperseded) {
		conditions.push("superseded_by_claim_id IS NULL");
		conditions.push("tombstoned_at IS NULL");
	}
	params.push(Math.max(1, Math.min(options.limit || 100, 1000)));
	const rows = db
		.prepare(
			`SELECT * FROM claims WHERE ${conditions.join(" AND ")}
         ORDER BY transaction_time DESC, id DESC LIMIT ?`,
		)
		.all(...params) as any[];
	return rows.map(row => rowToClaim(db, row));
}

export function rowToClaim(db: Database, row: any): MemoryClaim {
	const evidenceEventIds = (
		db
			.prepare(
				"SELECT evidence_event_id FROM claim_evidence WHERE claim_id = ? ORDER BY evidence_event_id",
			)
			.all(row.id) as Array<{ evidence_event_id: string }>
	).map(item => item.evidence_event_id);
	const certificate = safeParseJson(
		row.evidence_certificate || "{}",
	) as Partial<MemoryClaim["evidenceCertificate"]> | null;
	return {
		id: row.id,
		workspace: row.workspace,
		observationId: row.observation_id,
		sessionId: row.session_id,
		text: row.text,
		status: row.status,
		confidence: row.confidence,
		operation: row.operation,
		validFrom: row.valid_from,
		validTo: row.valid_to || undefined,
		transactionTime: row.transaction_time,
		source: row.source,
		trust: row.trust,
		extractorVersion: row.extractor_version,
		schemaVersion: row.schema_version,
		supersedesClaimId: row.supersedes_claim_id || undefined,
		supersededByClaimId: row.superseded_by_claim_id || undefined,
		tombstonedAt: row.tombstoned_at || undefined,
		evidenceEventIds,
		lifecycle: (row.lifecycle as MemoryClaim["lifecycle"]) || "probationary",
		validityPredicates: (safeParseJson(row.validity_predicates || "[]") ||
			[]) as MemoryClaim["validityPredicates"],
		evidenceCertificate: {
			extractorVersion:
				certificate?.extractorVersion || row.extractor_version,
			schemaVersion: certificate?.schemaVersion || row.schema_version,
			evidenceEventIds: certificate?.evidenceEventIds || evidenceEventIds,
			issuedAt: certificate?.issuedAt || row.transaction_time,
		},
	};
}

export function promoteClaim(
	db: Database,
	getWorkspace: () => string,
	claimId: string,
	evidenceEventIds: string[] = [],
): MemoryClaim | null {
	const claim = listClaims(db, getWorkspace, {
		includeSuperseded: true,
		limit: 1_000,
	}).find(candidate => candidate.id === claimId);
	if (!claim || claim.lifecycle !== "probationary") return null;
	const corroborated = new Set([
		...claim.evidenceEventIds,
		...evidenceEventIds.map(sanitizeString),
	]);
	// Promotion requires verified status, trusted provenance, and at least two
	// independent evidence identifiers. Model confidence alone is insufficient.
	if (
		claim.status !== "verified" ||
		claim.trust === "untrusted" ||
		corroborated.size < 2
	)
		return null;
	const issuedAt = now();
	db.transaction(() => {
		const insertEvidence = db.prepare(`INSERT OR IGNORE INTO claim_evidence
		  (claim_id, observation_id, evidence_event_id) VALUES (?, ?, ?)`);
		for (const eventId of evidenceEventIds)
			insertEvidence.run(
				claim.id,
				claim.observationId,
				sanitizeString(eventId),
			);
		db.prepare(
			`UPDATE claims SET lifecycle = 'durable', evidence_certificate = ?
		  WHERE id = ? AND lifecycle = 'probationary'`,
		).run(
			JSON.stringify({
				...claim.evidenceCertificate,
				evidenceEventIds: [...corroborated],
				issuedAt,
			}),
			claim.id,
		);
	})();
	const row = db.prepare("SELECT * FROM claims WHERE id = ?").get(claim.id);
	return row ? rowToClaim(db, row) : null;
}

export function searchObservations(
	db: Database,
	getWorkspace: () => string,
	query: string,
	limit: number = 20,
): SearchResult[] {
	const ftsQuery = toFtsQuery(query);
	if (!ftsQuery) return [];
	const rows = db
		.prepare(
			`
      SELECT o.*, bm25(observations_fts, 0, 8, 4, 2, 3, 3) AS lexical_rank
      FROM observations_fts
      JOIN observations o ON o.id = observations_fts.id
      WHERE observations_fts MATCH ? AND o.workspace = ?
      ORDER BY lexical_rank ASC, o.importance DESC, o.timestamp DESC
      LIMIT ?
    `,
		)
		.all(
			ftsQuery,
			getWorkspace(),
			Math.max(1, Math.min(limit, 1000)),
		) as any[];

	return rows.map(r => ({
		observation: {
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
			workspace: r.workspace || "",
			claims: parseObservationClaims(r.claims),
			provenance: parseObservationProvenance(r.provenance),
		},
		score: Number(
			(Math.max(0, -Number(r.lexical_rank || 0)) + r.importance / 10).toFixed(
				4,
			),
		),
		sessionId: r.session_id,
	}));
}

export function expandEntries(
	db: Database,
	getWorkspace: () => string,
	ids: string[],
): ExpandedMemoryEntry[] {
	const workspace = getWorkspace();
	const uniqueIds = [
		...new Set(ids.map(id => id.trim()).filter(Boolean)),
	].slice(0, 20);
	const entries = new Map<string, ExpandedMemoryEntry>();
	const observationStatement = db.prepare(
		"SELECT * FROM observations WHERE id = ? AND workspace = ?",
	);
	const memoryStatement = db.prepare(
		"SELECT * FROM memories WHERE id = ? AND workspace = ?",
	);
	for (const id of uniqueIds) {
		const observationRow = observationStatement.get(id, workspace) as any;
		if (observationRow) {
			const observation = rowToObservation(observationRow);
			entries.set(id, {
				id,
				kind: "observation",
				title: observation.title,
				content: [observation.narrative, ...observation.facts]
					.filter(Boolean)
					.join("\n"),
				type: observation.type,
				files: observation.files,
				concepts: observation.concepts,
				timestamp: observation.timestamp,
				sessionIds: [observation.sessionId],
			});
			continue;
		}
		const memoryRow = memoryStatement.get(id, workspace) as any;
		if (memoryRow) {
			const memory = rowToMemory(memoryRow);
			trackAccess(db, getWorkspace, memory.id);
			entries.set(id, {
				id,
				kind: "memory",
				title: memory.title,
				content: memory.content,
				type: memory.type,
				files: memory.files,
				concepts: memory.concepts,
				timestamp: memory.updatedAt,
				sessionIds: memory.sessionIds,
			});
		}
	}
	return uniqueIds.flatMap(id => (entries.get(id) ? [entries.get(id)!] : []));
}

export function clearObservations(
	db: Database,
	getWorkspace: () => string,
): number {
	const workspace = getWorkspace();
	const { count } = db
		.prepare("SELECT COUNT(*) AS count FROM observations WHERE workspace = ?")
		.get(workspace) as { count: number };
	db.prepare("DELETE FROM observations WHERE workspace = ?").run(workspace);
	db.prepare(
		`
      UPDATE sessions
      SET observation_count = (
        SELECT COUNT(*) FROM observations WHERE observations.session_id = sessions.id
      )
      WHERE workspace = ?
    `,
	).run(workspace);
	return count;
}
