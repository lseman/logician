/** Vector embedding storage and locality-sensitive-hash bucketed cosine
 * similarity search over observations/memories. */

import type { Database } from "bun:sqlite";
import type { EmbeddingMetadata, SemanticSearchResult } from "../../types.ts";
import { safeParseJson } from "../models/db-helpers.js";
import { now } from "../module-helpers.js";

const EMBEDDING_BUCKET_BITS = 12;

function embeddingBucket(vector: number[]): string {
	return vector
		.slice(0, Math.min(EMBEDDING_BUCKET_BITS, vector.length))
		.map(value => (value >= 0 ? "1" : "0"))
		.join("");
}

function embeddingProbeBuckets(vector: number[]): string[] {
	const primary = embeddingBucket(vector);
	const probes = [primary];
	for (let index = 0; index < primary.length; index++) {
		probes.push(
			`${primary.slice(0, index)}${primary[index] === "1" ? "0" : "1"}${primary.slice(index + 1)}`,
		);
	}
	return probes;
}

export function upsertEmbedding(
	db: Database,
	getWorkspace: () => string,
	id: string,
	kind: "observation" | "memory",
	vector: number[],
	sessionId?: string,
	metadata: EmbeddingMetadata = {
		model: "unknown",
		contentHash: "",
		creationVersion: 1,
	},
): void {
	if (!vector.length || vector.some(value => !Number.isFinite(value))) return;
	const vectorBucket = embeddingBucket(vector);
	db.prepare(
		`INSERT INTO memory_embeddings
	      (entity_id, entity_kind, session_id, workspace, dimensions, model,
	       content_hash, creation_version, vector_bucket, vector, updated_at)
	      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(entity_id) DO UPDATE SET
        entity_kind = excluded.entity_kind,
        session_id = excluded.session_id,
        workspace = excluded.workspace,
	        dimensions = excluded.dimensions,
	        model = excluded.model,
	        content_hash = excluded.content_hash,
	        creation_version = excluded.creation_version,
	        vector_bucket = excluded.vector_bucket,
        vector = excluded.vector,
        updated_at = excluded.updated_at`,
	).run(
		id,
		kind,
		sessionId || null,
		getWorkspace(),
		vector.length,
		metadata.model,
		metadata.contentHash,
		metadata.creationVersion,
		vectorBucket,
		JSON.stringify(vector),
		now(),
	);
}

export function searchEmbeddings(
	db: Database,
	getWorkspace: () => string,
	vector: number[],
	limit: number = 40,
): SemanticSearchResult[] {
	if (!vector.length || vector.some(value => !Number.isFinite(value)))
		return [];
	const workspace = getWorkspace();
	const buckets = embeddingProbeBuckets(vector);
	const placeholders = buckets.map(() => "?").join(", ");
	const rows = db
		.prepare(
			`SELECT entity_id, entity_kind, session_id, vector
	      FROM memory_embeddings WHERE workspace = ? AND dimensions = ?
	        AND vector_bucket IN (${placeholders})
	      ORDER BY updated_at DESC LIMIT 8000`,
		)
		.all(workspace, vector.length, ...buckets) as Array<{
		entity_id: string;
		entity_kind: "observation" | "memory";
		session_id: string | null;
		vector: string;
	}>;
	// Legacy rows are backfilled lazily. A bounded fallback keeps semantic
	// retrieval available during migration without reinstating a recency-only
	// cliff for indexed rows.
	if (rows.length < Math.min(limit, 20)) {
		const legacyRows = db
			.prepare(
				`SELECT entity_id, entity_kind, session_id, vector
			  FROM memory_embeddings WHERE workspace = ? AND dimensions = ?
			    AND vector_bucket = '' ORDER BY updated_at DESC LIMIT 1000`,
			)
			.all(workspace, vector.length) as typeof rows;
		rows.push(...legacyRows);
	}
	let queryNorm = 0;
	for (const value of vector) queryNorm += value * value;
	queryNorm = Math.sqrt(queryNorm);
	if (!queryNorm) return [];
	const results: SemanticSearchResult[] = [];
	for (const row of rows) {
		const candidate = safeParseJson(row.vector);
		if (!Array.isArray(candidate) || candidate.length !== vector.length)
			continue;
		let dot = 0;
		let norm = 0;
		for (let index = 0; index < vector.length; index++) {
			const value = Number(candidate[index]);
			if (!Number.isFinite(value)) {
				norm = 0;
				break;
			}
			dot += vector[index] * value;
			norm += value * value;
		}
		const score = norm ? dot / (queryNorm * Math.sqrt(norm)) : 0;
		if (score <= 0) continue;
		results.push({
			id: row.entity_id,
			kind: row.entity_kind,
			sessionId: row.session_id || undefined,
			score,
		});
	}
	return results
		.sort((a, b) => b.score - a.score)
		.slice(0, Math.max(1, Math.min(limit, 200)));
}

export function hasEmbedding(
	db: Database,
	getWorkspace: () => string,
	id: string,
	metadata: Partial<EmbeddingMetadata> = {},
): boolean {
	const conditions = ["entity_id = ?", "workspace = ?"];
	const params: Array<string | number> = [id, getWorkspace()];
	if (metadata.model !== undefined) {
		conditions.push("model = ?");
		params.push(metadata.model);
	}
	if (metadata.contentHash !== undefined) {
		conditions.push("content_hash = ?");
		params.push(metadata.contentHash);
	}
	if (metadata.creationVersion !== undefined) {
		conditions.push("creation_version = ?");
		params.push(metadata.creationVersion);
	}
	return Boolean(
		db
			.prepare(
				`SELECT 1 AS found FROM memory_embeddings WHERE ${conditions.join(" AND ")}`,
			)
			.get(...params),
	);
}
