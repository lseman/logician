/** Get/set a memory's working-memory tier, and batch re-tier all memories
 * from their current retention score. */

import type { Database } from "bun:sqlite";
import type { DecayConfigInput, WorkingMemoryTier } from "../types.js";
import { computeRetentionScore } from "./retention-scoring.ts";

export function getWorkingMemoryTier(
	db: Database,
	getWorkspace: () => string,
	entityId: string,
): WorkingMemoryTier {
	const row = db
		.prepare(
			"SELECT working_tier FROM memories WHERE id = ? AND workspace = ?",
		)
		.get(entityId, getWorkspace()) as { working_tier: string } | undefined;
	return (row?.working_tier as WorkingMemoryTier) || "cold";
}

export function setWorkingMemoryTier(
	db: Database,
	getWorkspace: () => string,
	entityId: string,
	tier: WorkingMemoryTier,
): void {
	db.prepare(
		"UPDATE memories SET working_tier = ? WHERE id = ? AND workspace = ?",
	).run(tier, entityId, getWorkspace());
}

export function autoTierMemories(
	db: Database,
	getWorkspace: () => string,
	config: DecayConfigInput = {},
): Record<string, WorkingMemoryTier> {
	const tiered: Record<string, WorkingMemoryTier> = {};
	const workspace = getWorkspace();

	const rows = db
		.prepare(
			"SELECT id FROM memories WHERE is_latest = 1 AND last_accessed IS NOT NULL AND workspace = ?",
		)
		.all(workspace) as { id: string }[];

	for (const row of rows) {
		// Retention scoring (exponential decay + access reinforcement,
		// weighted by memory-type salience) replaces naive last-accessed
		// bucketing so tiers reflect actual relevance, not just recency.
		const retention = computeRetentionScore(db, getWorkspace, row.id, config);
		const tier: WorkingMemoryTier = retention?.tier ?? "cold";

		db.prepare("UPDATE memories SET working_tier = ? WHERE id = ?").run(
			tier,
			row.id,
		);
		tiered[row.id] = tier;
	}

	// Mark memories with no access as archived
	db.prepare(
		"UPDATE memories SET working_tier = 'archived' WHERE is_latest = 1 AND last_accessed IS NULL AND workspace = ?",
	).run(workspace);

	return tiered;
}
