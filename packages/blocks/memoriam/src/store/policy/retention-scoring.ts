/** Computes a decay/reinforcement-weighted retention score per memory and
 * derives its working-memory tier from that score. */

import type { Database } from "bun:sqlite";
import type {
	DecayConfigInput,
	RetentionScore,
	WorkingMemoryTier,
} from "../../types.ts";
import { getAccessStats } from "../access-tracker.js";
import { get, rowToMemory } from "../models/memories.js";

function resolveDecayConfig(input?: DecayConfigInput): {
	lambda: number;
	sigma: number;
	tierThresholds: { hot: number; warm: number; cold: number };
} {
	const tierThresholds = {
		hot: input?.tierThresholds?.hot ?? 0.7,
		warm: input?.tierThresholds?.warm ?? 0.4,
		cold: input?.tierThresholds?.cold ?? 0.15,
	};
	return {
		lambda: input?.lambda ?? 0.01,
		sigma: input?.sigma ?? 0.3,
		tierThresholds,
	};
}

export function computeRetentionScore(
	db: Database,
	getWorkspace: () => string,
	id: string,
	config: DecayConfigInput = {},
): RetentionScore | null {
	const memory = get(db, getWorkspace, id);
	if (!memory) return null;

	const resolved = resolveDecayConfig(config);
	const now = Date.now();

	// Time decay
	const deltaT =
		(now - new Date(memory.createdAt).getTime()) / (1000 * 60 * 60 * 24);
	const temporalDecay = Math.exp(-resolved.lambda * deltaT);

	// Reinforcement from access recency: peaks at sigma right when accessed
	// and decays smoothly, rather than 1/x (which is undefined at 0 and
	// blows up for any access within the same second).
	const accessStats = getAccessStats(db, getWorkspace, id);
	let reinforcementBoost = 0;
	if (accessStats?.lastAccessed) {
		const daysSinceAccess = Math.max(
			0,
			(now - new Date(accessStats.lastAccessed).getTime()) /
				(1000 * 60 * 60 * 24),
		);
		reinforcementBoost = resolved.sigma * Math.exp(-daysSinceAccess);
	}

	// Salience from memory type and access count
	const typeWeights: Record<string, number> = {
		architecture: 0.9,
		bug: 0.7,
		pattern: 0.8,
		preference: 0.85,
		workflow: 0.6,
		fact: 0.5,
	};
	const baseSalience = typeWeights[memory.type] || 0.5;
	const accessBonus = Math.min(0.2, (accessStats?.accessCount || 0) * 0.02);
	const salience = Math.min(1, baseSalience + accessBonus);

	// Final retention score
	const score = Math.min(1, salience * temporalDecay + reinforcementBoost);

	// Determine tier
	let tier: WorkingMemoryTier = "cold";
	if (score >= resolved.tierThresholds.hot) tier = "hot";
	else if (score >= resolved.tierThresholds.warm) tier = "warm";

	return {
		id: memory.id,
		score,
		decayFactor: temporalDecay,
		reinforcementBoost,
		tier,
		type: memory.type,
		strength: memory.strength,
	};
}

export function rescoreAll(
	db: Database,
	getWorkspace: () => string,
	config: DecayConfigInput = {},
): RetentionScore[] {
	const allMemories = db
		.prepare(`SELECT * FROM memories WHERE is_latest = 1 AND workspace = ?`)
		.all(getWorkspace()) as any[];
	const scores: RetentionScore[] = [];

	for (const row of allMemories) {
		const memory = rowToMemory(row);
		const score = computeRetentionScore(db, getWorkspace, memory.id, config);
		if (score) scores.push(score);
	}

	scores.sort((a, b) => b.score - a.score);
	return scores;
}

export function listByRetentionScore(
	db: Database,
	getWorkspace: () => string,
	config: DecayConfigInput = {},
	limit: number = 50,
): RetentionScore[] {
	return rescoreAll(db, getWorkspace, config).slice(0, limit);
}
export type { RetentionScore } from "../../types.ts";
