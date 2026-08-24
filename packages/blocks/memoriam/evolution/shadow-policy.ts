import type { ContextBlock, ShadowMemoryPolicy } from "../src/types.js";

const FEATURE_COUNT = 6;

export function policyFeatures(candidate: {
	type: ContextBlock["type"];
	score: number;
	tokens: number;
	reasons: string[];
}): number[] {
	return [
		1,
		Math.tanh(candidate.score / 20),
		Math.min(1, candidate.tokens / 1_000),
		candidate.type === "claim" ? 1 : 0,
		candidate.type === "memory" ? 1 : 0,
		candidate.reasons.includes("dense") ? 1 : 0,
	];
}

export function shadowDecision(
	policy: ShadowMemoryPolicy,
	features: number[],
): { action: "inject" | "withhold"; score: number; policyVersion: number } {
	const score = features.reduce(
		(sum, feature, index) => sum + feature * (policy.weights[index] || 0),
		0,
	);
	return {
		action: score >= 0 ? "inject" : "withhold",
		score: Number(score.toFixed(4)),
		policyVersion: policy.version,
	};
}

/** Bounded online contextual update. It never controls selection in shadow mode. */
export function learnShadowPolicy(
	policy: ShadowMemoryPolicy,
	featureRows: number[][],
	reward: number,
): ShadowMemoryPolicy {
	if (!featureRows.length) return policy;
	const mean = Array.from(
		{ length: FEATURE_COUNT },
		(_, index) =>
			featureRows.reduce((sum, row) => sum + (row[index] || 0), 0) /
			featureRows.length,
	);
	const learningRate = Math.min(0.05, 1 / Math.sqrt(policy.samples + 1));
	return {
		...policy,
		version: policy.version + 1,
		weights: policy.weights.map((weight, index) =>
			Math.max(
				-2,
				Math.min(2, weight + learningRate * reward * (mean[index] || 0)),
			),
		),
		samples: policy.samples + 1,
		updatedAt: new Date().toISOString(),
	};
}

export function initialShadowPolicy(): ShadowMemoryPolicy {
	return {
		version: 1,
		mode: "shadow",
		weights: Array.from({ length: FEATURE_COUNT }, () => 0),
		samples: 0,
		updatedAt: new Date(0).toISOString(),
	};
}
