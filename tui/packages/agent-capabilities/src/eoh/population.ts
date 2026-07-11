// ── EoH population management ─────────────────────────────────────────────────
// Selection, ranking, fitness-proportionate sampling with rank weighting.

import type { Heuristic } from "./types.ts";

/**
 * Rank-based selection probability from EoH paper:
 *   p_i ∝ 1 / (rank_i + N)
 * where rank_i is 0-based (best = 0) and N = population size.
 * Returns selected heuristic indices.
 */
function rankWeights(population: Heuristic[]): number[] {
	const N = population.length;
	return population.map((_, i) => 1 / (i + N + 1));
}

function weightedSample(population: Heuristic[], weights: number[], count: number): Heuristic[] {
	const totalWeight = weights.reduce((a, b) => a + b, 0);
	const selected: Heuristic[] = [];
	const used = new Set<number>();

	for (let s = 0; s < count && used.size < population.length; s++) {
		let r = Math.random() * totalWeight;
		for (let i = 0; i < population.length; i++) {
			if (used.has(i)) continue;
			r -= weights[i];
			if (r <= 0) {
				selected.push(population[i]);
				used.add(i);
				break;
			}
		}
		// fallback: pick first unused
		if (selected.length <= s) {
			for (let i = 0; i < population.length; i++) {
				if (!used.has(i)) {
					selected.push(population[i]);
					used.add(i);
					break;
				}
			}
		}
	}
	return selected;
}

/** Sort population by descending fitness (best first). */
export function rankPopulation(population: Heuristic[]): Heuristic[] {
	return [...population].sort((a, b) => b.fitness - a.fitness);
}

/** Select p parents using rank-weighted sampling. Population must be sorted (best first). */
export function selectParents(sortedPopulation: Heuristic[], count: number): Heuristic[] {
	if (sortedPopulation.length === 0) return [];
	const count_ = Math.min(count, sortedPopulation.length);
	const weights = rankWeights(sortedPopulation);
	return weightedSample(sortedPopulation, weights, count_);
}

/** Select top-N heuristics from union of current population + candidates. */
export function selectNextGeneration(
	current: Heuristic[],
	candidates: Heuristic[],
	populationSize: number,
): Heuristic[] {
	const all = [...current, ...candidates];
	const ranked = rankPopulation(all);
	// Deduplicate by thought+code fingerprint
	const seen = new Set<string>();
	const unique: Heuristic[] = [];
	for (const h of ranked) {
		const key = h.thought.slice(0, 80) + h.code.slice(0, 80);
		if (!seen.has(key)) {
			seen.add(key);
			unique.push(h);
		}
	}
	return unique.slice(0, populationSize);
}

/** Stats for display. */
export function populationStats(population: Heuristic[]): {
	best: number;
	worst: number;
	mean: number;
	size: number;
} {
	if (population.length === 0) return { best: 0, worst: 0, mean: 0, size: 0 };
	const fitnesses = population.map((h) => h.fitness);
	const best = Math.max(...fitnesses);
	const worst = Math.min(...fitnesses);
	const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
	return { best, worst, mean, size: population.length };
}
