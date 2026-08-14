import { test } from "node:test";
import assert from "node:assert";
import {
	rankPopulation,
	selectParents,
	selectNextGeneration,
	populationStats,
} from "../src/population.ts";

function makeHeuristic(id, fitness) {
	return { id, thought: `thought ${id}`, code: `code ${id}`, fitness, generation: 0, createdBy: "init", parentIds: [] };
}

test("rankPopulation sorts by descending fitness", () => {
	const pop = [makeHeuristic(1, 0.3), makeHeuristic(2, 0.9), makeHeuristic(3, 0.5)];
	const ranked = rankPopulation(pop);
	assert.equal(ranked[0].fitness, 0.9);
	assert.equal(ranked[1].fitness, 0.5);
	assert.equal(ranked[2].fitness, 0.3);
});

test("rankPopulation handles empty population", () => {
	assert.deepStrictEqual(rankPopulation([]), []);
});

test("selectParents selects correct number of parents", () => {
	const sorted = [
		makeHeuristic(1, 0.9),
		makeHeuristic(2, 0.7),
		makeHeuristic(3, 0.5),
		makeHeuristic(4, 0.3),
	];
	const parents = selectParents(sorted, 3);
	assert.equal(parents.length, 3);
});

test("selectParents handles fewer parents than population", () => {
	const sorted = [makeHeuristic(1, 0.9), makeHeuristic(2, 0.7)];
	const parents = selectParents(sorted, 5);
	assert.equal(parents.length, 2);
});

test("selectNextGeneration selects top-N from union", () => {
	const current = [makeHeuristic(1, 0.5)];
	const candidates = [makeHeuristic(2, 0.8), makeHeuristic(3, 0.3)];
	const next = selectNextGeneration(current, candidates, 2);
	assert.equal(next.length, 2);
	assert.equal(next[0].fitness, 0.8);
	assert.equal(next[1].fitness, 0.5);
});

test("selectNextGeneration deduplicates by fingerprint", () => {
	const current = [makeHeuristic(1, 0.5)];
	const candidates = [makeHeuristic(2, 0.9)];
	const next = selectNextGeneration(current, candidates, 2);
	assert.equal(next.length, 2);
});

test("populationStats returns correct stats", () => {
	const pop = [makeHeuristic(1, 0.3), makeHeuristic(2, 0.7), makeHeuristic(3, 0.5)];
	const stats = populationStats(pop);
	assert.equal(stats.best, 0.7);
	assert.equal(stats.worst, 0.3);
	assert.equal(stats.mean, 0.5);
	assert.equal(stats.size, 3);
});

test("populationStats handles empty population", () => {
	const stats = populationStats([]);
	assert.equal(stats.best, 0);
	assert.equal(stats.size, 0);
});
