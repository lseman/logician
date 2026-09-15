/**
 * JSONL persistence for EoH sessions.
 *
 * Each line in .eoh/log.jsonl is either:
 *   - A config entry: { type: "eoh_config", name, populationSize, ... }
 *   - A run entry:     { run, thought, code, fitness, generation, createdBy, ... }
 *
 * This mirrors the autoresearch jsonl.ts pattern — reconstructs full state
 * from the persisted log so the session can resume after a restart.
 */

import type {
	EohConfigEntry,
	EohOperator,
	EohRunEntry,
	ReconstructedEohState,
} from "./types.ts";

export type { ReconstructedEohState };

type JsonlEntry = Record<string, unknown>;

const DEFAULT_POPULATION_SIZE = 10;
const DEFAULT_MAX_GENERATIONS = 0;
const DEFAULT_DIRECTION = "lower" as const;

function isObjectRecord(value: unknown): value is JsonlEntry {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyLines(text: string): string[] {
	return text.split("\n").filter(Boolean);
}

function statusFrom(value: unknown): EohRunEntry["status"] {
	if (value === "discard") return "discard";
	if (value === "crash") return "crash";
	return "keep";
}

function operatorFrom(value: unknown): EohOperator {
	switch (value) {
		case "e1_diversity":
		case "e2_convergence":
		case "m1_improve":
		case "m2_tune":
		case "m3_simplify":
			return value;
		default:
			return "init";
	}
}

function directionFrom(value: unknown): ReconstructedEohState["bestDirection"] {
	return value === "higher" ? "higher" : DEFAULT_DIRECTION;
}

function reconstructedState(): ReconstructedEohState {
	return {
		name: null,
		populationSize: DEFAULT_POPULATION_SIZE,
		maxGenerations: DEFAULT_MAX_GENERATIONS,
		bestDirection: DEFAULT_DIRECTION,
		currentSegment: 0,
		results: [],
	};
}

function updateConfig(
	state: ReconstructedEohState,
	entry: EohConfigEntry,
): void {
	if (typeof entry.name === "string") state.name = entry.name;
	if (typeof entry.populationSize === "number" && entry.populationSize > 0) {
		state.populationSize = entry.populationSize;
	}
	if (typeof entry.maxGenerations === "number" && entry.maxGenerations >= 0) {
		state.maxGenerations = entry.maxGenerations;
	}
	state.bestDirection = directionFrom(entry.bestDirection);
}

function nextSegment(state: ReconstructedEohState, segment: number): number {
	if (state.results.length === 0) return segment;
	return segment + 1;
}

function runFrom(entry: JsonlEntry, segment: number): EohRunEntry {
	return {
		run: typeof entry.run === "number" ? entry.run : 0,
		thought: typeof entry.thought === "string" ? entry.thought : "",
		code: typeof entry.code === "string" ? entry.code : "",
		fitness: typeof entry.fitness === "number" ? entry.fitness : 0,
		generation: typeof entry.generation === "number" ? entry.generation : 0,
		createdBy: operatorFrom(entry.createdBy),
		parentIds: Array.isArray(entry.parentIds)
			? (entry.parentIds as string[])
			: [],
		status: statusFrom(entry.status),
		description: typeof entry.description === "string" ? entry.description : "",
		timestamp: typeof entry.timestamp === "number" ? entry.timestamp : 0,
		segment,
	};
}

export function parseJsonlEntry(line: string): JsonlEntry | null {
	try {
		const parsed = JSON.parse(line);
		return isObjectRecord(parsed) ? parsed : null;
	} catch {
		return null;
	}
}

export function isEohConfigEntry(entry: unknown): entry is EohConfigEntry {
	return isObjectRecord(entry) && entry.type === "eoh_config";
}

export function isEohRunEntry(entry: unknown): entry is EohRunEntry {
	return isObjectRecord(entry) && typeof entry.run === "number";
}

function firstConfigEntry(jsonlContent: string): EohConfigEntry | null {
	for (const line of nonEmptyLines(jsonlContent)) {
		const entry = parseJsonlEntry(line);
		if (isEohConfigEntry(entry)) return entry;
	}
	return null;
}

export function hasEohConfigHeader(jsonlContent: string): boolean {
	return firstConfigEntry(jsonlContent) !== null;
}

export function extractEohSessionName(jsonlContent: string): string {
	return firstConfigEntry(jsonlContent)?.name || "EoH";
}

/**
 * Reconstruct full EoH session state from persisted JSONL content.
 * Returns a ReconstructedEohState that can be used to resume a session.
 */
export function reconstructEohState(
	jsonlContent: string,
): ReconstructedEohState {
	const state = reconstructedState();
	let segment = 0;

	for (const line of nonEmptyLines(jsonlContent)) {
		const entry = parseJsonlEntry(line);
		if (!entry) continue;

		if (isEohConfigEntry(entry)) {
			updateConfig(state, entry);
			segment = nextSegment(state, segment);
			state.currentSegment = segment;
			continue;
		}

		if (!isEohRunEntry(entry)) continue;

		const run = runFrom(entry, segment);
		state.results.push(run);
	}

	return state;
}
