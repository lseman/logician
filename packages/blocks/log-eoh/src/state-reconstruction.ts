/** Rebuilds EohState from the persisted .eoh/log.jsonl — used on
 * EohSession.reload() (e.g. after a restart or context reset). */

import * as fs from "node:fs";
import { reconstructEohState } from "./jsonl.ts";
import { sessionFilePath } from "./paths.ts";

interface EohRunResult {
	thought: string;
	code: string;
	fitness: number;
	generation: number;
	createdBy: string;
	parentIds: string[];
	status: "keep" | "discard" | "crash";
	description: string;
	timestamp: number;
	segment: number;
	asi?: Record<string, unknown>;
}

export interface EohState {
	results: EohRunResult[];
	bestDirection: "lower" | "higher";
	populationSize: number;
	maxGenerations: number;
	currentSegment: number;
	name: string | null;
}

export function createEohState(): EohState {
	return {
		results: [],
		bestDirection: "lower",
		populationSize: 10,
		maxGenerations: 0,
		currentSegment: 0,
		name: null,
	};
}

export function eohJsonlPath(dir: string): string {
	return sessionFilePath(dir, "log");
}

export function eohPromptPath(dir: string): string {
	return sessionFilePath(dir, "prompt");
}

export function reconstructState(cwd: string): EohState {
	const state = createEohState();

	const jsonlPath = eohJsonlPath(cwd);
	const hasPersistedLog = fs.existsSync(jsonlPath);

	try {
		if (hasPersistedLog) {
			const reconstructed = reconstructEohState(
				fs.readFileSync(jsonlPath, "utf-8"),
			);
			state.name = reconstructed.name;
			state.populationSize = reconstructed.populationSize;
			state.maxGenerations = reconstructed.maxGenerations;
			state.currentSegment = reconstructed.currentSegment;
			state.results = reconstructed.results.map(r => ({
				...r,
			}));
		}
	} catch {
		// Fall through
	}

	return state;
}
