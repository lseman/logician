import type { MemoryStore } from "./store.ts";
import type { Observation, Reflection, Relevance } from "./types.ts";

export type MemorySearchMatch =
	| {
			kind: "observation";
			id: string;
			content: string;
			score: number;
			relevance: Relevance;
	  }
	| {
			kind: "reflection";
			id: string;
			content: string;
			score: number;
			supportingObservationIds: string[];
	  };

export interface MemorySearchOptions {
	limit?: number;
}

const RELEVANCE_SCORE: Record<Relevance, number> = {
	low: 0,
	medium: 0.1,
	high: 0.2,
	critical: 0.3,
};

export function searchMemory(
	query: string,
	observations: Observation[],
	reflections: Reflection[],
	options: MemorySearchOptions = {},
): MemorySearchMatch[] {
	const limit = normalizeLimit(options.limit);
	const normalizedQuery = normalize(query);
	const queryTerms = terms(normalizedQuery);
	const candidates: MemorySearchMatch[] = [
		...observations.map((observation) => ({
			kind: "observation" as const,
			id: observation.id,
			content: observation.content,
			score:
				textScore(normalizedQuery, queryTerms, observation.content) +
				RELEVANCE_SCORE[observation.relevance],
			relevance: observation.relevance,
		})),
		...reflections.map((reflection) => ({
			kind: "reflection" as const,
			id: reflection.id,
			content: reflection.content,
			score: textScore(normalizedQuery, queryTerms, reflection.content) + 0.15,
			supportingObservationIds: [...reflection.supportingObservationIds],
		})),
	];

	return candidates
		.filter(
			(candidate) =>
				normalizedQuery.length === 0 ||
				textScore(normalizedQuery, queryTerms, candidate.content) > 0,
		)
		.sort((a, b) => b.score - a.score)
		.slice(0, limit);
}

export function searchMemoryStore(
	store: MemoryStore,
	query: string,
	options?: MemorySearchOptions,
): MemorySearchMatch[] {
	return searchMemory(
		query,
		store.getActiveObservations(),
		store.getReflections(),
		options,
	);
}

export function formatMemoryContext(
	store: MemoryStore,
	query: string,
	options: MemorySearchOptions & { maxTokens?: number } = {},
): string {
	const matches = searchMemoryStore(store, query, { limit: options.limit ?? 8 });
	if (matches.length === 0) return "";
	const maxChars = Math.max(200, (options.maxTokens ?? 1_000) * 4);
	const lines = [
		"<observational-memory>",
		"Potentially relevant retained context. Treat it as fallible; use recall with an ID for supporting evidence.",
	];
	for (const match of matches) {
		const label =
			match.kind === "observation"
				? `observation:${match.relevance}`
				: "reflection";
		const line = `- [${match.id}] (${label}) ${match.content}`;
		if (lines.join("\n").length + line.length + 25 > maxChars) break;
		lines.push(line);
	}
	if (lines.length === 2) return "";
	lines.push("</observational-memory>");
	return lines.join("\n");
}

function textScore(
	normalizedQuery: string,
	queryTerms: Set<string>,
	content: string,
): number {
	if (!normalizedQuery) return 0;
	const normalizedContent = normalize(content);
	let score = normalizedContent.includes(normalizedQuery) ? 2 : 0;
	const contentTerms = terms(normalizedContent);
	for (const term of queryTerms) {
		if (contentTerms.has(term)) score += 1 / queryTerms.size;
	}
	return score;
}

function normalize(value: string): string {
	return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function terms(value: string): Set<string> {
	return new Set(value.match(/[\p{L}\p{N}_-]{2,}/gu) ?? []);
}

function normalizeLimit(value: number | undefined): number {
	if (value === undefined || !Number.isFinite(value)) return 8;
	return Math.min(20, Math.max(1, Math.floor(value)));
}
