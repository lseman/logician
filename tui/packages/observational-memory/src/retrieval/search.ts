import { KnowledgeGraphManager } from "../storage/knowledge-graph.ts";
import type { MemoryStore } from "../storage/store.ts";
import type { Observation, Reflection, Relevance } from "../types.ts";

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

const STOP_WORDS = new Set([
	"and",
	"are",
	"for",
	"from",
	"how",
	"that",
	"the",
	"this",
	"was",
	"what",
	"when",
	"where",
	"with",
]);

export function searchMemory(
	query: string,
	observations: Observation[],
	reflections: Reflection[],
	options: MemorySearchOptions = {},
): MemorySearchMatch[] {
	const limit = normalizeLimit(options.limit);
	const normalizedQuery = normalize(query);
	const queryTerms = terms(normalizedQuery);
	const newestTimestamp = observations.reduce(
		(latest, item) => Math.max(latest, Date.parse(item.timestamp) || 0),
		0,
	);
	const candidates: MemorySearchMatch[] = [
		...observations.map((observation) => ({
			kind: "observation" as const,
			id: observation.id,
			content: observation.content,
			score:
				textScore(normalizedQuery, queryTerms, observation.content) +
				RELEVANCE_SCORE[observation.relevance] +
				recencyScore(observation.timestamp, newestTimestamp),
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
				hasSearchMatch(normalizedQuery, queryTerms, candidate.content),
		)
		.sort((a, b) => b.score - a.score)
		.slice(0, limit);
}

export function searchMemoryStore(
	store: MemoryStore,
	query: string,
	options?: MemorySearchOptions,
): MemorySearchMatch[] {
	const limit = normalizeLimit(options?.limit);
	const active = store.getActiveObservations();
	const reflections = store.getReflections();
	const direct = searchMemory(query, active, reflections, { limit });
	const graphData = store.fold().knowledgeGraph;
	if (!graphData) return direct;
	const graph = new KnowledgeGraphManager(graphData);
	const byId = new Map(active.map((item) => [item.id, item]));
	const expanded = [...direct];
	const seen = new Set(direct.map((item) => item.id));
	for (const match of direct) {
		if (match.kind !== "reflection") continue;
		for (const node of graph.getRelatedNodes(match.id)) {
			const observation = byId.get(node.id);
			if (!observation || seen.has(observation.id)) continue;
			seen.add(observation.id);
			expanded.push({
				kind: "observation",
				id: observation.id,
				content: observation.content,
				score: Math.max(0, match.score - 0.05),
				relevance: observation.relevance,
			});
		}
	}
	return expanded.sort((a, b) => b.score - a.score).slice(0, limit);
}

export function formatMemoryContext(
	store: MemoryStore,
	query: string,
	options: MemorySearchOptions & { maxTokens?: number } = {},
): string {
	const matches = searchMemoryStore(store, query, {
		limit: options.limit ?? 8,
	});
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

function hasSearchMatch(
	normalizedQuery: string,
	queryTerms: Set<string>,
	content: string,
): boolean {
	const normalizedContent = normalize(content);
	if (normalizedContent.includes(normalizedQuery)) return true;
	const contentTerms = terms(normalizedContent);
	for (const term of queryTerms) if (contentTerms.has(term)) return true;
	return bigramSimilarity(normalizedQuery, normalizedContent) >= 0.55;
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
	score += bigramSimilarity(normalizedQuery, normalizedContent) * 0.5;
	return score;
}

function normalize(value: string): string {
	return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function terms(value: string): Set<string> {
	return new Set(
		(value.match(/[\p{L}\p{N}_-]{2,}/gu) ?? []).filter(
			(term) => !STOP_WORDS.has(term),
		),
	);
}

function recencyScore(timestamp: string, newestTimestamp: number): number {
	const parsed = Date.parse(timestamp);
	if (!Number.isFinite(parsed) || newestTimestamp <= 0) return 0;
	const ageDays = Math.max(0, newestTimestamp - parsed) / 86_400_000;
	return 0.15 / (1 + ageDays / 30);
}

function bigramSimilarity(left: string, right: string): number {
	const leftBigrams = bigrams(left);
	const rightBigrams = bigrams(right);
	if (leftBigrams.size === 0 || rightBigrams.size === 0) return 0;
	let overlap = 0;
	for (const gram of leftBigrams) if (rightBigrams.has(gram)) overlap++;
	return (2 * overlap) / (leftBigrams.size + rightBigrams.size);
}

function bigrams(value: string): Set<string> {
	const compact = value.replace(/\s+/g, " ");
	const result = new Set<string>();
	for (let index = 0; index < compact.length - 1; index++) {
		result.add(compact.slice(index, index + 2));
	}
	return result;
}

function normalizeLimit(value: number | undefined): number {
	if (value === undefined || !Number.isFinite(value)) return 8;
	return Math.min(20, Math.max(1, Math.floor(value)));
}
