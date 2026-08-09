// ── Query Rewriting & Expansion ──────────────────────────────────────────────
// Transforms user queries into better retrieval queries.
//
// Strategies:
// 1. Query expansion: add synonyms and related terms
// 2. Query decomposition: split multi-hop queries into sub-queries
// 3. Query rewriting: improve phrasing for better matching
//
// Without an LLM, we use dictionary-based expansion and heuristic decomposition.
// When an LLM API is configured, we use it for better rewriting.

import type { RewrittenQuery } from "./types.ts";

// ── Synonym dictionary ────────────────────────────────────────────────────────

const SYNONYM_MAP: Record<string, string[]> = {
	// ML / AI terms
	machine: ["ml", "ai", "artificial intelligence"],
	learning: ["training", "optimization"],
	neural: ["deep learning", "dnn", "nn"],
	deep: ["neural network", "dnn"],
	transformer: ["attention", "self-attention", "encoder", "decoder"],
	attention: ["transformer", "multi-head attention"],
	embedding: ["word embedding", "token embedding", "vector representation"],
	retrieval: ["search", "retrieving", "information retrieval"],
	ranking: ["re-ranking", "rerank", "relevance ranking"],
	classification: ["categorization", "labeling", "prediction"],
	clustering: ["grouping", "unsupervised learning"],
	optimization: ["training", "minimization", "maximization"],
	loss: ["cost function", "objective function", "error function"],
	inference: ["prediction", "generation", "forward pass"],
	training: ["fine-tuning", "pre-training", "supervised learning"],
	generalization: ["robustness", "transfer", "adaptation"],
	// General academic terms
	paper: ["article", "publication", "work", "study"],
	method: ["approach", "technique", "framework", "algorithm"],
	result: ["finding", "outcome", "observation", "evidence"],
	performance: ["effectiveness", "accuracy", "quality", "metrics"],
	state: ["state-of-the-art", "sota", "current best"],
	baseline: ["comparison", "reference", "standard"],
	dataset: ["corpus", "benchmark", "evaluation set", "data"],
	model: ["architecture", "system", "method", "framework"],
	// Domain-independent
	used: ["utilized", "employed", "applied", "proposed"],
	framework: ["system", "architecture", "methodology"],
	approach: ["method", "technique", "strategy"],
	problem: ["task", "challenge", "issue", "difficulty"],
	solution: ["approach", "method", "technique"],
	effect: ["impact", "influence", "effectiveness"],
	improve: ["enhance", "boost", "increase", "achieve better"],
};

/** Expand a single query with synonyms. */
export function expandQuery(query: string): string[] {
	const terms = query.toLowerCase().split(/\s+/);
	const expansions: string[] = [];

	for (const term of terms) {
		const synonyms = SYNONYM_MAP[term];
		if (synonyms) {
			for (const syn of synonyms) {
				if (syn !== term) {
					expansions.push(syn);
				}
			}
		}
	}

	// Generate expanded queries
	const expanded: string[] = [];
	if (expansions.length > 0) {
		// Original query
		expanded.push(query);
		// Each synonym as a separate query
		for (const syn of expansions) {
			expanded.push(syn);
		}
		// Combined expansion (top 3 synonyms + original key terms)
		const keyTerms = terms.filter(
			(t) => !SYNONYM_MAP[t] || SYNONYM_MAP[t].length === 0,
		);
		if (keyTerms.length > 0 && expansions.length > 0) {
			expanded.push(
				[...keyTerms, expansions.slice(0, 2)].join(" "),
			);
		}
	}

	return expanded;
}

/**
 * Rewrite a query for better retrieval.
 * Without an LLM, uses basic expansion and rephrasing.
 * With an LLM API, sends a structured prompt for high-quality rewriting.
 */
export async function rewriteQuery(
	query: string,
	options?: {
		/** LLM API endpoint for rewriting (optional). */
		rewriteEndpoint?: string;
		/** LLM API key (optional). */
		rewriteApiKey?: string;
		/** Number of expansions to generate. */
		nExpansions?: number;
	},
): Promise<RewrittenQuery> {
	const nExpansions = options?.nExpansions ?? 3;

	// Expand with synonyms
	const expansions = expandQuery(query);

	// Take top N unique expansions
	const uniqueExpansions = [...new Set(expansions)].slice(0, nExpansions);

	// If no synonym expansions, still return the original query
	if (uniqueExpansions.length <= 1) {
		return {
			original: query,
			rewritten: query,
			expansions: [],
			combined: query,
		};
	}

	// Combine all expansions for broader retrieval
	const combined = uniqueExpansions.join(" OR ");

	return {
		original: query,
		rewritten: uniqueExpansions[0],
		expansions: uniqueExpansions.slice(1),
		combined,
	};
}

/**
 * Multi-query retrieval: expand the query into sub-queries and search each.
 * Returns a set of hits from all sub-queries, deduplicated.
 */
export async function multiQuerySearch(
	query: string,
	searchFn: (q: string, k: number) => Promise<Array<{ chunk: { id: string }; score: number }>>,
	options?: {
		nSubQueries?: number;
		perQueryK?: number;
	},
): Promise<Map<string, number>> {
	const { nSubQueries = 3, perQueryK = 10 } = options ?? {};

	const rewritten = await rewriteQuery(query, { nExpansions: nSubQueries });

	const hitScores = new Map<string, number>();

	// Search with each sub-query
	const queriesToSearch = [
		rewritten.original,
		rewritten.rewritten,
		...rewritten.expansions,
	];

	for (const q of queriesToSearch) {
		const hits = await searchFn(q, perQueryK);
		for (const hit of hits) {
			const existing = hitScores.get(hit.chunk.id) ?? 0;
			hitScores.set(hit.chunk.id, Math.max(existing, hit.score));
		}
	}

	return hitScores;
}

// ── LLM-powered rewriting ─────────────────────────────────────────────────────

/**
 * Rewrite a query using an LLM API for higher-quality expansion.
 * Sends a structured prompt and parses the response.
 */
export async function llmRewriteQuery(
	query: string,
	options?: {
		endpoint: string;
		apiKey?: string;
		model?: string;
		nExpansions?: number;
	},
): Promise<RewrittenQuery> {
	const {
		endpoint,
		apiKey,
		model = "gpt-3.5-turbo",
		nExpansions = 3,
	} = options ?? {};

	if (!endpoint) {
		// Fallback to rule-based rewriting
		return rewriteQuery(query, { nExpansions });
	}

	const prompt = `You are a query expansion expert for academic literature retrieval.

Given a user's search query, generate ${nExpansions} expanded versions that:
1. Use synonyms and related terminology
2. Broaden the scope slightly to catch more relevant papers
3. Include domain-specific terms that might appear in paper titles/abstracts

Original query: "${query}"

Return ONLY a JSON array of ${nExpansions} expanded queries. No explanation.

Example:
Input: "deep learning for time series forecasting"
Output: ["deep learning time series prediction", "neural network sequence forecasting", "transformer models temporal data"]

Output:`;

	const headers: Record<string, string> = {
		"Content-Type": "application/json",
	};
	if (apiKey) {
		headers["Authorization"] = `Bearer ${apiKey}`;
	}

	const response = await fetch(endpoint, {
		method: "POST",
		headers,
		body: JSON.stringify({
			model,
			messages: [{ role: "user", content: prompt }],
			max_tokens: 200,
			temperature: 0.7,
		}),
	});

	const data = await response.json();
	const content = data.choices?.[0]?.message?.content ?? "";

	// Parse JSON array from response
	try {
		const jsonMatch = content.match(/\[[\s\S]*\]/);
		if (jsonMatch) {
			const expansions = JSON.parse(jsonMatch[0]) as string[];
			return {
				original: query,
				rewritten: expansions[0] ?? query,
				expansions: expansions.slice(1, nExpansions),
				combined: [query, ...expansions].join(" "),
			};
		}
	} catch {
		// Parse failed — return original
	}

	return {
		original: query,
		rewritten: query,
		expansions: [],
		combined: query,
	};
}
