// ── Cross-Encoder Reranker ───────────────────────────────────────────────────
// Reranks retrieval candidates using a cross-encoder model (query + document
// scored jointly). Much more accurate than re-ranking with independent encoders.
// Uses @huggingface/transformers in-process.

import type { IReranker, RAGChunk, RerankerConfig } from "./types.ts";

type TextPairClassificationPipeline = (
	pairs: [string, string][],
	options: { batch_size?: number },
) => Promise<{ data: number[] | Float32Array; dims: number[] }>;

const DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base";

/**
 * Cross-encoder reranker running locally via @huggingface/transformers.
 *
 * The cross-encoder takes [query, document] pairs and outputs a relevance score
 * (0–1 after sigmoid). This is significantly more accurate than scoring query
 * and document independently, because it can model interactions between them.
 */
export class CrossEncoderReranker implements IReranker {
	readonly name = "cross-encoder";
	private modelId: string;
	private pipeline: TextPairClassificationPipeline | null = null;
	private loading: Promise<TextPairClassificationPipeline> | null = null;
	private batchSize: number;

	constructor(options?: { modelId?: string; batchSize?: number }) {
		this.modelId = options?.modelId ?? DEFAULT_RERANKER_MODEL;
		this.batchSize = options?.batchSize ?? 32;
	}

	private async getPipeline(): Promise<TextPairClassificationPipeline> {
		if (this.pipeline) return this.pipeline;
		if (!this.loading) {
			this.loading = import("@huggingface/transformers").then(async ({ pipeline }) => {
				const p = (await pipeline(
					"text-classification",
					this.modelId,
				)) as unknown as TextPairClassificationPipeline;
				this.pipeline = p;
				return p;
			});
		}
		return this.loading;
	}

	/**
	 * Rerank query + document pairs.
	 *
	 * @param query - The search query
	 * @param pairs - Array of { chunk, score } from initial retrieval
	 * @returns Ranked results with cross-encoder rerank scores
	 */
	async rerank(
		query: string,
		pairs: Array<{ chunk: RAGChunk; score: number }>,
	): Promise<Array<{ chunk: RAGChunk; score: number; rerankScore: number }>> {
		if (!pairs.length) return [];

		const pipeline = await this.getPipeline();

		// Build pairs for cross-encoder
		const pairTexts: Array<[string, string]> = pairs.map((p) => [query, p.chunk.text]);

		// Batch process
		const results: Array<{ chunk: RAGChunk; score: number; rerankScore: number }> = [];

		for (let i = 0; i < pairTexts.length; i += this.batchSize) {
			const batch = pairTexts.slice(i, i + this.batchSize);
		 const output = await pipeline(batch, { batch_size: batch.length });

			const scores =
				output.data instanceof Float32Array
					? output.data
					: new Float32Array(output.data);

			for (let j = 0; j < batch.length; j++) {
				const rawScore = scores[j];
				// Cross-encoder outputs raw logits; apply sigmoid for [0, 1]
				const rerankScore = 1 / (1 + Math.exp(-rawScore));
				const idx = i + j;
				results.push({
					chunk: pairs[idx].chunk,
					score: pairs[idx].score,
					rerankScore,
				});
			}
		}

		// Sort by rerank score (descending)
		results.sort((a, b) => b.rerankScore - a.rerankScore);
		return results;
	}
}

/**
 * Lightweight BM25-based reranker that doesn't need a model.
 * Useful when cross-encoder is too slow or unavailable.
 * Reranks by BM25 score of query vs candidate text.
 */
export class BM25Reranker implements IReranker {
	readonly name = "bm25";
	private batchSize = 1000;

	async rerank(
		query: string,
		pairs: Array<{ chunk: RAGChunk; score: number }>,
	): Promise<Array<{ chunk: RAGChunk; score: number; rerankScore: number }>> {
		if (!pairs.length) return [];

		const queryTerms = tokenize(query);
		if (!queryTerms.length) {
			// No query terms — return original ranking
			return pairs.map((p) => ({ ...p, rerankScore: p.score }));
		}

		const scored = pairs.map((p) => {
			const docTerms = tokenize(p.chunk.text);
			const bm25Score = scoreBM25(queryTerms, docTerms);
			return {
				chunk: p.chunk,
				score: p.score,
				rerankScore: bm25Score,
			};
		});

		scored.sort((a, b) => b.rerankScore - a.rerankScore);
		return scored;
	}
}

// ── Inline helpers ───────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
	"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
	"of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
	"have", "has", "had", "do", "does", "did", "will", "would", "could",
	"should", "may", "might", "shall", "can", "this", "that", "these",
	"those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
	"he", "she", "they", "them", "their", "what", "which", "who", "how",
	"when", "where", "why", "not", "no", "yes", "so", "if", "as",
]);

function tokenize(text: string): string[] {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9\s]/g, " ")
		.split(/\s+/)
		.filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

function scoreBM25(queryTerms: string[], docTerms: string[]): number {
	const k1 = 1.5;
	const b = 0.75;
	const avgLen = 200; // approximate
	const dl = docTerms.length || 1;

	// Compute IDF from corpus (approximate using term frequency as proxy)
	const termCounts = new Map<string, number>();
	for (const t of docTerms) {
		termCounts.set(t, (termCounts.get(t) ?? 0) + 1);
	}

	let score = 0;
	for (const qTerm of queryTerms) {
		const tf = (termCounts.get(qTerm) ?? 0) / dl;
		const idf = Math.log(200 / (termCounts.get(qTerm) ?? 1) + 1); // approximate IDF
		const denom = tf + k1 * (1 - b + b * dl / avgLen);
		if (denom > 0) {
			score += idf * (tf * (k1 + 1)) / denom;
		}
	}

	return score;
}
