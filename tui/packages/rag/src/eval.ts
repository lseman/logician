// ── RAG Evaluation ───────────────────────────────────────────────────────────
// Metrics: Precision@K, Recall, MRR, NDCG@K, faithfulness.
// Evaluates retrieval quality against ground-truth relevance judgments.

import type { EvalResult, EvalSummary, SearchHit } from "./types.ts";

// ── Retrieval Metrics ────────────────────────────────────────────────────────

/**
 * Precision@K: fraction of retrieved items that are relevant.
 */
export function precisionAtK(
	retrieved: string[],
	relevant: string[],
	k: number,
): number {
	const kItems = retrieved.slice(0, k);
	const relevantSet = new Set(relevant);
	const hits = kItems.filter(id => relevantSet.has(id)).length;
	return kItems.length > 0 ? hits / kItems.length : 0;
}

/**
 * Recall: fraction of all relevant items that were retrieved.
 */
export function recall(retrieved: string[], relevant: string[]): number {
	const relevantSet = new Set(relevant);
	const hits = retrieved.filter(id => relevantSet.has(id)).length;
	return relevant.length > 0 ? hits / relevant.length : 0;
}

/**
 * Mean Reciprocal Rank (MRR): 1/rank of first relevant item.
 */
export function mrr(retrieved: string[], relevant: string[]): number {
	const relevantSet = new Set(relevant);
	for (let i = 0; i < retrieved.length; i++) {
		if (relevantSet.has(retrieved[i])) {
			return 1 / (i + 1);
		}
	}
	return 0;
}

/**
 * Normalized Discounted Cumulative Gain at K (NDCG@K).
 * Approximate: assumes all retrieved items have equal relevance.
 */
export function ndcgAtK(
	retrieved: string[],
	relevant: string[],
	k: number,
): number {
	const relevantSet = new Set(relevant);
	const dcg = retrieved.slice(0, k).reduce((sum, id, rank) => {
		const rel = relevantSet.has(id) ? 1 : 0;
		return sum + rel / Math.log2(rank + 2);
	}, 0);

	// Ideal DCG: all relevant items at top positions
	const idealHits = Math.min(relevant.length, k);
	const idcg =
		idealHits > 0
			? Array.from(
					{ length: idealHits },
					(_, i) => 1 / Math.log2(i + 2),
				).reduce((a, b) => a + b, 0)
			: 1;

	return dcg / idcg;
}

// ── Evaluation Runner ────────────────────────────────────────────────────────

/**
 * Evaluate retrieval system against ground truth.
 *
 * @param testQueries - Array of { query, expectedChunkIds }
 * @param searchFn - Function that searches and returns hits
 * @param options - Configuration
 */
export async function evaluateRetrieval(
	testQueries: Array<{
		query: string;
		expectedIds: string[];
	}>,
	searchFn: (query: string, topK: number) => Promise<SearchHit[]>,
	options?: {
		topK?: number;
		/** Number of candidates to retrieve for evaluation. */
		candidates?: number;
	},
): Promise<EvalSummary> {
	const { topK = 10, candidates = 50 } = options ?? {};

	const results: EvalResult[] = [];
	let totalPrecision = 0;
	let totalRecall = 0;
	let totalMRR = 0;
	let totalNDCG = 0;
	let retrievalMs = 0;

	for (const test of testQueries) {
		const start = performance.now();

		// Run retrieval
		const hits = await searchFn(test.query, candidates);
		retrievalMs += performance.now() - start;

		const retrievedIds = hits.map(h => h.chunk.id);

		// Compute metrics
		const p = precisionAtK(retrievedIds, test.expectedIds, topK);
		const r = recall(retrievedIds, test.expectedIds);
		const mrrScore = mrr(retrievedIds, test.expectedIds);
		const ndcg = ndcgAtK(retrievedIds, test.expectedIds, topK);

		totalPrecision += p;
		totalRecall += r;
		totalMRR += mrrScore;
		totalNDCG += ndcg;

		results.push({
			query: test.query,
			expectedIds: test.expectedIds,
			retrievedIds,
			precision: p,
			recall: r,
			mrr: mrrScore,
			nDCG: ndcg,
		});
	}

	return {
		queryCount: testQueries.length,
		avgPrecision: totalPrecision / testQueries.length,
		avgRecall: totalRecall / testQueries.length,
		avgMRR: totalMRR / testQueries.length,
		avgNDCG: totalNDCG / testQueries.length,
		results,
		latency: {
			retrieval_ms: retrievalMs,
		},
	};
}

// ── Comparison Benchmark ──────────────────────────────────────────────────────

/**
 * Compare two retrieval configurations on the same test set.
 * Returns a diff showing which is better per metric.
 */
export function compareEvaluations(a: EvalSummary, b: EvalSummary): string {
	const metrics = [
		{ key: "avgPrecision", label: "Precision@K" },
		{ key: "avgRecall", label: "Recall" },
		{ key: "avgMRR", label: "MRR" },
		{ key: "avgNDCG", label: "NDCG@K" },
	] as const;

	let output = "=== RAG Evaluation Comparison ===\n\n";

	for (const m of metrics) {
		const valA = a[m.key];
		const valB = b[m.key];
		const diff = valB - valA;
		const delta = Math.abs(valA) > 0 ? ((diff / valA) * 100).toFixed(1) : "N/A";
		const arrow = diff > 0 ? "↑" : diff < 0 ? "↓" : "→";
		output += `${m.label}: A=${valA.toFixed(4)} B=${valB.toFixed(4)} (${arrow}${delta}%)\n`;
	}

	// Per-query comparison
	const worseQueries: string[] = [];
	const betterQueries: string[] = [];

	for (let i = 0; i < a.results.length; i++) {
		const aScore = a.results[i].mrr;
		const bScore = b.results[i]?.mrr ?? 0;
		if (bScore > aScore)
			betterQueries.push(`"${a.results[i].query.slice(0, 50)}..."`);
		else if (aScore > bScore)
			worseQueries.push(`"${a.results[i].query.slice(0, 50)}..."`);
	}

	if (betterQueries.length) {
		output += `\nBetter with B: ${betterQueries.join(", ")}\n`;
	}
	if (worseQueries.length) {
		output += `\nBetter with A: ${worseQueries.join(", ")}\n`;
	}

	return output;
}

// ── Utility: Create synthetic test queries ──────────────────────────────────────

/**
 * Create synthetic test queries from indexed documents.
 * Uses document metadata to generate retrieval tasks.
 */
export function createSyntheticTests(
	docs: Array<{
		id: string;
		filename: string;
		chunks: Array<{ id: string; text: string }>;
		meta: Record<string, unknown>;
	}>,
	options?: { queriesPerDoc?: number },
): Array<{ query: string; expectedIds: string[] }> {
	const { queriesPerDoc = 2 } = options ?? {};
	const tests: Array<{ query: string; expectedIds: string[] }> = [];

	for (const doc of docs) {
		for (let i = 0; i < queriesPerDoc; i++) {
			// Generate query from first few sentences of first chunk
			const chunk = doc.chunks[0];
			if (!chunk) continue;

			const sentences = chunk.text
				.split(/[.!?]+\s+/)
				.filter(s => s.trim().length > 10);
			const query =
				sentences.slice(0, 2).join(". ").replace(/\. $/, "") ||
				chunk.text.slice(0, 100);

			tests.push({
				query: query.toLowerCase(),
				expectedIds: doc.chunks.map(c => c.id).slice(0, 5),
			});
		}
	}

	return tests;
}

// ── Summary Report ──────────────────────────────────────────────────────────────

export function summaryReport(summary: EvalSummary): string {
	let report = `# RAG Evaluation Report\n\n`;
	report += `## Overview\n\n`;
	report += `- **Queries tested:** ${summary.queryCount}\n`;
	report += `- **Avg Precision@K:** ${summary.avgPrecision.toFixed(4)}\n`;
	report += `- **Avg Recall:** ${summary.avgRecall.toFixed(4)}\n`;
	report += `- **Avg MRR:** ${summary.avgMRR.toFixed(4)}\n`;
	report += `- **Avg NDCG@K:** ${summary.avgNDCG.toFixed(4)}\n`;

	if (summary.latency?.retrieval_ms) {
		report += `- **Total retrieval time:** ${summary.latency.retrieval_ms.toFixed(0)}ms\n`;
	}

	report += `\n## Per-Query Results\n\n`;
	report += `| Query | P@K | Recall | MRR | NDCG |
|---|---|---|---|---|`;

	for (const r of summary.results) {
		const q = r.query.length > 40 ? `${r.query.slice(0, 40)}...` : r.query;
		report += `\n| ${q} | ${r.precision.toFixed(3)} | ${r.recall.toFixed(3)} | ${r.mrr.toFixed(3)} | ${r.nDCG.toFixed(3)} |`;
	}

	report += `\n\n## Interpretation\n\n`;
	report += `- **Precision@K > 0.5**: Good — most top results are relevant\n`;
	report += `- **Recall > 0.7**: Good — most relevant items found\n`;
	report += `- **MRR > 0.6**: Good — first relevant item is near top\n`;
	report += `- **NDCG > 0.7**: Good — ranking quality is high\n`;

	return report;
}
