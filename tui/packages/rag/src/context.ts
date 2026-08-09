// ── Context Window Management ────────────────────────────────────────────────
// Manages how retrieved chunks are assembled into a prompt context.
//
// Strategies:
// - Keep all chunks (naive)
// - Prune by relevance score
// - Compress with LLM summarization
// - Maintain parent context for small chunks
// - Respect token budget

import type { ContextWindowConfig, ParentContext, RAGChunk, SearchHit } from "./types.ts";

const DEFAULT_CONTEXT: ContextWindowConfig = {
	maxTokens: 128_000,
	estimateTokens: (text) => Math.ceil(text.length / 4),
};

/**
 * Assemble retrieved chunks into a context window respecting token limits.
 * Prioritizes by score, maintains order, adds parent context for small chunks.
 */
export function assembleContext(
	hits: SearchHit[],
	config?: Partial<ContextWindowConfig>,
	parents?: Map<string, ParentContext>,
): string {
	const { maxTokens, estimateTokens } = { ...DEFAULT_CONTEXT, ...config };

	// Sort by score descending
	const sorted = [...hits].sort((a, b) => b.score - a.score);

	// Build context with token budget
	const contextParts: string[] = [];
	let totalTokens = 0;

	for (const hit of sorted) {
		const chunkTokens = estimateTokens(hit.chunk.text);

		// If this chunk alone exceeds budget, skip it
		if (chunkTokens > maxTokens) continue;

		// If adding this chunk exceeds budget, stop
		if (totalTokens + chunkTokens > maxTokens) {
			// Try partial fit
			const remainingTokens = maxTokens - totalTokens;
			if (remainingTokens < 32) break; // Too small to fit another chunk

			// Use parent context if available (better than truncating)
			const parentId = hit.chunk.parentId;
			if (parentId && parents?.has(parentId)) {
				const parent = parents.get(parentId)!;
				const parentTokens = estimateTokens(parent.text);
				if (totalTokens + parentTokens <= maxTokens) {
					contextParts.push(formatChunk(hit.chunk, parent.text));
					totalTokens += parentTokens;
					continue;
				}
			}

			// Skip remaining chunks
			break;
		}

		// Use parent context if this is a small chunk and parent is available
		const parentId = hit.chunk.parentId;
		if (parentId && parents?.has(parentId)) {
			const parent = parents.get(parentId)!;
			const parentTokens = estimateTokens(parent.text);
			if (totalTokens + parentTokens <= maxTokens) {
				contextParts.push(formatChunk(hit.chunk, parent.text));
				totalTokens += parentTokens;
				continue;
			}
		}

		contextParts.push(formatChunk(hit.chunk));
		totalTokens += chunkTokens;
	}

	return contextParts.join("\n\n---\n\n");
}

/**
 * Format a chunk with metadata for LLM consumption.
 */
function formatChunk(chunk: RAGChunk, parentText?: string): string {
	const source = String(chunk.metadata?.source ?? "unknown");
	const docId = chunk.documentId ?? "unknown";

	if (parentText && parentText !== chunk.text) {
		return `[Context from ${source} (doc: ${docId})]\n${parentText}\n\n[Retrieved excerpt]:\n${chunk.text}`;
	}

	return `[Context from ${source} (doc: ${docId})]\n${chunk.text}`;
}

/**
 * Compress context using a simple extraction summary.
 * Without an LLM, uses title-based extraction (first sentence of each chunk).
 */
export function compressContext(
	hits: SearchHit[],
	config?: Partial<ContextWindowConfig>,
): { compressed: string; originalTokens: number; compressedTokens: number } {
	const { maxTokens, estimateTokens } = { ...DEFAULT_CONTEXT, ...config };

	const sorted = [...hits].sort((a, b) => b.score - a.score);

	const summaries: string[] = [];
	let totalTokens = 0;

	for (const hit of sorted) {
		const text = hit.chunk.text;
		// Extract key sentence (first sentence or highest-scoring sentence)
		const sentences = text.split(/[.!?]+\s+/).filter((s) => s.trim().length > 20);
		const keySentence = sentences[0] ?? text.slice(0, 200);

		const summaryTokens = estimateTokens(keySentence);
		if (totalTokens + summaryTokens > maxTokens) break;

		summaries.push(keySentence);
		totalTokens += summaryTokens;
	}

	return {
		compressed: summaries.join("\n"),
		originalTokens: estimateTokens(hits.map((h) => h.chunk.text).join("\n")),
		compressedTokens: totalTokens,
	};
}

/**
 * Build a context summary with source attribution.
 */
export function buildContextSummary(
	hits: SearchHit[],
	config?: Partial<ContextWindowConfig>,
): {
	context: string;
	sources: Array<{ source: string; docId: string; relevance: number }>;
	tokens: number;
} {
	const { maxTokens, estimateTokens } = { ...DEFAULT_CONTEXT, ...config };

	const sorted = [...hits].sort((a, b) => b.score - a.score);
	const sources: Array<{ source: string; docId: string; relevance: number }> = [];
	const contextParts: string[] = [];
	let totalTokens = 0;

	for (const hit of sorted) {
		const source = String(hit.chunk.metadata?.source ?? "unknown");
		const docId = hit.chunk.documentId ?? "unknown";
		sources.push({ source, docId, relevance: hit.score });

		const chunkTokens = estimateTokens(hit.chunk.text);
		if (totalTokens + chunkTokens > maxTokens) break;

		contextParts.push(hit.chunk.text);
		totalTokens += chunkTokens;
	}

	return {
		context: contextParts.join("\n\n"),
		sources,
		tokens: totalTokens,
	};
}
