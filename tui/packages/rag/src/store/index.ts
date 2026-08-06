// ── Vector Store ─────────────────────────────────────────────────────────────
// In-memory vector store with cosine similarity search and persistence.

import { cosineSimilarity } from "../embedder.ts";
import type { IVectorStore, RAGChunk, SearchHit } from "../types.ts";

export { SQLiteVectorStore } from "./sqlite-store.ts";

/**
 * Simple in-memory vector store using pre-computed embeddings.
 * Chunks must have their `vector` field populated before calling add().
 */
export class MemoryVectorStore implements IVectorStore {
	private chunks: RAGChunk[] = [];

	constructor(dimension = 384) {
		this.dimension = dimension;
	}

	/** Add pre-embedded chunks to the store. */
	async add(chunks: RAGChunk[]): Promise<void> {
		for (const chunk of chunks) {
			if (!chunk.vector) {
				console.warn(`Skipping chunk ${chunk.id}: no vector`);
				continue;
			}
			this.chunks.push(chunk);
		}
	}

	/** Search by embedding vectors → topK most similar chunks. */
	async searchByVector(queryVec: number[], topK = 5): Promise<SearchHit[]> {
		const scored: Array<{ chunk: RAGChunk; score: number }> = this.chunks.map(
			chunk => ({
				chunk,
				score: cosineSimilarity(chunk.vector!, queryVec),
			}),
		);

		return scored.sort((a, b) => b.score - a.score).slice(0, topK);
	}

	/** Search by text → embeds then searches (caller must provide an embedder). */
	async search(_query: string, _topK = 5): Promise<SearchHit[]> {
		throw new Error(
			"MemoryVectorStore.search() requires embedding the query first. Use searchByVector() with a pre-computed vector.",
		);
	}

	async clear(): Promise<void> {
		this.chunks = [];
	}

	async count(): Promise<number> {
		return this.chunks.length;
	}

	async documentIds(): Promise<string[]> {
		const ids = new Set<string>();
		for (const c of this.chunks) {
			if (c.documentId) ids.add(c.documentId);
		}
		return Array.from(ids);
	}
}
