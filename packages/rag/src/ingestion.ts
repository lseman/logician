// ── Ingestion Pipeline ───────────────────────────────────────────────────────
// End-to-end: ingest document → extract via Python Docling → smart chunking →
// store in HybridVectorStore.

import path from "node:path";
import type { EmbeddingModel, RAGConfig } from "./config.ts";
import type { IEmbedder } from "./embedder.ts";
import { RAGPipeline } from "./pipeline/index.ts";
import { HybridVectorStore } from "./store/hybrid-store.ts";
import type { ExtractedDocument, RAGChunk, SearchHit } from "./types.ts";

export interface IngestionConfig {
	/** Embedding model to use. */
	embeddingModel?: EmbeddingModel;
	/** Embedder instance (if already initialized). */
	embedder?: IEmbedder;
	/** RAG configuration (overrides defaults). */
	config?: Partial<RAGConfig>;
	dbName?: string;
	pythonPath?: string;
	scriptPath?: string;
	/** Project directory for storage paths. */
	projectDir?: string;
}

/**
 * Full ingestion pipeline:
 * - Hybrid search (dense + BM25)
 * - Smart chunking (recursive, semantic, parent-child)
 * - Cross-encoder reranking (optional)
 * - Query rewriting
 */
export class IngestionPipeline {
	private store: HybridVectorStore;
	private pipeline: RAGPipeline;

	constructor(projectDir: string, config: IngestionConfig) {
		// Determine embedder
		const embedder = config.embedder;
		if (!embedder) {
			throw new Error(
				"embedder is required; provide IEmbedder instance or use IngestionPipeline with embedder option",
			);
		}

		const dimension = embedder.dimension;

		// Create hybrid vector store
		// HybridVectorStore implements IVectorStore
		this.store = new HybridVectorStore(projectDir, {
			dbName: config.dbName,
			dimension,
		});

		// Create pipeline
		this.pipeline = new RAGPipeline(
			{
				embedder,
				vectorStore: this.store,
				chunkingConfig: config.config?.chunking,
				enableReranking: config.config?.enableReranking,
				rerankerModel: config.config?.rerankerModel,
				contextWindow: config.config?.contextWindow,
				pythonPath: config.pythonPath,
				scriptPath: config.scriptPath,
			},
			{ pythonPath: config.pythonPath, scriptPath: config.scriptPath },
		);
	}

	/** Ingest a file (PDF, DOCX, etc.) via Docling → smart chunking → hybrid store. */
	async ingestFile(
		filePath: string,
		docId?: string,
	): Promise<ExtractedDocument> {
		const abs = path.resolve(filePath);
		return this.pipeline.indexFile(abs, docId);
	}

	/** Ingest raw text with smart chunking. */
	async ingestText(
		text: string,
		source: string,
		docId?: string,
	): Promise<ExtractedDocument> {
		return this.pipeline.indexText(text, source, docId);
	}

	/**
	 * Search: query rewrite → hybrid retrieval → rerank → context.
	 */
	async search(
		query: string,
		topK = 5,
		options?: {
			expand?: boolean;
			rewriteEndpoint?: string;
			denseWeight?: number;
			sparseWeight?: number;
		},
	): Promise<SearchHit[]> {
		return this.pipeline.search(query, topK, options);
	}

	/** Search with context assembly for LLM prompts. */
	async searchWithContext(
		query: string,
		topK = 5,
		options?: {
			expand?: boolean;
			assembleContext?: boolean;
			compress?: boolean;
		},
	): Promise<{ hits: SearchHit[]; context: string }> {
		return this.pipeline.searchWithContext(query, topK, options);
	}

	/** List all document IDs. */
	async listDocuments(): Promise<string[]> {
		return this.store.documentIds();
	}

	/** Count total chunks. */
	async countChunks(): Promise<number> {
		return this.store.count();
	}

	/** Get raw chunk by ID. */
	getChunkById(chunkId: string): RAGChunk | null {
		return this.store.getChunkById(chunkId);
	}

	/** Remove a document. */
	async deleteDocument(docId: string): Promise<void> {
		await this.store.deleteDocument(docId);
	}

	/** Clear all data. */
	async clear(): Promise<void> {
		await this.store.clear();
	}

	/** Close database connections. */
	close(): void {
		this.store.close();
	}
}
