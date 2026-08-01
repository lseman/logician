// ── Ingestion Pipeline ────────────────────────────────────────────────────────
// End-to-end: ingest document → extract via Python Docling → store in SQLite.

import path from "node:path";

import type { ExtractedDocument, SearchHit, RAGChunk } from "./types.ts";
import type { IEmbedder } from "./embedder.ts";
import { RAGPipeline } from "./pipeline/index.ts";
import { SQLiteVectorStore } from "./store/sqlite-store.ts";

export interface IngestionConfig {
	embedder: IEmbedder;
	dbName?: string;
	pythonPath?: string;
	scriptPath?: string;
}

/**
 * Full ingestion pipeline that persists to SQLite and supports re-ingestion.
 */
export class IngestionPipeline {
	private db: SQLiteVectorStore;
	private pipeline: RAGPipeline;

	constructor(projectDir: string, config: IngestionConfig) {
		this.db = new SQLiteVectorStore(projectDir, { dbName: config.dbName, dimension: config.embedder.dimension });

		this.pipeline = new RAGPipeline(
			{ embedder: config.embedder, vectorStore: this.db },
			{ pythonPath: config.pythonPath, scriptPath: config.scriptPath }
		);
	}

	/** Ingest a file (PDF, DOCX, etc.) via Docling → SQLite. */
	async ingestFile(filePath: string, docId?: string): Promise<ExtractedDocument> {
		const abs = path.resolve(filePath);
		return this.pipeline.indexFile(abs, docId);
	}

	/** Ingest raw text with a source identifier. */
	async ingestText(text: string, source: string, docId?: string): Promise<ExtractedDocument> {
		return this.pipeline.indexText(text, source, docId);
	}

	/** Search across all indexed documents. */
	async search(query: string, topK = 5): Promise<SearchHit[]> {
		return this.pipeline.search(query, topK);
	}

	/** List all document IDs in the store. */
	async listDocuments(): Promise<string[]> {
		return this.db.documentIds();
	}

	/** Count total chunks. */
	async countChunks(): Promise<number> {
		return this.db.count();
	}

	/** Get raw chunk by ID (for debugging). */
	getChunkById(chunkId: string): RAGChunk | null {
		return this.db.getChunkById(chunkId);
	}

	/** Remove a single document by ID. */
	async deleteDocument(docId: string): Promise<void> {
		await this.db.deleteDocument(docId);
	}

	/** Close the database connection. */
	close(): void {
		this.db.close();
	}
}
