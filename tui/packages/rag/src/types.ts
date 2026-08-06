// ── RAG Core Types ───────────────────────────────────────────────────────────

/** A single extracted text chunk ready for vectorization. */
export interface RAGChunk {
	id: string;
	text: string;
	metadata: Record<string, unknown>;
	documentId?: string;
	vector?: number[];
}

/** Document representation after extraction by Docling. */
export interface ExtractedDocument {
	id: string;
	filename: string;
	content: string;
	meta: { title?: string; author?: string; format: string; pageCount?: number };
	chunks: RAGChunk[];
	extractedAt: number;
}

/** Result of a similarity search. */
export interface SearchHit {
	chunk: RAGChunk;
	score: number;
}

/** Configuration for the vector store. */
export interface VectorStoreConfig {
	name: string;
	dimension: number;
	indexPath?: string;
	metric?: "cosine" | "euclidean" | "dot";
}

/** Interface for any vector storage backend. */
export interface IVectorStore {
	add(chunks: RAGChunk[]): Promise<void>;
	search(query: string, topK?: number): Promise<SearchHit[]>;
	searchByVector(vector: number[], topK?: number): Promise<SearchHit[]>;
	clear(): Promise<void>;
	count(): Promise<number>;
	documentIds(): Promise<string[]>;
}

/** Pipeline step interface. */
export interface PipelineStep<TIn, TOut> {
	name: string;
	process(input: TIn): Promise<TOut>;
}
