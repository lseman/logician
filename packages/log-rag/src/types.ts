// ── RAG Core Types ───────────────────────────────────────────────────────────

/** Text chunk after splitting. */
export interface RAGChunk {
	id: string;
	text: string;
	metadata: Record<string, unknown>;
	documentId?: string;
	vector?: number[];
	/** Parent chunk ID if this is a child chunk (parent-child retrieval). */
	parentId?: string;
	/** Token/character count for window management. */
	approxTokens?: number;
}

/** A parent context block that wraps child chunks. */
export interface ParentContext {
	id: string;
	text: string;
	childIds: string[];
	documentId?: string;
}

/** Document representation after extraction. */
export interface ExtractedDocument {
	id: string;
	filename: string;
	content: string;
	meta: {
		title?: string;
		author?: string;
		format: string;
		pageCount?: number;
		/** BM25-style term frequency index (for hybrid search). */
		termFrequencies?: Record<string, number>;
	};
	chunks: RAGChunk[];
	extractedAt: number;
}

/** Result of a similarity search. */
export interface SearchHit {
	chunk: RAGChunk;
	score: number;
	/** Dense vector similarity component. */
	denseScore?: number;
	/** BM25 component. */
	sparseScore?: number;
	/** Reranked cross-encoder score. */
	rerankScore?: number;
	/** Query variants/routes that independently retrieved this chunk. */
	retrievalRoutes?: string[];
}

/** Observable retrieval quality signals for abstention and debugging. */
export interface RetrievalDiagnostics {
	queryVariants: string[];
	candidateCount: number;
	selectedCount: number;
	/** Fraction of selected hits supported by more than one retrieval route. */
	routeAgreement: number;
	/** 0..1 heuristic confidence; intended for calibration, not truth claims. */
	confidence: number;
	insufficientEvidence: boolean;
	reasons: string[];
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
	/** Search with metadata filters. */
	searchByVector(
		vector: number[],
		topK: number,
		options?: { filter?: Record<string, string | number | boolean> },
	): Promise<SearchHit[]>;
	clear(): Promise<void>;
	count(): Promise<number>;
	documentIds(): Promise<string[]>;
	/** Get BM25 term frequencies for a document. */
	getTermFrequencies(docId: string): Record<string, number> | null;
	/** Store BM25 term frequencies for a document. */
	setTermFrequencies(docId: string, tf: Record<string, number>): Promise<void>;
}

/** Store capabilities required by the end-to-end RAG pipeline. */
export interface RAGStore extends IVectorStore {
	searchHybrid?(
		queryText: string,
		queryVector: number[],
		topK?: number,
		options?: {
			filter?: Record<string, string | number | boolean>;
			denseWeight?: number;
			sparseWeight?: number;
		},
	): Promise<SearchHit[]>;
	deleteDocument(documentId: string): Promise<void>;
	getChunksByDocument(documentId: string): RAGChunk[];
	close?(): void;
}

/** Pipeline step interface. */
export interface PipelineStep<TIn, TOut> {
	name: string;
	process(input: TIn): Promise<TOut>;
}

// ── Chunking ─────────────────────────────────────────────────────────────────

/** Chunking strategy selection. */
export enum ChunkStrategy {
	/** Recursive splitting by heading with configurable chunk size. */
	Recursive = "recursive",
	/** Semantic splitting (boundary detection via embedding discontinuity). */
	Semantic = "semantic",
	/** Fixed-size with overlap. */
	FixedSize = "fixed",
}

/** Chunking configuration. */
export interface ChunkingConfig {
	strategy: ChunkStrategy;
	/** Target chunk size in characters. */
	chunkSize?: number;
	/** Overlap between chunks in characters. */
	overlap?: number;
	/** Minimum chunk size to avoid tiny chunks. */
	minChunkSize?: number;
	/** Maximum depth of recursive splitting. */
	maxDepth?: number;
	/** Separator patterns for splitting. */
	separators?: string[];
}

/** Default recursive chunking config. */
export const DEFAULT_CHUNKING_CONFIG: ChunkingConfig = {
	strategy: ChunkStrategy.Recursive,
	chunkSize: 512,
	overlap: 128,
	minChunkSize: 64,
	maxDepth: 3,
	separators: ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
};

// ── Query ────────────────────────────────────────────────────────────────────

/** A query that has been rewritten and expanded. */
export interface RewrittenQuery {
	/** Original user query. */
	original: string;
	/** Rewritten query (e.g. improved phrasing). */
	rewritten: string;
	/** Expanded sub-queries for multi-hop retrieval. */
	expansions: string[];
	/** Combined query for retrieval. */
	combined: string;
}

// ── Reranking ────────────────────────────────────────────────────────────────

/** Reranking configuration. */
export interface RerankerConfig {
	modelId: string;
	/** Max pairs to score per call (batch size). */
	batchSize?: number;
}

/** Reranker interface. */
export interface IReranker {
	name: string;
	/** Rerank query+document pairs, return ranked results. */
	rerank(
		query: string,
		pairs: Array<{ chunk: RAGChunk; score: number }>,
	): Promise<Array<{ chunk: RAGChunk; score: number; rerankScore: number }>>;
}

// ── Context Management ───────────────────────────────────────────────────────

/** Context window configuration. */
export interface ContextWindowConfig {
	/** Max tokens allowed in final context. */
	maxTokens: number;
	/** Tokenizer function (chars → approximate tokens). */
	estimateTokens: (text: string) => number;
}

/** Default: ~4 chars per token, 128K max. */
export const DEFAULT_CONTEXT_WINDOW: ContextWindowConfig = {
	maxTokens: 128_000,
	estimateTokens: text => Math.ceil(text.length / 4),
};

// ── Metadata Filtering ───────────────────────────────────────────────────────

export interface MetadataFilter {
	field: string;
	operator: "eq" | "neq" | "in" | "gte" | "lte" | "contains";
	value: string | number | string[];
}

// ── Evaluation ───────────────────────────────────────────────────────────────

/** Single retrieval evaluation result. */
export interface EvalResult {
	query: string;
	/** Expected relevant chunk IDs. */
	expectedIds: string[];
	/** Retrieved chunk IDs. */
	retrievedIds: string[];
	/** Precision at K. */
	precision: number;
	/** Recall. */
	recall: number;
	/** MRR (mean reciprocal rank). */
	mrr: number;
	/** NDCG@K (approximate). */
	nDCG: number;
}

/** Evaluation summary. */
export interface EvalSummary {
	/** Number of test queries. */
	queryCount: number;
	/** Average precision. */
	avgPrecision: number;
	/** Average recall. */
	avgRecall: number;
	/** Average MRR. */
	avgMRR: number;
	/** Average NDCG. */
	avgNDCG: number;
	/** Per-query results. */
	results: EvalResult[];
	/** Latency breakdown. */
	latency?: {
		rewrite_ms?: number;
		chunking_ms?: number;
		embedding_ms?: number;
		retrieval_ms?: number;
		reranking_ms?: number;
	};
}
