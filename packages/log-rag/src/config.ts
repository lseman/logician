// ── RAG Configuration ────────────────────────────────────────────────────────
// Centralized configuration for all RAG components.

import { DEFAULT_CHUNKING_CONFIG, DEFAULT_CONTEXT_WINDOW } from "./types.ts";

export { DEFAULT_CHUNKING_CONFIG, DEFAULT_CONTEXT_WINDOW };

/** Embedding model options. */
export enum EmbeddingModel {
	/** BGE-M3 — best quality (568M params, 1024-dim). SOTA on MTEP. */
	BGE_M3 = "BAAI/bge-m3",
	/** GTE-large — strong quality, reasonable speed (335M params, 1024-dim). */
	GTE_LARGE = "thenlper/gte-large",
	/** Nomic-embed-text — fast, good quality (137M params, 768-dim). */
	NOMIC_EMBED_V2 = "nomic-ai/nomic-embed-text-v1.5",
	/** MiniLM-L6 — fast baseline (22M params, 384-dim). Current default. */
	MINILM_L6 = "Xenova/all-MiniLM-L6-v2",
}

/** Model dimension mapping. */
export const EMBEDDING_DIMENSIONS: Record<EmbeddingModel, number> = {
	[EmbeddingModel.BGE_M3]: 1024,
	[EmbeddingModel.GTE_LARGE]: 1024,
	[EmbeddingModel.NOMIC_EMBED_V2]: 768,
	[EmbeddingModel.MINILM_L6]: 384,
};

/** Cross-encoder reranking models. */
export enum RerankingModel {
	/** BGE-reranker-large — best quality reranker. */
	BGE_RERANKER_LARGE = "BAAI/bge-reranker-v2-m3",
	/** BGE-reranker-base — good quality, faster. */
	BGE_RERANKER_BASE = "BAAI/bge-reranker-base",
}

/** Reranker dimension mapping. */
export const RERANKER_MODEL_MAP: Record<RerankingModel, string> = {
	[RerankingModel.BGE_RERANKER_LARGE]: "BAAI/bge-reranker-v2-m3",
	[RerankingModel.BGE_RERANKER_BASE]: "BAAI/bge-reranker-base",
};

/** Hybrid search fusion parameters. */
export interface HybridConfig {
	/** Weight for dense (vector) component. */
	denseWeight: number;
	/** Weight for sparse (BM25) component. */
	sparseWeight: number;
	/** Number of candidate chunks to retrieve before reranking. */
	candidatesBeforeRerank: number;
	/** Number of results to return after reranking. */
	resultsAfterRerank: number;
}

export const DEFAULT_HYBRID_CONFIG: HybridConfig = {
	denseWeight: 0.6,
	sparseWeight: 0.4,
	candidatesBeforeRerank: 50,
	resultsAfterRerank: 5,
};

/** RAG system configuration. */
export interface RAGConfig {
	/** Embedding model to use. */
	embeddingModel: EmbeddingModel;
	/** Chunking configuration. */
	chunking: typeof DEFAULT_CHUNKING_CONFIG;
	/** Hybrid search configuration. */
	hybrid: typeof DEFAULT_HYBRID_CONFIG;
	/** Whether to use reranking. */
	enableReranking: boolean;
	/** Reranking model (if enabled). */
	rerankerModel?: RerankingModel;
	/** Python path for extraction subprocess. */
	pythonPath?: string;
	/** Python extraction script path. */
	scriptPath?: string;
	/** Context window config. */
	contextWindow?: typeof DEFAULT_CONTEXT_WINDOW;
}

/** Default configuration. */
export const DEFAULT_RAG_CONFIG: RAGConfig = {
	embeddingModel: EmbeddingModel.BGE_M3,
	chunking: DEFAULT_CHUNKING_CONFIG,
	hybrid: DEFAULT_HYBRID_CONFIG,
	enableReranking: true,
	rerankerModel: RerankingModel.BGE_RERANKER_BASE,
	contextWindow: DEFAULT_CONTEXT_WINDOW,
};

/** Create a lightweight config (no reranking, faster). */
export function lightConfig(): RAGConfig {
	return {
		...DEFAULT_RAG_CONFIG,
		embeddingModel: EmbeddingModel.NOMIC_EMBED_V2,
		enableReranking: false,
	};
}

/** Create a maximum-quality config. */
export function heavyConfig(): RAGConfig {
	return {
		...DEFAULT_RAG_CONFIG,
		enableReranking: true,
		rerankerModel: RerankingModel.BGE_RERANKER_LARGE,
		chunking: {
			...DEFAULT_RAG_CONFIG.chunking,
			chunkSize: 384,
			overlap: 96,
		},
		hybrid: {
			...DEFAULT_RAG_CONFIG.hybrid,
			candidatesBeforeRerank: 80,
		},
	};
}
