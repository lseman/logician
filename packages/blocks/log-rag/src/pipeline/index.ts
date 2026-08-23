// ── RAG Pipeline ─────────────────────────────────────────────────────────────
// Orchestrates: extract (Docling) → smart chunking → embed → store →
// hybrid search → rerank → context assembly.
// - Hybrid search (dense + BM25 via RRF)
// - Smart chunking (recursive, semantic, parent-child)
// - Cross-encoder reranking
// - Query rewriting/expansion
// - Context window management

import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { recursiveChunk, smartChunk } from "../chunking.ts";
import {
	DEFAULT_CHUNKING_CONFIG,
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_RAG_CONFIG,
	type RerankingModel,
} from "../config.ts";
import { assembleContext, compressContext } from "../context.ts";
import type { IEmbedder } from "../embedder.ts";
import { llmRewriteQuery, rewriteQuery } from "../query.ts";
import { CrossEncoderReranker } from "../reranker.ts";
import {
	diagnoseRetrieval,
	fuseRankedHits,
	selectDiverseHits,
} from "../retrieval.ts";
import type {
	ChunkingConfig,
	ContextWindowConfig,
	ExtractedDocument,
	IReranker,
	MetadataFilter,
	RAGChunk,
	RAGStore,
	RetrievalDiagnostics,
	RewrittenQuery,
	SearchHit,
} from "../types.ts";

// packages/rag/src/pipeline/index.ts -> repo root
const REPO_ROOT = path.resolve(
	fileURLToPath(new URL(".", import.meta.url)),
	"../../../..",
);
const DEFAULT_PYTHON_PATH = path.join(REPO_ROOT, ".venv", "bin", "python");
const DEFAULT_SCRIPT_PATH = path.join(
	REPO_ROOT,
	"rag-python",
	"src",
	"rag_extract",
	"cli.py",
);

const execFileAsync = promisify(execFile);

/** Parsed extraction result from Python Docling subprocess. */
interface ExtractedDocumentJSON {
	id: string;
	filename: string;
	content: string;
	meta: Record<string, unknown>;
	chunks: Array<{
		id: string;
		text: string;
		metadata?: Record<string, unknown>;
		document_id?: string;
	}>;
	extracted_at: number;
}

/** Pipeline configuration. */
export interface RAGPipelineConfig {
	/** Embedding model. */
	embedder: IEmbedder;
	/** Vector store (HybridVectorStore for BM25, SQLiteVectorStore for basic). */
	vectorStore: RAGStore;
	/** Chunking configuration. */
	chunkingConfig?: ChunkingConfig;
	/** Reranking model (if enabled). */
	rerankerModel?: RerankingModel;
	/** Enable reranking. */
	enableReranking?: boolean;
	/** Context window configuration. */
	contextWindow?: ContextWindowConfig;
	/** Python path for extraction. */
	pythonPath?: string;
	/** Python extraction script path. */
	scriptPath?: string;
}

export interface RAGSearchOptions {
	/** Expand query into sub-queries. */
	expand?: boolean;
	/** LLM endpoint for query rewriting. */
	rewriteEndpoint?: string;
	/** Reranker to use (overrides config). */
	reranker?: IReranker;
	/** Metadata filters. */
	filter?: MetadataFilter[];
	/** Number of candidates before reranking. */
	candidatesBeforeRerank?: number;
	/** Dense/sparse weights for hybrid search. */
	denseWeight?: number;
	sparseWeight?: number;
	/** Retrieve rewritten variants independently, then fuse their rankings. */
	multiQuery?: boolean;
	/** Remove redundant evidence while preserving relevance. */
	diversify?: boolean;
}

export interface RAGContextSearchOptions extends RAGSearchOptions {
	assembleContext?: boolean;
	includeParents?: boolean;
	compress?: boolean;
}

/**
 * RAG pipeline.
 *
 * Workflow:
 * 1. Index: Docling extraction → smart chunking → embed → hybrid store
 * 2. Search: query rewrite → hybrid retrieval → rerank → context assembly
 */
export class RAGPipeline {
	private embedder: IEmbedder;
	private vectorStore: RAGStore;
	private chunkingConfig: ChunkingConfig;
	private enableReranking: boolean;
	private reranker: IReranker | null;
	private contextWindow: ContextWindowConfig;

	readonly pythonPath: string;
	readonly scriptPath: string;

	constructor(
		config: RAGPipelineConfig,
		options?: { pythonPath?: string; scriptPath?: string },
	) {
		this.embedder = config.embedder;
		this.vectorStore = config.vectorStore;
		this.chunkingConfig = config.chunkingConfig ?? DEFAULT_CHUNKING_CONFIG;
		this.enableReranking =
			config.enableReranking ?? DEFAULT_RAG_CONFIG.enableReranking;
		this.contextWindow = config.contextWindow ?? DEFAULT_CONTEXT_WINDOW;

		// Initialize reranker if enabled
		if (this.enableReranking) {
			const modelId = config.rerankerModel
				? config.rerankerModel === "BAAI/bge-reranker-v2-m3"
					? "BAAI/bge-reranker-v2-m3"
					: "BAAI/bge-reranker-base"
				: "BAAI/bge-reranker-base";
			this.reranker = new CrossEncoderReranker({ modelId });
		} else {
			this.reranker = null;
		}

		this.pythonPath = options?.pythonPath || DEFAULT_PYTHON_PATH;
		this.scriptPath = options?.scriptPath || DEFAULT_SCRIPT_PATH;
	}

	// ── Indexing ──────────────────────────────────────────────────────────────

	/** Index a document file via Docling → smart chunking → store. */
	async indexFile(
		filePath: string,
		docId?: string,
	): Promise<ExtractedDocument> {
		const json = await this._extractViaPython(
			"extract",
			filePath,
			docId ? { "--doc-id": docId } : {},
		);
		return this.processAndStore(json);
	}

	/** Index raw text with smart chunking. */
	async indexText(
		text: string,
		source?: string,
		docId?: string,
	): Promise<ExtractedDocument> {
		const json = await this._extractViaPython(
			"extract-from-text",
			text.slice(0, 64_000),
			{
				...{ "--source": source || "manual" },
				...(docId ? { "--doc-id": docId } : {}),
			},
		);
		return this.processAndStore(json);
	}

	private async _extractViaPython(
		command: string,
		arg: string,
		extraArgs?: Record<string, string>,
	): Promise<ExtractedDocumentJSON> {
		const args = [this.scriptPath, command, arg];
		if (extraArgs) {
			for (const [k, v] of Object.entries(extraArgs)) {
				if (v) args.push(k, v);
			}
		}

		try {
			const { stdout } = await execFileAsync(this.pythonPath, args, {
				timeout: 120_000,
			});
			return JSON.parse(stdout.trim()) as ExtractedDocumentJSON;
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			throw new Error(`Docling extraction failed: ${msg}`);
		}
	}

	private async processAndStore(
		json: ExtractedDocumentJSON,
	): Promise<ExtractedDocument> {
		// Smart chunking
		const chunks = await this._smartChunkText(json.content, json.id, json.meta);

		// Embed all chunks
		const chunkTexts = chunks.map(c => c.text);
		const vectors = await this.embedder.embedBatch(chunkTexts);

		for (let i = 0; i < chunks.length; i++) {
			chunks[i].vector = vectors[i];
		}

		// Store in hybrid or vector store
		await this.vectorStore.add(chunks);

		return {
			id: json.id,
			filename: json.filename,
			content: json.content,
			meta: json.meta as ExtractedDocument["meta"],
			chunks,
			extractedAt: new Date(json.extracted_at).getTime(),
		};
	}

	private async _smartChunkText(
		text: string,
		docId: string,
		meta: Record<string, unknown>,
	): Promise<RAGChunk[]> {
		const strategy = this.chunkingConfig.strategy;

		if (strategy === "semantic") {
			const chunks = await smartChunk(text, this.chunkingConfig, t =>
				this.embedder.embed(t).then(v => v),
			);
			return chunks.map(c => ({
				...c,
				id: `${docId}:${c.id}`,
				parentId: c.parentId ? `${docId}:${c.parentId}` : undefined,
				metadata: { ...c.metadata, source: meta?.title || docId },
				documentId: docId,
			}));
		}

		if (strategy === "fixed") {
			// Fixed-size with overlap
			const chunks: RAGChunk[] = [];
			const size = this.chunkingConfig.chunkSize ?? 512;
			const overlap = this.chunkingConfig.overlap ?? 128;
			let start = 0;
			let idx = 0;

			while (start < text.length) {
				const end = Math.min(start + size, text.length);
				let breakPoint = end;
				if (end < text.length) {
					const lookBack = Math.min(overlap + 50, end - start);
					const seg = text.slice(end - lookBack, end);
					const spaceIdx = seg.lastIndexOf(" ");
					if (spaceIdx > 0) breakPoint = end - lookBack + spaceIdx;
				}
				const chunkText = text.slice(start, breakPoint).trim();
				if (chunkText.length > (this.chunkingConfig.minChunkSize ?? 64)) {
					chunks.push({
						id: `${docId}:chunk_${idx++}`,
						text: chunkText,
						metadata: { source: meta?.title || docId },
						documentId: docId,
						approxTokens: Math.ceil(chunkText.length / 4),
					});
				}
				start = breakPoint - overlap;
				if (start <= 0) start = end;
			}
			return chunks;
		}

		// Default: recursive chunking
		const chunks = recursiveChunk(text, this.chunkingConfig) as RAGChunk[];
		return chunks.map(c => ({
			...c,
			id: `${docId}:${c.id}`,
			parentId: c.parentId ? `${docId}:${c.parentId}` : undefined,
			metadata: { ...c.metadata, source: meta?.title || docId },
			documentId: docId,
		}));
	}

	// ── Search ──────────────────────────────────────────────────────────────────

	/**
	 * Search: query rewrite → hybrid retrieval → rerank → context.
	 */
	async search(
		query: string,
		topK = 5,
		options?: RAGSearchOptions,
	): Promise<SearchHit[]> {
		const {
			expand = true,
			rewriteEndpoint,
			reranker,
			filter,
			candidatesBeforeRerank = 50,
			denseWeight = 0.6,
			sparseWeight = 0.4,
			multiQuery = true,
			diversify = true,
		} = options ?? {};

		// Step 1: Query rewriting/expansion
		let effectiveQuery = query;
		let rewritten: RewrittenQuery | undefined;

		if (expand) {
			if (rewriteEndpoint) {
				rewritten = await llmRewriteQuery(query, {
					endpoint: rewriteEndpoint,
					nExpansions: 3,
				});
				effectiveQuery = rewritten.combined;
			} else {
				rewritten = await rewriteQuery(query, { nExpansions: 3 });
				effectiveQuery = rewritten.combined;
			}
		}

		// Embed and retrieve variants independently. Embedding a synthetic "OR"
		// string blurs intent; rank fusion preserves evidence from each route.
		const queryVariants =
			multiQuery && rewritten
				? [...new Set([query, rewritten.rewritten, ...rewritten.expansions])]
				: [effectiveQuery];
		const queryVectors = await this.embedder.embedBatch(queryVariants);
		const routes = await Promise.all(
			queryVariants.map(async (variant, index) => {
				const routeHits = this.vectorStore.searchHybrid
					? await this.vectorStore.searchHybrid(
							variant,
							queryVectors[index],
							candidatesBeforeRerank,
							{
								denseWeight,
								sparseWeight,
								filter: this._filterToMap(filter),
							},
						)
					: await this.vectorStore.searchByVector(
							queryVectors[index],
							candidatesBeforeRerank,
							{
								filter: this._filterToMap(filter),
							},
						);
				return { name: variant, hits: routeHits };
			}),
		);
		let hits = fuseRankedHits(routes);

		// Step 4: Reranking
		if (this.enableReranking) {
			const rr = reranker ?? this.reranker;
			if (rr && hits.length > 0) {
				const retrievalById = new Map(hits.map(hit => [hit.chunk.id, hit]));
				const reranked = await rr.rerank(query, hits);
				hits = reranked.map(r => ({
					...retrievalById.get(r.chunk.id),
					chunk: r.chunk,
					score: r.rerankScore,
					rerankScore: r.rerankScore,
				}));
			}
		}

		// Step 5: Return top-K
		return diversify ? selectDiverseHits(hits, topK) : hits.slice(0, topK);
	}

	/** Search plus calibrated, machine-readable evidence diagnostics. */
	async searchWithDiagnostics(
		query: string,
		topK = 5,
		options?: RAGSearchOptions,
	): Promise<{ hits: SearchHit[]; diagnostics: RetrievalDiagnostics }> {
		const hits = await this.search(query, topK, options);
		const variants = [
			...new Set(hits.flatMap(hit => hit.retrievalRoutes ?? [])),
		];
		return {
			hits,
			diagnostics: diagnoseRetrieval(
				variants.length ? variants : [query],
				hits,
				hits,
			),
		};
	}

	/**
	 * Search with context assembly.
	 * Returns formatted context string suitable for LLM prompts.
	 */
	async searchWithContext(
		query: string,
		topK = 5,
		options?: RAGContextSearchOptions,
	): Promise<{ hits: SearchHit[]; context: string }> {
		const searchOptions = options;
		const hits = await this.search(query, topK, {
			...searchOptions,
			expand: searchOptions?.expand ?? true,
		});

		let context: string;

		if (searchOptions?.compress) {
			context = compressContext(hits, this.contextWindow).compressed;
		} else {
			context = assembleContext(hits, this.contextWindow);
		}

		return { hits, context };
	}

	// ── Document management ───────────────────────────────────────────────────

	/** List all indexed document IDs. */
	async listDocuments(): Promise<string[]> {
		return this.vectorStore.documentIds();
	}

	/** Count total chunks. */
	async countChunks(): Promise<number> {
		return this.vectorStore.count();
	}

	/** Remove a document. */
	async deleteDocument(docId: string): Promise<void> {
		await this.vectorStore.deleteDocument(docId);
	}

	/** Get chunks for a document (debugging). */
	getChunksByDocument(docId: string): RAGChunk[] {
		return this.vectorStore.getChunksByDocument(docId);
	}

	/** Clear all data. */
	async clear(): Promise<void> {
		await this.vectorStore.clear();
	}

	// ── Helpers ─────────────────────────────────────────────────────────────────

	private _filterToMap(
		filters?: MetadataFilter[],
	): Record<string, string | number | boolean> | undefined {
		if (!filters) return undefined;
		const result: Record<string, string | number | boolean> = {};
		for (const f of filters) {
			result[f.field] = Array.isArray(f.value) ? f.value[0] : f.value;
		}
		return result;
	}

	/** Close resources. */
	close(): void {
		this.vectorStore.close?.();
	}
}
