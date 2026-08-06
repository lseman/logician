// ── RAG Pipeline ─────────────────────────────────────────────────────────────
// Orchestrates: extract (via Python Docling) → chunk → embed → store operations.

import { execFile } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import type { IEmbedder } from "../embedder.ts";
import type { ExtractedDocument, IVectorStore, RAGChunk } from "../types.ts";

// tui/packages/rag/src/pipeline/index.ts -> repo root (5 levels up)
const REPO_ROOT = path.resolve(
	fileURLToPath(new URL(".", import.meta.url)),
	"../../../../..",
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

/** Pipeline configuration. */
export interface RAGPipelineConfig {
	embedder: IEmbedder;
	vectorStore: IVectorStore;
}

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

/**
 * The main RAG pipeline. Extraction uses Python subprocess (Docling);
 * embedding + storage are in-process.
 */
export class RAGPipeline {
	private embedder: IEmbedder;
	private store: IVectorStore;
	readonly pythonPath: string;
	readonly scriptPath: string;

	constructor(
		config: RAGPipelineConfig,
		options?: { pythonPath?: string; scriptPath?: string },
	) {
		this.embedder = config.embedder;
		this.store = config.vectorStore;
		this.pythonPath = options?.pythonPath || DEFAULT_PYTHON_PATH;
		this.scriptPath = options?.scriptPath || DEFAULT_SCRIPT_PATH;
	}

	/** Add a document file via Docling (PDF, DOCX, PPTX, etc.). */
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

	/** Add raw text as a single chunk. */
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

	/** Search the indexed documents. */
	async search(
		text: string,
		topK = 5,
	): Promise<Array<{ chunk: RAGChunk; score: number }>> {
		const vectors = await this.embedder.embedBatch([text]);
		return this.store.searchByVector(vectors[0], topK);
	}

	/** List all indexed document IDs. */
	async listDocuments(): Promise<string[]> {
		return this.store.documentIds();
	}

	/** Count total chunks in the store. */
	async countChunks(): Promise<number> {
		return this.store.count();
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
		// Normalize chunk keys (Python uses snake_case, TS interface uses camelCase)
		const chunks: RAGChunk[] = json.chunks.map(c => ({
			id: c.id,
			text: c.text,
			metadata: c.metadata || {},
			documentId: c.document_id || json.id,
		}));

		// Embed all chunks in batch
		const chunkTexts = chunks.map(c => c.text);
		const vectors = await this.embedder.embedBatch(chunkTexts);

		for (let i = 0; i < chunks.length; i++) {
			chunks[i].vector = vectors[i];
		}

		// Store them
		await this.store.add(chunks);

		return {
			id: json.id,
			filename: json.filename,
			content: json.content,
			meta: json.meta as ExtractedDocument["meta"],
			chunks,
			extractedAt: new Date(json.extracted_at).getTime(),
		};
	}
}
