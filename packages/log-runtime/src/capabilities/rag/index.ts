// ── RAG (Retrieval-Augmented Generation) Tools ───────────────────────────────
// Built-in tools backed by @logician/log-rag:
//   - rag_ingest_pdf(path, docId?) — ingest a document into the vector store
//   - rag_search_docs(query, k?) — search indexed documents
//   - rag_list_docs() — list indexed document IDs
//   - rag_delete_doc(docId) — remove a document and its chunks

import type { Tool, ToolContext } from "@logician/log-core";
import {
	IngestionPipeline,
	type SearchHit,
	TransformersEmbedder,
} from "@logician/log-rag";

const embedder = new TransformersEmbedder();
const pipelines = new Map<string, IngestionPipeline>();

function getPipeline(cwd: string): IngestionPipeline {
	let pipeline = pipelines.get(cwd);
	if (!pipeline) {
		pipeline = new IngestionPipeline(cwd, { embedder, dbName: "logician-rag" });
		pipelines.set(cwd, pipeline);
	}
	return pipeline;
}

function requireCwd(ctx: ToolContext): string {
	if (!ctx.cwd) throw new Error("cwd is required for RAG tools");
	return ctx.cwd;
}

export const rag_ingest_pdf: Tool = {
	name: "rag_ingest_pdf",
	label: "RAG: Ingest PDF",
	description:
		"Ingest a document (PDF, DOCX, etc.) into the RAG vector store for later retrieval. Path must be absolute or relative to cwd.",
	promptSnippet: "Ingest a document into the RAG vector store",
	parameters: {
		type: "object",
		properties: {
			path: {
				type: "string",
				description: "Absolute or relative path to the document file to ingest",
			},
			docId: {
				type: "string",
				description:
					"Optional document ID. If omitted, a default is derived from the file.",
			},
		},
		required: ["path"],
	},
	execute: async (args, ctx) => {
		const filePath = args.path as string;
		if (!filePath) return { content: "Error: path is required", isError: true };

		try {
			const pipeline = getPipeline(requireCwd(ctx));
			const doc = await pipeline.ingestFile(
				filePath,
				args.docId as string | undefined,
			);
			return {
				content: JSON.stringify(
					{
						success: true,
						id: doc.id,
						filename: doc.filename,
						chunks: doc.chunks.length,
						extractedAt: new Date(doc.extractedAt).toISOString(),
					},
					null,
					2,
				),
			};
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			return { content: `Error ingesting document: ${msg}`, isError: true };
		}
	},
};

export const rag_search_docs: Tool = {
	name: "rag_search_docs",
	label: "RAG: Search Docs",
	description:
		"Search indexed documents in the RAG vector store. Returns top-k most similar chunks by cosine similarity.",
	promptSnippet: "Search indexed documents in the RAG vector store",
	readOnly: true,
	parameters: {
		type: "object",
		properties: {
			query: { type: "string", description: "Search query text" },
			k: {
				type: "number",
				description: "Number of results to return (default 5)",
			},
		},
		required: ["query"],
	},
	execute: async (args, ctx) => {
		const query = args.query as string;
		if (!query) return { content: "Error: query is required", isError: true };

		try {
			const pipeline = getPipeline(requireCwd(ctx));
			const k = Number(args.k ?? 5);
			const results = await pipeline.search(query, k);
			const hits = results.map((h: SearchHit) => ({
				id: h.chunk.id,
				documentId: h.chunk.documentId,
				text: h.chunk.text.slice(0, 500),
				score: parseFloat(h.score.toFixed(4)),
				metadata: h.chunk.metadata,
			}));

			return {
				content: JSON.stringify(
					{ query, results: hits, totalFound: results.length },
					null,
					2,
				),
			};
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			return { content: `Error searching docs: ${msg}`, isError: true };
		}
	},
};

export const rag_list_docs: Tool = {
	name: "rag_list_docs",
	label: "RAG: List Docs",
	description: "List document IDs currently indexed in the RAG vector store.",
	promptSnippet: "List documents indexed in the RAG vector store",
	readOnly: true,
	parameters: { type: "object", properties: {} },
	execute: async (_args, ctx) => {
		try {
			const pipeline = getPipeline(requireCwd(ctx));
			const documentIds = await pipeline.listDocuments();
			const chunkCount = await pipeline.countChunks();
			return { content: JSON.stringify({ documentIds, chunkCount }, null, 2) };
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			return { content: `Error listing docs: ${msg}`, isError: true };
		}
	},
};

export const rag_delete_doc: Tool = {
	name: "rag_delete_doc",
	label: "RAG: Delete Doc",
	description:
		"Remove a document and all its chunks from the RAG vector store.",
	promptSnippet: "Delete a document from the RAG vector store",
	parameters: {
		type: "object",
		properties: {
			docId: { type: "string", description: "Document ID to delete" },
		},
		required: ["docId"],
	},
	execute: async (args, ctx) => {
		const docId = args.docId as string;
		if (!docId) return { content: "Error: docId is required", isError: true };

		try {
			const pipeline = getPipeline(requireCwd(ctx));
			await pipeline.deleteDocument(docId);
			return { content: JSON.stringify({ success: true, docId }, null, 2) };
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			return { content: `Error deleting doc: ${msg}`, isError: true };
		}
	},
};

export const rag_tools: Tool[] = [
	rag_ingest_pdf,
	rag_search_docs,
	rag_list_docs,
	rag_delete_doc,
];
