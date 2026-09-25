// ── rag:// protocol handler ────────────────────────────────────────────────────
// RAG (Retrieval-Augmented Generation) operations backed by @logician/log-rag.
// URL forms:
//   rag://                  — shows usage help
//   rag://list              — list indexed document IDs (via read)
//   rag://search — write JSON {"query": "...", k?: number} to search
//   rag://ingest — write JSON {"path": "/path/to/doc", docId?: "..."} to ingest
//   rag://delete — write JSON {"docId": "..."} to delete a document
//
// Example:
//   read rag://              — see help
//   read rag://list           — list docs
//   write rag://search        — '{"query":"error handling"}'
//   write rag://ingest        — '{"path":"doc.pdf"}'
//   write rag://delete        — '{"docId":"abc123"}'

import type { SearchHit } from "@logician/log-rag";
import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	WriteContext,
} from "./types";

/** Parsed JSON argument for write operations. */
interface JsonArgs {
	docId?: string;
	path?: string;
	query?: string;
	k?: number;
}

/** Parse and validate JSON content from a write call. */
function parseJsonArgs(url: InternalUrl, content: string): JsonArgs {
	let parsed: unknown;
	try {
		parsed = JSON.parse(content);
	} catch (err) {
		throw new Error(
			`rag://${url.host}: invalid JSON — ${err instanceof Error ? err.message : String(err)}`,
		);
	}
	if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
		throw new Error(
			`rag://${url.host}: expected a JSON object with {docId, path, query, k}, got ${Array.isArray(parsed) ? "array" : typeof parsed}.`,
		);
	}
	return parsed as JsonArgs;
}

/**
 * Lazily load the RAG pipeline for a cwd. Dynamic import is intentional:
 * `capabilities/rag` pulls in the full `@logician/log-rag` graph (native
 * sqlite bindings, transformers.js), which the internal-URL router must not
 * force onto every process at module-load time.
 */
async function loadPipeline(cwd: string) {
	const { getPipeline } = await import("../../../../capabilities/rag/index.ts");
	return getPipeline(cwd);
}

export class RagProtocolHandler implements ProtocolHandler {
	readonly scheme = "rag";
	readonly immutable = false;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

		// rag:// — show usage help
		if (full === "") {
			return {
				url: url.href,
				content: [
					"# RAG (Retrieval-Augmented Generation)",
					"",
					"Search and manage documents indexed in the vector store.",
					"",
					"## Usage",
					"",
					"- `read rag://list` — list indexed document IDs",
					'- `write rag://search` — JSON: `{"query": "...", k?: number}`',
					'- `write rag://ingest` — JSON: `{"path": "file.pdf", docId?: "..."}`',
					'- `write rag://delete` — JSON: `{"docId": "..."}`',
					"",
				].join("\n"),
				contentType: "text/markdown",
			};
		}

		// rag://list — list indexed documents
		if (full === "list") {
			return this.#listDocuments(url, context);
		}

		throw new Error(
			`Unknown rag:// path: ${full}. Use rag:// for help, rag://list to list documents, or write to search/ingest/delete.`,
		);
	}

	/** List indexed document IDs from the RAG store for this cwd. */
	async #listDocuments(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const cwd = context?.cwd ?? process.cwd();
		try {
			const pipeline = await loadPipeline(cwd);
			const [documentIds, chunkCount] = await Promise.all([
				pipeline.listDocuments(),
				pipeline.countChunks(),
			]);
			if (documentIds.length === 0) {
				return {
					url: url.href,
					content: [
						"# RAG",
						"",
						"No documents indexed.",
						'Ingest one with: `write rag://ingest` — JSON: `{"path": "doc.pdf"}`',
					].join("\n"),
					contentType: "text/markdown",
				};
			}
			const lines = documentIds.map(id => `- \`${id}\``).join("\n");
			return {
				url: url.href,
				content: `# RAG\n\n${documentIds.length} document(s), ${chunkCount} chunk(s) indexed:\n\n${lines}\n`,
				contentType: "text/markdown",
			};
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			throw new Error(`rag://list failed: ${msg}`);
		}
	}

	async write(
		url: InternalUrl,
		content: string,
		ctx?: WriteContext,
	): Promise<string | void> {
		const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

		const args = parseJsonArgs(url, content);
		const cwd = ctx?.cwd;

		if (!cwd) {
			throw new Error("cwd is required for rag:// operations.");
		}

		const pipeline = await loadPipeline(cwd);

		// rag://search — search indexed documents
		if (full === "search") {
			if (!args.query) {
				throw new Error("rag://search requires {query}.");
			}
			const k = Number(args.k ?? 5);
			const results = await pipeline.search(args.query, k);
			const hits = results.map((h: SearchHit) => {
				const truncated = h.chunk.text.length > 500;
				return {
					id: h.chunk.id,
					documentId: h.chunk.documentId,
					text: h.chunk.text.slice(0, 500),
					...(truncated ? { truncated: true } : {}),
					score: parseFloat(h.score.toFixed(4)),
					metadata: h.chunk.metadata,
				};
			});
			return JSON.stringify(
				{ query: args.query, results: hits, totalFound: results.length },
				null,
				2,
			);
		}

		// rag://ingest — ingest a document
		if (full === "ingest") {
			if (!args.path) {
				throw new Error("rag://ingest requires {path}.");
			}
			const doc = await pipeline.ingestFile(args.path, args.docId);
			return JSON.stringify(
				{
					success: true,
					id: doc.id,
					filename: doc.filename,
					chunks: doc.chunks.length,
					extractedAt: new Date(doc.extractedAt).toISOString(),
				},
				null,
				2,
			);
		}

		// rag://delete — delete a document
		if (full === "delete") {
			if (!args.docId) {
				throw new Error("rag://delete requires {docId}.");
			}
			await pipeline.deleteDocument(args.docId);
			return JSON.stringify({ success: true, docId: args.docId });
		}

		throw new Error(
			`Unknown rag:// action: ${full}. Use search, ingest, or delete.`,
		);
	}
}
