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

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
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
	const parsed = JSON.parse(content);
	if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
		throw new Error(
			`Invalid JSON for ${url.host}://<action>. Expected an object with {docId, path, query, k}.`,
		);
	}
	return parsed as JsonArgs;
}

export class RagProtocolHandler implements ProtocolHandler {
	readonly scheme = "rag";
	readonly immutable = false;

	async resolve(
		url: InternalUrl,
	): Promise<InternalResource> {
		const full =
			url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

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
			return {
				url: url.href,
				content: "# RAG\n\nUse `write` to operate on indexed documents.\n\nUse `read rag://` for usage help.",
				contentType: "text/markdown",
			};
		}

		throw new Error(
			`Unknown rag:// path: ${full}. Use rag:// for help, rag://list to list documents, or write to search/ingest/delete.`,
		);
	}

	async write(
		url: InternalUrl,
		content: string,
		ctx?: WriteContext,
	): Promise<string | void> {
		const full =
			url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

		const args = parseJsonArgs(url, content);
		const cwd = ctx?.cwd;

		if (!cwd) {
			throw new Error("cwd is required for rag:// operations.");
		}

		const { getPipeline } = await import("../../../../capabilities/rag/index.ts");
		const pipeline = getPipeline(cwd);

		// rag://search — search indexed documents
		if (full === "search") {
			if (!args.query) {
				throw new Error("rag://search requires {query}.");
			}
			const k = Number(args.k ?? 5);
			const results = await pipeline.search(args.query, k);
			const hits = results.map((h: import("../../../capabilities/rag/index.ts").SearchHit) => ({
				id: h.chunk.id,
				documentId: h.chunk.documentId,
				text: h.chunk.text.slice(0, 500),
				score: parseFloat(h.score.toFixed(4)),
				metadata: h.chunk.metadata,
			}));
			return JSON.stringify({ query: args.query, results: hits, totalFound: results.length }, null, 2);
		}

		// rag://ingest — ingest a document
		if (full === "ingest") {
			if (!args.path) {
				throw new Error("rag://ingest requires {path}.");
			}
			const doc = await pipeline.ingestFile(args.path, args.docId);
			return JSON.stringify({
				success: true,
				id: doc.id,
				filename: doc.filename,
				chunks: doc.chunks.length,
				extractedAt: new Date(doc.extractedAt).toISOString(),
			}, null, 2);
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
