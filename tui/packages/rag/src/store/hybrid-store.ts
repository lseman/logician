// ── BM25 Hybrid Vector Store ─────────────────────────────────────────────────
// Combines dense vector similarity (USearch) with sparse BM25 retrieval.
// BM25 term frequencies stored in SQLite; IDF computed on demand.
// Scores fused via reciprocal rank fusion (RRF) for robust ranking.

import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Index, MetricKind } from "usearch";
import type {
	MetadataFilter,
	RAGChunk,
	SearchHit,
} from "../types.ts";

// ── SQLite helpers ─────────────────────────────────────────────────────────────

const SCHEMA_VERSION = 2;

interface SqliteStatement {
	run(...args: unknown[]): unknown;
	get(...args: unknown[]): unknown;
	all(...args: unknown[]): unknown[];
}

interface SqliteDatabase {
	exec(sql: string): unknown;
	prepare(sql: string): SqliteStatement;
	close(): void;
}

type SqliteDatabaseConstructor = new (path: string) => SqliteDatabase;

function resolveSqliteDatabase(): SqliteDatabaseConstructor {
	const runtimeRequire = createRequire(import.meta.url);
	const isBun = "Bun" in globalThis;
	const mod = isBun ? runtimeRequire("bun:sqlite") : runtimeRequire("node:sqlite");
	return (isBun ? mod.Database : mod.DatabaseSync) as SqliteDatabaseConstructor;
}

function resolveStoragePaths(
	projectDir: string,
	dbName = "rag",
): { dbPath: string; indexPath: string; bm25Path: string } {
	const base = "tui/rag-storage";
	const storageRoot = process.env.XDG_DATA_HOME
		? join(process.env.XDG_DATA_HOME, base)
		: join(process.env.HOME || ".", ".local", "share", base);
	const key = `${createHash("sha256")
		.update(projectDir.toLowerCase())
		.digest("hex")
		.slice(0, 8)}-${dbName}`;
	return {
		dbPath: join(storageRoot, `${key}.db`),
		indexPath: join(storageRoot, `${key}.usearch`),
		bm25Path: join(storageRoot, `${key}.bm25`),
	};
}

// ── Tokenizer ──────────────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
	"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
	"of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
	"have", "has", "had", "do", "does", "did", "will", "would", "could",
	"should", "may", "might", "shall", "can", "this", "that", "these",
	"those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
	"he", "she", "they", "them", "their", "what", "which", "who", "how",
	"when", "where", "why", "not", "no", "yes", "so", "if", "as",
]);

export function tokenize(text: string): string[] {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9\s]/g, " ")
		.split(/\s+/)
		.filter((t) => t.length > 1 && !STOP_WORDS.has(t));
}

// ── BM25 Scorer (in-process) ──────────────────────────────────────────────────

interface BM25State {
	/** term → set of chunk rowids that contain it */
	docFreq: Map<string, Set<number>>;
	/** rowid → list of terms in that chunk */
	docTerms: Map<number, string[]>;
	/** total number of chunks */
	nDocs: number;
	/** average doc length (total tokens / nDocs) */
	avgDocLen: number;
}

export class BM25Scorer {
	private state: BM25State = {
		docFreq: new Map(),
		docTerms: new Map(),
		nDocs: 0,
		avgDocLen: 1,
	};
	private idfCache = new Map<string, number>();

	addChunk(rowId: number, terms: string[]): void {
		this.state.docTerms.set(rowId, terms);
		this.state.nDocs = Math.max(this.state.nDocs, rowId + 1);
		for (const term of terms) {
			let s = this.state.docFreq.get(term);
			if (!s) {
				s = new Set();
				this.state.docFreq.set(term, s);
			}
			s.add(rowId);
		}
	}

	addBulk(rows: Array<{ rowId: number; terms: string[] }>): void {
		for (const { rowId, terms } of rows) {
			this.addChunk(rowId, terms);
		}
	}

	recomputeAvgLen(totalTokens: number): void {
		this.state.avgDocLen =
			this.state.nDocs > 0 ? totalTokens / this.state.nDocs : 1;
	}

	private idf(term: string): number {
		if (this.idfCache.has(term)) return this.idfCache.get(term)!;
		const df = this.state.docFreq.get(term)?.size ?? 0;
		const n = this.state.nDocs;
		const score = Math.log((n - df + 0.5) / (df + 0.5) + 1);
		this.idfCache.set(term, score);
		return score;
	}

	scoreQuery(queryTerms: string[], targetRowIds: Set<number>): Map<number, number> {
		const k1 = 1.5;
		const b = 0.75;
		const avgLen = this.state.avgDocLen;
		const scores = new Map<number, number>();

		for (const rowId of targetRowIds) {
			const terms = this.state.docTerms.get(rowId);
			if (!terms || !terms.length) continue;

			const dl = terms.length;
			let score = 0;

			for (const qTerm of queryTerms) {
				const tf = terms.filter((t) => t === qTerm).length / dl;
				const idf = this.idf(qTerm);
				const denom = tf + k1 * (1 - b + b * dl / avgLen);
				if (denom > 0) {
					score += idf * (tf * (k1 + 1)) / denom;
				}
			}

			if (score > 0) {
				scores.set(rowId, score);
			}
		}

		return scores;
	}

	/** Top-K rows by BM25 score for a query. */
	topK(queryTerms: string[], k: number): Array<{ id: number; score: number }> {
		const scores = new Map<number, number>();
		for (const [term, rowIds] of this.state.docFreq.entries()) {
			const idf = this.idf(term);
			const k1 = 1.5;
			const b = 0.75;
			const avgLen = this.state.avgDocLen;

			for (const rowId of rowIds) {
				const terms = this.state.docTerms.get(rowId);
				if (!terms) continue;
				const dl = terms.length;
				const tf = terms.filter((t) => t === term).length / dl;
				const denom = tf + k1 * (1 - b + b * dl / avgLen);
				if (denom > 0) {
					const contrib = idf * (tf * (k1 + 1)) / denom;
					scores.set(rowId, (scores.get(rowId) ?? 0) + contrib);
				}
			}
		}

		const entries = Array.from(scores.entries())
			.sort((a, b) => b[1] - a[1]);
		return entries.slice(0, k).map(([id, score]) => ({ id, score }));
	}
}

// ── Chunk Row interface ────────────────────────────────────────────────────────

interface ChunkRow {
	id: string;
	document_id: string | null;
	filename: string;
	text: string;
	metadata_json: string;
	chunk_index: number;
	created_at: string;
	rowid: number;
}

// ── HybridVectorStore ──────────────────────────────────────────────────────────

export class HybridVectorStore {
	private db: SqliteDatabase;
	private index: Index;
	private bm25 = new BM25Scorer();
	private readonly indexPath: string;
	private readonly bm25Path: string;
	private dimension: number;

	constructor(
		projectDir: string,
		options?: { dbName?: string; dimension?: number },
	) {
		this.dimension = options?.dimension ?? 384;
		const { dbPath, indexPath, bm25Path } = resolveStoragePaths(
			projectDir,
			options?.dbName || "chunks",
		);
		this.indexPath = indexPath;
		this.bm25Path = bm25Path;

		const dir = dirname(dbPath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}

		const Database = resolveSqliteDatabase();
		this.db = new Database(dbPath);
		this.db.exec("PRAGMA journal_mode = WAL");
		this.db.exec("PRAGMA synchronous = normal");
		this.db.exec("PRAGMA busy_timeout = 5000");

		this.initSchema();
		this.prepareStatements();

		this.index = new Index(this.dimension, MetricKind.Cos);
		if (existsSync(this.indexPath)) {
			try {
				this.index.load(this.indexPath);
			} catch {
				this.rebuildIndexFromDb();
			}
		}

		this.loadBM25FromDB();
	}

	private initSchema(): void {
		this.db.exec(`
			CREATE TABLE IF NOT EXISTS chunks (
				rowid INTEGER PRIMARY KEY AUTOINCREMENT,
				id TEXT NOT NULL UNIQUE,
				document_id TEXT NOT NULL,
				filename TEXT NOT NULL DEFAULT '',
				text TEXT NOT NULL,
				metadata_json TEXT NOT NULL DEFAULT '{}',
				chunk_index INTEGER NOT NULL DEFAULT 0,
				created_at TEXT NOT NULL DEFAULT (datetime('now'))
			);
			CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
			CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename);
			CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at DESC);

			CREATE TABLE IF NOT EXISTS bm25_terms (
				chunk_rowid INTEGER NOT NULL,
				term TEXT NOT NULL,
				count INTEGER NOT NULL DEFAULT 1,
				PRIMARY KEY (chunk_rowid, term)
			);
		`);

		const current = (
			this.db.prepare("PRAGMA user_version").get() as { user_version: number }
		).user_version;
		if (current < SCHEMA_VERSION) {
			this.db.exec(`PRAGMA user_version = ${SCHEMA_VERSION}`);
		}
	}

	private prepareStatements(): void {
		this.statements = {
			insertChunk: this.db.prepare(`
				INSERT OR REPLACE INTO chunks (id, document_id, filename, text, metadata_json, chunk_index)
				VALUES (?, ?, ?, ?, ?, ?)`),
			getRowId: this.db.prepare("SELECT rowid FROM chunks WHERE id = ?"),
			deleteById: this.db.prepare("DELETE FROM chunks WHERE id = ?"),
			selectByDocId: this.db.prepare("SELECT rowid, id FROM chunks WHERE document_id = ?"),
			clearAll: this.db.prepare("DELETE FROM chunks"),
			countChunks: this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks"),
			getAllDocs: this.db.prepare("SELECT DISTINCT document_id FROM chunks ORDER BY created_at DESC"),
			getChunkById: this.db.prepare("SELECT * FROM chunks WHERE id = ?"),
			getByRowIds: this.db.prepare(
				"SELECT * FROM chunks WHERE rowid IN (SELECT value FROM json_each(?))",
			),
			listChunksByDoc: this.db.prepare(
				"SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
			),
			deleteByRowId: this.db.prepare("DELETE FROM bm25_terms WHERE chunk_rowid = ?"),
			getBM25Terms: this.db.prepare("SELECT term, count FROM bm25_terms WHERE chunk_rowid = ?"),
			deleteAllBM25: this.db.prepare("DELETE FROM bm25_terms"),
			insertBM25Term: this.db.prepare(
				"INSERT OR REPLACE INTO bm25_terms (chunk_rowid, term, count) VALUES (?, ?, ?)",
			),
		};
	}

	private statements!: Record<string, SqliteStatement>;

	private rebuildIndexFromDb(): void {
		this.index = new Index(this.dimension, MetricKind.Cos);
	}

	private saveIndex(): void {
		mkdirSync(dirname(this.indexPath), { recursive: true });
		this.index.save(this.indexPath);
	}

	// ── BM25 persistence ────────────────────────────────────────────────────────

	private loadBM25FromDB(): void {
		const rows = this.db.prepare(
			"SELECT chunk_rowid, term, count FROM bm25_terms",
		).all() as Array<{ chunk_rowid: number; term: string; count: number }>;

		// Group by rowid
		const byRowId = new Map<number, Array<{ term: string; count: number }>>();
		for (const r of rows) {
			let list = byRowId.get(r.chunk_rowid);
			if (!list) {
				list = [];
				byRowId.set(r.chunk_rowid, list);
			}
			list.push(r);
		}

		const bulk: Array<{ rowId: number; terms: string[] }> = [];
		let totalTokens = 0;
		for (const [rowId, terms] of byRowId) {
			const expanded = terms.flatMap((t) => Array(t.count).fill(t.term));
			bulk.push({ rowId, terms: expanded });
			totalTokens += expanded.length;
		}

		if (bulk.length > 0) {
			this.bm25.addBulk(bulk);
			this.bm25.recomputeAvgLen(totalTokens);
		}
	}

	private saveBM25ToDB(rows: Array<{ rowId: number; termCounts: Map<string, number> }>): void {
		this.statements.deleteAllBM25.run();
		for (const { rowId, termCounts } of rows) {
			for (const [term, count] of termCounts) {
				this.statements.insertBM25Term.run(rowId, term, count);
			}
		}
	}

	// ── Chunk operations ────────────────────────────────────────────────────────

	async add(chunks: RAGChunk[]): Promise<void> {
		if (!chunks.length) return;

		const insertBM25Term = this.statements.insertBM25Term;
		const getRowId = this.statements.getRowId;
		const insertChunk = this.statements.insertChunk;

		// BM25 data per chunk
		const bm25Data: Array<{ rowId: number; termCounts: Map<string, number> }> = [];

		for (let i = 0; i < chunks.length; i++) {
			const c = chunks[i];
			if (!c.vector)
				throw new Error(`Chunk ${c.id} has no vector; embed before storing.`);

			insertChunk.run(
				c.id,
				c.documentId || "unknown",
				(c.metadata?.source as string) || "",
				c.text,
				JSON.stringify(c.metadata || {}),
				i,
			);

			const row = getRowId.get(c.id) as { rowid: number };
			const rowId = row.rowid;

			// Compute BM25 terms
			const terms = tokenize(c.text);
			const termCounts = new Map<string, number>();
			for (const t of terms) {
				termCounts.set(t, (termCounts.get(t) ?? 0) + 1);
			}
			bm25Data.push({ rowId, termCounts });

			// Store in USearch
			const key = BigInt(rowId);
			if (this.index.contains(key)) this.index.remove(key);
			this.index.add(key, Float32Array.from(c.vector));
		}

		this.saveIndex();
		this.saveBM25ToDB(bm25Data);
	}

	async clear(): Promise<void> {
		this.statements.clearAll.run();
		this.rebuildIndexFromDb();
		this.saveIndex();
		this.statements.deleteAllBM25.run();
		this.bm25 = new BM25Scorer();
	}

	// ── Hybrid search ──────────────────────────────────────────────────────────────

	/**
	 * Hybrid search: dense vector + BM25 via Reciprocal Rank Fusion (RRF).
	 * RRF is robust because it doesn't require score normalization —
	 * it only cares about the relative rank of each document.
	 */
	async searchByVector(
		vector: number[],
		topK = 5,
		options?: {
			filter?: Record<string, string | number | boolean>;
			denseWeight?: number;
			sparseWeight?: number;
		},
	): Promise<SearchHit[]> {
		const size = this.index.size();
		if (size === 0) return [];

		const {
			denseWeight = 0.6,
			sparseWeight = 0.4,
			filter,
		} = { ...options };

		const kDense = Math.min(topK * 8, size);

		// Step 1: Dense retrieval
		const denseMatches = this.index.search(Float32Array.from(vector), kDense, 0);
		const denseRowIds = Array.from(denseMatches.keys, (k) => Number(k));

		if (!denseRowIds.length) return [];

		// Step 2: Get candidate chunks with texts for BM25
		const rows = this.statements.getByRowIds.all(
			JSON.stringify(denseRowIds),
		) as ChunkRow[];

		let candidates = rows.map((r) => ({
			rowId: r.rowid,
			text: r.text,
			metadata: JSON.parse(r.metadata_json),
		}));

		// Apply metadata filters
		if (filter) {
			candidates = candidates.filter((c) => {
				for (const [key, value] of Object.entries(filter)) {
					const metaVal = c.metadata[key];
					if (typeof value === "string") {
						if (String(metaVal ?? "") !== value) return false;
					} else if (typeof value === "number") {
						if (Number(metaVal ?? 0) !== value) return false;
					} else if (typeof value === "boolean") {
						if (Boolean(metaVal) !== value) return false;
					}
				}
				return true;
			});
		}

		// Step 3: BM25 scoring
		const queryTerms = tokenize(
			this._vectorToQueryText(vector),
		);
		const bm25Scores = new Map<number, number>();
		if (queryTerms.length > 0) {
			const bm25Top = this.bm25.topK(queryTerms, candidates.length);
			for (const { id, score } of bm25Top) {
				bm25Scores.set(id, score);
			}
		}

		// Step 4: Reciprocal Rank Fusion
		return this._rrfFusion(
			denseRowIds,
			denseMatches,
			candidates,
			queryTerms,
			topK,
			denseWeight,
			sparseWeight,
		);
	}

	/**
	 * Reciprocal Rank Fusion (RRF) combines dense and BM25 rankings.
	 * Formula: score(d) = Σ_{r ∈ rankings} 1 / (k + rank_r(d))
	 * where k is a constant (typically 60) and rank_r(d) is d's rank in ranking r.
	 * This is robust to score distribution differences between modalities.
	 */
	private _rrfFusion(
		denseRowIds: number[],
		denseMatches: { keys: BigUint64Array; distances: Float32Array },
		candidates: Array<{ rowId: number; text: string; metadata: Record<string, unknown> }>,
		queryTerms: string[],
		topK: number,
		denseWeight: number,
		sparseWeight: number,
	): SearchHit[] {
		const rrfK = 60;
		const rankScores = new Map<number, number>();

		// Dense ranks (higher score = better rank)
		const denseSorted = Array.from(denseMatches.keys).map((k, i) => ({
			rowId: Number(k),
			rank: i,
			distance: denseMatches.distances[i],
		}));
		for (const { rowId, rank } of denseSorted) {
			rankScores.set(rowId, (rankScores.get(rowId) ?? 0) + denseWeight / (rrfK + rank));
		}

		// BM25 ranks
		const bm25Sorted = this.bm25.topK(queryTerms, candidates.length);
		for (let rank = 0; rank < bm25Sorted.length; rank++) {
			const rowId = bm25Sorted[rank].id;
			rankScores.set(rowId, (rankScores.get(rowId) ?? 0) + sparseWeight / (rrfK + rank));
		}

		// Sort by fused score
		const fused = Array.from(rankScores.entries())
			.sort((a, b) => b[1] - a[1]);

		// Fetch top-K rows
		const topRowIds = fused.slice(0, topK).map(([id]) => id);
		const rows = this.statements.getByRowIds.all(
			JSON.stringify(topRowIds),
		) as ChunkRow[];

		const byRowId = new Map(rows.map((r) => [r.rowid, r]));
		const hits: SearchHit[] = [];

		for (const [rowId, rrfScore] of fused.slice(0, topK)) {
			const row = byRowId.get(rowId);
			if (!row) continue;

			const denseIdx = denseSorted.findIndex((d) => d.rowId === rowId);
			const denseSim = denseIdx >= 0 ? 1 - denseMatches.distances[denseIdx] : 0;
			const bm25Score = this.bm25.scoreQuery(queryTerms, new Set([rowId])).get(rowId) ?? 0;
			const maxBm25 = bm25Sorted[0]?.score ?? 1;
			const bm25Norm = maxBm25 > 0 ? bm25Score / maxBm25 : 0;

			hits.push({
				chunk: toRAGChunk(row),
				score: rrfScore,
				denseScore: denseSim,
				sparseScore: bm25Norm,
			});
		}

		return hits;
	}

	/** Convert a vector back to approximate query text (for BM25).
	 * In practice, the query text is passed separately — this is a fallback. */
	private _vectorToQueryText(_vector: number[]): string {
		// This is a placeholder — in real usage, the query text is available
		// from the pipeline. For hybrid search, the pipeline passes the query text.
		return "";
	}

	// ── Full hybrid search with query text ──────────────────────────────────────────

	/**
	 * Full hybrid search: query text + pre-computed vector for the query.
	 * Uses RRF to combine dense vector and BM25 scores.
	 */
	async searchHybrid(
		queryText: string,
		queryVector: number[],
		topK = 5,
		options?: {
			filter?: Record<string, string | number | boolean>;
			denseWeight?: number;
			sparseWeight?: number;
		},
	): Promise<SearchHit[]> {
		const size = this.index.size();
		if (size === 0) return [];

		const {
			denseWeight = 0.6,
			sparseWeight = 0.4,
			filter,
		} = { ...options };

		// Step 1: Dense retrieval
		const kDense = Math.min(topK * 8, size);
		const denseMatches = this.index.search(Float32Array.from(queryVector), kDense, 0);
		const denseRowIds = Array.from(denseMatches.keys, (k) => Number(k));

		if (!denseRowIds.length) return [];

		// Step 2: Get candidate chunks
		const rows = this.statements.getByRowIds.all(
			JSON.stringify(denseRowIds),
		) as ChunkRow[];

		let candidates = rows.map((r) => ({
			rowId: r.rowid,
			text: r.text,
			metadata: JSON.parse(r.metadata_json),
		}));

		// Apply metadata filters
		if (filter) {
			candidates = candidates.filter((c) => {
				for (const [key, value] of Object.entries(filter)) {
					const metaVal = c.metadata[key];
					if (typeof value === "string") {
						if (String(metaVal ?? "") !== value) return false;
					} else if (typeof value === "number") {
						if (Number(metaVal ?? 0) !== value) return false;
					} else if (typeof value === "boolean") {
						if (Boolean(metaVal) !== value) return false;
					}
				}
				return true;
			});
		}

		// Step 3: BM25 scoring
		const queryTerms = tokenize(queryText);
		const bm25Top = queryTerms.length > 0
			? this.bm25.topK(queryTerms, candidates.length)
			: [];

		// Step 4: RRF fusion
		const rankScores = new Map<number, number>();

		// Dense ranks
		for (let rank = 0; rank < denseRowIds.length; rank++) {
			const rowId = denseRowIds[rank];
			rankScores.set(rowId, (rankScores.get(rowId) ?? 0) + denseWeight / (60 + rank));
		}

		// BM25 ranks
		for (let rank = 0; rank < bm25Top.length; rank++) {
			const rowId = bm25Top[rank].id;
			rankScores.set(rowId, (rankScores.get(rowId) ?? 0) + sparseWeight / (60 + rank));
		}

		// Sort by fused score
		const fused = Array.from(rankScores.entries())
			.sort((a, b) => b[1] - a[1]);

		const topRowIds = fused.slice(0, topK).map(([id]) => id);
		const topRows = this.statements.getByRowIds.all(
			JSON.stringify(topRowIds),
		) as ChunkRow[];

		const byRowId = new Map(topRows.map((r) => [r.rowid, r]));
		const denseByIdx = new Map(denseRowIds.map((k, i) => [k, 1 - denseMatches.distances[i]]));

		const hits: SearchHit[] = [];
		for (const [rowId, rrfScore] of fused.slice(0, topK)) {
			const row = byRowId.get(rowId);
			if (!row) continue;

			const denseSim = denseByIdx.get(rowId) ?? 0;
			const bm25Score = bm25Top.find((b) => b.id === rowId)?.score ?? 0;
			const maxBm25 = bm25Top[0]?.score ?? 1;
			const bm25Norm = maxBm25 > 0 ? bm25Score / maxBm25 : 0;

			hits.push({
				chunk: toRAGChunk(row),
				score: rrfScore,
				denseScore: denseSim,
				sparseScore: bm25Norm,
			});
		}

		return hits;
	}

	// ── Legacy methods ──────────────────────────────────────────────────────────

	async search(query: string, _topK = 5): Promise<SearchHit[]> {
		throw new Error(
			"Use searchByVector() with pre-computed embeddings. For text search use the RAGPipeline.",
		);
	}

	async count(): Promise<number> {
		const result = this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks").get();
		return (result as { cnt: number }).cnt;
	}

	async documentIds(): Promise<string[]> {
		const rows = this.db.prepare(
			"SELECT DISTINCT document_id FROM chunks ORDER BY created_at DESC",
		).all() as Array<{ document_id: string }>;
		return rows.map((r) => r.document_id);
	}

	async deleteDocument(docId: string): Promise<void> {
		const rows = this.db.prepare(
			"SELECT rowid, id FROM chunks WHERE document_id = ?",
		).all(docId) as Array<{ rowid: number; id: string }>;
		for (const row of rows) {
			const key = BigInt(row.rowid);
			if (this.index.contains(key)) this.index.remove(key);
			this.statements.deleteById.run(row.id);
			this.statements.deleteByRowId.run(row.rowid);
		}
		if (rows.length) {
			this.saveIndex();
		}
	}

	getChunksByDocument(docId: string): RAGChunk[] {
		const rows = this.db.prepare(
			"SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
		).all(docId) as ChunkRow[];
		return rows.map(toRAGChunk);
	}

	getChunkById(chunkId: string): RAGChunk | null {
		const row = this.db.prepare(
			"SELECT * FROM chunks WHERE id = ?",
		).get(chunkId) as ChunkRow | undefined;
		return row ? toRAGChunk(row) : null;
	}

	getTermFrequencies(docId: string): Record<string, number> | null {
		const rows = this.db.prepare(
			"SELECT t.term, t.count FROM bm25_terms t " +
			"JOIN chunks c ON t.chunk_rowid = c.rowid " +
			"WHERE c.document_id = ?",
		).all(docId) as Array<{ term: string; count: number }>;
		if (!rows.length) return null;
		const tf: Record<string, number> = {};
		for (const r of rows) {
			tf[r.term] = (tf[r.term] ?? 0) + r.count;
		}
		return tf;
	}

	async setTermFrequencies(
		docId: string,
		tf: Record<string, number>,
	): Promise<void> {
		const rowId = this.db.prepare(
			"SELECT rowid FROM chunks WHERE document_id = ? LIMIT 1",
		).get(docId) as { rowid: number } | undefined;
		if (!rowId) return;
		for (const [term, count] of Object.entries(tf)) {
			this.statements.insertBM25Term.run(rowId.rowid, term, count);
		}
	}

	close(): void {
		this.db.close();
	}
}

function toRAGChunk(row: ChunkRow): RAGChunk {
	return {
		id: row.id,
		documentId: row.document_id || undefined,
		text: row.text,
		metadata: JSON.parse(row.metadata_json),
	};
}
