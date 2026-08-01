// ── SQLite + USearch Vector Store ────────────────────────────────────────────
// Chunk text/metadata persisted in SQLite; vectors indexed in a USearch ANN
// index persisted alongside it. Search queries the ANN index, then hydrates
// hits from SQLite by id.

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { existsSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { Index, MetricKind } from "usearch";
import type { IVectorStore, RAGChunk, SearchHit } from "../types.ts";

// ── Database helpers ──────────────────────────────────────────────────────────

const SCHEMA_VERSION = 1;

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

/** Resolve storage paths (SQLite db + USearch index) using XDG data dir or fallback to user home. */
function resolveStoragePaths(projectDir: string, dbName = "rag"): { dbPath: string; indexPath: string } {
	const base = "tui/rag-storage";
	const storageRoot = process.env.XDG_DATA_HOME
		? join(process.env.XDG_DATA_HOME, base)
		: join(process.env.HOME || ".", ".local", "share", base);
	const key = `${createHash("sha256").update(projectDir.toLowerCase()).digest("hex").slice(0, 8)}-${dbName}`;
	return {
		dbPath: join(storageRoot, `${key}.db`),
		indexPath: join(storageRoot, `${key}.usearch`),
	};
}

// ── ChunkRow interface ────────────────────────────────────────────────────────

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

// ── SQLiteVectorStore ─────────────────────────────────────────────────────────

export class SQLiteVectorStore implements IVectorStore {
	private db: SqliteDatabase;
	private statements: Record<string, SqliteStatement> = {};
	private index: Index;
	private readonly indexPath: string;
	private dimension: number;

	constructor(projectDir: string, options?: { dbName?: string; dimension?: number }) {
		this.dimension = options?.dimension ?? 384;
		const { dbPath, indexPath } = resolveStoragePaths(projectDir, options?.dbName || "chunks");
		this.indexPath = indexPath;

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
				// Corrupt/incompatible index (e.g. dimension change) — rebuild from SQLite.
				this.rebuildIndexFromDb();
			}
		}
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
		`);

		const current = (this.db.prepare("PRAGMA user_version").get() as { user_version: number }).user_version;
		if (current < SCHEMA_VERSION) {
			this.db.exec(`PRAGMA user_version = ${SCHEMA_VERSION}`);
		}
	}

	private p(key: string, sql: string): void {
		this.statements[key] = this.db.prepare(sql);
	}

	private prepareStatements(): void {
		this.p("insertChunk", `
			INSERT OR REPLACE INTO chunks (id, document_id, filename, text, metadata_json, chunk_index)
			VALUES (?, ?, ?, ?, ?, ?)
		`);
		this.p("getRowId", `SELECT rowid FROM chunks WHERE id = ?`);
		this.p("deleteById", `DELETE FROM chunks WHERE id = ?`);
		this.p("selectByDocId", `SELECT rowid, id FROM chunks WHERE document_id = ?`);
		this.p("clearAll", "DELETE FROM chunks");
		this.p("countChunks", "SELECT COUNT(*) AS cnt FROM chunks");
		this.p("getAllDocs", `SELECT DISTINCT document_id FROM chunks ORDER BY created_at DESC`);
		this.p("getChunkById", "SELECT * FROM chunks WHERE id = ?");
		this.p("getByRowIds", `SELECT * FROM chunks WHERE rowid IN (SELECT value FROM json_each(?))`);
		this.p("listChunksByDoc", `SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC`);
		this.p("listAll", `SELECT * FROM chunks`);
	}

	/** Reset to an empty ANN index. SQLite rows remain but have no vectors — caller must re-ingest to repopulate. */
	private rebuildIndexFromDb(): void {
		this.index = new Index(this.dimension, MetricKind.Cos);
	}

	private saveIndex(): void {
		mkdirSync(dirname(this.indexPath), { recursive: true });
		this.index.save(this.indexPath);
	}

	async add(chunks: RAGChunk[]): Promise<void> {
		if (!chunks.length) return;

		for (let i = 0; i < chunks.length; i++) {
			const c = chunks[i];
			if (!c.vector) throw new Error(`Chunk ${c.id} has no vector; embed before storing.`);

			this.statements.insertChunk.run(
				c.id,
				c.documentId || "unknown",
				(c.metadata?.source as string) || "",
				c.text,
				JSON.stringify(c.metadata || {}),
				i,
			);

			const row = this.statements.getRowId.get(c.id) as { rowid: number };
			const key = BigInt(row.rowid);
			if (this.index.contains(key)) this.index.remove(key);
			this.index.add(key, Float32Array.from(c.vector));
		}

		this.saveIndex();
	}

	async searchByVector(vector: number[], topK = 5): Promise<SearchHit[]> {
		const size = this.index.size();
		if (size === 0) return [];

		const k = Math.min(topK, size);
		const matches = this.index.search(Float32Array.from(vector), k, 0);
		const rowIds = Array.from(matches.keys, (k) => Number(k));
		if (!rowIds.length) return [];

		const rows = this.statements.getByRowIds.all(JSON.stringify(rowIds)) as ChunkRow[];
		const byRowId = new Map(rows.map((r) => [r.rowid, r]));

		const hits: SearchHit[] = [];
		for (let i = 0; i < rowIds.length; i++) {
			const row = byRowId.get(rowIds[i]);
			if (!row) continue; // stale index entry for a deleted row
			// USearch cosine metric returns distance (1 - similarity); convert back.
			hits.push({ chunk: toRAGChunk(row), score: 1 - matches.distances[i] });
		}
		return hits;
	}

	async search(_query: string, _topK = 5): Promise<SearchHit[]> {
		throw new Error("Use searchByVector() with pre-computed embeddings. For text search use the RAGPipeline.");
	}

	async clear(): Promise<void> {
		this.statements.clearAll.run();
		this.rebuildIndexFromDb();
		this.saveIndex();
	}

	async count(): Promise<number> {
		const result = this.statements.countChunks.get() as { cnt: number };
		return result.cnt;
	}

	async documentIds(): Promise<string[]> {
		const rows = this.statements.getAllDocs.all() as Array<{ document_id: string }>;
		return rows.map((r) => r.document_id);
	}

	/** Remove all chunks belonging to a document, from both SQLite and the ANN index. */
	async deleteDocument(docId: string): Promise<void> {
		const rows = this.statements.selectByDocId.all(docId) as Array<{ rowid: number; id: string }>;
		for (const row of rows) {
			const key = BigInt(row.rowid);
			if (this.index.contains(key)) this.index.remove(key);
			this.statements.deleteById.run(row.id);
		}
		if (rows.length) this.saveIndex();
	}

	/** List all chunks for a document (for debugging/inspection). */
	getChunksByDocument(docId: string): RAGChunk[] {
		const rows = this.statements.listChunksByDoc.all(docId) as ChunkRow[];
		return rows.map(toRAGChunk);
	}

	getChunkById(chunkId: string): RAGChunk | null {
		const row = this.statements.getChunkById.get(chunkId) as ChunkRow | undefined;
		return row ? toRAGChunk(row) : null;
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
