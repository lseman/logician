// ── Shared SQLite store plumbing ─────────────────────────────────────────────
// Common to both SQLiteVectorStore (sqlite-store.ts) and HybridVectorStore
// (hybrid-store.ts): the runtime-agnostic SQLite binding, storage path
// resolution, and the chunk-row-to-RAGChunk mapping. Each store keeps its own
// schema version and table layout — only the identical parts live here.

import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { join } from "node:path";
import type { RAGChunk } from "../types.ts";

export interface SqliteStatement {
	run(...args: unknown[]): unknown;
	get(...args: unknown[]): unknown;
	all(...args: unknown[]): unknown[];
}

export interface SqliteDatabase {
	exec(sql: string): unknown;
	prepare(sql: string): SqliteStatement;
	close(): void;
}

export type SqliteDatabaseConstructor = new (path: string) => SqliteDatabase;

export function resolveSqliteDatabase(): SqliteDatabaseConstructor {
	const runtimeRequire = createRequire(import.meta.url);
	const isBun = "Bun" in globalThis;
	const mod = isBun
		? runtimeRequire("bun:sqlite")
		: runtimeRequire("node:sqlite");
	return (isBun ? mod.Database : mod.DatabaseSync) as SqliteDatabaseConstructor;
}

/** Resolve storage paths (SQLite db + USearch index) using XDG data dir or fallback to user home. */
export function resolveStoragePaths(
	projectDir: string,
	dbName = "rag",
): { dbPath: string; indexPath: string } {
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
	};
}

export interface ChunkRow {
	id: string;
	document_id: string | null;
	filename: string;
	text: string;
	metadata_json: string;
	chunk_index: number;
	created_at: string;
	rowid: number;
}

export function toRAGChunk(row: ChunkRow): RAGChunk {
	return {
		id: row.id,
		documentId: row.document_id || undefined,
		text: row.text,
		metadata: JSON.parse(row.metadata_json),
	};
}
