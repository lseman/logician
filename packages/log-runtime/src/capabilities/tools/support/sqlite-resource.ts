/**
 * SQLite table/row read/write, addressed via `db.sqlite:table` (schema +
 * sample rows) and `db.sqlite:table:rowid` (one row). Rowid-addressed only —
 * tables declared `WITHOUT ROWID` and arbitrary composite-primary-key lookup
 * are out of scope. Uses `node:sqlite`'s `DatabaseSync`, which works
 * identically under plain Node (>=22.19) and bun — no new dependency.
 */

import { DatabaseSync } from "node:sqlite";
import * as fs from "node:fs";

const SQLITE_MAGIC = "SQLite format 3\0";
const SAMPLE_ROW_LIMIT = 20;
const ROW_COUNT_PROBE_CAP = 50_000;
const CELL_CHAR_LIMIT = 300;

export function isSqliteFile(absolutePath: string): boolean {
	let fd: number;
	try {
		fd = fs.openSync(absolutePath, "r");
	} catch {
		return false;
	}
	try {
		const buffer = Buffer.alloc(SQLITE_MAGIC.length);
		const read = fs.readSync(fd, buffer, 0, buffer.length, 0);
		return read === buffer.length && buffer.toString("latin1") === SQLITE_MAGIC;
	} catch {
		return false;
	} finally {
		fs.closeSync(fd);
	}
}

type Row = Record<string, unknown>;

function sanitizeCell(value: unknown): string {
	if (value === null || value === undefined) return "NULL";
	let text: string;
	if (Buffer.isBuffer(value)) text = `<blob ${value.length}B>`;
	else if (value instanceof Uint8Array) text = `<blob ${value.length}B>`;
	else text = String(value);
	text = text.replace(/\r\n/g, "\\r\\n").replace(/\n/g, "\\n").replace(/\r/g, "\\r");
	if (text.length > CELL_CHAR_LIMIT) {
		text = `${text.slice(0, CELL_CHAR_LIMIT)}... [truncated]`;
	}
	return text;
}

function renderRows(rows: Row[]): string {
	const [first] = rows;
	if (!first) return "(no rows)";
	const columns = Object.keys(first);
	const lines = [columns.join(" | ")];
	for (const row of rows) {
		lines.push(columns.map(col => sanitizeCell(row[col])).join(" | "));
	}
	return lines.join("\n");
}

const SAFE_IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

function assertValidTableName(table: string): void {
	if (!SAFE_IDENTIFIER_RE.test(table)) {
		throw new Error(`Invalid table name: ${table}`);
	}
}

/**
 * JSON body keys become column names and are interpolated directly into SQL
 * identifier position (quoted identifiers only escape embedded quotes, they
 * don't sanitize arbitrary content) — reject anything that isn't a safe
 * identifier before it ever reaches a query string.
 */
function assertValidColumnName(column: string): void {
	if (!SAFE_IDENTIFIER_RE.test(column)) {
		throw new Error(`Invalid column name: ${column}`);
	}
}

function tableExists(db: DatabaseSync, table: string): boolean {
	const row = db
		.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?`)
		.get(table);
	return row !== undefined;
}

function assertHasRowid(db: DatabaseSync, table: string): void {
	const row = db
		.prepare(`SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?`)
		.get(table) as { sql?: string } | undefined;
	if (row?.sql && /\bWITHOUT\s+ROWID\b/i.test(row.sql)) {
		throw new Error(`Table ${table} has no rowid (WITHOUT ROWID); not supported in this scope.`);
	}
}

function withDatabase<T>(absolutePath: string, readOnly: boolean, fn: (db: DatabaseSync) => T): T {
	const db = new DatabaseSync(absolutePath, readOnly ? { readOnly: true } : {});
	try {
		return fn(db);
	} finally {
		db.close();
	}
}

// ── reads ───────────────────────────────────────────────────────────────────

export function listSqliteTables(absolutePath: string): string {
	return withDatabase(absolutePath, true, db => {
		const tables = db
			.prepare(
				`SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name`,
			)
			.all() as Array<{ name: string }>;
		if (tables.length === 0) return "(no tables)";
		return tables
			.map(t => {
				const count = db
					.prepare(`SELECT COUNT(*) AS n FROM (SELECT 1 FROM "${t.name}" LIMIT ?)`)
					.get(ROW_COUNT_PROBE_CAP + 1) as { n: number };
				const rows = count.n > ROW_COUNT_PROBE_CAP ? `${ROW_COUNT_PROBE_CAP}+` : String(count.n);
				return `${t.name} (${rows} rows)`;
			})
			.join("\n");
	});
}

export function readSqliteTable(absolutePath: string, table: string): string {
	assertValidTableName(table);
	return withDatabase(absolutePath, true, db => {
		if (!tableExists(db, table)) throw new Error(`No such table: ${table}`);
		const schema = db
			.prepare(`SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?`)
			.get(table) as { sql: string };
		const rows = db.prepare(`SELECT rowid AS rowid, * FROM "${table}" LIMIT ?`).all(SAMPLE_ROW_LIMIT) as Row[];
		return [schema.sql, "", renderRows(rows)].join("\n");
	});
}

export function readSqliteRow(absolutePath: string, table: string, rowid: string): string | undefined {
	assertValidTableName(table);
	return withDatabase(absolutePath, true, db => {
		if (!tableExists(db, table)) throw new Error(`No such table: ${table}`);
		assertHasRowid(db, table);
		const row = db.prepare(`SELECT rowid AS rowid, * FROM "${table}" WHERE rowid = ?`).get(rowid) as
			| Row
			| undefined;
		if (!row) return undefined;
		return renderRows([row]);
	});
}

// ── writes ──────────────────────────────────────────────────────────────────

function inferColumnType(value: unknown): string {
	if (typeof value === "number") return Number.isInteger(value) ? "INTEGER" : "REAL";
	if (typeof value === "boolean") return "INTEGER";
	return "TEXT";
}

/** DatabaseSync.run()/get() only accept null/number/bigint/string/Buffer bind values. */
function toBindValue(value: unknown): null | number | string | Buffer {
	if (value === null || value === undefined) return null;
	if (typeof value === "boolean") return value ? 1 : 0;
	if (typeof value === "number" || typeof value === "string") return value;
	if (Buffer.isBuffer(value)) return value;
	return JSON.stringify(value);
}

function parseJsonObject(content: string): Row {
	let parsed: unknown;
	try {
		parsed = JSON.parse(content);
	} catch {
		throw new Error("Content must be a JSON object encoded as a string.");
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("Content must be a JSON object encoded as a string.");
	}
	return parsed as Row;
}

export function insertSqliteRow(absolutePath: string, table: string, content: string): string {
	assertValidTableName(table);
	const values = parseJsonObject(content);
	const columns = Object.keys(values);
	if (columns.length === 0) throw new Error("Insert requires at least one column in the JSON body.");
	for (const col of columns) assertValidColumnName(col);
	return withDatabase(absolutePath, false, db => {
		if (!tableExists(db, table)) {
			const columnDefs = columns.map(col => `"${col}" ${inferColumnType(values[col])}`).join(", ");
			db.exec(`CREATE TABLE IF NOT EXISTS "${table}" (${columnDefs})`);
		}
		const placeholders = columns.map(() => "?").join(", ");
		const columnList = columns.map(col => `"${col}"`).join(", ");
		const stmt = db.prepare(`INSERT INTO "${table}" (${columnList}) VALUES (${placeholders})`);
		const result = stmt.run(...(columns.map(col => toBindValue(values[col])) as never[]));
		return `Inserted into ${table} (rowid ${result.lastInsertRowid})`;
	});
}

export function updateSqliteRow(
	absolutePath: string,
	table: string,
	rowid: string,
	content: string,
): string {
	assertValidTableName(table);
	const values = parseJsonObject(content);
	const columns = Object.keys(values);
	if (columns.length === 0) throw new Error("Update requires at least one column in the JSON body.");
	for (const col of columns) assertValidColumnName(col);
	return withDatabase(absolutePath, false, db => {
		if (!tableExists(db, table)) throw new Error(`No such table: ${table}`);
		assertHasRowid(db, table);
		const assignments = columns.map(col => `"${col}" = ?`).join(", ");
		const stmt = db.prepare(`UPDATE "${table}" SET ${assignments} WHERE rowid = ?`);
		const result = stmt.run(
			...(columns.map(col => toBindValue(values[col])) as never[]),
			rowid as unknown as never,
		);
		if (result.changes === 0) throw new Error(`No such row: ${table}:${rowid}`);
		return `Updated ${table}:${rowid} (${result.changes} row changed)`;
	});
}

export function deleteSqliteRow(absolutePath: string, table: string, rowid: string): string {
	assertValidTableName(table);
	return withDatabase(absolutePath, false, db => {
		if (!tableExists(db, table)) throw new Error(`No such table: ${table}`);
		assertHasRowid(db, table);
		const stmt = db.prepare(`DELETE FROM "${table}" WHERE rowid = ?`);
		const result = stmt.run(rowid as unknown as never);
		if (result.changes === 0) throw new Error(`No such row: ${table}:${rowid}`);
		return `Deleted ${table}:${rowid}`;
	});
}
