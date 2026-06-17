// ── Extension state persistence ──────────────────────────────────────────────
// Per-extension key-value store backed by SQLite. State survives across turns
// and session restarts. Stored under ~/.logician/extensions/<ext-id>/state.db.

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

interface StateDB {
	get(key: string): string | null;
	set(key: string, value: string): void;
	delete(key: string): boolean;
	keys(): string[];
}

function openStateDB(extId: string): StateDB {
	const storageRoot = process.env.XDG_DATA_HOME
		? join(process.env.XDG_DATA_HOME, "logician", "extensions")
		: join(homedir(), ".local", "share", "logician", "extensions");

	const dbPath = join(storageRoot, extId, "state.db");
	const dir = dirname(dbPath);
	if (!existsSync(dir)) {
		mkdirSync(dir, { recursive: true });
	}

	const db = new Database(dbPath);
	db.exec("PRAGMA journal_mode = WAL");
	db.exec(`
		CREATE TABLE IF NOT EXISTS kv (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)
	`);

	return {
		get(key: string): string | null {
			const row = db.prepare("SELECT value FROM kv WHERE key = ?").get(key) as { value: string } | undefined;
			return row?.value ?? null;
		},
		set(key: string, value: string): void {
			db.prepare("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)").run(key, value);
		},
		delete(key: string): boolean {
			const result = db.prepare("DELETE FROM kv WHERE key = ?").run(key);
			return (result as { changes: number }).changes > 0;
		},
		keys(): string[] {
			const rows = db.prepare("SELECT key FROM kv ORDER BY key").all() as Array<{ key: string }>;
			return rows.map((r) => r.key);
		},
	};
}

export function createExtensionState(extId: string): StateDB {
	return openStateDB(extId);
}
