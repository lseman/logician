// ── Extension state persistence ──────────────────────────────────────────────
// Per-extension key-value store backed by JSON. State survives across turns
// and session restarts. Stored under ~/.logician/extensions/<ext-id>/state.db.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
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

	const dbPath = join(storageRoot, extId, "state.json");
	const dir = dirname(dbPath);
	if (!existsSync(dir)) {
		mkdirSync(dir, { recursive: true });
	}

	const read = (): Record<string, string> => {
		if (!existsSync(dbPath)) return {};
		try {
			const parsed = JSON.parse(readFileSync(dbPath, "utf8")) as unknown;
			return parsed && typeof parsed === "object" && !Array.isArray(parsed)
				? parsed as Record<string, string>
				: {};
		} catch (e: unknown) {
			return {};
		}
	};
	const write = (data: Record<string, string>): void => {
		writeFileSync(dbPath, JSON.stringify(data, null, 2));
	};

	return {
		get(key: string): string | null {
			return read()[key] ?? null;
		},
		set(key: string, value: string): void {
			const data = read();
			data[key] = value;
			write(data);
		},
		delete(key: string): boolean {
			const data = read();
			if (!(key in data)) return false;
			delete data[key];
			write(data);
			return true;
		},
		keys(): string[] {
			return Object.keys(read()).sort();
		},
	};
}

export function createExtensionState(extId: string): StateDB {
	return openStateDB(extId);
}
