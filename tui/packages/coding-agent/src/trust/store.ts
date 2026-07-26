// ── Trust store — per-directory trust decisions persisted in JSON ─────────────
// Stores trust decisions at ~/.logician/trust.json. Supports walking up the
// directory tree to find the nearest ancestor decision. Null = undecided.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

type TrustFile = Record<string, boolean | null | undefined>;

const TRUST_DIR = join(homedir(), ".logician");
const TRUST_FILE = join(TRUST_DIR, "trust.json");

function readTrustFile(): TrustFile {
	if (!existsSync(TRUST_FILE)) return {};
	try {
		const parsed = JSON.parse(readFileSync(TRUST_FILE, "utf-8"));
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
			return {};
		}
		const data: TrustFile = {};
		for (const [key, value] of Object.entries(parsed)) {
			if (value === true || value === false || value === null) {
				data[key] = value;
			}
		}
		return data;
	} catch (_e: unknown) {
		return {};
	}
}

function writeTrustFile(data: TrustFile): void {
	mkdirSync(TRUST_DIR, { recursive: true });
	const sorted: TrustFile = {};
	for (const key of Object.keys(data).sort()) {
		const value = data[key];
		if (value === true || value === false || value === null) {
			sorted[key] = value;
		}
	}
	writeFileSync(TRUST_FILE, JSON.stringify(sorted, null, 2) + "\n", "utf-8");
}

function normalizeCwd(cwd: string): string {
	return cwd.replace(/\/+$/, "") || "/";
}

function findNearestEntry(data: TrustFile, cwd: string): { path: string; decision: boolean } | null {
	let currentDir = normalizeCwd(cwd);
	while (true) {
		const value = data[currentDir];
		if (value === true || value === false) {
			return { path: currentDir, decision: value };
		}
		const parentDir = dirname(currentDir);
		if (parentDir === currentDir) return null;
		currentDir = parentDir;
	}
}

export interface TrustDecision {
	decision: boolean | null; // null = undecided
	savedPath: string;         // directory where the decision is stored
}

export class TrustStore {
	get(cwd: string): TrustDecision {
		const data = readTrustFile();
		const entry = findNearestEntry(data, cwd);
		if (!entry) return { decision: null, savedPath: cwd };
		return { decision: entry.decision, savedPath: entry.path };
	}

	set(cwd: string, decision: boolean): void {
		const data = readTrustFile();
		data[normalizeCwd(cwd)] = decision;
		writeTrustFile(data);
	}

	setMany(updates: { path: string; decision: boolean | null }[]): void {
		const data = readTrustFile();
		for (const { path, decision } of updates) {
			const key = normalizeCwd(path);
			if (decision === null) {
				delete data[key];
			} else {
				data[key] = decision;
			}
		}
		writeTrustFile(data);
	}

	clear(): void {
		writeTrustFile({});
	}
}
