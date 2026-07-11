// ── File-based persistence ───────────────────────────────────────────────
// JSON file storage for folded memory state. Mirrors pi-om's compaction details
// format but persists to a dedicated file.

import * as fs from "node:fs";
import * as path from "node:path";
import { isObservation, isReflection, type FoldedMemory } from "./types.ts";

const DEFAULT_DIR = ".logician/observational-memory";

function defaultPath(): string {
	return path.join(process.cwd(), DEFAULT_DIR, "memory.json");
}

export interface PersistenceOptions {
	path?: string;
}

export class FilePersistence {
	private filePath: string;

	constructor(options: PersistenceOptions = {}) {
		this.filePath = options.path ?? defaultPath();
	}

	/** Load folded memory from file. Returns undefined if file doesn't exist or is corrupt. */
	load(): FoldedMemory | undefined {
		try {
			if (!fs.existsSync(this.filePath)) return undefined;
			const raw = fs.readFileSync(this.filePath, "utf-8");
			const parsed = JSON.parse(raw) as unknown;
			return this.validateFolded(parsed);
		} catch {
			return undefined;
		}
	}

	/** Persist folded memory to file. Creates directories if needed. */
	save(memory: FoldedMemory): void {
		try {
			const dir = path.dirname(this.filePath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, { recursive: true });
			}
			fs.writeFileSync(this.filePath, JSON.stringify(memory, null, 2));
		} catch {
			// Ignore persistence errors
		}
	}

	/** Clear persisted memory. */
	clear(): void {
		try {
			if (fs.existsSync(this.filePath)) {
				fs.unlinkSync(this.filePath);
			}
		} catch {
			// Ignore
		}
	}

	/** Path to the storage file (for diagnostics). */
	getPath(): string {
		return this.filePath;
	}

	private validateFolded(value: unknown): FoldedMemory | undefined {
		if (!value || typeof value !== "object") return undefined;
		const obj = value as Record<string, unknown>;
		if (obj.type !== "om.folded" || obj.version !== 1) return undefined;
		if (typeof obj.fullFold !== "boolean") return undefined;
		if (!Array.isArray(obj.observations) || !Array.isArray(obj.reflections))
			return undefined;
		if (!obj.observations.every(isObservation) || !obj.reflections.every(isReflection))
			return undefined;
		if (
			obj.droppedObservationIds !== undefined &&
			(!Array.isArray(obj.droppedObservationIds) ||
				!obj.droppedObservationIds.every((id) => typeof id === "string"))
		)
			return undefined;
		return value as FoldedMemory;
	}
}
