// ── File-based persistence ───────────────────────────────────────────────
// JSON file storage for folded memory state. Mirrors pi-om's compaction details
// format but persists to a dedicated file.

import * as fs from "node:fs";
import * as path from "node:path";
import {
	type FoldedMemory,
	isObservation,
	isReflection,
	type MemoryProgress,
	type MemoryWorkerDiagnostics,
	type PersistedKnowledgeGraph,
} from "./types.ts";

const DEFAULT_DIR = ".logician/observational-memory";

function defaultPath(): string {
	return path.join(process.cwd(), DEFAULT_DIR, "memory.json");
}

export interface PersistenceOptions {
	path?: string;
}

export interface PersistenceDiagnostic {
	lastError?: string;
	recoveredFromBackup?: boolean;
}

export class FilePersistence {
	private filePath: string;
	private diagnostic: PersistenceDiagnostic = {};

	constructor(options: PersistenceOptions = {}) {
		this.filePath = options.path ?? defaultPath();
	}

	/** Load folded memory from file. Returns undefined if file doesn't exist or is corrupt. */
	load(): FoldedMemory | undefined {
		this.diagnostic = {};
		try {
			if (!fs.existsSync(this.filePath)) return undefined;
			const loaded = this.loadPath(this.filePath);
			if (loaded) return loaded;
			throw new Error("memory file failed validation");
		} catch (error) {
			const backupPath = this.backupPath();
			try {
				const backup = this.loadPath(backupPath);
				if (backup) {
					this.diagnostic = {
						lastError: errorMessage(error),
						recoveredFromBackup: true,
					};
					return backup;
				}
			} catch {
				// Preserve the primary failure below.
			}
			this.diagnostic = { lastError: errorMessage(error) };
			return undefined;
		}
	}

	/** Persist folded memory to file. Creates directories if needed. */
	save(memory: FoldedMemory): void {
		const tempPath =
			`${this.filePath}.tmp-${process.pid}-${Math.random().toString(36).slice(2)}`;
		try {
			const dir = path.dirname(this.filePath);
			if (!fs.existsSync(dir)) {
				fs.mkdirSync(dir, { recursive: true });
			}
			fs.writeFileSync(tempPath, JSON.stringify(memory, null, 2), "utf8");
			if (
				fs.existsSync(this.filePath) &&
				!this.diagnostic.recoveredFromBackup
			) {
				fs.copyFileSync(this.filePath, this.backupPath());
			}
			fs.renameSync(tempPath, this.filePath);
			this.diagnostic = {};
		} catch (error) {
			try {
				if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
			} catch {
				// Best-effort cleanup; the original failure is more useful.
			}
			this.diagnostic = { lastError: errorMessage(error) };
		}
	}

	/** Clear persisted memory. */
	clear(): void {
		try {
			if (fs.existsSync(this.filePath)) {
				fs.unlinkSync(this.filePath);
			}
			if (fs.existsSync(this.backupPath())) {
				fs.unlinkSync(this.backupPath());
			}
			this.diagnostic = {};
		} catch {
			// Ignore
		}
	}

	/** Path to the storage file (for diagnostics). */
	getPath(): string {
		return this.filePath;
	}

	getDiagnostic(): PersistenceDiagnostic {
		return { ...this.diagnostic };
	}

	private backupPath(): string {
		return `${this.filePath}.bak`;
	}

	private loadPath(filePath: string): FoldedMemory | undefined {
		if (!fs.existsSync(filePath)) return undefined;
		const raw = fs.readFileSync(filePath, "utf-8");
		const parsed = JSON.parse(raw) as unknown;
		return this.validateFolded(parsed);
	}

	private validateFolded(value: unknown): FoldedMemory | undefined {
		if (!value || typeof value !== "object") return undefined;
		const obj = value as Record<string, unknown>;
		if (obj.type !== "om.folded" || obj.version !== 1) return undefined;
		if (typeof obj.fullFold !== "boolean") return undefined;
		if (!Array.isArray(obj.observations) || !Array.isArray(obj.reflections))
			return undefined;
		if (
			!obj.observations.every(isObservation) ||
			!obj.reflections.every(isReflection)
		)
			return undefined;
		if (
			obj.droppedObservationIds !== undefined &&
			(!Array.isArray(obj.droppedObservationIds) ||
				!obj.droppedObservationIds.every((id) => typeof id === "string"))
		)
			return undefined;
		const progress = validateProgress(obj.progress);
		const diagnostics = validateDiagnostics(obj.diagnostics);
		const knowledgeGraph = validateKnowledgeGraph(obj.knowledgeGraph);
		if (
			(obj.progress !== undefined && !progress) ||
			(obj.diagnostics !== undefined && !diagnostics) ||
			(obj.knowledgeGraph !== undefined && !knowledgeGraph)
		)
			return undefined;
		return {
			type: "om.folded",
			version: 1,
			fullFold: obj.fullFold,
			observations: obj.observations,
			reflections: obj.reflections,
			...(obj.droppedObservationIds
				? { droppedObservationIds: obj.droppedObservationIds as string[] }
				: {}),
			...(progress ? { progress } : {}),
			...(diagnostics ? { diagnostics } : {}),
			...(knowledgeGraph ? { knowledgeGraph } : {}),
		};
	}
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function optionalString(value: unknown): value is string | undefined {
	return value === undefined || typeof value === "string";
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function validateProgress(value: unknown): MemoryProgress | undefined {
	if (value === undefined) return undefined;
	if (!isObject(value)) return undefined;
	if (
		!optionalString(value.observationCoverageId) ||
		!optionalString(value.reflectionCoverageId) ||
		!optionalString(value.dropCoverageId)
	)
		return undefined;
	return {
		...(value.observationCoverageId
			? { observationCoverageId: value.observationCoverageId }
			: {}),
		...(value.reflectionCoverageId
			? { reflectionCoverageId: value.reflectionCoverageId }
			: {}),
		...(value.dropCoverageId ? { dropCoverageId: value.dropCoverageId } : {}),
	};
}

function validateDiagnostics(
	value: unknown,
): MemoryWorkerDiagnostics | undefined {
	if (value === undefined) return undefined;
	if (!isObject(value)) return undefined;
	if (
		(value.lastStage !== undefined &&
			!["observer", "reflector", "dropper"].includes(
				String(value.lastStage),
			)) ||
		!optionalString(value.lastRunAt) ||
		!optionalString(value.lastError) ||
		!optionalString(value.lastPersistenceError) ||
		(value.recoveredFromBackup !== undefined &&
			typeof value.recoveredFromBackup !== "boolean")
	)
		return undefined;
	return {
		...(value.lastStage
			? {
					lastStage: value.lastStage as "observer" | "reflector" | "dropper",
				}
			: {}),
		...(value.lastRunAt ? { lastRunAt: value.lastRunAt } : {}),
		...(value.lastError ? { lastError: value.lastError } : {}),
		...(value.lastPersistenceError
			? { lastPersistenceError: value.lastPersistenceError }
			: {}),
		...(value.recoveredFromBackup !== undefined
			? { recoveredFromBackup: value.recoveredFromBackup }
			: {}),
	};
}

function validateKnowledgeGraph(
	value: unknown,
): PersistedKnowledgeGraph | undefined {
	if (value === undefined) return undefined;
	if (
		!isObject(value) ||
		!Array.isArray(value.nodes) ||
		!Array.isArray(value.edges)
	)
		return undefined;
	const nodes = value.nodes.filter(
		(node): node is PersistedKnowledgeGraph["nodes"][number] =>
			isObject(node) &&
			typeof node.id === "string" &&
			(node.type === "observation" || node.type === "reflection") &&
			typeof node.content === "string" &&
			isObject(node.metadata) &&
			typeof node.tokens === "number" &&
			Number.isFinite(node.tokens),
	);
	const edges = value.edges.filter(
		(edge): edge is PersistedKnowledgeGraph["edges"][number] =>
			isObject(edge) &&
			typeof edge.source === "string" &&
			typeof edge.target === "string" &&
			edge.relationship === "supported_by" &&
			typeof edge.weight === "number" &&
			Number.isFinite(edge.weight) &&
			isObject(edge.metadata),
	);
	if (
		nodes.length !== value.nodes.length ||
		edges.length !== value.edges.length
	)
		return undefined;
	return { nodes, edges };
}
