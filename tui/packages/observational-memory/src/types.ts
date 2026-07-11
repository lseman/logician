// ── Observational Memory Types (V3) ──────────────────────────────────────
// Structured memory model: observations, reflections, and drops.
// Mirrors pi-observational-memory V3 data model.

export const OM_OBSERVATIONS_RECORDED = "om.observations.recorded";
export const OM_REFLECTIONS_RECORDED = "om.reflections.recorded";
export const OM_OBSERVATIONS_DROPPED = "om.observations.dropped";
export const OM_FOLDED = "om.folded";

export const RELEVANCE_VALUES = ["low", "medium", "high", "critical"] as const;
export type Relevance = (typeof RELEVANCE_VALUES)[number];

export const MEMORY_ID_PATTERN = /^[a-f0-9]{12}$/;

export type Observation = {
	id: string;
	content: string;
	timestamp: string;
	relevance: Relevance;
	sourceEntryIds: string[];
	tokenCount: number;
};

export type Reflection = {
	id: string;
	content: string;
	supportingObservationIds: string[];
	tokenCount: number;
};

export type ObservationRecord = {
	observations: Observation[];
	coversUpToId: string;
};

export type ReflectionRecord = {
	reflections: Reflection[];
	coversUpToId: string;
};

export type DropRecord = {
	observationIds: string[];
	coversUpToId: string;
};

export type FoldedMemory = {
	type: typeof OM_FOLDED;
	version: 1;
	fullFold: boolean;
	observations: Observation[];
	reflections: Reflection[];
	/** Tombstones are persisted so dropped observations stay inactive after restart. */
	droppedObservationIds?: string[];
};

export type MemoryStoreEvent =
	| { type: "observations_added"; observations: Observation[] }
	| { type: "reflections_added"; reflections: Reflection[] }
	| { type: "observations_dropped"; observationIds: string[] }
	| { type: "cleared" };

// ── Validation helpers ───────────────────────────────────────────────────

function isNonEmptyString(value: unknown): value is string {
	return typeof value === "string" && value.length > 0;
}

function isRelevance(value: unknown): value is Relevance {
	return (
		typeof value === "string" &&
		(RELEVANCE_VALUES as readonly string[]).includes(value)
	);
}

function isMemoryId(value: unknown): value is string {
	return typeof value === "string" && MEMORY_ID_PATTERN.test(value);
}

function isTokenCount(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isStringArray(value: unknown): value is string[] {
	return (
		Array.isArray(value) && value.length > 0 && value.every(isNonEmptyString)
	);
}

export function isObservation(value: unknown): value is Observation {
	if (!value || typeof value !== "object") return false;
	const o = value as Record<string, unknown>;
	return (
		isMemoryId(o.id) &&
		isNonEmptyString(o.content) &&
		isNonEmptyString(o.timestamp) &&
		isRelevance(o.relevance) &&
		isStringArray(o.sourceEntryIds) &&
		isTokenCount(o.tokenCount)
	);
}

export function isReflection(value: unknown): value is Reflection {
	if (!value || typeof value !== "object") return false;
	const r = value as Record<string, unknown>;
	return (
		isMemoryId(r.id) &&
		isNonEmptyString(r.content) &&
		!/\r|\n/.test(r.content) &&
		isStringArray(r.supportingObservationIds) &&
		isTokenCount(r.tokenCount)
	);
}

// ── Memory Store Interface ───────────────────────────────────────────────

export type MemoryStatus = {
	observationCount: number;
	reflectionCount: number;
	droppedCount: number;
	activeObservationTokens: number;
	observationPoolTargetTokens: number;
};

export interface MemoryStore {
	/** Record observations from the observer stage. */
	recordObservations(observations: Observation[], coversUpToId: string): void;
	/** Record reflections from the reflector stage. */
	recordReflections(reflections: Reflection[], coversUpToId: string): void;
	/** Record dropped observations (tombstones, not deletion). */
	recordDrops(observationIds: string[], coversUpToId: string): void;
	/** Get the current folded memory state. */
	fold(): FoldedMemory;
	/** Get active observations (not dropped). */
	getActiveObservations(): Observation[];
	/** Get all reflections. */
	getReflections(): Reflection[];
	/** Check if an observation is dropped. */
	isDropped(id: string): boolean;
	/** Get status summary. */
	getStatus(): MemoryStatus;
	/** Load from file. */
	load(path?: string): void;
	/** Persist to file. */
	save(path?: string): void;
	/** Clear all memory. */
	clear(): void;
	/** Get all observations including dropped (for recall). */
	getAllObservations(): Observation[];
	/** Get all dropped observation IDs (for recall). */
	getAllDroppedIds(): Set<string>;
	/** Subscribe to durable memory changes. */
	subscribe(listener: (event: MemoryStoreEvent) => void): () => void;
}
