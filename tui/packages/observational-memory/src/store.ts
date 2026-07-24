// ── In-memory store ──────────────────────────────────────────────────────
// Core memory store implementing the MemoryStore interface.
// Manages observations, reflections, and drops in-memory.
// Persists to file via FilePersistence.

import type {
	FoldedMemory,
	MemoryStatus,
	MemoryStore as MemoryStoreInterface,
	Observation,
	ObservationRecord,
	Reflection,
	ReflectionRecord,
	MemoryStoreEvent,
} from "./types.ts";

// Re-export MemoryStore interface for consumers
export type MemoryStore = MemoryStoreInterface;
import { FilePersistence } from "./persistence.ts";
import { OM_FOLDED } from "./types.ts";

export interface StoreOptions {
	/** Persistence backend. Defaults to file-based. */
	persistence?: FilePersistence;
	/** Target token count for observation pool (below which no drops occur). */
	observationsPoolTargetTokens?: number;
}

const DEFAULT_TARGET_TOKENS = 10_000;

export class MemoryStoreImpl implements MemoryStoreInterface {
	private observations: Map<string, Observation> = new Map();
	private reflections: Map<string, Reflection> = new Map();
	private dropped: Set<string> = new Set();
	private records: Array<ObservationRecord | ReflectionRecord> = [];
	private dropRecords: string[][] = [];

	private persistence: FilePersistence;
	private targetTokens: number;
	private listeners = new Set<(event: MemoryStoreEvent) => void>();
	private persistScheduled = false;

	constructor(options: StoreOptions = {}) {
		this.persistence = options.persistence ?? new FilePersistence();
		this.targetTokens =
			options.observationsPoolTargetTokens ?? DEFAULT_TARGET_TOKENS;
	}

	// ── Record operations ────────────────────────────────────────────────

	recordObservations(observations: Observation[], coversUpToId: string): void {
		const existingContent = new Set(
			Array.from(this.observations.values(), (item) => normalizeContent(item.content)),
		);
		const added = observations.filter((observation) => {
			const normalized = normalizeContent(observation.content);
			if (this.observations.has(observation.id) || existingContent.has(normalized)) {
				return false;
			}
			existingContent.add(normalized);
			return true;
		});
		if (added.length === 0) return;
		for (const obs of added) {
			this.observations.set(obs.id, obs);
		}
		this.records.push({ observations: added, coversUpToId });
		this.persist();
		this.emit({ type: "observations_added", observations: added });
	}

	recordReflections(reflections: Reflection[], coversUpToId: string): void {
		const added = reflections.filter((reflection) => !this.reflections.has(reflection.id));
		if (added.length === 0) return;
		for (const ref of added) {
			this.reflections.set(ref.id, ref);
		}
		this.records.push({ reflections: added, coversUpToId });
		this.persist();
		this.emit({ type: "reflections_added", reflections: added });
	}

	recordDrops(observationIds: string[], _coversUpToId: string): void {
		const added = observationIds.filter(
			(id) => this.observations.has(id) && !this.dropped.has(id),
		);
		if (added.length === 0) return;
		for (const id of added) {
			this.dropped.add(id);
		}
		this.dropRecords.push(added);
		this.persist();
		this.emit({ type: "observations_dropped", observationIds: added });
	}

	// ── Query operations ─────────────────────────────────────────────────

	fold(): FoldedMemory {
		return {
			type: OM_FOLDED,
			version: 1,
			fullFold: true,
			observations: Array.from(this.observations.values()),
			reflections: Array.from(this.reflections.values()),
			droppedObservationIds: Array.from(this.dropped),
		};
	}

	getActiveObservations(): Observation[] {
		return Array.from(this.observations.values()).filter(
			(obs) => !this.dropped.has(obs.id),
		);
	}

	getReflections(): Reflection[] {
		return Array.from(this.reflections.values());
	}

	isDropped(id: string): boolean {
		return this.dropped.has(id);
	}

	getStatus(): MemoryStatus {
		const active = this.getActiveObservations();
		const activeTokens = active.reduce((sum, o) => sum + o.tokenCount, 0);
		return {
			observationCount: this.observations.size,
			reflectionCount: this.reflections.size,
			droppedCount: this.dropped.size,
			activeObservationTokens: activeTokens,
			observationPoolTargetTokens: this.targetTokens,
		};
	}

	// ── Persistence ──────────────────────────────────────────────────────

	load(path?: string): void {
		if (path) {
			this.persistence = new FilePersistence({ path });
		}
		const folded = this.persistence.load();
		if (!folded) return;
		this.observations.clear();
		this.reflections.clear();
		this.dropped.clear();

		for (const obs of folded.observations) {
			this.observations.set(obs.id, obs);
		}
		for (const ref of folded.reflections) {
			this.reflections.set(ref.id, ref);
		}
		for (const id of folded.droppedObservationIds ?? []) this.dropped.add(id);
	}

	save(path?: string): void {
		if (path) {
			this.persistence = new FilePersistence({ path });
		}
		this.flush();
	}

	/**
	 * Schedule a persist on the next microtask, coalescing writes that happen
	 * within the same synchronous burst (e.g. recordObservations +
	 * recordReflections + recordDrops from one consolidation cycle) into a
	 * single file write instead of one per call.
	 */
	private persist(): void {
		if (this.persistScheduled) return;
		this.persistScheduled = true;
		queueMicrotask(() => {
			this.persistScheduled = false;
			this.writeNow();
		});
	}

	/** Force an immediate, non-debounced write of the current state. */
	flush(): void {
		this.persistScheduled = false;
		this.writeNow();
	}

	private writeNow(): void {
		const folded = this.fold();
		this.persistence.save(folded);
	}

	clear(): void {
		this.observations.clear();
		this.reflections.clear();
		this.dropped.clear();
		this.records = [];
		this.dropRecords = [];
		this.persistScheduled = false;
		this.persistence.clear();
		this.emit({ type: "cleared" });
	}

	// ── Recall helpers ─────────────────────────────────────────────────

	getAllObservations(): Observation[] {
		return Array.from(this.observations.values());
	}

	getAllDroppedIds(): Set<string> {
		return new Set(this.dropped);
	}

	subscribe(listener: (event: MemoryStoreEvent) => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	private emit(event: MemoryStoreEvent): void {
		for (const listener of this.listeners) {
			try {
				listener(event);
			} catch (e: unknown) {
				// Observers must never make an already-persisted memory write fail.
			}
		}
	}
}

function normalizeContent(content: string): string {
	return content.trim().replace(/\s+/g, " ").toLowerCase();
}
