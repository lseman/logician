// ── Recall mechanism ─────────────────────────────────────────────────────
// Exact evidence recovery by 12-char hex memory ID.
// Mirrors pi-observational-memory's recall functionality.

import {
	MEMORY_ID_PATTERN,
	type Observation,
	type Reflection,
} from "./types.ts";

export type RecallStatus =
	| "ok"
	| "partial"
	| "invalid_id"
	| "not_found"
	| "no_source"
	| "source_unavailable";

export type RecalledObservation = {
	observation: Observation;
	/** Active / dropped (tombstoned) / source_unavailable / no_source */
	status: "active" | "dropped" | "source_unavailable" | "no_source";
	sourceEntries: RecallSourceEntry[];
	missingSourceEntryIds: string[];
	nonSourceEntryIds: string[];
};

export type RecalledReflection = {
	reflection: Reflection;
};

export type RecallSourceEntry = {
	id: string;
	origin: string;
	timestamp: string;
	tokens: number;
	qualifiers: string[];
	content?: string;
};

export type RecallResult =
	| {
			status: "not_found";
			memoryId: string;
			kind: undefined;
			reflections: [];
			observations: [];
			sourceEntries: [];
			missingSourceEntryIds: [];
			nonSourceEntryIds: [];
			missingSupportingObservationIds: [];
			collision: false;
			partial: false;
	  }
	| {
			status: "found";
			memoryId: string;
			kind: "observation" | "reflection" | "mixed";
			reflections: RecalledReflection[];
			observations: RecalledObservation[];
			sourceEntries: RecallSourceEntry[];
			missingSourceEntryIds: string[];
			nonSourceEntryIds: string[];
			missingSupportingObservationIds: string[];
			collision: boolean;
			partial: boolean;
	  };

export function isValidMemoryId(id: string): boolean {
	return MEMORY_ID_PATTERN.test(id);
}

/** Resolve source entries for an observation from available session data. */
function resolveSources(
	obs: Observation,
	entries: Array<{
		id: string;
		type: string;
		origin: string;
		timestamp: string;
		content?: string;
	}>,
): {
	sourceEntries: RecallSourceEntry[];
	missingSourceEntryIds: string[];
	nonSourceEntryIds: string[];
} {
	const byId = new Map(entries.map((e) => [e.id, e]));
	const sourceEntries: RecallSourceEntry[] = [];
	const missing: string[] = [];
	const nonSource: string[] = [];

	for (const sourceId of obs.sourceEntryIds) {
		const entry = byId.get(sourceId);
		if (!entry) {
			missing.push(sourceId);
			continue;
		}
		if (!["message", "custom_message", "branch_summary"].includes(entry.type)) {
			nonSource.push(sourceId);
			continue;
		}
		sourceEntries.push({
			id: entry.id,
			origin: entry.origin,
			timestamp: entry.timestamp,
			tokens: Math.ceil((entry.content ?? "").length / 4),
			qualifiers: [],
			content: entry.content,
		});
	}

	return {
		sourceEntries,
		missingSourceEntryIds: missing,
		nonSourceEntryIds: nonSource,
	};
}

/**
 * Recall a memory item by ID from the memory store data.
 *
 * @param memoryId - 12-char hex ID
 * @param observations - All observations (active + dropped)
 * @param reflections - All reflections
 * @param droppedIds - Set of dropped observation IDs
 * @param sourceEntries - Available source entries from session branch
 */
export function recallMemory(
	memoryId: string,
	observations: Observation[],
	reflections: Reflection[],
	droppedIds: Set<string>,
	sourceEntries: Array<{
		id: string;
		type: string;
		origin: string;
		timestamp: string;
		content?: string;
	}>,
): RecallResult {
	if (!MEMORY_ID_PATTERN.test(memoryId)) {
		return {
			status: "not_found",
			memoryId,
			kind: undefined,
			reflections: [],
			observations: [],
			sourceEntries: [],
			missingSourceEntryIds: [],
			nonSourceEntryIds: [],
			missingSupportingObservationIds: [],
			collision: false,
			partial: false,
		};
	}

	const directObs = observations.filter((o) => o.id === memoryId);
	const refMatches = reflections.filter((r) => r.id === memoryId);

	if (directObs.length === 0 && refMatches.length === 0) {
		return {
			status: "not_found",
			memoryId,
			kind: undefined,
			reflections: [],
			observations: [],
			sourceEntries: [],
			missingSourceEntryIds: [],
			nonSourceEntryIds: [],
			missingSupportingObservationIds: [],
			collision: false,
			partial: false,
		};
	}

	// Build lookup
	const obsById = new Map<string, Observation>();
	for (const o of observations) {
		if (!obsById.has(o.id)) obsById.set(o.id, o);
	}

	// Resolve direct observation matches
	const recalled: RecalledObservation[] = [];
	for (const obs of directObs) {
		const src = resolveSources(obs, sourceEntries);
		recalled.push({
			observation: obs,
			status: droppedIds.has(obs.id) ? "dropped" : "active",
			...src,
		});
	}

	// Resolve observations referenced by reflections
	const missingSupporting: string[] = [];
	for (const ref of refMatches) {
		for (const obsId of ref.supportingObservationIds) {
			const obs = obsById.get(obsId);
			if (!obs) {
				missingSupporting.push(obsId);
				continue;
			}
			const already = recalled.some((r) => r.observation.id === obsId);
			if (!already) {
				const src = resolveSources(obs, sourceEntries);
				recalled.push({
					observation: obs,
					status: droppedIds.has(obs.id) ? "dropped" : "active",
					...src,
				});
			}
		}
	}

	const reflectedRefs: RecalledReflection[] = refMatches.map((r) => ({
		reflection: r,
	}));

	// Deduplicate
	const allSrc = recalled.flatMap((o) => o.sourceEntries);
	const uniqueSources: RecallSourceEntry[] = [];
	const srcIds = new Set<string>();
	for (const s of allSrc) {
		if (!srcIds.has(s.id)) {
			srcIds.add(s.id);
			uniqueSources.push(s);
		}
	}

	const allMissing = [
		...new Set(recalled.flatMap((o) => o.missingSourceEntryIds)),
	];
	const allNonSource = [
		...new Set(recalled.flatMap((o) => o.nonSourceEntryIds)),
	];
	const uniqueMissingSupporting = [...new Set(missingSupporting)];

	const matchCount = directObs.length + refMatches.length;

	return {
		status: "found",
		memoryId,
		kind:
			directObs.length > 0 && refMatches.length > 0
				? "mixed"
				: refMatches.length > 0
					? "reflection"
					: "observation",
		reflections: reflectedRefs,
		observations: recalled,
		sourceEntries: uniqueSources,
		missingSourceEntryIds: allMissing,
		nonSourceEntryIds: allNonSource,
		missingSupportingObservationIds: uniqueMissingSupporting,
		collision: matchCount > 1,
		partial:
			allMissing.length > 0 ||
			allNonSource.length > 0 ||
			uniqueMissingSupporting.length > 0,
	};
}

// ── Rendering ────────────────────────────────────────────────────────────

export function formatRecallResult(result: RecallResult): string {
	if (result.status === "not_found") {
		return `No observation or reflection with id ${result.memoryId} was found.`;
	}

	const lines: string[] = [];

	// Header
	const obsCount = result.observations.length;
	const refCount = result.reflections.length;
	const parts = ["✓ success"];
	if (refCount > 0)
		parts.push(`${refCount} reflection${refCount > 1 ? "s" : ""}`);
	if (obsCount > 0)
		parts.push(`${obsCount} observation${obsCount > 1 ? "s" : ""}`);
	lines.push(parts.join(" · "));
	lines.push("");

	// Reflections
	for (const { reflection } of result.reflections) {
		lines.push(`[${reflection.id}] ${reflection.content}`);
	}

	// Observations
	for (const { observation, status } of result.observations) {
		const statusStr = status === "dropped" ? " [dropped]" : "";
		lines.push(
			`[${observation.id}]${statusStr} [${observation.relevance}] ${observation.content}`,
		);
	}

	// Notes
	const notes: string[] = [];
	if (result.collision)
		notes.push(`ID collision: multiple items share ${result.memoryId}`);
	if (result.observations.some((o) => o.status === "dropped")) {
		notes.push(
			"Some observations are dropped from active memory but remain recallable",
		);
	}
	if (result.missingSourceEntryIds.length > 0) {
		notes.push(
			`Missing source entries: ${result.missingSourceEntryIds.join(", ")}`,
		);
	}
	if (result.nonSourceEntryIds.length > 0) {
		notes.push(`Non-source entries: ${result.nonSourceEntryIds.join(", ")}`);
	}
	if (result.missingSupportingObservationIds.length > 0) {
		notes.push(
			`Missing supporting observations: ${result.missingSupportingObservationIds.join(", ")}`,
		);
	}
	if (notes.length > 0) {
		lines.push("");
		lines.push(...notes);
	}

	// Source entries
	if (result.sourceEntries.length > 0) {
		lines.push("");
		lines.push("Sources:");
		for (const src of result.sourceEntries) {
			const tag = src.origin.toLowerCase().includes("user")
				? "user"
				: src.origin.toLowerCase().includes("assistant")
					? "assistant"
					: "tool";
			lines.push(`  [${tag}] ${src.timestamp} ~${src.tokens}t — ${src.id}`);
			if (src.content) {
				const preview = src.content.slice(0, 200).replace(/\n/g, " ");
				lines.push(`    ${preview}${src.content.length > 200 ? "..." : ""}`);
			}
		}
	}

	return lines.join("\n");
}
