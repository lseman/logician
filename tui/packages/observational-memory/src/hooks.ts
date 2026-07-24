// ── Hook integration ─────────────────────────────────────────────────────
// Registers hooks on the extension event bus to trigger
// the consolidation pipeline.

import type { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import type { MemoryStore } from "./store.ts";
import type { ConsolidationPipeline } from "./consolidation.ts";
import { formatMemoryContext } from "./search.ts";

export interface HookOptions {
	/** Token thresholds */
	observeAfterTokens?: number;
	reflectAfterTokens?: number;
	compactAfterTokens?: number;
	/** Maximum observational-memory tokens injected before a turn. */
	memoryContextMaxTokens?: number;
}

export interface HookContext {
	/** Extension event bus for lifecycle events */
	extensionBus: ExtensionEventBus;
	/** Memory store instance */
	memoryStore: MemoryStore;
	/** Consolidation pipeline (observer/reflector/dropper) */
	pipeline: ConsolidationPipeline;
	/** Optional token thresholds */
	options?: HookOptions;
	/** Current raw token count in session */
	currentTokens: () => number;
	getSourceEntries?: () => Array<{ id: string; role: string; content: string }>;
}

/**
 * Register consolidation hooks on the extension event bus.
 * Triggers the pipeline on turn_end when thresholds are met.
 */
export function registerConsolidationHooks(ctx: HookContext): () => void {
	const { extensionBus, pipeline, options } = ctx;

	const observeThreshold = options?.observeAfterTokens ?? 10_000;
	const reflectThreshold = options?.reflectAfterTokens ?? 20_000;
	let lastObservationRunTokens = 0;
	let lastReflectedObservationCount = ctx.memoryStore.getActiveObservations().length;
	const processedSourceIds = new Set<string>();

	const launchConsolidation = async () => {
		try {
			const currentTokens = ctx.currentTokens();
			const observationDue =
				currentTokens - lastObservationRunTokens >= observeThreshold;
			const activeObservations = ctx.memoryStore.getActiveObservations();
			const reflectionDue =
				currentTokens >= reflectThreshold &&
				activeObservations.length > lastReflectedObservationCount;
			if (!observationDue && !reflectionDue) return undefined;
			const sourceEntries = (ctx.getSourceEntries?.() ?? []).filter(
				(entry) => !processedSourceIds.has(entry.id),
			);
			const result = await pipeline.maybeLaunch({
				observeThreshold: observationDue ? 0 : Number.POSITIVE_INFINITY,
				reflectThreshold: reflectionDue ? 0 : Number.POSITIVE_INFINITY,
				currentTokens,
				observations: activeObservations,
				reflections: ctx.memoryStore.getReflections(),
				sourceEntries,
			});
			if (!result?.ran) return undefined;
			const coverageId = sourceEntries.at(-1)?.id ?? "unknown";
			ctx.memoryStore.recordObservations(result.observations, coverageId);
			ctx.memoryStore.recordReflections(result.reflections, coverageId);
			ctx.memoryStore.recordDrops(result.droppedObservationIds, coverageId);
			if (observationDue) {
				lastObservationRunTokens = currentTokens;
				for (const entry of sourceEntries) processedSourceIds.add(entry.id);
			}
			if (reflectionDue) {
				lastReflectedObservationCount = ctx.memoryStore.getActiveObservations().length;
			}
		} catch (error) {
			console.error("[observational-memory] Consolidation error:", error);
		}
		// Return undefined — no result type modification needed
		return undefined;
	};

	// Register on turn_end
	const unsubTurnEnd = extensionBus.on("turn_end", () => {
		void launchConsolidation();
		return undefined;
	});

	// Register on before_agent_start (session boundary)
	const unsubBeforeAgentStart = extensionBus.on("before_agent_start", (event) => {
		const memoryContext = formatMemoryContext(ctx.memoryStore, event.prompt, {
			maxTokens: options?.memoryContextMaxTokens ?? 1_000,
		});
		if (!memoryContext) return undefined;
		return {
			systemPrompt: `${event.systemPrompt}\n\n${memoryContext}`,
		};
	});

	return () => {
		unsubTurnEnd();
		unsubBeforeAgentStart();
	};
}

/**
 * Register compaction hook to persist memory before context compaction.
 */
export function registerCompactionHook(
	extensionBus: ExtensionEventBus,
	memoryStore: MemoryStore,
): () => void {
	const off = extensionBus.on("session_before_compact", () => {
		// Persist folded memory before compaction
		memoryStore.save();
		return undefined;
	});
	return off;
}
