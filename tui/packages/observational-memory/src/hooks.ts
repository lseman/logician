// ── Hook integration ─────────────────────────────────────────────────────
// Registers hooks on the extension event bus to trigger
// the consolidation pipeline.

import type { ExtensionEventBus } from "@logician/agent-core/hooks/extensions";
import type { ConsolidationPipeline, SourceEntry } from "./consolidation.ts";
import { formatMemoryContext } from "./search.ts";
import type { MemoryStore } from "./store.ts";
import { estimateTokens } from "./tokens.ts";

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
	getSourceEntries?: () => SourceEntry[];
	/** Recent conversation text to enrich memory retrieval beyond the latest prompt. */
	getRetrievalContext?: () => string;
}

/**
 * Register consolidation hooks on the extension event bus.
 * Triggers the pipeline on turn_end when thresholds are met.
 */
export function registerConsolidationHooks(ctx: HookContext): () => void {
	const { extensionBus, pipeline, options } = ctx;

	const observeThreshold = options?.observeAfterTokens ?? 10_000;
	const reflectThreshold = options?.reflectAfterTokens ?? 20_000;
	const launchConsolidation = async () => {
		try {
			const progress = ctx.memoryStore.getProgress();
			const allSources = ctx.getSourceEntries?.() ?? [];
			const observationSources = entriesAfter(
				allSources,
				progress.observationCoverageId,
			);
			const reflectionSources = entriesAfter(
				allSources,
				progress.reflectionCoverageId,
			);
			const observationTokens = sourceTokens(observationSources);
			const reflectionTokens = sourceTokens(reflectionSources);
			const observationDue = observationTokens >= observeThreshold;
			const activeObservations = ctx.memoryStore.getActiveObservations();
			const reflectionDue =
				!observationDue &&
				reflectionTokens >= reflectThreshold &&
				activeObservations.length > 0;
			if (!observationDue && !reflectionDue) return undefined;
			const result = await pipeline.maybeLaunch({
				observeDue: observationDue,
				reflectDue: reflectionDue,
				observations: activeObservations,
				reflections: ctx.memoryStore.getReflections(),
				sourceEntries: observationSources,
			});
			if (!result?.ran) return undefined;
			const observationCoverageId = observationSources.at(-1)?.id;
			if (result.observations.length > 0 && observationCoverageId) {
				ctx.memoryStore.recordObservations(
					result.observations,
					observationCoverageId,
				);
				ctx.memoryStore.setProgress({ observationCoverageId });
			}
			const reflectionCoverageId =
				ctx.memoryStore.getProgress().observationCoverageId;
			if (result.reflections.length > 0 && reflectionCoverageId) {
				ctx.memoryStore.recordReflections(
					result.reflections,
					reflectionCoverageId,
				);
				ctx.memoryStore.setProgress({ reflectionCoverageId });
			}
			if (result.droppedObservationIds.length > 0 && reflectionCoverageId) {
				ctx.memoryStore.recordDrops(
					result.droppedObservationIds,
					reflectionCoverageId,
				);
				ctx.memoryStore.setProgress({ dropCoverageId: reflectionCoverageId });
			}
			const status = pipeline.getStatus();
			ctx.memoryStore.setDiagnostics({
				lastStage: status.stage,
				lastRunAt: status.lastRunAt,
			});
		} catch (error) {
			console.error("[observational-memory] Consolidation error:", error);
			const status = pipeline.getStatus();
			ctx.memoryStore.setDiagnostics({
				lastStage: status.stage,
				lastRunAt: status.lastRunAt ?? new Date().toISOString(),
				lastError:
					status.lastError ??
					(error instanceof Error ? error.message : String(error)),
			});
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
	const unsubBeforeAgentStart = extensionBus.on(
		"before_agent_start",
		(event) => {
			const query = [event.prompt, ctx.getRetrievalContext?.() ?? ""]
				.filter(Boolean)
				.join("\n");
			const memoryContext = formatMemoryContext(ctx.memoryStore, query, {
				maxTokens: options?.memoryContextMaxTokens ?? 1_000,
			});
			if (!memoryContext) return undefined;
			return {
				systemPrompt: `${event.systemPrompt}\n\n${memoryContext}`,
			};
		},
	);

	return () => {
		pipeline.cancel();
		unsubTurnEnd();
		unsubBeforeAgentStart();
	};
}

function entriesAfter(
	entries: readonly SourceEntry[],
	coverageId: string | undefined,
): SourceEntry[] {
	if (!coverageId) return [...entries];
	const index = entries.findIndex((entry) => entry.id === coverageId);
	return index < 0 ? [...entries] : entries.slice(index + 1);
}

function sourceTokens(entries: readonly SourceEntry[]): number {
	return entries.reduce(
		(sum, entry) => sum + (entry.tokenCount ?? estimateTokens(entry.content)),
		0,
	);
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
