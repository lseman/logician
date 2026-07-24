// ── Observational Memory (V3) — Public API ───────────────────────────────
// Structured memory: observations, reflections, drops with file-based persistence.
// Replaces the legacy in-process MemoryStore.

import { MemoryStoreImpl } from "./store.ts";
import { FilePersistence } from "./persistence.ts";
import { ConsolidationPipeline } from "./consolidation.ts";
import { registerConsolidationHooks } from "./hooks.ts";

export {
	MemoryStoreImpl,
	type MemoryStore,
	type StoreOptions,
} from "./store.ts";
export { FilePersistence, type PersistenceOptions } from "./persistence.ts";
export {
	ConsolidationPipeline,
	type ConsolidationConfig,
	type LaunchParams,
	type ConsolidationResult,
} from "./consolidation.ts";
export {
	registerConsolidationHooks,
	registerCompactionHook,
	type HookContext,
	type HookOptions,
} from "./hooks.ts";
export {
	recallMemory,
	formatRecallResult,
	isValidMemoryId,
	type RecallResult,
	type RecallSourceEntry,
	type RecalledObservation,
	type RecalledReflection,
} from "./recall.ts";
export {
	createRecallTool,
	createMemorySearchTool,
	type RecallToolOptions,
	type RecallToolResult,
	type MemorySearchToolResult,
	RECALL_TOOL_NAME,
	MEMORY_SEARCH_TOOL_NAME,
} from "./tool.ts";
export {
	searchMemory,
	searchMemoryStore,
	formatMemoryContext,
	type MemorySearchMatch,
	type MemorySearchOptions,
} from "./search.ts";
export { hashId } from "./ids.ts";
export {
	estimateTokens,
	estimateObservationTokens,
	estimateReflectionTokens,
} from "./tokens.ts";
export {
	OBSERVER_SYSTEM_PROMPT,
	REFLECTOR_SYSTEM_PROMPT,
	DROPPER_SYSTEM_PROMPT,
} from "./prompts.ts";
export type {
	Observation,
	Reflection,
	Relevance,
	FoldedMemory,
	MemoryStatus,
	MemoryStoreEvent,
} from "./types.ts";
export {
	KnowledgeGraphManager,
	type KnowledgeNode,
	type KnowledgeEdge,
	type KnowledgeGraph,
} from "./knowledge-graph.ts";
export {
	OM_OBSERVATIONS_RECORDED,
	OM_REFLECTIONS_RECORDED,
	OM_OBSERVATIONS_DROPPED,
	OM_FOLDED,
	MEMORY_ID_PATTERN,
	RELEVANCE_VALUES,
} from "./types.ts";

// ── Default config ───────────────────────────────────────────────────────

export const DEFAULT_CONFIG = {
	/** Token threshold to trigger observer stage */
	observeAfterTokens: 10_000,
	/** Token threshold to trigger reflector stage */
	reflectAfterTokens: 20_000,
	/** Token threshold for auto-compaction */
	compactAfterTokens: 81_000,
	/** Target token count for observation pool */
	observationsPoolTargetTokens: 10_000,
	/** Maximum token count before forced compaction */
	observationsPoolMaxTokens: 20_000,
} as const;

// ── Factory ──────────────────────────────────────────────────────────────

export interface MemoryFactoryOptions {
	/** Model name for LLM calls */
	model: string;
	/** API key for LLM calls */
	apiKey: string;
	/** OpenAI-compatible endpoint used for background consolidation. */
	baseUrl?: string;
	/** Optional request headers */
	headers?: Record<string, string>;
	/** Persistence path override */
	persistencePath?: string;
	/** Token thresholds override */
	config?: Partial<typeof DEFAULT_CONFIG>;
}

/**
 * Create a complete observational memory system.
 * Returns the store, pipeline, and hook registration function.
 */
export function createMemorySystem(opts: MemoryFactoryOptions) {
	const config = { ...DEFAULT_CONFIG, ...opts.config };

	// Persistence
	const persistence = new FilePersistence({ path: opts.persistencePath });

	// In-memory store
	const store = new MemoryStoreImpl({
		persistence,
		observationsPoolTargetTokens: config.observationsPoolTargetTokens,
	});

	// Consolidation pipeline
	const pipeline = new ConsolidationPipeline(
		{
			model: opts.model,
			apiKey: opts.apiKey,
			baseUrl: opts.baseUrl,
			headers: opts.headers,
			observationsPoolTargetTokens: config.observationsPoolTargetTokens,
		},
		{
			observeAfterTokens: config.observeAfterTokens,
			reflectAfterTokens: config.reflectAfterTokens,
		},
	);

	return {
		store,
		pipeline,
		config,
		/** Register hooks on an extension event bus */
		registerHooks: (extensionBus: Parameters<typeof registerConsolidationHooks>[0]["extensionBus"], runtime?: {
			currentTokens?: () => number;
			getSourceEntries?: () => Array<{ id: string; role: string; content: string }>;
		}) =>
			registerConsolidationHooks({
				extensionBus,
				memoryStore: store,
				pipeline,
				options: {
					observeAfterTokens: config.observeAfterTokens,
					reflectAfterTokens: config.reflectAfterTokens,
					compactAfterTokens: config.compactAfterTokens,
				},
				currentTokens: runtime?.currentTokens ?? (() => 0),
				getSourceEntries: runtime?.getSourceEntries,
			}),
	};
}
