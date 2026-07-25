// ── Observational Memory (V3) — Public API ───────────────────────────────
// Structured memory: observations, reflections, drops with file-based persistence.
// Replaces the legacy in-process MemoryStore.

import { ConsolidationPipeline } from "./consolidation/pipeline.ts";
import { registerConsolidationHooks } from "./integration/hooks.ts";
import { FilePersistence } from "./storage/persistence.ts";
import { MemoryStoreImpl } from "./storage/store.ts";

export {
	type ConsolidationConfig,
	ConsolidationPipeline,
	type ConsolidationResult,
	type LaunchParams,
} from "./consolidation/pipeline.ts";
export {
	type HookContext,
	type HookOptions,
	registerCompactionHook,
	registerConsolidationHooks,
} from "./integration/hooks.ts";
export { hashId } from "./ids.ts";
export {
	type KnowledgeEdge,
	type KnowledgeGraph,
	KnowledgeGraphManager,
	type KnowledgeNode,
} from "./storage/knowledge-graph.ts";
export {
	FilePersistence,
	type PersistenceOptions,
} from "./storage/persistence.ts";
export {
	DROPPER_SYSTEM_PROMPT,
	OBSERVER_SYSTEM_PROMPT,
	REFLECTOR_SYSTEM_PROMPT,
} from "./consolidation/prompts.ts";
export {
	formatRecallResult,
	isValidMemoryId,
	type RecalledObservation,
	type RecalledReflection,
	type RecallResult,
	type RecallSourceEntry,
	recallMemory,
} from "./retrieval/recall.ts";
export {
	formatMemoryContext,
	type MemorySearchMatch,
	type MemorySearchOptions,
	searchMemory,
	searchMemoryStore,
} from "./retrieval/search.ts";
export {
	type MemoryStore,
	MemoryStoreImpl,
	type StoreOptions,
} from "./storage/store.ts";
export {
	estimateObservationTokens,
	estimateReflectionTokens,
	estimateTokens,
} from "./tokens.ts";
export {
	createMemorySearchTool,
	createRecallTool,
	MEMORY_SEARCH_TOOL_NAME,
	type MemorySearchToolResult,
	RECALL_TOOL_NAME,
	type RecallToolOptions,
	type RecallToolResult,
} from "./integration/tools.ts";
export type {
	FoldedMemory,
	MemoryStatus,
	MemoryStoreEvent,
	Observation,
	Reflection,
	Relevance,
} from "./types.ts";
export {
	MEMORY_ID_PATTERN,
	OM_FOLDED,
	OM_OBSERVATIONS_DROPPED,
	OM_OBSERVATIONS_RECORDED,
	OM_REFLECTIONS_RECORDED,
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
	const pipeline = new ConsolidationPipeline({
		model: opts.model,
		apiKey: opts.apiKey,
		baseUrl: opts.baseUrl,
		headers: opts.headers,
		observationsPoolTargetTokens: config.observationsPoolTargetTokens,
	});

	return {
		store,
		pipeline,
		config,
		/** Register hooks on an extension event bus */
		registerHooks: (
			extensionBus: Parameters<
				typeof registerConsolidationHooks
			>[0]["extensionBus"],
			runtime?: {
				getSourceEntries?: () => Array<{
					id: string;
					role: string;
					content: string;
					tokenCount?: number;
				}>;
				getRetrievalContext?: () => string;
			},
		) =>
			registerConsolidationHooks({
				extensionBus,
				memoryStore: store,
				pipeline,
				options: {
					observeAfterTokens: config.observeAfterTokens,
					reflectAfterTokens: config.reflectAfterTokens,
					compactAfterTokens: config.compactAfterTokens,
				},
				getSourceEntries: runtime?.getSourceEntries,
				getRetrievalContext: runtime?.getRetrievalContext,
			}),
	};
}
