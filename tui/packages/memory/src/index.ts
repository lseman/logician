// ── @logician/memory Entry Point ────────────────────────────────────────────
// Persistent memory for Logician agent — SQLite-backed with observation capture,
// synthetic compression, consolidation, and context injection.

export {
	autoForget,
	autoTierMemories,
	consolidate,
	forget,
	getContext,
	listMemories,
	recall,
	recallWithTier,
	registerMemoryHooks,
	remember,
	searchObservations,
	setSessionId,
} from "./hook-adapter.js";
export { LocalMemoryEmbedder, type MemoryEmbedder } from "./local-embedder.js";
export { createMemoryHooks, type MemoryHooksConfig } from "./memory-hooks.js";
export type {
	SemanticExtractionRequest,
	SemanticExtractor,
} from "./semantic-extractor.js";
export { createMemoryStore } from "./store.js";

export type {
	AutoForgetConfig,
	CompressedObservation,
	ContextBlock,
	// Options
	CreateMemoryOptions,
	// Retention Scoring
	DecayConfig,
	DecayConfigInput,
	// Dedup / Auto-Forget
	DedupConfig,
	ExpandedMemoryEntry,
	// Export/Import
	ExportData,
	// File Context
	FileContextEntry,
	HookPayload,
	HookPhase,
	ImportData,
	ImportResult,
	// Memories
	Memory,
	// Retrieval
	MemoryQuery,
	// Memory Relations
	MemoryRelation,
	MemoryRelationType,
	// Store
	MemoryStore,
	MemoryType,
	ObservationType,
	ObserveOptions,
	// Observations
	RawObservation,
	RecallOptions,
	RetentionScore,
	SearchResult,
	SemanticSearchResult,
	// Sessions
	Session,
	// Working Memory
	WorkingMemoryTier,
} from "./types.js";
