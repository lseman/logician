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
} from "./hooks/hook-adapter.js";
export { LocalMemoryEmbedder, type MemoryEmbedder } from "./embeddings/local-embedder.js";
export { createMemoryHooks, type MemoryHooksConfig } from "./hooks/memory-hooks.js";
export type {
	SemanticExtractionRequest,
	SemanticExtractor,
} from "./episodes/semantic-extractor.js";
export { createMemoryStore } from "./store/index.js";
export {
	getBoundViewerPort,
	startViewerServer,
} from "./viewer/viewer-server.js";
export type { ViewerOptions } from "./viewer/viewer-server.js";

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
