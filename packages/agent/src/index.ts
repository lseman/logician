// ── Agent Core Beta Entry Point ─────────────────────────────────────────────
// Barrel re-exporting every module. This is the only supported way to import
// from @logician/agent — there is no other public entry point.
//
// Ported from pi coding agent's architecture (repos/pi/packages/agent),
// modernized incrementally. See module READMEs for design notes on where
// this deviates from pi.

// ── Agent (stateful loop wrapper) ────────────────────────────────────────────
export { Agent, type AgentOptions, type QueueMode } from "./agent/agent.ts";
export {
	type AgentEventSink,
	agentLoop,
	agentLoopContinue,
	buildProviderContext,
	runAgentLoop,
	runAgentLoopContinue,
} from "./agent/agent-loop.ts";
export { getDefaultStreamFn, setDefaultStreamFn } from "./agent/stream-fn.ts";
export type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	AgentToolUpdateCallback,
	BeforeToolCallContext,
	BeforeToolCallResult,
	CustomAgentMessages,
	PrepareNextTurnContext,
	ShouldStopAfterTurnContext,
	ToolExecutionMode,
} from "./agent/types.ts";
export {
	classifyHttpError,
	classifyNetworkError,
	TransportError,
	type TransportErrorCategory,
} from "./ai/errors.ts";
// ── AI transport helpers ─────────────────────────────────────────────────────
export {
	AssistantMessageEventStream as AssistantMessageEventStreamImpl,
	createAssistantMessageEventStream,
	EventStream,
} from "./ai/event-stream.ts";
export {
	normalizeProviderMessages,
	parseProviderUsage,
	streamOpenAiCompletions,
	streamSimple,
} from "./ai/openai-completions.ts";
export {
	isRetryableAssistantError,
	retryAssistantCall,
} from "./ai/retry-assistant.ts";
export { contentText } from "./ai/text.ts";
export {
	validateToolArguments,
	validateToolCall,
} from "./ai/tool-validation.ts";
// ── AI transport (OpenAI-compatible chat completions) ───────────────────────
export type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	CacheRetention,
	Context,
	ImageContent,
	KnownApi,
	KnownProvider,
	Message,
	Model,
	ProviderId,
	SimpleStreamOptions,
	StopReason,
	StreamFn,
	StreamOptions,
	TextContent,
	ThinkingContent,
	ThinkingLevel,
	Tool,
	ToolCall,
	ToolChoice,
	ToolResultMessage,
	Usage,
	UserMessage,
} from "./ai/types.ts";
export { isDeepEqual } from "./core/deep-equal.ts";
export {
	BranchSummaryError,
	type BranchSummaryErrorCode,
	CompactionError,
	type CompactionErrorCode,
	ExecutionError,
	type ExecutionErrorCode,
	FileError,
	type FileErrorCode,
} from "./core/errors.ts";
// ── Core (Result, typed errors) ─────────────────────────────────────────────
export {
	err,
	getOrThrow,
	getOrUndefined,
	ok,
	type Result,
	toError,
} from "./core/result.ts";
export type { RetryCallbacks, RetryPolicy } from "./core/retry.ts";
export { uuidv7 } from "./core/uuid.ts";
// ── Environment (filesystem/shell abstraction) ──────────────────────────────
export type {
	ExecutionEnv,
	FileInfo,
	FileKind,
	FileSystem,
	Shell,
	ShellExecOptions,
} from "./env/execution-env.ts";
export type {
	AbortRejected,
	AbortResult,
	ActionInfo,
	AgentHarnessOptions,
	AgentLane,
	CancelQueuedRejected,
	CancelQueuedResult,
	CompactionOutcome,
	CompactionRejected,
	CompactionResult,
	CreateLaneResult,
	EntryProjector,
	Events as HarnessEvents,
	HookName,
	Hooks,
	LaneInfo,
	LaneSnapshot,
	NavigateOptions,
	NavigationOutcome,
	NavigationRejected,
	NavigationResult,
	OperationError,
	QueuedItem,
	QueueRejected,
	QueueResult,
	RecordUsageResult,
	Resources as HarnessResources,
	ResumeOutcome,
	ResumeRejected,
	ResumeResult,
	RunOutcome,
	RunRejected,
	RunResult,
	SessionSnapshot,
	StreamOptions as HarnessStreamOptions,
	StreamOptionsPatch as HarnessStreamOptionsPatch,
	SuspendedOperation,
	WatchHandle,
} from "./harness/agent-harness.ts";
// ── Harness (multi-lane orchestration surface) ───────────────────────────────
export {
	AgentHarness,
	Closed,
	HarnessClosed,
	HarnessFault,
	HarnessNotImplemented,
	InvalidLane,
	InvalidMessage,
	LaneBusy,
	LaneExists,
	MissingIdentities,
	NoActiveOperation,
	NoActiveRun,
	NothingToCompact,
	NothingToResume,
	UnknownQueueItem,
	UnknownSkill,
	UnknownTarget,
	UnknownTemplate,
} from "./harness/agent-harness.ts";
export type {
	BranchPreparation,
	BranchSummaryDetails,
	BranchSummaryResult,
	CollectEntriesResult,
	GenerateBranchSummaryOptions,
} from "./harness/compaction/branch-summarization.ts";
export {
	collectEntriesForBranchSummary,
	generateBranchSummary,
	prepareBranchEntries,
} from "./harness/compaction/branch-summarization.ts";
export type {
	CompactionDetails,
	CompactionPreparation,
	CompactionSettings,
	CompactResult,
	ContextUsageEstimate,
	CutPointResult,
	MicroCompactResult,
} from "./harness/compaction/compaction.ts";
export {
	calculateContextTokens,
	compact,
	DEFAULT_COMPACTION_SETTINGS,
	estimateContextTokens,
	estimateTokens,
	findCutPoint,
	findTurnStartIndex,
	generateSummary,
	generateSummaryWithUsage,
	getLastAssistantUsage,
	microCompactMessages,
	prepareCompaction,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
	shouldCompact,
	truncateMiddle,
} from "./harness/compaction/compaction.ts";
export type { FileOperations } from "./harness/compaction/utils.ts";
export {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	formatFileOperations,
} from "./harness/compaction/utils.ts";
export type {
	Events as HarnessEventsInterface,
	HarnessEvent,
	HarnessEventListener,
	HarnessEventOfType,
	HarnessEventType,
	RunEndEvent,
	RunStartEvent,
	WatchHandle as HarnessEventWatchHandle,
} from "./harness/events.ts";
export { HarnessEventBus } from "./harness/events.ts";
export { parseFrontmatter } from "./harness/frontmatter.ts";
export type { HookEventMap, HookResultMap } from "./harness/hooks.ts";
export { HookRegistry } from "./harness/hooks.ts";
export type {
	BashExecutionMessage,
	BranchSummaryMessage,
	CompactionSummaryMessage,
	CustomMessage,
} from "./harness/messages.ts";
export {
	BRANCH_SUMMARY_PREFIX,
	BRANCH_SUMMARY_SUFFIX,
	bashExecutionToText,
	COMPACTION_SUMMARY_PREFIX,
	COMPACTION_SUMMARY_SUFFIX,
	convertToLlm as harnessConvertToLlm,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
	createCustomMessage,
} from "./harness/messages.ts";
export type {
	PermissionMode,
	PermissionRules,
	PermissionVerdict,
} from "./harness/permissions.ts";
export {
	createPermissionHook,
	PermissionManager,
	primaryArgString,
} from "./harness/permissions.ts";
export type {
	PromptTemplateDiagnostic,
	PromptTemplateDiagnosticCode,
} from "./harness/prompt-templates.ts";
export {
	formatPromptTemplateInvocation,
	loadPromptTemplates,
	loadSourcedPromptTemplates,
	parseCommandArgs,
	substituteArgs,
} from "./harness/prompt-templates.ts";
export type {
	EffectiveLaneConfiguration,
	LaneReductionInput,
	LaneReductionResult,
	LaneState,
	RecordLogCorruptionReason,
	RecordLogSlice,
	TerminalFailureState,
	ToolBatchState,
} from "./harness/reducer.ts";
export {
	RecordLogCorruption,
	reduceLaneState,
	validateRecordLog,
} from "./harness/reducer.ts";
export type {
	ErrorMatchers,
	TaggedErrorFactory,
	TaggedErrorValue,
} from "./harness/result.ts";
export {
	matchError,
	Result as HarnessResult,
	TaggedError,
} from "./harness/result.ts";
export type {
	ContextEntryTransform,
	CustomEntryContextMessageProjector,
	SessionContext,
	SessionContextBuildOptions,
} from "./harness/session/context.ts";
export {
	buildContextEntries,
	buildSessionContext,
	defaultContextEntryTransform,
	sessionEntryToContextMessages,
} from "./harness/session/context.ts";
export type {
	JsonlSessionCreateOptions,
	JsonlSessionHeader,
	JsonlSessionListOptions,
	JsonlSessionMetadata,
	JsonlSessionRepoFileSystem,
	JsonlSessionRepoOptions,
} from "./harness/session/jsonl.ts";
export {
	encodeHeader,
	encodeMutation,
	fileResult,
	invalidFile,
	JsonlDecodeError,
	JsonlSessionRepo,
	JsonlSessionStorage,
	listJsonlSessionMetadata,
	loadJsonlSessionStorage,
	metadataFromHeader,
	parseHeader,
	parseMutation,
} from "./harness/session/jsonl.ts";
export {
	InMemorySessionRepo,
	InMemorySessionStorage,
} from "./harness/session/memory.ts";
export { assertJsonSerializable, Session } from "./harness/session/session.ts";
export type { SessionMutation } from "./harness/session/state.ts";
export { SessionState } from "./harness/session/state.ts";
export type {
	AbortRequestedRecord,
	ActiveToolsEntry,
	BranchBounds,
	BranchSummaryEntry,
	CompactionEntry,
	CompactionReason,
	CustomEntry,
	Entry,
	EntryBase,
	EntryCursor,
	EntryOrder,
	EntryQuery,
	ForkOptions,
	IdGenerator,
	JsonValue as SessionJsonValue,
	LanePointer,
	LogItem,
	LogOptions,
	MessageEntry,
	ModelChangeEntry,
	NewRecord,
	OperationFinishedRecord,
	OperationStartedRecord,
	ProvisionedEntry,
	QueueCancelledRecord,
	QueueEnqueuedRecord,
	RecordBase,
	RecordQuery,
	SessionCreateOptions,
	SessionErrorCode as SessionLogErrorCode,
	SessionMetadata,
	SessionRepo,
	SessionStats,
	SessionStopReason,
	SessionStorage,
	SessionTree,
	StepAttemptRecord,
	ThinkingLevelEntry,
	ToolStartedRecord,
	UsageRecord,
	WriteDeferredRecord,
} from "./harness/session/types.ts";
export { SessionError as SessionLogError } from "./harness/session/types.ts";
export type { SkillDiagnostic, SkillDiagnosticCode } from "./harness/skills.ts";
export {
	formatSkillInvocation,
	formatSkillsForSystemPrompt,
	loadSkills,
	loadSourcedSkills,
} from "./harness/skills.ts";
export type {
	AttributeValue,
	SpanAttributes,
	SpanOptions,
	SpanStatus,
	TelemetryContext,
	TelemetrySpan,
} from "./harness/telemetry.ts";
export { NOOP_TELEMETRY_CONTEXT } from "./harness/telemetry.ts";
export type { AtomicWriteOptions } from "./harness/tools/atomic-write.ts";
export { appendToFile, atomicWriteFile } from "./harness/tools/atomic-write.ts";
// ── Tools (bash/edit/read/write) ─────────────────────────────────────────────
export type { BashToolOptions } from "./harness/tools/bash.ts";
export { createBashTool } from "./harness/tools/bash.ts";
export type { EditDiffResult } from "./harness/tools/diff-utils.ts";
export {
	generateDiffString,
	generateUnifiedPatch,
	summarizeDiff,
	syntheticUnifiedDiff,
} from "./harness/tools/diff-utils.ts";
export type { EditToolDetails, EditToolOptions } from "./harness/tools/edit.ts";
export { createEditTool } from "./harness/tools/edit.ts";
export type { ApplyEditsResult, Edit } from "./harness/tools/edit-matching.ts";
export {
	applyEditsToNormalizedContent,
	fuzzyFindText,
	normalizeForFuzzyMatch,
} from "./harness/tools/edit-matching.ts";
export { withFileMutationQueue } from "./harness/tools/file-mutation-queue.ts";
export {
	ensureInsideCwd,
	resolveReadPath,
	resolveToCwd,
} from "./harness/tools/path-utils.ts";
export type { ReadToolOptions } from "./harness/tools/read.ts";
export { createReadTool } from "./harness/tools/read.ts";
export {
	hasBeenRead,
	isStaleSinceRead,
	recordRead,
	refreshAfterWrite,
} from "./harness/tools/read-tracker.ts";
export type { ToolRegistryOptions } from "./harness/tools/registry.ts";
export {
	DEFAULT_MAX_RESULT_CHARS,
	ToolRegistry,
} from "./harness/tools/registry.ts";
export {
	detectLineEnding,
	mutationSummary,
	normalizeToLF,
	restoreLineEndings,
	stripBom,
} from "./harness/tools/text-helpers.ts";
export type {
	OutputAccumulatorOptions,
	OutputSnapshot,
	TruncationOptions,
	TruncationResult,
} from "./harness/tools/truncate.ts";
export {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	GREP_MAX_LINE_LENGTH,
	OutputAccumulator,
	sanitizeBinaryOutput,
	truncateHead,
	truncateLine,
	truncateTail,
} from "./harness/tools/truncate.ts";
export type { WriteToolOptions } from "./harness/tools/write.ts";
export { createWriteTool } from "./harness/tools/write.ts";
export type {
	AgentHarnessResources,
	AgentHarnessStreamOptions,
	AgentHarnessStreamOptionsPatch,
	AgentHarnessTool,
	AgentHarnessToolContextSource,
	PromptTemplate,
	Skill,
} from "./harness/types.ts";
