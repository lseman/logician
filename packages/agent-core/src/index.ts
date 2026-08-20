// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Barrel re-exporting every module. This is the only supported way to import
// from @logician/agent-core — there is no other public entry point.
//
// Module layout:
//   application    — bridge and high-level orchestration
//   core           — agent engine, harness, hooks, extensions, and shared types
//   features       — commands, MCP, prompts, and skills
//   infrastructure — tools, guards, configuration, trust, and diagnostics
//   runtime        — sessions, queues, tasks, summaries, and runtime events

// ── Core ────────────────────────────────────────────────────────────────────
export * from "./core/execution/agent-loop-runner.ts";
export * from "./core/configuration/agent-settings.ts";
export * from "./core/provider/backend.ts";
export {
	type ContinuationDecision,
	type ContinuationLimits,
	type ContinuationState,
	ContinuationTracker,
	type RunBudgetStatus,
} from "./core/policy/continuation-tracker.ts";
export * from "./core/policy/execution-policy.ts";
export * from "./core/session/file-checkpoints.ts";
export * from "./core/policy/intervention-controller.ts";
export * from "./core/provider/messages.ts";
export {
	createRuntimeState,
	reduceRuntimeState,
	type AgentRuntimeState,
	type HarnessPhase,
} from "./core/state/runtime-state.ts";
export * from "./core/session/session.ts";
export * from "./core/state/tool-cache.ts";
export * from "./core/policy/exit-path.ts";
export * from "./core/policy/run-budget.ts";
export * from "./core/policy/conclusion-policy.ts";
export * from "./core/execution/tool-batch-controller.ts";

// ── Harness ─────────────────────────────────────────────────────────────────
export * from "./core/harness/agent-harness.ts";
export type {
	BranchInfo,
	BranchSummaryData,
} from "./runtime/summaries/types.ts";
export type {
	AbortResult,
	AgentHarnessOptions,
	HarnessQueues,
} from "./core/harness/types.ts";
export { HarnessBusyError } from "./core/harness/runtime/phase.ts";

// ── Guards ──────────────────────────────────────────────────────────────────
export * from "./infrastructure/guards/index.ts";

// ── Types ───────────────────────────────────────────────────────────────────
export * from "./core/types/index.ts";

// ── Tasks ───────────────────────────────────────────────────────────────────
export type {
	Task,
	TaskState,
	TaskStatus,
} from "./runtime/todo-state.ts";
export {
	allocateTaskId,
	getTasks,
	getTasksMutable,
	notifyTodosChanged,
	onTodosChanged,
	replaceTasks,
} from "./runtime/todo-state.ts";

// ── Compaction ──────────────────────────────────────────────────────────────
export {
	type CompactionSettings,
	compactToFit,
	DEFAULT_COMPACTION_SETTINGS,
	estimateContextTokens,
	shouldCompact,
} from "./core/compaction/index.ts";

// ── Extension ───────────────────────────────────────────────────────────────
export * from "./core/extension/index.ts";

// ── Hooks ───────────────────────────────────────────────────────────────────
export * from "./core/hooks/builtin/index.ts";
export * from "./core/hooks/index.ts";

// ── Queue ───────────────────────────────────────────────────────────────────
export * from "./runtime/queue/index.ts";

// ── Tools ───────────────────────────────────────────────────────────────────
export * from "./infrastructure/tools/index.ts";

// ── TUI Utilities ───────────────────────────────────────────────────────────
export {
	formatContextSize,
	formatTokenCount,
	formatDelay,
	escapeTable,
	tableRow,
	parseInterval,
} from "./tui-utils.ts";
