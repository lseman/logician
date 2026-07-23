// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Thin barrel re-exporting the three sub-modules plus the tools barrel.
// External consumers import from here; internals import from sub-paths.

// Core: loop, harness, types, backend, messages, events, session
export * from "./core/backend.ts";
export * from "./core/agent-loop-runner.ts";
export * from "./core/file-checkpoints.ts";
export * from "./core/harness.ts";
export * from "./core/loop-detector.ts";
export * from "./core/messages.ts";
export * from "./core/runtime-state.ts";
export * from "./core/session.ts";
export * from "./core/tool-cache.ts";
export * from "./core/types.ts";

// Hooks: hook bus, builtin hooks, budget
export * from "./hooks/builtin/index.ts";
export * from "./hooks/native/index.ts";
export * from "./compatibility/claude-code/hook-layer.ts";
export * from "./compatibility/claude-code/index.ts";

// Shared (registry, parser, permissions, plugins, subagent, skills, mcp, utils)
export * from "./tools/shared/permissions.ts";
export * from "./tools/shared/parser.ts";
export * from "./tools/shared/plugins.ts";
export * from "./compatibility/claude-code/plugin-executor.ts";
export * from "./compatibility/claude-code/plugin-manager.ts";
export * from "./tools/shared/async-utils.ts";
export * from "./tools/shared/path-utils.ts";
export * from "./tools/shared/syntax-highlighter.ts";
export * from "./tools/shared/registry.ts";
export * from "./tools/shared/frontmatter.ts";
export * from "./tools/shared/json-utils.ts";

// Task/todo state read by built-in hooks. The task_status and todo Tool
// objects that mutate this state live in @logician/agent-capabilities, which depends
// on this package — the state stays here to avoid a circular dependency.
export {
	getTaskStatus,
	resetTaskStatus,
} from "./core/task-status-state.ts";
export type { TaskStatusRecord } from "./core/task-status-state.ts";
export type { Task, TaskStatus } from "./core/todo-state.ts";
export { getTasks, onTodosChanged } from "./core/todo-state.ts";

// Compaction (kept at top level for backwards compat).
// Note: core/messages.ts and compaction/compaction.ts both define
// estimateTokens (string→number vs message→number) and compactToFit.
// The messages.ts versions are exported through the barrel; the
// compaction.ts versions are internal (no export) to avoid collisions.
export {
	estimateContextTokens,
	shouldCompact,
	type CompactionSettings,
	DEFAULT_COMPACTION_SETTINGS,
	recoverFromContextFull,
} from "./compaction/index.ts";

// Extensions: TypeScript extension system
export * from "./extensions/index.ts";

// Message queue: steering and follow-up messages
export * from "./message-queue/index.ts";
