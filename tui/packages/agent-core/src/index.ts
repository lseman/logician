// ── Agent Core Entry Point ─────────────────────────────────────────────────
// Thin barrel re-exporting the three sub-modules plus the tools barrel.
// External consumers import from here; internals import from sub-paths.

// Core: loop, harness, types, backend, messages, events, session
export * from "./agent/backend.ts";
export * from "./agent/agent-loop-runner.ts";
export * from "./agent/execution-policy.ts";
export * from "./agent/file-checkpoints.ts";
export * from "./agent/harness.ts";
export * from "./agent/guards/loop-detector.ts";
export * from "./agent/messages.ts";
export * from "./agent/runtime-state.ts";
export * from "./agent/session.ts";
export * from "./agent/tool-cache.ts";
export * from "./agent/tasks/task-state-controller.ts";
export * from "./agent/types.ts";

// Hooks: hook bus, builtin hooks, budget
export * from "./hooks/builtin/index.ts";
export * from "./hooks/native/index.ts";
export * from "./plugins/claude-code/hook-layer.ts";
export * from "./plugins/claude-code/index.ts";

// Shared (registry, parser, permissions, plugins, subagent, skills, mcp, utils)
export * from "./tools/shared/permissions.ts";
export * from "./tools/shared/parser.ts";
export * from "./tools/shared/plugins.ts";
export * from "./plugins/claude-code/plugin-executor.ts";
export * from "./plugins/claude-code/plugin-manager.ts";
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
} from "./agent/tasks/task-status-state.ts";
export type { TaskStatusRecord } from "./agent/tasks/task-status-state.ts";
export type { Task, TaskStatus } from "./agent/tasks/todo-state.ts";
export { getTasks, onTodosChanged } from "./agent/tasks/todo-state.ts";

// Compaction: single engine shared by harness compact, the loop's
// context-full retry, and the builtin proactive hook.
export {
	compactToFit,
	estimateContextTokens,
	shouldCompact,
	type CompactionSettings,
	DEFAULT_COMPACTION_SETTINGS,
} from "./compaction/index.ts";

// Extensions: TypeScript extension system
export * from "./extensions/index.ts";

// Message queue: steering and follow-up messages
export * from "./queue/index.ts";
