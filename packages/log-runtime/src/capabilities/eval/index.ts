// ── Eval capability ──────────────────────────────────────────────────────────
// Persistent eval kernel with Python and JS support, workpool batching.

export {
	type CompletionToolDeps,
	createCompletionTool,
} from "./completion-tool.ts";
export { createEvalTool, type EvalToolDeps } from "./eval-tool.ts";
export {
	createKernelManager,
	type EvalKernelConfig,
	type EvalResult,
	type KernelManager,
	type KernelManagerConfig,
	type KernelState,
} from "./kernel-manager.ts";
export {
	createWaitTool,
	type WaitToolDeps,
} from "./wait-tool.ts";
export {
	createWorkpoolTool,
	type WorkpoolDeps,
	type WorkpoolItem,
} from "./workpool-tool.ts";
