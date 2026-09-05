// ── eval tool ────────────────────────────────────────────────────────────────
// Persistent code execution in Python or JS kernels.
// State survives across calls: imports, variables, and computed data persist.

import type { Tool, ToolContext } from "@logician/log-core";
import type { KernelManager, KernelManagerConfig } from "./kernel-manager.ts";

export interface EvalToolDeps {
	kernel: KernelManager;
	config?: KernelManagerConfig;
}

const evalSchema = {
	type: "object",
	properties: {
		language: {
			type: "string",
			enum: ["python", "js"],
			description: "Execution language: python or js.",
		},
		code: {
			type: "string",
			description:
				"Code to execute. State persists across calls within a session.",
		},
		timeout_ms: {
			type: "integer",
			minimum: 100,
			description: "Timeout in milliseconds (default: 30000).",
		},
		reset: {
			type: "boolean",
			description: "Reset kernel state before execution (clears variables).",
		},
	},
	required: ["language", "code"],
} as const;

/**
 * Execute code in a persistent Python or JS kernel.
 *
 * Python kernel: imports, variables, and functions persist across calls.
 * JS kernel: Bun/Node globals and eval scope persist across calls.
 *
 * Use cases:
 * - Data processing with libraries (pandas, numpy, etc.)
 * - Iterative computation without re-importing
 * - Running scripts that need shared state
 * - Testing code snippets before deploying
 */
export function createEvalTool(deps: EvalToolDeps): Tool {
	const timeoutMs = deps.config?.defaultTimeoutMs ?? 30_000;

	return {
		name: "eval",
		label: "Eval",
		description:
			"Execute code in a persistent Python or JS kernel. State persists across calls — imports, " +
			"variables, and computed data carry forward. Use reset=true to clear state.",
		promptSnippet: "Execute Python or JS code in persistent kernel",
		promptGuidelines: [
			"Use eval for iterative computation where state matters",
			"Python: import libraries once, reuse across calls",
			"JS: Bun globals persist; use for Node/Bun scripts",
			"Set timeout_ms for long-running computations",
			"Use reset=true to start fresh if state is corrupted",
		],
		readOnly: false,
		executionMode: "sequential",
		parameters: evalSchema,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<string> => {
			const language = args.language as "python" | "js";
			const code = String(args.code);
			const timeout = Number(args.timeout_ms) || timeoutMs;
			const reset = Boolean(args.reset);

			if (!code.trim()) {
				return "Error: code is required and must be non-empty.";
			}

			if (language !== "python" && language !== "js") {
				return `Error: language must be "python" or "js", got "${language}".`;
			}

			const result = await deps.kernel.eval({
				language,
				code,
				timeoutMs: timeout,
				reset,
			});

			const output = result.output?.trim();
			const error = result.error;

			if (result.status === "timeout") {
				return `[timeout after ${timeout}ms]\n${error ?? "Execution timed out"}`;
			}

			if (result.status === "error") {
				return `[error]\n${error ?? "Unknown error"}`;
			}

			if (output) {
				return output;
			}

			return "(completed with no output)";
		},
	};
}

export const evalTool = createEvalTool({
	kernel: {
		python: {
			eval: () =>
				Promise.resolve({
					status: "error",
					output: "",
					error: "No kernel configured",
				}),
			state: { available: false, launched: false, requestCount: 0 },
		} as never,
		js: {
			eval: () =>
				Promise.resolve({
					status: "error",
					output: "",
					error: "No kernel configured",
				}),
			state: { available: false, launched: false, requestCount: 0 },
		} as never,
		eval: () =>
			Promise.resolve({
				status: "error",
				output: "",
				error: "No kernel configured",
			}),
		pythonState: () => ({ available: false, launched: false, requestCount: 0 }),
		jsState: () => ({ available: false, launched: false, requestCount: 0 }),
		stop: () => Promise.resolve(),
	},
});
