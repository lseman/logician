// ── wait tool ────────────────────────────────────────────────────────────────
// Wait for pending completion handles to resolve.
// Polls the kernel's persistent completion registry for results.

import type { Tool, ToolContext } from "@logician/log-core";
import type { KernelManager, KernelManagerConfig } from "./kernel-manager.ts";

export interface WaitToolDeps {
	kernel: KernelManager;
	config?: KernelManagerConfig;
}

const waitSchema = {
	type: "object",
	properties: {
		handles: {
			type: "array",
			items: { type: "string" },
			description:
				"Completion handles to wait for. Use handles returned from the completion tool.",
		},
		timeout_ms: {
			type: "number",
			description:
				"Maximum wait time in milliseconds. Defaults to 30000.",
		},
	},
	required: ["handles"],
} as const;

const WAIT_KERNEL_CODE = `
// Wait for pending completions in the kernel's persistent registry.
if (typeof __ompCompletions === 'undefined') {
    __ompCompletions = new Map();
}

const handles = %HANDLES%;
const timeoutMs = %TIMEOUT_MS%;
const results = [];

const start = performance.now();
let allDone = false;

while (!allDone && (performance.now() - start) < timeoutMs) {
    allDone = true;
    for (const id of handles) {
        if (!__ompCompletions.has(id)) {
            allDone = false;
            await new Promise(r => setTimeout(r, 100));
            break;
        }
    }
    if (allDone) break;
}

for (const id of handles) {
    const result = __ompCompletions.get(id);
    results.push({
        handle: id,
        status: result ? 'completed' : 'timeout',
        text: result?.text || '',
        model: result?.model,
        finishReason: result?.finishReason,
        validation: result?.validation,
    });
}

JSON.stringify(results);
`;

/**
 * Wait for pending completion handles to resolve.
 *
 * Polls the kernel's persistent completion registry. Returns results
 * for each handle in the same order as input.
 */
export function createWaitTool(deps: WaitToolDeps): Tool {
	return {
		name: "wait",
		label: "Wait",
		description:
			"Wait for pending completion handles to resolve. " +
			"Polls the kernel's completion registry and returns results " +
			"in the same order as input handles.",
		promptSnippet: "wait(handles=['omp_xxxxx'])",
		promptGuidelines: [
			"Pass completion handles from the completion tool output",
			"Results return in input order",
			"Timed-out handles show status: 'timeout'",
		],
		readOnly: true,
		executionMode: "sequential",
		parameters: waitSchema,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<string> => {
			const handles = args.handles as string[];
			const timeoutMs = (Number(args.timeout_ms) || 30_000) as number;

			if (!handles?.length) {
				return "Error: handles must be a non-empty array of handle IDs.";
			}

			const code = WAIT_KERNEL_CODE
				.replace("%HANDLES%", JSON.stringify(handles))
				.replace("%TIMEOUT_MS%", String(timeoutMs));

			const result = await deps.kernel.js.eval(code, timeoutMs);

			if (result.status === "success") {
				try {
					const parsed = JSON.parse(result.output);
					return JSON.stringify(parsed, null, 2);
				} catch {
					return result.output;
				}
			}

			return `Error: ${result.error ?? "Unknown error"}`;
		},
	};
}
