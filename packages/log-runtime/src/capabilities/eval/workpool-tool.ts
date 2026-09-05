// ── workpool tool ────────────────────────────────────────────────────────────
// Batch parallel eval: run multiple code snippets concurrently across kernels.
// Results return in the same order as input items.

import type { Tool, ToolContext, ToolResult } from "@logician/log-core";
import type {
	EvalResult,
	KernelManager,
	KernelManagerConfig,
} from "./kernel-manager.ts";

export interface WorkpoolItem {
	language: "python" | "js";
	code: string;
	timeoutMs?: number;
}

export interface WorkpoolDeps {
	kernel: KernelManager;
	config?: KernelManagerConfig;
}

const workpoolSchema = {
	type: "object",
	properties: {
		items: {
			type: "array",
			minItems: 1,
			items: {
				type: "object",
				properties: {
					language: {
						type: "string",
						enum: ["python", "js"],
						description: "Execution language.",
					},
					code: {
						type: "string",
						description: "Code to execute.",
					},
					timeout_ms: {
						type: "integer",
						minimum: 100,
						description: "Timeout in milliseconds.",
					},
				},
				required: ["language", "code"],
			},
			description: "Array of code items to execute in parallel.",
		},
	},
	required: ["items"],
} as const;

/**
 * Execute multiple code snippets in parallel across Python/JS kernels.
 *
 * Results are returned in the same order as input items. Each item runs
 * independently — failures in one item don't affect others.
 *
 * Use cases:
 * - Run the same analysis in multiple languages
 * - Execute independent computations concurrently
 * - Test multiple code variants simultaneously
 */
export function createWorkpoolTool(deps: WorkpoolDeps): Tool {
	const timeoutMs = deps.config?.defaultTimeoutMs ?? 30_000;

	return {
		name: "workpool",
		label: "Workpool",
		description:
			"Execute multiple code snippets in parallel across Python/JS kernels. " +
			"Results return in input order — failures in one item don't affect others.",
		promptSnippet: "Run multiple evals in parallel",
		promptGuidelines: [
			"Use workpool for independent computations that can run concurrently",
			"Results are ordered by input position, not completion time",
			"Each item runs independently — one failure doesn't block others",
		],
		readOnly: false,
		executionMode: "parallel",
		parameters: workpoolSchema,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<ToolResult> => {
			const rawItems = args.items as unknown[];

			if (!Array.isArray(rawItems) || rawItems.length === 0) {
				return {
					content:
						"Error: items array is required and must have at least one item.",
					isError: true,
				};
			}

			const items: WorkpoolItem[] = [];
			for (const raw of rawItems) {
				if (
					typeof raw !== "object" ||
					raw === null ||
					!("language" in raw) ||
					!("code" in raw)
				)
					continue;
				const item = raw as Record<string, unknown>;
				items.push({
					language:
						typeof item.language === "string"
							? (item.language as "python" | "js")
							: "python",
					code: String(item.code),
					timeoutMs:
						typeof item.timeoutMs === "number" ? item.timeoutMs : timeoutMs,
				});
			}

			if (items.length === 0) {
				return {
					content:
						"Error: all items must have 'language' (python|js) and 'code' (string).",
					isError: true,
				};
			}

			const results: Array<{ index: number; result: EvalResult }> = [];

			// Run all items in parallel
			await Promise.all(
				items.map(async (item, index) => {
					try {
						const result = await deps.kernel.eval({
							language: item.language,
							code: item.code,
							timeoutMs: item.timeoutMs,
						});
						results.push({ index, result });
					} catch (err) {
						results.push({
							index,
							result: {
								status: "error",
								output: "",
								error: err instanceof Error ? err.message : String(err),
							},
						});
					}
				}),
			);

			// Sort by original index
			results.sort((a, b) => a.index - b.index);

			// Format results
			const formatted = results.map((r, i) => {
				const label = items[i]?.language ?? "?";
				if (r.result.status === "success") {
					const output = r.result.output?.trim();
					if (output) {
						return `## Result ${i + 1} (${label}): OK\n\n\`\`\`\n${output}\n\`\`\``;
					}
					return `## Result ${i + 1} (${label}): OK (no output)`;
				}
				const errorMsg = r.result.error ?? "Unknown error";
				return `## Result ${i + 1} (${label}): ${r.result.status.toUpperCase()}\n\n${errorMsg}`;
			});

			const total = results.length;
			const success = results.filter(r => r.result.status === "success").length;
			const failed = results.filter(r => r.result.status !== "success").length;

			return {
				content: formatted.join("\n\n"),
				details: {
					results: results.map(r => ({
						status: r.result.status,
						output: r.result.output?.trim(),
						error: r.result.error,
					})),
					total,
					success,
					failed,
				},
			};
		},
	};
}
