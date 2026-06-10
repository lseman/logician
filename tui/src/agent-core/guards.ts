// ── Tool guards ────────────────────────────────────────────────────────────
// Stateful guard engine that watches tool calls/results within a single run and
// blocks the model when it is stuck in a loop. Wired into the agent-loop
// contract hooks (`beforeToolCall` observes pending calls, `afterToolCall`
// records outcomes). Mirrors openclaude's toolFailureLoopGuard + a
// duplicate-call guard.

import type { ToolCall } from "./types.ts";

export interface GuardDecision {
	block: boolean;
	// Message recorded as the (synthetic) tool result when blocked.
	message?: string;
}

const DEFAULT_DUPLICATE_THRESHOLD = 3;
const DEFAULT_FAILURE_THRESHOLD = 3;
const MAX_CATEGORY_LEN = 120;

export interface GuardEngineOptions {
	duplicateThreshold?: number;
	failureThreshold?: number;
}

// Stable signature for a tool call: name + canonical args.
function signature(call: ToolCall): string {
	let argsKey = call.arguments || "";
	try {
		argsKey = JSON.stringify(sortKeys(JSON.parse(call.arguments || "{}")));
	} catch {
		// Non-JSON args: use the raw string.
	}
	return `${call.name} ${argsKey}`;
}

function sortKeys(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(sortKeys);
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const k of Object.keys(value as Record<string, unknown>).sort()) {
			out[k] = sortKeys((value as Record<string, unknown>)[k]);
		}
		return out;
	}
	return value;
}

// Target path from common arg names — used to bucket failures by file.
function callPath(call: ToolCall): string {
	try {
		const args = JSON.parse(call.arguments || "{}") as Record<string, unknown>;
		const raw = args.path ?? args.file_path ?? args.filename ?? "";
		return String(raw).trim();
	} catch {
		return "";
	}
}

// Coarse error bucket so distinct-but-equivalent failures collapse together.
// tui has no structured error payloads, so we key on the leading
// slice of the error text under the tool name.
function failureCategory(toolName: string, result: string): string {
	const body = result.replace(/^Error:\s*/i, "").trim();
	return `${toolName} ${body.slice(0, MAX_CATEGORY_LEN)}`;
}

function inc(map: Map<string, number>, key: string): number {
	const next = (map.get(key) || 0) + 1;
	map.set(key, next);
	return next;
}

export class GuardEngine {
	private duplicateThreshold: number;
	private failureThreshold: number;

	// Counts over the whole run.
	private callSignatureCounts = new Map<string, number>();
	private failSignatureCounts = new Map<string, number>();
	private failCategoryCounts = new Map<string, number>();
	private failPathCounts = new Map<string, number>();

	constructor(options: GuardEngineOptions = {}) {
		this.duplicateThreshold = Math.max(
			0,
			options.duplicateThreshold ?? DEFAULT_DUPLICATE_THRESHOLD,
		);
		this.failureThreshold = Math.max(
			0,
			options.failureThreshold ?? DEFAULT_FAILURE_THRESHOLD,
		);
	}

	// Called before a tool runs. Counts the pending call and decides whether to
	// block it. Returns block=true with a message the loop records instead of
	// executing the tool.
	inspect(call: ToolCall): GuardDecision {
		const sig = signature(call);
		const callCount = inc(this.callSignatureCounts, sig);

		if (this.duplicateThreshold > 0 && callCount >= this.duplicateThreshold) {
			return {
				block: true,
				message: `Error: blocked — \`${call.name}\` was called with identical arguments ${callCount} times. Stop repeating the same call; change your approach.`,
			};
		}

		if (this.failureThreshold > 0) {
			const path = callPath(call);
			if ((this.failSignatureCounts.get(sig) || 0) >= this.failureThreshold) {
				return this.tripFailure(call.name, "the same call");
			}
			if (
				path &&
				(this.failPathCounts.get(path) || 0) >= this.failureThreshold
			) {
				return this.tripFailure(call.name, `\`${path}\``);
			}
			// Category bucket: distinct-but-equivalent failures (e.g. three
			// different search patterns that all error the same way) collapse to
			// one category. Trip when any category for this tool crosses the
			// threshold, even though each individual call/path differs.
			for (const [cat, count] of this.failCategoryCounts) {
				if (count >= this.failureThreshold && cat.startsWith(`${call.name} `)) {
					return this.tripFailure(call.name, "this kind of operation");
				}
			}
		}

		return { block: false };
	}

	// Called after a tool runs. Records failures for the loop guard.
	record(call: ToolCall, isError: boolean, result: string): void {
		if (!isError || this.failureThreshold === 0) return;
		inc(this.failSignatureCounts, signature(call));
		const path = callPath(call);
		if (path) inc(this.failPathCounts, path);
		const cat = failureCategory(call.name, result);
		inc(this.failCategoryCounts, cat);
	}

	private tripFailure(toolName: string, target: string): GuardDecision {
		return {
			block: true,
			message: `Error: blocked — \`${toolName}\` has failed on ${target} ${this.failureThreshold} times. Stop retrying the same approach; inspect the actual error, fix the root cause, or use a different tool.`,
		};
	}
}
