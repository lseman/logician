// ── Bash Tool Debugger ────────────────────────────────────────────────────────
// Traces bash tool calls through the full pipeline to diagnose why arguments
// may be empty or malformed. Logs at 4 stages:
//   Stage 0: Raw call.arguments from model (JSON string)
//   Stage 1: After parseToolInput() — first parse
//   Stage 2: After prepareArguments() — normalized args
//   Stage 3: Final args passed to execute()
//
// Usage: Enable via `setBashDebugger(true)` or set LOGICIAN_BASH_DEBUG=1 env var.
// Query results via `getBashDebuggerReport()` which returns a summary of the
// last N bash calls with their pipeline state at each stage.

export interface BashDebugEntry {
	id: string;
	toolName: string;
	stage0_rawArgs: string | null;       // Raw JSON string from model
	stage1_parsed: Record<string, unknown> | null;  // After parseToolInput
	stage2_prepared: Record<string, unknown> | null; // After prepareArguments
	stage3_final: Record<string, unknown> | null;    // Final args for execute
	stage0_timestamp: number;
	stage3_timestamp?: number;
	result?: string;
	isEmptyArgs: boolean;
}

let enabled = false;
const MAX_ENTRIES = 50;
const entries: BashDebugEntry[] = [];

/** Enable or disable the bash debugger. */
export function setBashDebugger(on: boolean): void {
	enabled = on;
	if (on) {
		console.log("[bash-debugger] ENABLED — tracing all tool calls");
	}
}

/** Check if the debugger is enabled. */
export function isBashDebuggerEnabled(): boolean {
	return enabled;
}

/** Log a tool call at a specific pipeline stage. */
export function logStage(
	entryId: string,
	toolName: string,
	stage: 0 | 1 | 2 | 3,
	value: string | Record<string, unknown> | null,
): void {
	if (!enabled) return;

	let entry = entries.find(e => e.id === entryId);
	if (!entry) {
		entry = {
			id: entryId,
			toolName,
			stage0_rawArgs: null,
			stage1_parsed: null,
			stage2_prepared: null,
			stage3_final: null,
			stage0_timestamp: Date.now(),
			isEmptyArgs: false,
		};
		entries.unshift(entry);
		if (entries.length > MAX_ENTRIES) entries.pop();
	}

	switch (stage) {
		case 0:
			entry.stage0_rawArgs = typeof value === "string" ? value : JSON.stringify(value);
			break;
		case 1:
			entry.stage1_parsed = value as Record<string, unknown> | null;
			if (!entry.isEmptyArgs && value && Object.keys(value).length === 0) {
				entry.isEmptyArgs = true;
			}
			break;
		case 2:
			entry.stage2_prepared = value as Record<string, unknown> | null;
			if (!entry.isEmptyArgs && value && Object.keys(value).length === 0) {
				entry.isEmptyArgs = true;
			}
			break;
		case 3:
			entry.stage3_final = value as Record<string, unknown> | null;
			entry.stage3_timestamp = Date.now();
			if (!entry.isEmptyArgs && value && Object.keys(value).length === 0) {
				entry.isEmptyArgs = true;
			}
			break;
	}

	// Log to stderr for visibility
	const stageLabel = ["raw", "parsed", "prepared", "final"][stage];
	const displayValue = typeof value === "string" ? value : JSON.stringify(value, null, 2);
	console.error(
		`[bash-debugger] ${toolName} [${entryId}] stage=${stageLabel}: ${displayValue.slice(0, 500)}`,
	);
}

/** Set the result for a completed tool call. */
export function setResult(entryId: string, result: string): void {
	const entry = entries.find(e => e.id === entryId);
	if (entry) {
		entry.result = result;
	}
}

/** Get a human-readable report of recent bash calls. */
export function getBashDebuggerReport(limit: number = 10): string {
	const relevant = entries.filter(e => e.toolName === "bash").slice(0, limit);

	if (relevant.length === 0) {
		return "[bash-debugger] No bash calls recorded. Enable with setBashDebugger(true).";
	}

	const lines: string[] = [];
	lines.push(`[bash-debugger] Report: ${relevant.length} recent bash call(s)`);
	lines.push("=".repeat(80));

	for (const entry of relevant) {
		const elapsed = entry.stage3_timestamp
			? `${entry.stage3_timestamp - entry.stage0_timestamp}ms`
			: "incomplete";

		lines.push(`\n── ${entry.id.slice(0, 12)}... (${elapsed}) ──`);
		lines.push(`  Stage 0 (raw):     ${entry.stage0_rawArgs || "(none)"}`);
		lines.push(`  Stage 1 (parsed):  ${JSON.stringify(entry.stage1_parsed) || "(none)"}`);
		lines.push(`  Stage 2 (prepared):${JSON.stringify(entry.stage2_prepared) || "(none)"}`);
		lines.push(`  Stage 3 (final):   ${JSON.stringify(entry.stage3_final) || "(none)"}`);
		lines.push(`  isEmptyArgs:       ${entry.isEmptyArgs ? "YES ⚠️" : "no"}`);
		if (entry.result) {
			const preview = entry.result.slice(0, 200);
			lines.push(`  result:            ${preview}${entry.result.length > 200 ? "..." : ""}`);
		}
	}

	// Summary
	const emptyCount = relevant.filter(e => e.isEmptyArgs).length;
	if (emptyCount > 0) {
		lines.push("\n" + "=".repeat(80));
		lines.push(`SUMMARY: ${emptyCount}/${relevant.length} bash calls had empty arguments.`);
		lines.push("This means the model sent {} or a string that couldn't be parsed.");
		lines.push("Check: (1) tool schema in system prompt, (2) model's understanding of bash usage.");
	}

	return lines.join("\n");
}

/** Clear all debug entries. */
export function clearBashDebugger(): void {
	entries.length = 0;
	console.log("[bash-debugger] Cleared.");
}

// Auto-enable from environment variable
if (process.env.LOGICIAN_BASH_DEBUG === "1") {
	setBashDebugger(true);
}
