// ── bash tool ──────────────────────────────────────────────────────────────────────
// Execute shell commands with timeout, process tree tracking, and OutputAccumulator.
// Ported from Pi with logician integration.

import { spawn } from "node:child_process";
import { constants, access as fsAccess } from "node:fs/promises";

import type { Tool, ToolResult } from "@logician/log-core";
import { defaultPersistentTerminalManager } from "./support/utils/terminal-pool.ts";
import {
	getShellConfig,
	getShellEnv,
	killProcessTree,
	trackDetachedChildPid,
	untrackDetachedChildPid,
} from "./support/utils/shell.ts";
import { defaultTaskManager } from "./support/utils/background-task-registry.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	OutputAccumulator,
	type TruncationResult,
} from "./support/utils/truncate.ts";

const bashSchema = {
	type: "object",
	properties: {
		command: { type: "string", description: "Bash command to execute" },
		commands: {
			type: "array",
			description: "Structured commands to execute as a batch",
			minItems: 1,
			maxItems: 32,
			items: {
				type: "object",
				properties: {
					id: {
						type: "string",
						description: "Optional identifier used in results",
					},
					command: { type: "string", description: "Bash command to execute" },
					timeout: { type: "number", description: "Timeout in seconds" },
				},
				required: ["command"],
			},
		},
		mode: {
			type: "string",
			enum: ["sequential", "parallel"],
			description: "Batch execution mode (default: sequential)",
		},
		stopOnFailure: {
			type: "boolean",
			description: "Skip remaining sequential commands after a failure",
		},
		maxConcurrency: {
			type: "number",
			description: "Parallel concurrency (default: 4, maximum: 16)",
		},
		timeout: {
			type: "number",
			description: "Timeout for a single command, or default for batch entries",
		},
		waitMsBeforeAsync: {
			type: "number",
			description:
				"Optional wait duration in milliseconds before moving command to background task (e.g. 2000). If command finishes within this time, returns synchronous output. If still running, detaches to background and returns task ID.",
		},
		runPersistent: {
			type: "boolean",
			description:
				"Optional flag to run this command in a persistent shell that retains environment variables and working directory across calls.",
		},
		terminalId: {
			type: "string",
			description:
				"Optional terminal session identifier to reuse (defaults to 'default').",
		},
	},
	anyOf: [{ required: ["command"] }, { required: ["commands"] }],
} as const;

type SingleBashArgs = {
	command: string;
	timeout?: number;
	waitMsBeforeAsync?: number;
	runPersistent?: boolean;
	terminalId?: string;
};

type BashBatchEntry = SingleBashArgs & { id?: string };

type BashArgs = {
	command?: string;
	commands?: BashBatchEntry[];
	mode?: "sequential" | "parallel";
	stopOnFailure?: boolean;
	maxConcurrency?: number;
	timeout?: number;
	waitMsBeforeAsync?: number;
	runPersistent?: boolean;
	terminalId?: string;
};

interface CommandExecutionResult {
	content: string;
	details?: BashDetails;
	exitCode: number | null;
	signal: string | null;
	status: "completed" | "failed" | "timed_out" | "aborted";
}

interface BashBatchResult {
	id: string;
	command: string;
	content: string;
	exitCode: number | null;
	signal: string | null;
	status: CommandExecutionResult["status"] | "skipped";
	details?: BashDetails;
}

export interface BashDetails {
	truncation?: {
		truncated: boolean;
		totalLines: number;
		totalBytes: number;
		outputLines: number;
		outputBytes: number;
		lastLinePartial?: boolean;
		truncatedBy?: "lines" | "bytes";
		maxLines: number;
		maxBytes: number;
	};
	fullOutputPath?: string;
	[key: string]: unknown;
}

// ── Destructive command guard ────────────────────────────────────────────────
// Not a sandbox — a denylist for a short list of unambiguously destructive
// patterns that have no legitimate use in agent-driven development (wiping
// home/root, force-pushing over history, fork bombs, curl-pipe-to-shell).
// Matches git.ts's allowlist shape but inverted: block instead of allow, since
// bash must stay broadly usable.
const DESTRUCTIVE_PATTERNS: Array<{ pattern: RegExp; reason: string }> = [
	{
		pattern:
			/\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(-[a-zA-Z]*\s+)*(\/|~|\$HOME|\.\.?\/?\s*$)/,
		reason:
			"recursive/force delete of home, root, or relative parent directory",
	},
	{
		pattern: /\bgit\s+push\b[^|;&\n]*(--force(?!-with-lease)|(?<!--)\s-f\b)/,
		reason: "force-push (overwrites remote history)",
	},
	{
		pattern: /:\(\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;\s*:/,
		reason: "fork bomb",
	},
	{
		pattern:
			/\b(curl|wget)\b[^|;&\n]*\|\s*(sudo\s+)?(bash|sh|zsh|python[23]?)\b/,
		reason: "curl/wget piped directly into a shell interpreter",
	},
	{
		pattern: /\bmkfs(\.\w+)?\b/,
		reason: "filesystem format",
	},
	{
		pattern: /\bdd\b[^|;&\n]*\bof=\/dev\/(sd|nvme|hd|disk)/,
		reason: "raw disk write via dd",
	},
	{
		pattern: />\s*\/dev\/(sd|nvme|hd)[a-z0-9]*\b/,
		reason: "direct write to a block device",
	},
	{
		pattern: /\bchmod\s+(-R\s+)?[0-7]*777\s+\//,
		reason: "world-writable permissions on root or system path",
	},
];

function findDestructiveMatch(command: string): string | undefined {
	for (const { pattern, reason } of DESTRUCTIVE_PATTERNS) {
		if (pattern.test(command)) return reason;
	}
	return undefined;
}

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (typeof raw === "string") return { command: raw };
	if (!raw || typeof raw !== "object") return {};
	const args = raw as Record<string, unknown>;
	const command = args.command ?? args.cmd ?? args.script ?? args.input;
	return {
		...args,
		...(command === undefined ? {} : { command: String(command) }),
	};
}

// ── Update throttling ──────────────────────────────────────────────────────────
// Throttle TUI updates to prevent flooding the interface.

function makeUpdateThrottler(): (callback: () => void) => void {
	let lastUpdate = 0;
	const THROTTLE_MS = 100;

	return function throttledUpdate(callback: () => void) {
		const now = Date.now();
		if (now - lastUpdate < THROTTLE_MS) {
			// Debounce: skip this update
			return;
		}
		lastUpdate = now;
		callback();
	};
}

export const bash: Tool = {
	name: "bash",
	executionMode: "sequential",
	label: "Bash",
	hookAliases: ["Bash"],
	description: `Execute bash commands with timeout. Output is streamed and truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB. Uses process tree tracking for proper cleanup.`,
	promptSnippet:
		"Execute shell commands in a managed subprocess with timeout and approval policy",
	promptGuidelines: [
		"Use bash for file operations like ls, grep, find; use read for file content instead of cat",
	],
	parameters: bashSchema,
	prepareArguments,
	// The registry enforces a default execution timeout; when the model passes
	// an explicit timeout, allow it plus grace for the tool's own kill+cleanup.
	resolveTimeoutMs: args => {
		const entries = Array.isArray(args.commands) ? args.commands : [];
		const entryTimeouts = entries.map(entry =>
			typeof entry === "object" && entry !== null
				? Number((entry as Record<string, unknown>).timeout)
				: 0,
		);
		const timeout = Math.max(Number(args.timeout) || 0, ...entryTimeouts);
		return timeout > 0 ? timeout * 1000 + 30_000 : undefined;
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const parsed = args as BashArgs;
		if (parsed.command !== undefined && parsed.commands !== undefined) {
			return "Error: provide either command or commands, not both.";
		}
		if (parsed.commands !== undefined) return executeBatch(parsed, ctx);
		if (!parsed.command) return "Error: command or commands is required.";
		const blockedReason = findDestructiveMatch(parsed.command);
		if (blockedReason) {
			return {
				content: `Error: command blocked (${blockedReason}): ${parsed.command}`,
				isError: true,
			};
		}

		if (parsed.runPersistent) {
			const res = await defaultPersistentTerminalManager.execute(
				parsed.terminalId ?? "default",
				parsed.command,
				{
					cwd: ctx.cwd,
					timeout: parsed.timeout,
					signal: ctx.signal,
					onUpdate: ctx.onUpdate,
				},
			);
			return {
				content: res.content,
				details: {
					terminalId: res.terminalId,
					exitCode: res.exitCode,
					status: res.status,
				},
				isError:
					res.status === "failed" ||
					(res.exitCode !== 0 && res.exitCode !== null),
			};
		}

		const result = await executeSingleCommand(
			{
				command: parsed.command,
				timeout: parsed.timeout,
				waitMsBeforeAsync: parsed.waitMsBeforeAsync,
			},
			ctx,
		);
		return { content: result.content, details: result.details };
	},
};

async function executeSingleCommand(
	{ command, timeout, waitMsBeforeAsync }: SingleBashArgs,
	ctx: Parameters<Tool["execute"]>[1],
): Promise<CommandExecutionResult> {
	// Resolve shell
	const { shell, args: shellArgs } = getShellConfig();

	// Resolve cwd
	const cwd = ctx.cwd || process.cwd();
	try {
		await fsAccess(cwd, constants.F_OK);
	} catch {
		return {
			content: `Error: Working directory does not exist: ${cwd}`,
			exitCode: null,
			signal: null,
			status: "failed",
		};
	}

	const shellEnv = getShellEnv(cwd);
	const output = new OutputAccumulator({ tempFilePrefix: "logician-bash" });
	const timeoutSeconds = timeout;
	const throttledUpdate = makeUpdateThrottler();
	const startTime = Date.now();

	return new Promise<CommandExecutionResult>(resolve => {
		let settled = false;
		let transitionedToBackground = false;
		let asyncTimer: NodeJS.Timeout | undefined;

		const settle = (fn: () => void) => {
			if (!settled) {
				settled = true;
				if (asyncTimer) clearTimeout(asyncTimer);
				fn();
			}
		};

		const child = spawn(shell, [...shellArgs, command], {
			cwd,
			stdio: ["pipe", "pipe", "pipe"],
			env: shellEnv,
			detached: process.platform !== "win32",
		});

		// Track detached child for cleanup
		if (child.pid) {
			if (process.platform !== "win32") {
				trackDetachedChildPid(child.pid);
			}
		}

		let timedOut = false;
		let timeoutHandle: NodeJS.Timeout | undefined;
		let hasError = false;

		const onAbort = () => {
			if (child.pid) killProcessTree(child.pid);
		};

		// Abort signal
		if (ctx.signal) {
			if (ctx.signal.aborted) {
				onAbort();
			} else {
				ctx.signal.addEventListener("abort", onAbort, { once: true });
			}
		}

		// Timeout
		if (timeoutSeconds && timeoutSeconds > 0) {
			timeoutHandle = setTimeout(() => {
				timedOut = true;
				if (child.pid) killProcessTree(child.pid);
			}, timeoutSeconds * 1000);
		}

		// Background async timer
		if (waitMsBeforeAsync !== undefined && waitMsBeforeAsync > 0) {
			asyncTimer = setTimeout(() => {
				if (!settled) {
					transitionedToBackground = true;
					const taskEntry = defaultTaskManager.registerTask({
						command,
						cwd,
						child,
						output,
						startTime,
					});

					settle(() => {
						resolve({
							content: `Command is running in the background as task "${taskEntry.id}" (PID: ${child.pid ?? "unknown"}).\nLog file: ${taskEntry.logFilePath}\nUse manage_task with action="status", taskId="${taskEntry.id}" to check output, or action="kill" to terminate.`,
							details: {
								taskId: taskEntry.id,
								pid: child.pid,
								logFilePath: taskEntry.logFilePath,
								status: "running",
								background: true,
							},
							exitCode: null,
							signal: null,
							status: "completed",
						});
					});
				}
			}, waitMsBeforeAsync);
		}

		// Stream output
		const handleData = (data: Buffer) => {
			output.append(data);
			// Stream updates to TUI (throttled)
			const snapshot = output.snapshot();
			if (snapshot.content && ctx.onUpdate) {
				throttledUpdate(() => {
					ctx.onUpdate?.(snapshot.content);
				});
			}
		};

		child.stdout?.on("data", handleData);
		child.stderr?.on("data", handleData);

		// Handle errors
		child.on("error", err => {
			hasError = true;
			if (asyncTimer) clearTimeout(asyncTimer);
			if (transitionedToBackground) return;

			output.finish();
			if (child.pid) untrackDetachedChildPid(child.pid);
			settle(() => {
				if (timeoutHandle) clearTimeout(timeoutHandle);
				ctx.signal?.removeEventListener("abort", onAbort);
				resolve({
					content: `Error: ${err.message || "Command failed"}`,
					exitCode: null,
					signal: null,
					status: "failed",
				});
			});
		});

		// Handle exit
		child.on("close", (code, signal) => {
			if (asyncTimer) clearTimeout(asyncTimer);
			if (transitionedToBackground) return;

			output.finish();
			if (child.pid) untrackDetachedChildPid(child.pid);
			if (timeoutHandle) clearTimeout(timeoutHandle);
			ctx.signal?.removeEventListener("abort", onAbort);

			settle(() => {
				const snapshot = output.snapshot({ persistIfTruncated: true });
				output.closeTempFile().catch(() => {});

				// If we already had an error (e.g. abort during close), skip
				if (hasError) return;

				// Check for abort
				if (ctx.signal?.aborted) {
					const { text, details } = formatOutput(
						snapshot,
						code,
						signal,
						output.getLastLineBytes(),
					);
					resolve({
						content: appendStatus(
							text === "(no output)" ? "" : text,
							"Command aborted",
						),
						details,
						exitCode: code,
						signal,
						status: "aborted",
					});
					return;
				}

				// Check for timeout
				if (timedOut) {
					const { text, details } = formatOutput(
						snapshot,
						code,
						signal,
						output.getLastLineBytes(),
					);
					resolve({
						content: appendStatus(
							text === "(no output)" ? "" : text,
							`Command timed out after ${timeoutSeconds} seconds`,
						),
						details,
						exitCode: code,
						signal,
						status: "timed_out",
					});
					return;
				}

				// Build result
				const { text, details } = formatOutput(
					snapshot,
					code,
					signal,
					output.getLastLineBytes(),
				);
				resolve({
					content: text,
					details,
					exitCode: code,
					signal,
					status: code === 0 ? "completed" : "failed",
				});
			});
		});
	});
}

async function executeBatch(
	args: BashArgs,
	ctx: Parameters<Tool["execute"]>[1],
): Promise<ToolResult> {
	const entries = validateBatchEntries(args.commands);
	if (typeof entries === "string") return { content: entries, isError: true };
	const mode = args.mode ?? "sequential";
	if (mode !== "sequential" && mode !== "parallel") {
		return {
			content: 'Error: mode must be "sequential" or "parallel".',
			isError: true,
		};
	}
	const concurrency = normalizeConcurrency(args.maxConcurrency);
	if (typeof concurrency === "string")
		return { content: concurrency, isError: true };
	const normalized = entries.map((entry, index) => ({
		id: entry.id ?? `command-${index + 1}`,
		command: entry.command,
		timeout: entry.timeout ?? args.timeout,
	}));
	const results: BashBatchResult[] = [];

	if (mode === "sequential") {
		for (const entry of normalized) {
			const blockedReason = findDestructiveMatch(entry.command);
			if (
				ctx.signal?.aborted ||
				(args.stopOnFailure && results.some(isBatchFailure))
			) {
				results.push(
					skippedResult(
						entry,
						ctx.signal?.aborted ? "Batch aborted" : "Skipped after failure",
					),
				);
			} else if (blockedReason) {
				results.push(blockedResult(entry, blockedReason));
			} else {
				results.push(
					toBatchResult(entry, await executeSingleCommand(entry, ctx)),
				);
			}
		}
	} else {
		let nextIndex = 0;
		const ordered = new Array<BashBatchResult>(normalized.length);
		const worker = async () => {
			while (nextIndex < normalized.length) {
				const index = nextIndex++;
				const entry = normalized[index];
				const blockedReason = findDestructiveMatch(entry.command);
				if (ctx.signal?.aborted) {
					ordered[index] = skippedResult(entry, "Batch aborted");
				} else if (blockedReason) {
					ordered[index] = blockedResult(entry, blockedReason);
				} else {
					ordered[index] = toBatchResult(
						entry,
						await executeSingleCommand(entry, ctx),
					);
				}
			}
		};
		await Promise.all(
			Array.from({ length: Math.min(concurrency, normalized.length) }, () =>
				worker(),
			),
		);
		results.push(...ordered);
	}

	return {
		content: results
			.map(
				result =>
					`[${result.id}] ${result.status}: ${result.command}\n${result.content}`,
			)
			.join("\n\n"),
		details: {
			mode,
			commands: results,
			summary: {
				total: results.length,
				completed: results.filter(result => result.status === "completed")
					.length,
				failed: results.filter(isBatchFailure).length,
				skipped: results.filter(result => result.status === "skipped").length,
			},
		},
	};
}

function validateBatchEntries(value: unknown): BashBatchEntry[] | string {
	if (!Array.isArray(value) || value.length === 0) {
		return "Error: commands must be a non-empty array.";
	}
	if (value.length > 32) return "Error: commands supports at most 32 entries.";
	const entries: BashBatchEntry[] = [];
	const ids = new Set<string>();
	for (let index = 0; index < value.length; index++) {
		const raw = value[index];
		if (!raw || typeof raw !== "object")
			return `Error: commands[${index}] must be an object.`;
		const entry = raw as Record<string, unknown>;
		if (typeof entry.command !== "string" || entry.command.trim() === "") {
			return `Error: commands[${index}].command must be a non-empty string.`;
		}
		if (
			entry.id !== undefined &&
			(typeof entry.id !== "string" || entry.id.trim() === "")
		) {
			return `Error: commands[${index}].id must be a non-empty string.`;
		}
		if (typeof entry.id === "string") {
			if (ids.has(entry.id))
				return `Error: duplicate command id "${entry.id}".`;
			ids.add(entry.id);
		}
		if (
			entry.timeout !== undefined &&
			(!Number.isFinite(entry.timeout) || Number(entry.timeout) <= 0)
		) {
			return `Error: commands[${index}].timeout must be a positive number.`;
		}
		entries.push({
			command: entry.command,
			...(typeof entry.id === "string" ? { id: entry.id } : {}),
			...(entry.timeout === undefined
				? {}
				: { timeout: Number(entry.timeout) }),
		});
	}
	return entries;
}

function normalizeConcurrency(value: unknown): number | string {
	if (value === undefined) return 4;
	const concurrency = Number(value);
	return Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= 16
		? concurrency
		: "Error: maxConcurrency must be an integer between 1 and 16.";
}

function skippedResult(
	entry: { id: string; command: string },
	content: string,
): BashBatchResult {
	return {
		id: entry.id,
		command: entry.command,
		content,
		exitCode: null,
		signal: null,
		status: "skipped",
	};
}

function blockedResult(
	entry: { id: string; command: string },
	reason: string,
): BashBatchResult {
	return {
		id: entry.id,
		command: entry.command,
		content: `Error: command blocked (${reason}): ${entry.command}`,
		exitCode: null,
		signal: null,
		status: "failed",
	};
}

function toBatchResult(
	entry: { id: string; command: string },
	result: CommandExecutionResult,
): BashBatchResult {
	return {
		id: entry.id,
		command: entry.command,
		content: result.content,
		exitCode: result.exitCode,
		signal: result.signal,
		status: result.status,
		...(result.details ? { details: result.details } : {}),
	};
}

function isBatchFailure(result: BashBatchResult): boolean {
	return ["failed", "timed_out", "aborted"].includes(result.status);
}

function formatOutput(
	snapshot: {
		content: string;
		truncation: TruncationResult;
		fullOutputPath?: string;
	},
	code: number | null,
	signal: string | null,
	lastLineBytes = 0,
): { text: string; details: BashDetails | undefined } {
	const truncation = snapshot.truncation;
	const emptyText = "(no output)";
	let text = snapshot.content || emptyText;
	let details: BashDetails | undefined;

	if (truncation.truncated) {
		const startLine = truncation.totalLines - truncation.outputLines + 1;
		const endLine = truncation.totalLines;
		const fullPath = snapshot.fullOutputPath;

		const notices: string[] = [];
		if (truncation.lastLinePartial) {
			notices.push(
				`Showing last ${formatSize(truncation.outputBytes)} of line ${endLine} (line is ${formatSize(lastLineBytes)}). Full output: ${fullPath}`,
			);
		} else if (truncation.truncatedBy === "lines") {
			notices.push(
				`Showing lines ${startLine}-${endLine} of ${truncation.totalLines}. Full output: ${fullPath}`,
			);
		} else {
			notices.push(
				`Showing lines ${startLine}-${endLine} of ${truncation.totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Full output: ${fullPath}`,
			);
		}

		text += `\n\n[${notices.join(". ")}]`;
		details = {
			truncation: {
				truncated: true,
				totalLines: truncation.totalLines,
				totalBytes: truncation.totalBytes,
				outputLines: truncation.outputLines,
				outputBytes: truncation.outputBytes,
				lastLinePartial: truncation.lastLinePartial,
				truncatedBy: truncation.truncatedBy ?? undefined,
				maxLines: truncation.maxLines,
				maxBytes: truncation.maxBytes,
			},
			fullOutputPath: fullPath,
		};
	}

	if (code !== 0 && code !== null) {
		text = appendStatus(text, `Command exited with code ${code}`);
	} else if (code === null && signal) {
		text = appendStatus(text, `Command terminated by signal ${signal}`);
	}

	return { text, details };
}

function appendStatus(text: string, status: string): string {
	return `${text ? `${text}\n\n` : ""}${status}`;
}
