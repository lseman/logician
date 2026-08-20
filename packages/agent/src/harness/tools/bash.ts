// ── bash tool ─────────────────────────────────────────────────────────────
// Execute a shell command with timeout, streaming output, and truncation.
// Ported from coding-agent's tools/bash.ts, rewritten as a thin wrapper over
// ExecutionEnv.exec — which already owns spawn/process-group-kill/timeout/
// abort-signal handling — instead of driving node:child_process directly.
// Batch mode (commands[]/mode/stopOnFailure/maxConcurrency), a coding-agent-
// specific extension beyond a single AgentTool call, is not ported.

import { type Static, Type } from "@sinclair/typebox";
import type { AgentTool } from "../../agent/types.ts";
import type { ExecutionEnv } from "../../env/execution-env.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	OutputAccumulator,
} from "./truncate.ts";

const bashSchema = Type.Object({
	command: Type.String({ description: "Bash command to execute" }),
	timeout: Type.Optional(Type.Number({ description: "Timeout in seconds" })),
});

// ── Destructive command guard ────────────────────────────────────────────
// Not a sandbox — a denylist for a short list of unambiguously destructive
// patterns that have no legitimate use in agent-driven development (wiping
// home/root, force-pushing over history, fork bombs, curl-pipe-to-shell).
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

function makeUpdateThrottler(): (callback: () => void) => void {
	let lastUpdate = 0;
	const THROTTLE_MS = 100;
	return callback => {
		const now = Date.now();
		if (now - lastUpdate < THROTTLE_MS) return;
		lastUpdate = now;
		callback();
	};
}

function appendStatus(text: string, status: string): string {
	return `${text ? `${text}\n\n` : ""}${status}`;
}

function formatOutput(
	snapshot: ReturnType<OutputAccumulator["snapshot"]>,
	exitCode: number,
	lastLineBytes: number,
): string {
	const truncation = snapshot.truncation;
	let text = snapshot.content || "(no output)";

	if (truncation.truncated) {
		const startLine = truncation.totalLines - truncation.outputLines + 1;
		const endLine = truncation.totalLines;
		const fullPath = snapshot.fullOutputPath;
		let notice: string;
		if (truncation.lastLinePartial) {
			notice = `Showing last ${formatSize(truncation.outputBytes)} of line ${endLine} (line is ${formatSize(lastLineBytes)}). Full output: ${fullPath}`;
		} else if (truncation.truncatedBy === "lines") {
			notice = `Showing lines ${startLine}-${endLine} of ${truncation.totalLines}. Full output: ${fullPath}`;
		} else {
			notice = `Showing lines ${startLine}-${endLine} of ${truncation.totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Full output: ${fullPath}`;
		}
		text += `\n\n[${notice}]`;
	}

	if (exitCode !== 0) {
		text = appendStatus(text, `Command exited with code ${exitCode}`);
	}
	return text;
}

export interface BashToolOptions {
	env: ExecutionEnv;
}

export function createBashTool(
	options: BashToolOptions,
): AgentTool<typeof bashSchema, undefined> {
	const { env } = options;
	return {
		name: "bash",
		label: "Bash",
		executionMode: "sequential",
		description: `Execute bash commands with timeout. Output is streamed and truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB.`,
		parameters: bashSchema,
		prepareArguments: (raw): unknown => {
			if (typeof raw === "string") return { command: raw };
			if (!raw || typeof raw !== "object") return {};
			const args = raw as Record<string, unknown>;
			const command = args.command ?? args.cmd ?? args.script ?? args.input;
			return {
				...args,
				...(command === undefined ? {} : { command: String(command) }),
			};
		},
		async execute(_toolCallId, rawParams, signal, onUpdate) {
			const { command, timeout } = rawParams as Static<typeof bashSchema>;
			if (!command) throw new Error("command is required.");

			const blockedReason = findDestructiveMatch(command);
			if (blockedReason)
				throw new Error(`Command blocked (${blockedReason}): ${command}`);

			const output = new OutputAccumulator({
				tempFilePrefix: "agent-bash",
			});
			const throttledUpdate = makeUpdateThrottler();
			const handleChunk = (chunk: string) => {
				output.append(Buffer.from(chunk, "utf-8"));
				const snapshot = output.snapshot();
				if (snapshot.content && onUpdate) {
					throttledUpdate(() =>
						onUpdate({
							content: [{ type: "text", text: snapshot.content }],
							details: undefined,
						}),
					);
				}
			};

			const result = await env.exec(command, {
				timeout,
				abortSignal: signal,
				onStdout: handleChunk,
				onStderr: handleChunk,
			});
			output.finish();

			if (!result.ok) {
				await output.closeTempFile().catch(() => {});
				if (result.error.code === "timeout") {
					const snapshot = output.snapshot({ persistIfTruncated: true });
					const text = formatOutput(snapshot, 0, output.getLastLineBytes());
					throw new Error(
						appendStatus(
							text === "(no output)" ? "" : text,
							`Command timed out after ${timeout} seconds`,
						),
					);
				}
				if (result.error.code === "aborted") {
					const snapshot = output.snapshot({ persistIfTruncated: true });
					const text = formatOutput(snapshot, 0, output.getLastLineBytes());
					throw new Error(
						appendStatus(text === "(no output)" ? "" : text, "Command aborted"),
					);
				}
				throw new Error(result.error.message);
			}

			const snapshot = output.snapshot({ persistIfTruncated: true });
			await output.closeTempFile().catch(() => {});
			const text = formatOutput(
				snapshot,
				result.value.exitCode,
				output.getLastLineBytes(),
			);

			return { content: [{ type: "text", text }], details: undefined };
		},
	};
}
