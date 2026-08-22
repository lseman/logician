// ── git tool ───────────────────────────────────────────────────────────────────────
// Execute git commands with output limiting.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { Tool } from "@logician/log-core";
import { formatTruncationNotice } from "./support/utils/truncate.ts";

const execFileAsync = promisify(execFile);

export const git: Tool = {
	name: "git",
	executionMode: "sequential",
	label: "Git",
	description: "Execute a git command (safe subset of git operations).",
	promptSnippet: "Run git commands for version control operations",
	parameters: {
		type: "object",
		properties: {
			command: {
				type: "string",
				description: "Git subcommand (status, diff, log, add, commit, etc.)",
			},
		},
		required: ["command"],
	},
	execute: async (args, ctx): Promise<string> => {
		const command = String(args.command);
		const maxOutput = ctx.maxOutputChars || 8192;

		// Safe git commands only (no destructive operations)
		const allowedPatterns = [
			/^status\b/,
			/^diff\b/,
			/^log\b/,
			/^show\b/,
			/^add\b/,
			/^commit\b/,
			/^checkout\b/,
			/^switch\b/,
			/^branch\b/,
			/^tag\b/,
			/^merge\b/,
			/^pull\b/,
			/^push\b/,
			/^fetch\b/,
			/^remote\b/,
			/^reset\b.*--soft$/,
		];

		const isAllowed = allowedPatterns.some(p => p.test(command));
		if (!isAllowed) {
			return `Error: Command not allowed: ${command}. Use safe git operations only.`;
		}

		try {
			const { stdout } = await execFileAsync("git", command.split(" "), {
				cwd: ctx.cwd,
				timeout: 10000,
				maxBuffer: 1024 * 1024,
				signal: ctx.signal,
				killSignal: "SIGKILL",
			});
			const trimmed = stdout.trim().slice(0, maxOutput);
			return (
				trimmed +
				(stdout.length > maxOutput
					? formatTruncationNotice(
							'add a path filter (e.g. "diff -- <path>") or a limit (e.g. "log -n 20") to narrow the output',
						)
					: "")
			);
		} catch (err: unknown) {
			const error = err as {
				name?: string;
				code?: number | string;
				message?: string;
				stderr?: string;
			};
			if (error.name === "AbortError" || error.code === "ABORT_ERR") {
				return "Error: Command aborted";
			}
			const detail = (error.stderr || error.message || "").trim();
			return detail
				? `Error: git ${command} failed: ${detail}`
				: `Error: git ${command} failed with no output. Check that the command and arguments are valid for this repository.`;
		}
	},
};
