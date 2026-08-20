// ── write tool ────────────────────────────────────────────────────────────
// Create or overwrite a complete file. Creates parent directories.
// Overwriting an existing file requires it to have been read first (and not
// modified since), so the model can never blind-clobber content. Ported from
// coding-agent's tools/write-file.ts, rewritten against ExecutionEnv.
// Syntax-highlighted output (coding-agent's highlightAuto) is dropped for
// this pass — returns plain line-numbered content instead.

import { type Static, Type } from "@sinclair/typebox";
import type { AgentTool } from "../../agent/types.ts";
import type { ExecutionEnv } from "../../env/execution-env.ts";
import { atomicWriteFile } from "./atomic-write.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import { ensureInsideCwd, resolveToCwd } from "./path-utils.ts";
import {
	hasBeenRead,
	isStaleSinceRead,
	refreshAfterWrite,
} from "./read-tracker.ts";

const writeSchema = Type.Object({
	path: Type.String({ description: "File path to write" }),
	content: Type.String({ description: "Complete file contents" }),
});

export interface WriteToolOptions {
	env: ExecutionEnv;
	allowedPaths?: string[];
	allowAllPaths?: boolean;
}

export function createWriteTool(
	options: WriteToolOptions,
): AgentTool<typeof writeSchema, undefined> {
	const { env, allowedPaths, allowAllPaths } = options;
	return {
		name: "write",
		label: "Write File",
		executionMode: "parallel",
		description:
			"Create or overwrite a complete file. Creates parent directories. Overwriting an existing file requires reading it with read first. For very large files, use write_append in chunks instead.",
		parameters: writeSchema,
		prepareArguments: (raw): unknown => {
			if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
			const args = raw as Record<string, unknown>;
			return {
				...args,
				path: args.path ?? args.file_path ?? args.filename,
				content: args.content ?? args.text,
			};
		},
		async execute(_toolCallId, rawParams) {
			const { path, content } = rawParams as Static<typeof writeSchema>;
			const resolved = await resolveToCwd(env, path);
			await ensureInsideCwd(env, resolved, allowedPaths, allowAllPaths);

			return withFileMutationQueue(env, resolved, async () => {
				const beforeResult = await env.readTextFile(resolved);
				const before = beforeResult.ok ? beforeResult.value : null;
				if (before !== null) {
					if (!(await hasBeenRead(env, resolved))) {
						throw new Error(
							`${resolved} already exists but has not been read. Read it with read before overwriting, or use edit for targeted changes.`,
						);
					}
					if (await isStaleSinceRead(env, resolved)) {
						throw new Error(
							`${resolved} has been modified since it was last read. Read it again before overwriting.`,
						);
					}
				}
				if (before === content) {
					return {
						content: [
							{ type: "text" as const, text: `No changes made: ${resolved}` },
						],
						details: undefined,
					};
				}

				await atomicWriteFile(env, resolved, content, {
					expectedContent: before ?? undefined,
					expectedMissing: before === null,
				});
				await refreshAfterWrite(env, resolved);

				const lineCount = content === "" ? 0 : content.split("\n").length;
				const byteLen = new TextEncoder().encode(content).byteLength;
				if (before === null) {
					return {
						content: [
							{
								type: "text" as const,
								text: `Created ${resolved} (${lineCount} lines, ${byteLen} bytes)`,
							},
						],
						details: undefined,
					};
				}

				const gutterWidth = String(lineCount).length + 1;
				const lines = content.split("\n");
				const header = `Wrote ${resolved} (${lineCount} lines, ${byteLen} bytes)`;
				const out: string[] = [header];
				for (let i = 0; i < lines.length; i++) {
					const num = String(i + 1).padStart(gutterWidth, " ");
					out.push(`${num}|${lines[i]}`);
				}
				return {
					content: [{ type: "text" as const, text: out.join("\n") }],
					details: undefined,
				};
			});
		},
	};
}
