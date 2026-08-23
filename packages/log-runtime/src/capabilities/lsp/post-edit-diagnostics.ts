import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AgentHooks } from "@logician/log-core";
import { ensureInsideCwd } from "../tools/support/utils/path-utils.ts";
import type { LspClientPool } from "./lsp-client-pool.ts";

const MAX_SOURCE_BYTES = 1_000_000;
const EDIT_TOOLS = new Set(["edit_file", "write_file"]);

export interface PostEditDiagnostic {
	line: number;
	column: number;
	message: string;
	code?: number | string;
	source?: string;
}

function successfulMutation(toolName: string, result: string): boolean {
	if (toolName === "edit_file")
		return result.startsWith("Successfully replaced ");
	if (toolName === "write_file") {
		return result.startsWith("Created ") || result.startsWith("Wrote ");
	}
	return false;
}

function collectJsonDiagnostics(source: string): PostEditDiagnostic[] {
	try {
		JSON.parse(source);
		return [];
	} catch (error: unknown) {
		const message = error instanceof Error ? error.message : String(error);
		const offsetMatch = /\bposition\s+(\d+)/i.exec(message);
		const offset = Math.min(source.length, Number(offsetMatch?.[1] ?? 0));
		const before = source.slice(0, offset);
		const lines = before.split("\n");
		return [
			{
				line: lines.length,
				column: (lines.at(-1)?.length ?? 0) + 1,
				message,
			},
		];
	}
}

export async function diagnoseEditedFile(
	cwd: string,
	fileName: string,
	allowedPaths?: string[],
	allowAllPaths?: boolean,
): Promise<PostEditDiagnostic[]> {
	const resolved = path.resolve(cwd, fileName);
	ensureInsideCwd(cwd, resolved, allowedPaths, allowAllPaths);
	const extension = path.extname(resolved).toLowerCase();
	if (extension !== ".json") return [];

	const source = await readFile(resolved, "utf8");
	if (Buffer.byteLength(source, "utf8") > MAX_SOURCE_BYTES) return [];
	return collectJsonDiagnostics(source);
}

function formatDiagnostics(
	fileName: string,
	diagnostics: PostEditDiagnostic[],
): string {
	const lines = diagnostics.map(diagnostic => {
		const metadata = diagnostic.source
			? [diagnostic.source, diagnostic.code].filter(item => item !== undefined)
			: diagnostic.code === undefined
				? []
				: [`TS${diagnostic.code}`];
		const label = metadata.length > 0 ? ` ${metadata.join(" ")}` : "";
		return `- ${fileName}:${diagnostic.line}:${diagnostic.column}${label}: ${diagnostic.message}`;
	});
	return [
		"",
		`<post_edit_diagnostics file="${fileName}">`,
		"Fix these project diagnostics before continuing:",
		...lines,
		"</post_edit_diagnostics>",
	].join("\n");
}

export function createPostEditDiagnosticHooks(
	cwd: string,
	isEnabled: () => boolean = () => true,
	lspManager?: LspClientPool,
	pathPolicy?: {
		allowedPaths?: string[];
		allowAllPaths?: boolean;
	},
): AgentHooks {
	return {
		afterToolCall: async ({ toolCall, args, result, isError }) => {
			if (
				!isEnabled() ||
				isError ||
				!EDIT_TOOLS.has(toolCall.name) ||
				!successfulMutation(toolCall.name, result)
			) {
				return undefined;
			}
			const fileName = typeof args.path === "string" ? args.path : "";
			if (!fileName) return undefined;

			try {
				const resolved = path.resolve(cwd, fileName);
				const lspDiagnostics = await lspManager?.diagnosticsFor(resolved);
				const diagnostics = lspDiagnostics?.length
					? lspDiagnostics
					: await diagnoseEditedFile(
							cwd,
							fileName,
							pathPolicy?.allowedPaths,
							pathPolicy?.allowAllPaths,
						);
				if (diagnostics.length === 0) return undefined;
				return { content: result + formatDiagnostics(fileName, diagnostics) };
			} catch {
				// Diagnostics are advisory and must never turn a successful edit into
				// a failed tool call.
				return undefined;
			}
		},
	};
}
