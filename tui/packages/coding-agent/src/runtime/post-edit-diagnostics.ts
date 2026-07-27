import { readFile } from "node:fs/promises";
import path from "node:path";
import type { AgentHooks } from "@logician/agent-core";
import { ensureInsideCwd } from "@logician/agent-core/tools/shared/path-utils.ts";
import ts from "typescript";
import type { LspManager } from "./lsp-manager.ts";

const MAX_SOURCE_BYTES = 1_000_000;
const MAX_DIAGNOSTICS = 10;
const EDIT_TOOLS = new Set(["edit_file", "write_file"]);
const TS_EXTENSIONS = new Set([
	".js",
	".jsx",
	".mjs",
	".cjs",
	".ts",
	".tsx",
	".mts",
	".cts",
]);

export interface PostEditDiagnostic {
	line: number;
	column: number;
	message: string;
	code?: number | string;
	source?: string;
}

function successfulMutation(toolName: string, result: string): boolean {
	if (toolName === "edit_file") return result.startsWith("Successfully replaced ");
	if (toolName === "write_file") {
		return result.startsWith("Created ") || result.startsWith("Wrote ");
	}
	return false;
}

function scriptKindFor(extension: string): ts.ScriptKind {
	switch (extension) {
		case ".js":
		case ".mjs":
		case ".cjs":
			return ts.ScriptKind.JS;
		case ".jsx":
			return ts.ScriptKind.JSX;
		case ".tsx":
			return ts.ScriptKind.TSX;
		default:
			return ts.ScriptKind.TS;
	}
}

function flattenMessage(message: string | ts.DiagnosticMessageChain): string {
	return ts.flattenDiagnosticMessageText(message, "\n");
}

function collectTypeScriptDiagnostics(
	filePath: string,
	source: string,
	extension: string,
): PostEditDiagnostic[] {
	const output = ts.transpileModule(source, {
		fileName: filePath,
		reportDiagnostics: true,
		compilerOptions: {
			allowJs: true,
			checkJs: false,
			noEmit: true,
			target: ts.ScriptTarget.Latest,
			module: ts.ModuleKind.ESNext,
			jsx: ts.JsxEmit.Preserve,
		},
	});
	const sourceFile = ts.createSourceFile(
		filePath,
		source,
		ts.ScriptTarget.Latest,
		true,
		scriptKindFor(extension),
	);
	return (output.diagnostics ?? []).slice(0, MAX_DIAGNOSTICS).map((diagnostic) => {
		const position = sourceFile.getLineAndCharacterOfPosition(diagnostic.start ?? 0);
		return {
			line: position.line + 1,
			column: position.character + 1,
			message: flattenMessage(diagnostic.messageText),
			code: diagnostic.code,
		};
	});
}

function collectProjectDiagnostics(
	cwd: string,
	filePath: string,
): PostEditDiagnostic[] {
	const configPath = ts.findConfigFile(path.dirname(filePath), ts.sys.fileExists);
	if (!configPath) return [];
	const relativeConfig = path.relative(path.resolve(cwd), path.resolve(configPath));
	if (relativeConfig.startsWith("..") || path.isAbsolute(relativeConfig)) return [];

	const loaded = ts.readConfigFile(configPath, ts.sys.readFile);
	if (loaded.error) return [];
	const parsed = ts.parseJsonConfigFileContent(
		loaded.config,
		ts.sys,
		path.dirname(configPath),
		{ noEmit: true },
		configPath,
	);
	const program = ts.createProgram(parsed.fileNames, parsed.options);
	const sourceFile = program.getSourceFile(filePath);
	if (!sourceFile) return [];

	return [
		...program.getSyntacticDiagnostics(sourceFile),
		...program.getSemanticDiagnostics(sourceFile),
	]
		.slice(0, MAX_DIAGNOSTICS)
		.map((diagnostic) => {
			const position = sourceFile.getLineAndCharacterOfPosition(
				diagnostic.start ?? 0,
			);
			return {
				line: position.line + 1,
				column: position.character + 1,
				message: flattenMessage(diagnostic.messageText),
				code: diagnostic.code,
			};
		});
}

function collectJsonDiagnostics(
	source: string,
): PostEditDiagnostic[] {
	try {
		JSON.parse(source);
		return [];
	} catch (error: unknown) {
		const message = error instanceof Error ? error.message : String(error);
		const offsetMatch = /\bposition\s+(\d+)/i.exec(message);
		const offset = Math.min(source.length, Number(offsetMatch?.[1] ?? 0));
		const before = source.slice(0, offset);
		const lines = before.split("\n");
		return [{
			line: lines.length,
			column: (lines.at(-1)?.length ?? 0) + 1,
			message,
		}];
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
	if (!TS_EXTENSIONS.has(extension) && extension !== ".json") return [];

	const source = await readFile(resolved, "utf8");
	if (Buffer.byteLength(source, "utf8") > MAX_SOURCE_BYTES) return [];
	if (extension === ".json") return collectJsonDiagnostics(source);
	const projectDiagnostics = collectProjectDiagnostics(cwd, resolved);
	return projectDiagnostics.length > 0
		? projectDiagnostics
		: collectTypeScriptDiagnostics(resolved, source, extension);
}

function formatDiagnostics(
	fileName: string,
	diagnostics: PostEditDiagnostic[],
): string {
	const lines = diagnostics.map((diagnostic) => {
		const metadata = diagnostic.source
			? [diagnostic.source, diagnostic.code].filter((item) => item !== undefined)
			: diagnostic.code === undefined ? [] : [`TS${diagnostic.code}`];
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
	lspManager?: LspManager,
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
