import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import type { AgentHooks } from "../../core/types/index.ts";
import { ensureInsideCwd } from "../tools/utils/path-utils.ts";
import type { LspManager } from "./lsp-manager.ts";

const execFileAsync = promisify(execFile);
const runtimeRequire = createRequire(import.meta.url);

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
	if (toolName === "edit_file")
		return result.startsWith("Successfully replaced ");
	if (toolName === "write_file") {
		return result.startsWith("Created ") || result.startsWith("Wrote ");
	}
	return false;
}

let cachedTscBin: string | null = null;

/** Resolve the tsc CLI shipped with the installed typescript package.
 *  "typescript/bin/tsc" isn't in the package's exports map, so resolve the
 *  package root via its (exported) package.json and join the bin path on
 *  disk rather than through module resolution. */
function tscBin(): string {
	if (cachedTscBin) return cachedTscBin;
	const packageJsonPath = runtimeRequire.resolve("typescript/package.json");
	cachedTscBin = path.join(path.dirname(packageJsonPath), "bin", "tsc");
	return cachedTscBin;
}

// Matches non-pretty `tsc` diagnostic lines, e.g.:
//   broken.ts(1,15): error TS1109: Expression expected.
const TSC_DIAGNOSTIC_LINE =
	/^(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)$/;

function parseTscOutput(
	output: string,
	fileBaseName: string,
): PostEditDiagnostic[] {
	const diagnostics: PostEditDiagnostic[] = [];
	for (const line of output.split("\n")) {
		const match = TSC_DIAGNOSTIC_LINE.exec(line.trim());
		if (!match) continue;
		const [, reportedFile, lineStr, columnStr, code, message] = match;
		if (path.basename(reportedFile) !== fileBaseName) continue;
		diagnostics.push({
			line: Number(lineStr),
			column: Number(columnStr),
			message,
			code: Number(code.slice(2)),
		});
		if (diagnostics.length >= MAX_DIAGNOSTICS) break;
	}
	return diagnostics;
}

async function runTsc(args: string[], cwd: string): Promise<string> {
	try {
		const result = await execFileAsync(
			process.execPath,
			[tscBin(), ...args, "--pretty", "false"],
			{ cwd, timeout: 10_000, maxBuffer: 4 * 1024 * 1024 },
		);
		return result.stdout ?? "";
	} catch (error) {
		// tsc exits non-zero when it reports diagnostics — that's the
		// normal, expected outcome here, not a failure to run it.
		const withOutput = error as { stdout?: string };
		return withOutput.stdout ?? "";
	}
}

async function collectTypeScriptDiagnostics(
	filePath: string,
	source: string,
): Promise<PostEditDiagnostic[]> {
	const tmpDir = mkdtempSync(path.join(tmpdir(), "logician-tsc-"));
	const fileBaseName = path.basename(filePath);
	const tmpFile = path.join(tmpDir, fileBaseName);
	try {
		await writeFile(tmpFile, source, "utf8");
		const output = await runTsc(
			[
				"--noEmit",
				"--target",
				"ESNext",
				"--module",
				"ESNext",
				"--jsx",
				"preserve",
				"--allowJs",
				"--checkJs",
				"false",
				fileBaseName,
			],
			tmpDir,
		);
		return parseTscOutput(output, fileBaseName);
	} finally {
		rmSync(tmpDir, { recursive: true, force: true });
	}
}

/** Walk upward from `startDir` looking for the nearest tsconfig.json. */
function findConfigFile(startDir: string): string | undefined {
	let dir = startDir;
	while (true) {
		const candidate = path.join(dir, "tsconfig.json");
		if (existsSync(candidate)) return candidate;
		const parent = path.dirname(dir);
		if (parent === dir) return undefined;
		dir = parent;
	}
}

async function collectProjectDiagnostics(
	cwd: string,
	filePath: string,
): Promise<PostEditDiagnostic[]> {
	const configPath = findConfigFile(path.dirname(filePath));
	if (!configPath) return [];
	const relativeConfig = path.relative(
		path.resolve(cwd),
		path.resolve(configPath),
	);
	if (relativeConfig.startsWith("..") || path.isAbsolute(relativeConfig))
		return [];

	const output = await runTsc(
		["--noEmit", "-p", configPath],
		path.dirname(configPath),
	);
	return parseTscOutput(output, path.basename(filePath));
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
	if (!TS_EXTENSIONS.has(extension) && extension !== ".json") return [];

	const source = await readFile(resolved, "utf8");
	if (Buffer.byteLength(source, "utf8") > MAX_SOURCE_BYTES) return [];
	if (extension === ".json") return collectJsonDiagnostics(source);
	const projectDiagnostics = await collectProjectDiagnostics(cwd, resolved);
	return projectDiagnostics.length > 0
		? projectDiagnostics
		: collectTypeScriptDiagnostics(resolved, source);
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
