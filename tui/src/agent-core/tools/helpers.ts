import * as fs from "node:fs";
import * as path from "node:path";

export function resolvePath(
	cwd: string | undefined,
	inputPath: string,
): string {
	const base = cwd || process.cwd();
	return path.isAbsolute(inputPath)
		? path.normalize(inputPath)
		: path.resolve(base, inputPath);
}

export function ensureInsideCwd(
	cwd: string | undefined,
	absolutePath: string,
): void {
	const base = path.resolve(cwd || process.cwd());
	const rel = path.relative(base, absolutePath);
	if (rel.startsWith("..") || path.isAbsolute(rel)) {
		throw new Error(`Path is outside the working directory: ${absolutePath}`);
	}
}

export function readUtf8IfExists(filePath: string): string | null {
	if (!fs.existsSync(filePath)) return null;
	const stat = fs.statSync(filePath);
	if (stat.isDirectory()) throw new Error(`Path is a directory: ${filePath}`);
	return fs.readFileSync(filePath, "utf-8");
}

export function syntheticUnifiedDiff(
	filePath: string,
	before: string | null,
	after: string,
): string {
	const beforeLines = (before ?? "").split("\n");
	const afterLines = after.split("\n");
	const beforeLabel =
		before === null ? "/dev/null" : `a/${path.basename(filePath)}`;
	const afterLabel = `b/${path.basename(filePath)}`;

	let prefix = 0;
	while (
		prefix < beforeLines.length &&
		prefix < afterLines.length &&
		beforeLines[prefix] === afterLines[prefix]
	) {
		prefix++;
	}

	let beforeSuffix = beforeLines.length - 1;
	let afterSuffix = afterLines.length - 1;
	while (
		beforeSuffix >= prefix &&
		afterSuffix >= prefix &&
		beforeLines[beforeSuffix] === afterLines[afterSuffix]
	) {
		beforeSuffix--;
		afterSuffix--;
	}

	const contextBefore = Math.max(0, prefix - 3);
	const contextAfterBefore = Math.min(beforeLines.length - 1, beforeSuffix + 3);
	const contextAfterAfter = Math.min(afterLines.length - 1, afterSuffix + 3);
	const oldStart = contextBefore + 1;
	const newStart = contextBefore + 1;
	const oldCount = Math.max(0, contextAfterBefore - contextBefore + 1);
	const newCount = Math.max(0, contextAfterAfter - contextBefore + 1);

	const out = [
		`--- ${beforeLabel}`,
		`+++ ${afterLabel}`,
		`@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`,
	];

	for (let i = contextBefore; i < prefix; i++)
		out.push(` ${beforeLines[i] ?? ""}`);
	for (let i = prefix; i <= beforeSuffix; i++)
		out.push(`-${beforeLines[i] ?? ""}`);
	for (let i = prefix; i <= afterSuffix; i++)
		out.push(`+${afterLines[i] ?? ""}`);

	const afterContextStart = Math.max(prefix, afterSuffix + 1);
	for (let i = afterContextStart; i <= contextAfterAfter; i++)
		out.push(` ${afterLines[i] ?? ""}`);

	return out.join("\n");
}

export function summarizeDiff(diff: string, maxChars = 12000): string {
	if (!diff.trim()) return "(no diff)";
	if (diff.length <= maxChars) return diff;
	return (
		diff.slice(0, maxChars) +
		`\n... [diff truncated, ${diff.length} chars total]`
	);
}

export async function mutationSummary(
	_cwd: string | undefined,
	filePath: string,
	before: string | null,
	after: string,
): Promise<string> {
	// Always use synthetic diff from before/after to show the exact edit.
	// Git diff against HEAD would show unrelated uncommitted changes, not the
	// actual edit being reported to the LLM.
	return summarizeDiff(syntheticUnifiedDiff(filePath, before, after));
}
