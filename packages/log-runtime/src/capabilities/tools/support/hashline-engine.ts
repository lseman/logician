// Hashline line edits are planned before writing. Preview remains the default;
// callers must explicitly request application and may constrain the target path.
import * as fs from "node:fs";
import * as path from "node:path";
import type { EditStore } from "./edit-store.js";
import { createEditStore } from "./edit-store.js";
import {
	type HashlineEdit,
	type HashlineEditResult,
	hashlineHash,
	parseHashlineEdit,
	splitAddressableFileLines,
} from "./hashline.js";
import { generateEditDiffs } from "./utils/diff-utils.js";
import { atomicWriteFile } from "./utils/atomic-write.js";
import { detectLineEnding, normalizeToLF, restoreLineEndings, stripBom } from "./utils/helpers.ts";

interface FileEdit {
	path: string;
	tag: string;
	edits: HashlineEdit[];
}

function parseDocument(input: string, cwd: string, targetPath?: string): FileEdit[] {
	const files: FileEdit[] = [];
	for (const line of input.split(/\r?\n/)) {
		if (!line.trim()) continue;
		const header = /^\[(.+)#([a-fA-F0-9]{4})\]$/.exec(line.trim());
		if (header) {
			const resolved = path.resolve(cwd, header[1]);
			if (targetPath && resolved !== targetPath) {
				throw new Error("Hashline header must target the edit_file path.");
			}
			if (files.some(file => file.path === resolved)) {
				throw new Error("Use one header per file, followed by its operations.");
			}
			files.push({ path: resolved, tag: header[2].toLowerCase(), edits: [] });
			continue;
		}
		const file = files.at(-1);
		const edit = parseHashlineEdit(line);
		if (!file || !edit) throw new Error(`Invalid hashline input: ${line}`);
		if ((edit.operation !== "PUT" && edit.operation !== "CUT") || edit.register || edit.block || !edit.range || edit.range.includes("*")) {
			throw new Error("Unsupported hashline operation. Use PUT/CUT line edits; use another tool for moves, removal, or registers.");
		}
		file.edits.push(edit);
	}
	if (!files.length || files.some(file => !file.edits.length)) {
		throw new Error("Provide a [path#hash] header followed by at least one line edit.");
	}
	return files;
}

function applyLineEdits(original: string, edits: HashlineEdit[]): { content: string; linesChanged: number } {
	const { bom, text } = stripBom(original);
	const ending = detectLineEnding(text);
	const normalized = normalizeToLF(text);
	const lines = normalized === "" ? [] : splitAddressableFileLines(normalized);
	let linesChanged = 0;
	// Each operation addresses the result of the preceding operation.
	for (const edit of edits) {
		const range = edit.range ?? "";
		const insertion = /^([<>])(\d+)$/.exec(range);
		const replacement = /^(\d+)(?:-(\d+))?$/.exec(range);
		let start: number;
		let count: number;
		if (insertion && edit.operation === "PUT") {
			start = Number(insertion[2]) - (insertion[1] === "<" ? 1 : 0);
			count = 0;
		} else if (replacement) {
			start = Number(replacement[1]) - 1;
			count = Number(replacement[2] ?? replacement[1]) - start;
		} else {
			throw new Error(`Unsupported line range: ${range}`);
		}
		if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count < 0 || (replacement && count === 0) || start + count > lines.length) {
			throw new Error(`Line range out of bounds: ${range}`);
		}
		const body = edit.operation === "CUT" ? [] : edit.body ?? [];
		const before = lines.slice(start, start + count);
		if (before.length === body.length && before.every((line, i) => line === body[i])) continue;
		lines.splice(start, count, ...body);
		linesChanged += Math.max(count, body.length);
	}
	const content = bom + restoreLineEndings(lines.join("\n") + (lines.length > 0 && normalized.endsWith("\n") ? "\n" : ""), ending);
	return { content, linesChanged };
}

export async function executeHashlineEdit(
	input: string,
	store: EditStore,
	cwd: string,
	dryRun = true,
	targetPath?: string,
): Promise<HashlineEditResult> {
	let filesAffected = 0;
	let linesChanged = 0;
	let diff = "";
	try {
		const files = parseDocument(input, cwd, targetPath);
		const plans = files.map(file => {
			const original = fs.readFileSync(file.path, "utf8");
			const stale = store.checkStale(file.path);
			if (stale) throw new Error(stale);
			if (hashlineHash(original) !== file.tag) {
				throw new Error(`Stale hashline anchor for ${file.path}. Read it again before editing.`);
			}
			const result = applyLineEdits(original, file.edits);
			return { ...file, original, ...result };
		}).filter(plan => plan.original !== plan.content);
		if (!plans.length) return { applied: false, filesAffected: 0, linesChanged: 0, diff: "" };
		if (dryRun) {
			return {
				applied: false,
				filesAffected: plans.length,
				linesChanged: plans.reduce((sum, plan) => sum + plan.linesChanged, 0),
				diff: plans.map(plan => generateEditDiffs(plan.path, plan.original, plan.content).diff).join("\n"),
			};
		}
		for (const plan of plans) {
			if (fs.readFileSync(plan.path, "utf8") !== plan.original) {
				throw new Error(`${plan.path} changed while preparing the edit. Read it again.`);
			}
			await atomicWriteFile(plan.path, plan.content, { expectedContent: plan.original });
			store.clearSnapshot(plan.path);
			filesAffected++;
			linesChanged += plan.linesChanged;
			diff += generateEditDiffs(plan.path, plan.original, plan.content).diff + "\n";
		}
		return { applied: true, filesAffected, linesChanged, diff: diff.trimEnd() };
	} catch (error) {
		return {
			applied: false, filesAffected, linesChanged, diff,
			error: error instanceof Error ? error.message : String(error),
		};
	}
}

export function createDefaultEditStore(): EditStore {
	return createEditStore();
}
