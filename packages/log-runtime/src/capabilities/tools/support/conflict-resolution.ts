// ── Conflict resolution for git merge conflicts ───────────────────────────────
//
// When a file contains git merge conflict markers, this module:
// 1. Parses the conflict blocks from the file
// 2. Provides @ours/@theirs/@base resolution helpers
// 3. Exposes conflict://N internal URL support

import * as fs from "node:fs";

// ── Types ──────────────────────────────────────────────────────────────────────

/** A single merge conflict block from a file. */
export interface ConflictBlock {
	/** Block index (0-based). */
	index: number;
	/** The conflict markers and full content. */
	content: string;
	/** The ours section content. */
	ours: string;
	/** The theirs section content. */
	theirs: string;
	/** Label from the <<<<<<< marker. */
	oursLabel: string;
	/** Label from the >>>>>>> marker. */
	theirsLabel: string;
	/** Byte offset in the file where this conflict starts. */
	offset: number;
	/** File path. */
	file: string;
}

/**
 * Parse conflict markers from file content.
 * Standard git conflict format:
 *   <<<<<<< ours
 *   content
 *   =======
 *   content
 *   >>>>>>> theirs
 */
export function parseConflictBlocks(
	content: string,
	filePath: string,
): ConflictBlock[] {
	const blocks: ConflictBlock[] = [];
	const lines = content.split("\n");
	let i = 0;

	while (i < lines.length) {
		const line = lines[i];

		if (line.startsWith("<<<<<<< ")) {
			const oursLabel = line.slice(10).trim();
			let oursEnd = i + 1;

			while (oursEnd < lines.length && !lines[oursEnd].startsWith("=======")) {
				oursEnd++;
			}

			if (oursEnd >= lines.length) break;

			let theirsEnd = oursEnd + 1;
			while (theirsEnd < lines.length && !lines[theirsEnd].startsWith(">>>>>>> ")) {
				theirsEnd++;
			}

			if (theirsEnd >= lines.length) break;

			const theirsLabel = lines[theirsEnd].slice(10).trim();

			const ours = lines.slice(i + 1, oursEnd).join("\n");
			const theirs = lines.slice(oursEnd + 1, theirsEnd).join("\n");
			const fullBlock = lines.slice(i, theirsEnd + 1).join("\n");

			const offset = content
				.split("\n")
				.slice(0, i)
				.reduce((sum: number, l: string) => sum + l.length + 1, 0);

			blocks.push({
				index: blocks.length,
				content: fullBlock,
				ours,
				theirs,
				oursLabel,
				theirsLabel,
				offset,
				file: filePath,
			});

			i = theirsEnd + 1;
		} else {
			i++;
		}
	}

	return blocks;
}

/**
 * Resolve a conflict block by choosing ours, theirs, base, or both.
 */
export function resolveConflictBlock(
	block: ConflictBlock,
	strategy: "ours" | "theirs" | "ours+theirs" | "base",
	baseContent?: string,
): string {
	switch (strategy) {
		case "ours":
			return block.ours;
		case "theirs":
			return block.theirs;
		case "ours+theirs":
			return block.ours + "\n" + block.theirs;
		case "base":
			return baseContent ?? block.ours;
		default:
			return block.ours;
	}
}

/**
 * Apply conflict resolution to a file and return the resolved content.
 */
export function resolveConflictsInFile(
	filePath: string,
	strategy: "ours" | "theirs" | "ours+theirs" | "base",
	baseContent?: string,
): string {
	const content = fs.readFileSync(filePath, "utf-8");
	const blocks = parseConflictBlocks(content, filePath);

	if (blocks.length === 0) {
		return content;
	}

	let result = content;

	for (const block of blocks) {
		const resolved = resolveConflictBlock(block, strategy, baseContent);
		result =
			result.slice(0, block.offset) +
			resolved +
			result.slice(block.offset + block.content.length);
	}

	return result;
}

/**
 * Check if a file has unresolved merge conflicts.
 */
export function hasConflicts(filePath: string): boolean {
	const content = fs.readFileSync(filePath, "utf-8");
	return (
		content.includes("<<<<<<<") &&
		content.includes("=======") &&
		content.includes(">>>>>>>")
	);
}

/**
 * Get all conflict blocks for a file.
 */
export function getConflictBlocks(filePath: string): ConflictBlock[] {
	const content = fs.readFileSync(filePath, "utf-8");
	return parseConflictBlocks(content, filePath);
}

/**
 * Get a specific conflict block by index.
 */
export function getConflictBlock(
	filePath: string,
	index: number,
): ConflictBlock | null {
	const blocks = getConflictBlocks(filePath);
	return blocks[index] ?? null;
}

/**
 * Format conflict blocks for display (like the :conflicts selector).
 */
export function formatConflictBlocks(blocks: ConflictBlock[]): string {
	if (blocks.length === 0) {
		return "(No merge conflicts)";
	}

	const lines: string[] = [];
	const fileLabel = blocks[0].file;
	lines.push(
		`# Merge Conflicts in \`${fileLabel}\` (${blocks.length} block${blocks.length > 1 ? "s" : ""})`,
	);
	lines.push("");

	for (const block of blocks) {
		lines.push(`## Conflict #${block.index}`);
		lines.push("");
		lines.push(`<<<<<<< ${block.oursLabel}`);
		lines.push(block.ours);
		lines.push("=======");
		lines.push(block.theirs);
		lines.push(`>>>>>>> ${block.theirsLabel}`);
		lines.push("");
	}

	return lines.join("\n");
}
