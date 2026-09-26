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
			// "<<<<<<< " is exactly 8 characters; slice(10) cut 2 characters
			// into the label itself (e.g. "ours" -> "rs").
			const oursLabel = line.slice(8).trim();
			let oursEnd = i + 1;

			while (oursEnd < lines.length && !lines[oursEnd].startsWith("=======")) {
				oursEnd++;
			}

			if (oursEnd >= lines.length) break;

			let theirsEnd = oursEnd + 1;
			while (
				theirsEnd < lines.length &&
				!lines[theirsEnd].startsWith(">>>>>>> ")
			) {
				theirsEnd++;
			}

			if (theirsEnd >= lines.length) break;

			// Same off-by-two as oursLabel above: ">>>>>>> " is 8 characters.
			const theirsLabel = lines[theirsEnd].slice(8).trim();

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
function resolveConflictBlock(
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
			return `${block.ours}\n${block.theirs}`;
		case "base":
			return baseContent ?? block.ours;
		default:
			return block.ours;
	}
}

/**
 * Apply conflict resolution to a file and return the resolved content. When
 * `index` is given, only that one block is resolved and the rest of the
 * file's conflict markers are left untouched; when omitted, every block is
 * resolved with the same strategy.
 */
export function resolveConflictsInFile(
	filePath: string,
	strategy: "ours" | "theirs" | "ours+theirs" | "base",
	baseContent?: string,
	index?: number,
): string {
	const content = fs.readFileSync(filePath, "utf-8");
	const blocks = parseConflictBlocks(content, filePath);

	if (blocks.length === 0) {
		return content;
	}

	if (index !== undefined && !blocks.some(b => b.index === index)) {
		throw new Error(
			`No conflict block at index ${index} in ${filePath} (${blocks.length} block${blocks.length === 1 ? "" : "s"} found)`,
		);
	}

	let result = content;
	let shift = 0;

	for (const block of blocks) {
		if (index !== undefined && block.index !== index) continue;
		const resolved = resolveConflictBlock(block, strategy, baseContent);
		const start = block.offset + shift;
		result =
			result.slice(0, start) +
			resolved +
			result.slice(start + block.content.length);
		shift += resolved.length - block.content.length;
	}

	return result;
}

/**
 * Get all conflict blocks for a file.
 */
export function getConflictBlocks(filePath: string): ConflictBlock[] {
	const content = fs.readFileSync(filePath, "utf-8");
	return parseConflictBlocks(content, filePath);
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
