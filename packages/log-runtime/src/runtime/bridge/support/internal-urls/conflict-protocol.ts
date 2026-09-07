// ── conflict:// protocol handler ─────────────────────────────────────────────
// Resolves merge conflict blocks from a file.
// URL forms:
//   conflict://<file>:<index>  — get conflict block <index> from <file>
//   conflict://<file>          — list conflict blocks in <file>

import type { InternalResource, InternalUrl, ProtocolHandler } from "./types";
import {
	getConflictBlock,
	getConflictBlocks,
} from "../../../../capabilities/tools/support/conflict-resolution.js";

export class ConflictProtocolHandler implements ProtocolHandler {
	readonly scheme = "conflict";

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const hostname = url.rawHost || url.hostname;
		const pathname = url.pathname;

		if (!hostname) {
			throw new Error("conflict:// URL requires a file path: conflict://<file>");
		}

		const filePath = hostname;
		const blocks = getConflictBlocks(filePath);

		if (blocks.length === 0) {
			return {
				url: url.href,
				content: `# No conflicts in ${hostname}\n\nThis file has no unresolved merge conflicts.`,
				contentType: "text/markdown",
			};
		}

		// conflict://<file> — list all conflicts
		if (!pathname || pathname === "/" || pathname === "/0") {
			const lines = blocks.map((b, i) =>
				`- Block ${i}: ${b.index} at offset ${b.offset}`,
			);
			return {
				url: url.href,
				content: `# Merge Conflicts in ${hostname}\n\n${lines.join("\n")}`,
				contentType: "text/markdown",
			};
		}

		// conflict://<file>/<index> — get specific block
		const indexStr = pathname.slice(1);
		const index = Number(indexStr);
		if (Number.isNaN(index)) {
			throw new Error(`conflict://<file>/<index> requires a numeric index, got: ${indexStr}`);
		}

		const block = getConflictBlock(filePath, index);
		if (!block) {
			throw new Error(`No conflict block at index ${index} in ${hostname}`);
		}

		const content = [
			`# Conflict Block ${index} in ${hostname}`,
			``,
			`**Conflict:** ${block.oursLabel} ... ${block.theirsLabel}`,
			`**Offset:** ${block.offset}`,
			``,
			`--- ours ---`,
			``,
			block.ours,
			``,
			`--- theirs ---`,
			``,
			block.theirs,
			``,
			`--- base ---`,
			``,
			`No base section available.`,
		].join("\n");

		return {
			url: url.href,
			content,
			contentType: "text/plain",
			sourcePath: filePath,
		};
	}
}
