// ── Thinking panel component ────────────────────────────────────────────────────
// Collapsible panel showing thinking blocks

import type { Component, Focusable } from "../tui-core.ts";
import { theme } from "../theme.ts";
import { highlightAuto } from "../agent-core/tools/shared/syntax-highlighter.ts";

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";

interface ThinkingBlock {
	content: string;
	isComplete: boolean;
}

const getThinkingHeader = (): string =>
	theme.fg("thinkingText", "") + BOLD + "⚡ THINKING" + RESET;
const getThinkingPrefix = (): string =>
	theme.fg("thinkingText", "") + "  " + RESET;
const getBlockBg = (): string => theme.bg("mdCodeBlockBg", "");
const CODE_BLOCK_RESET = "\x1b[0m";

export class ThinkingPanel implements Component, Focusable {
	public focused = false;

	private blocks: ThinkingBlock[] = [];
	private isVisible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	// Regex to strip <think>…</think> tags and their content.
	private static readonly THINK_TAG_RE = /<think>[\s\S]*?<\/think>/gi;

	setBlocks(blocks: ThinkingBlock[]): void {
		this.blocks = blocks.map((b) => ({
			...b,
			content: b.content.replace(ThinkingPanel.THINK_TAG_RE, ""),
		}));
		this.isVisible = this.blocks.some((b) => b.content.trim().length > 0);
		this.invalidate();
	}

	addBlock(content: string): void {
		// Strip <think>…</think> tags from incoming content.
		content = content.replace(ThinkingPanel.THINK_TAG_RE, "");
		const last = this.blocks[this.blocks.length - 1];
		if (last && !last.isComplete) {
			last.content += content;
		} else {
			this.blocks.push({ content, isComplete: false });
		}
		this.isVisible = this.blocks.some((b) => b.content.trim().length > 0);
		this.invalidate();
	}

	completeLastBlock(): void {
		const last = this.blocks[this.blocks.length - 1];
		if (last) {
			last.isComplete = true;
		}
		// Update visibility: hide if all blocks are empty after stripping.
		this.isVisible = this.blocks.some((b) => b.content.trim().length > 0);
		this.invalidate();
	}

	clear(): void {
		this.blocks = [];
		this.isVisible = false;
		this.invalidate();
	}

	isVisiblePanel(): boolean {
		return this.isVisible;
	}

	invalidate(): void {
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;

		if (!this.isVisible || this.blocks.length === 0) {
			this.cachedLines = [];
			return [];
		}

		const contentWidth = Math.max(1, width - 2);
		const lines: string[] = [];

		// Header
		lines.push(getThinkingHeader());

		// Each block
		for (const block of this.blocks) {
			const truncated =
				block.content.length > 2000
					? `${block.content.slice(0, 2000)}...`
					: block.content;
			const rawLines = truncated.split("\n");

			let inCodeBlock = false;
			let codeContent = "";
			let codeBlockLang: string | null = null;

			for (const rawLine of rawLines) {
				if (rawLine.startsWith("```")) {
					if (inCodeBlock) {
						// Flush code block with syntax highlighting
						const lang = codeBlockLang || null;
						if (lang) {
							const highlighted = highlightAuto(codeContent);
							const langLabel = highlighted.language
								? ` ${highlighted.language} · ${codeContent.split("\n").length} lines`
								: "";
							lines.push(
								`${getBlockBg()}${DIM}  \`${rawLine}\`${langLabel}${CODE_BLOCK_RESET}`,
							);
							for (const cl of highlighted.value.split("\n")) {
								lines.push(`${getBlockBg()}${DIM}  ${cl}${CODE_BLOCK_RESET}`);
							}
						} else {
							const codeLines = codeContent.split("\n");
							for (const cl of codeLines) {
								lines.push(`${getBlockBg()}${DIM}  ${cl}${CODE_BLOCK_RESET}`);
							}
						}
						inCodeBlock = false;
						codeContent = "";
						codeBlockLang = null;
					} else {
						inCodeBlock = true;
						codeBlockLang = rawLine.slice(3).trim() || null;
					}
					continue;
				}

				if (inCodeBlock) {
					codeContent += rawLine + "\n";
				} else {
					// Word wrap
					if (rawLine.length > contentWidth) {
						const words = rawLine.split(/\s+/);
						let current = getThinkingPrefix();
						for (const word of words) {
							if (current.length <= getThinkingPrefix().length) {
								current = getThinkingPrefix() + word;
							} else if (current.length + 1 + word.length <= contentWidth) {
								current += ` ${word}`;
							} else {
								lines.push(current);
								current = getThinkingPrefix() + word;
							}
						}
						if (current !== getThinkingPrefix()) {
							lines.push(current);
						}
					} else {
						lines.push(getThinkingPrefix() + rawLine);
					}
				}
			}

			lines.push(""); // Spacer between blocks
		}

		this.cachedLines = lines;
		return lines;
	}
}
