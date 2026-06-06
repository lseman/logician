// ── Thinking panel component ────────────────────────────────────────────────────
// Collapsible panel showing thinking blocks

import type { Component, Focusable } from "../tui-core.ts";

interface ThinkingBlock {
    content: string;
    isComplete: boolean;
}

const THINKING_HEADER = "\x1b[38;5;220m\x1b[1m⚡ THINKING\x1b[0m";
const THINKING_PREFIX = "\x1b[38;5;220m  \x1b[0m";

export class ThinkingPanel implements Component, Focusable {
    public focused = false;

    private blocks: ThinkingBlock[] = [];
    private isVisible = false;
    private cachedLines: string[] | null = null;
    private cachedWidth = -1;

    setBlocks(blocks: ThinkingBlock[]): void {
        this.blocks = blocks;
        this.isVisible = blocks.length > 0;
        this.invalidate();
    }

    addBlock(content: string): void {
        const last = this.blocks[this.blocks.length - 1];
        if (last && !last.isComplete) {
            last.content += content;
        } else {
            this.blocks.push({ content, isComplete: false });
        }
        this.isVisible = true;
        this.invalidate();
    }

    completeLastBlock(): void {
        const last = this.blocks[this.blocks.length - 1];
        if (last) {
            last.isComplete = true;
        }
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
        lines.push(THINKING_HEADER);

        // Each block
        for (const block of this.blocks) {
            const truncated =
                block.content.length > 2000
                    ? block.content.slice(0, 2000) + "..."
                    : block.content;
            const rawLines = truncated.split("\n");

            for (const line of rawLines) {
                // Word wrap
                if (line.length > contentWidth) {
                    const words = line.split(/\s+/);
                    let current = THINKING_PREFIX;
                    for (const word of words) {
                        if (current.length <= THINKING_PREFIX.length) {
                            current = THINKING_PREFIX + word;
                        } else if (
                            current.length + 1 + word.length <=
                            contentWidth
                        ) {
                            current += " " + word;
                        } else {
                            lines.push(current);
                            current = THINKING_PREFIX + word;
                        }
                    }
                    if (current !== THINKING_PREFIX) {
                        lines.push(current);
                    }
                } else {
                    lines.push(THINKING_PREFIX + line);
                }
            }

            lines.push(""); // Spacer between blocks
        }

        this.cachedLines = lines;
        return lines;
    }
}
