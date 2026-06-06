// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering messages shown directly above the input bar.
// Renders nothing when the queue is empty, so it only takes vertical space while
// there are pending steering messages.
// Mirrors Pi's updatePendingMessagesDisplay() pattern.

import { visibleWidth, type Component } from "../tui-core.ts";

export interface SteerQueueItem {
    label: "Steering" | "Follow-up";
    messages: string[];
}

export class SteerQueue implements Component {
    private steering: string[] = [];
    private followUp: string[] = [];
    private onInvalidate: (() => void) | null = null;

    setOnInvalidate(cb: () => void): void {
        this.onInvalidate = cb;
    }

    setItems(steering: string[], followUp: string[] = []): void {
        this.steering = steering;
        this.followUp = followUp;
        this.onInvalidate?.();
    }

    invalidate(): void {
        this.onInvalidate?.();
    }

    render(width: number): string[] {
        const hasItems = this.steering.length > 0 || this.followUp.length > 0;
        if (!hasItems) return [];

        const lines: string[] = [];
        lines.push(""); // blank spacer before items

        const items: SteerQueueItem[] = [];
        if (this.steering.length > 0) {
            items.push({ label: "Steering", messages: this.steering });
        }
        if (this.followUp.length > 0) {
            items.push({ label: "Follow-up", messages: this.followUp });
        }

        for (const item of items) {
            for (const msg of item.messages) {
                const trimmed = msg.length > 100 ? msg.slice(0, 100) + "\u2026" : msg;
                const text = "\x1b[2m" + item.label + ": " + trimmed + "\x1b[0m";
                lines.push(pad(clampLine(text, width), width));
            }
        }

        const dequeueHint = "\x1b[2m↳ ctrl+u to edit queued messages\x1b[0m";
        lines.push(pad(clampLine(dequeueHint, width), width));

        return lines;
    }
}

function clampLine(text: string, maxW: number): string {
    let out = "";
    let w = 0;
    for (const ch of text) {
        const cw = visibleWidth(ch);
        if (w + cw > maxW) break;
        out += ch;
        w += cw;
    }
    return out;
}

function pad(line: string, width: number): string {
    const w = visibleWidth(line);
    return w < width ? line + " ".repeat(width - w) : line;
}
