// ── Todo bar component ─────────────────────────────────────────────────────────
// Pinned task list shown directly above the input bar. Renders nothing when the
// list is empty, so it only takes vertical space while there are active todos.

import type { Component } from "../tui-core.ts";
import { visibleWidth } from "../tui-core.ts";

export interface TodoBarItem {
    content: string;
    status: "pending" | "in_progress" | "completed";
}

const MARK: Record<TodoBarItem["status"], string> = {
    // ✔ done (green), ▸ in progress (yellow), ○ pending (dim)
    completed: "\x1b[32m✔\x1b[0m",
    in_progress: "\x1b[33m▸\x1b[0m",
    pending: "\x1b[38;5;240m○\x1b[0m",
};

const STYLE: Record<TodoBarItem["status"], (s: string) => string> = {
    completed: (s) => `\x1b[38;5;240m\x1b[9m${s}\x1b[0m`, // dim + strikethrough
    in_progress: (s) => `\x1b[1m${s}\x1b[0m`, // bold
    pending: (s) => `\x1b[38;5;250m${s}\x1b[0m`,
};

const MAX_ROWS = 6;

export class TodoBar implements Component {
    private todos: TodoBarItem[] = [];
    private onInvalidate: (() => void) | null = null;

    setOnInvalidate(cb: () => void): void {
        this.onInvalidate = cb;
    }

    setTodos(todos: TodoBarItem[]): void {
        this.todos = todos;
        this.onInvalidate?.();
    }

    invalidate(): void {
        this.onInvalidate?.();
    }

    render(width: number): string[] {
        if (this.todos.length === 0) return [];

        const done = this.todos.filter((t) => t.status === "completed").length;
        const header = `\x1b[38;5;245mTodos ${done}/${this.todos.length}\x1b[0m`;
        const rows: string[] = [pad(header, width)];

        const shown = this.todos.slice(0, MAX_ROWS);
        for (const t of shown) {
            const line = ` ${MARK[t.status]} ${STYLE[t.status](t.content)}`;
            rows.push(pad(clamp(line, width), width));
        }
        const hidden = this.todos.length - shown.length;
        if (hidden > 0) {
            rows.push(pad(`\x1b[38;5;240m … ${hidden} more\x1b[0m`, width));
        }
        return rows;
    }
}

function clamp(line: string, width: number): string {
    if (visibleWidth(line) <= width) return line;
    // Trim by visible width, leaving room for an ellipsis. Keep it simple: slice
    // raw and let the renderer's own clamp guard the physical last column.
    let out = "";
    let w = 0;
    for (const ch of line) {
        const cw = visibleWidth(ch);
        if (w + cw > width - 1) break;
        out += ch;
        w += cw;
    }
    return `${out}…\x1b[0m`;
}

function pad(line: string, width: number): string {
    const w = visibleWidth(line);
    return w < width ? line + " ".repeat(width - w) : line;
}
