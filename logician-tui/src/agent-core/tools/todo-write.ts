// ── todo_write tool ───────────────────────────────────────────────────────────────
// Explicit task tracking. The model overwrites the full todo list each call; the
// current list is held in a module store the TUI can subscribe to and pin above
// the input bar.

import type { Tool } from "../types.ts";

export type TodoStatus = "pending" | "in_progress" | "completed";

export interface TodoItem {
    content: string;
    status: TodoStatus;
}

const VALID_STATUS: ReadonlySet<string> = new Set([
    "pending",
    "in_progress",
    "completed",
]);

let todos: TodoItem[] = [];
const listeners = new Set<(todos: TodoItem[]) => void>();

/** Subscribe to todo list changes. Returns an unsubscribe function. */
export function onTodosChanged(cb: (todos: TodoItem[]) => void): () => void {
    listeners.add(cb);
    return () => listeners.delete(cb);
}

export function getTodos(): TodoItem[] {
    return todos;
}

function setTodos(next: TodoItem[]): void {
    todos = next;
    for (const cb of listeners) cb(todos);
}

function normalizeTodos(raw: unknown): TodoItem[] | string {
    let value = raw;
    if (typeof value === "string") {
        try {
            value = JSON.parse(value);
        } catch {
            return "Error: todos must be an array of { content, status }.";
        }
    }
    if (!Array.isArray(value)) {
        return "Error: todos must be an array of { content, status }.";
    }
    const out: TodoItem[] = [];
    for (const [idx, item] of value.entries()) {
        if (!item || typeof item !== "object") {
            return `Error: todo ${idx + 1} is not an object.`;
        }
        const obj = item as Record<string, unknown>;
        const content = String(obj.content ?? "").trim();
        if (!content) return `Error: todo ${idx + 1} has empty content.`;
        const status = String(obj.status ?? "pending");
        if (!VALID_STATUS.has(status)) {
            return `Error: todo ${idx + 1} has invalid status '${status}' (pending|in_progress|completed).`;
        }
        out.push({ content, status: status as TodoStatus });
    }
    return out;
}

const STATUS_MARK: Record<TodoStatus, string> = {
    completed: "[x]",
    in_progress: "[~]",
    pending: "[ ]",
};

export const todo_write: Tool = {
    name: "todo_write",
    description:
        "Replace the current todo list to track multi-step work. Pass the full list every " +
        "call. Mark exactly one item in_progress while working on it, completed when done.",
    parameters: {
        type: "object",
        properties: {
            todos: {
                type: "array",
                description: "Full todo list. Each item: { content, status }.",
                items: {
                    type: "object",
                    properties: {
                        content: {
                            type: "string",
                            description: "Task description",
                        },
                        status: {
                            type: "string",
                            enum: ["pending", "in_progress", "completed"],
                            description: "Task status",
                        },
                    },
                    required: ["content", "status"],
                },
            },
        },
        required: ["todos"],
    },
    execute: async (args): Promise<string> => {
        const result = normalizeTodos(args.todos);
        if (typeof result === "string") return result;
        setTodos(result);
        if (result.length === 0) return "Todo list cleared.";
        const lines = result.map(
            (t) => `${STATUS_MARK[t.status]} ${t.content}`,
        );
        const done = result.filter((t) => t.status === "completed").length;
        return `Updated todos (${done}/${result.length} done):\n${lines.join("\n")}`;
    },
};
