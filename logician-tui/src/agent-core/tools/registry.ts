// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.

import type { Tool, ToolCall, ToolContext } from "../types.ts";

export class ToolRegistry {
    private tools = new Map<string, Tool>();
    private ctx: ToolContext;

    constructor(ctx?: ToolContext) {
        this.ctx = ctx || {};
    }

    register(tool: Tool): void {
        this.tools.set(tool.name, tool);
    }

    registerMany(tools: Tool[]): void {
        for (const tool of tools) {
            this.register(tool);
        }
    }

    unregister(name: string): void {
        this.tools.delete(name);
    }

    has(name: string): boolean {
        return this.tools.has(name);
    }

    list(): Tool[] {
        return Array.from(this.tools.values());
    }

    async execute(call: ToolCall, context?: ToolContext): Promise<string> {
        const tool = this.tools.get(call.name);
        if (!tool) {
            return `Error: Unknown tool: ${call.name}`;
        }

        try {
            let args: Record<string, unknown> = {};
            try {
                args = JSON.parse(call.arguments || "{}");
            } catch {
                // Try YAML-style (simple key: value)
                args = {};
                const lines = (call.arguments || "").split("\n");
                for (const line of lines) {
                    const match = line.match(/^\s*(\w+)\s*:\s*(.+)\s*$/);
                    if (match) {
                        args[match[1]] = match[2].replace(/^"|"$/g, "");
                    }
                }
            }

            return await tool.execute(args, { ...this.ctx, ...context });
        } catch (e: unknown) {
            const error = e as Error;
            return `Error executing ${call.name}: ${error.message}`;
        }
    }

    toToolDefinitions(): Record<string, unknown>[] {
        return this.list().map((tool) => ({
            type: "function",
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }));
    }
}
