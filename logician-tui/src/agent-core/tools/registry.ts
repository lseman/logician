// ── Tool registry ──────────────────────────────────────────────────────────────────
// Manages tool registration and execution. Mirrors Python ToolRegistry.

import type { Tool, ToolCall, ToolContext } from "../types.ts";
import { parseToolInput } from "../parser.ts";

export interface PreparedToolCall {
    call: ToolCall;
    args: Record<string, unknown>;
    error?: string;
}

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

    get(name: string): Tool | undefined {
        return this.tools.get(name);
    }

    list(): Tool[] {
        return Array.from(this.tools.values());
    }

    prepare(call: ToolCall): PreparedToolCall {
        const tool = this.tools.get(call.name);
        if (!tool) {
            return {
                call,
                args: parseToolInput(call.arguments),
                error: `Error: Unknown tool: ${call.name}`,
            };
        }

        try {
            let args = parseToolInput(call.arguments);
            if (tool.prepareArguments) {
                args = tool.prepareArguments(args);
            }
            return {
                call: { ...call, arguments: JSON.stringify(args) },
                args,
            };
        } catch (e: unknown) {
            const error = e as Error;
            return {
                call,
                args: parseToolInput(call.arguments),
                error: `Error preparing ${call.name}: ${error.message}`,
            };
        }
    }

    async execute(
        call: ToolCall,
        context?: ToolContext,
        preparedArgs?: Record<string, unknown>,
    ): Promise<string> {
        const tool = this.tools.get(call.name);
        if (!tool) {
            return `Error: Unknown tool: ${call.name}`;
        }

        try {
            const args = preparedArgs ?? this.prepare(call).args;
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
