// ── bash tool ──────────────────────────────────────────────────────────────────────
// Execute shell commands with timeout and output limiting.

import { spawn } from "node:child_process";
import type { Tool } from "../types.ts";
import {
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    formatSize,
    truncateTail,
} from "./truncate.ts";

export const bash: Tool = {
    name: "bash",
    description: `Execute a bash command with timeout (default 30s). Output truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB, keeping the end.`,
    parameters: {
        type: "object",
        properties: {
            command: { type: "string", description: "Bash command to execute" },
            timeout: {
                type: "number",
                description: "Timeout in ms (default 30000)",
            },
        },
        required: ["command"],
    },
    execute: async (args, ctx): Promise<string> => {
        const command = String(args.command);
        const timeout = Number(args.timeout) || 30000;

        return new Promise((resolve) => {
            const child = spawn("bash", ["-c", command], {
                cwd: ctx.cwd,
                stdio: ["ignore", "pipe", "pipe"],
            });
            let output = "";
            let settled = false;
            const timer = setTimeout(() => {
                child.kill("SIGKILL");
                finish(`Error: Command timed out after ${timeout}ms`);
            }, timeout);

            const onAbort = () => {
                child.kill("SIGKILL");
                finish("Error: Command aborted");
            };
            ctx.signal?.addEventListener("abort", onAbort, { once: true });

            const append = (chunk: Buffer) => {
                output += chunk.toString("utf8");
                const streamed = truncateTail(output);
                ctx.onUpdate?.(streamed.truncated ? streamed.content : output);
            };

            const finish = (result: string) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                ctx.signal?.removeEventListener("abort", onAbort);
                resolve(result);
            };

            child.stdout.on("data", append);
            child.stderr.on("data", append);
            child.on("error", (error) => {
                finish(`Error: ${error.message || "Command failed"}`);
            });
            child.on("close", (code, signal) => {
                if (settled) return;
                const t = truncateTail(output);
                const finalOutput = t.truncated
                    ? `${truncationNote(t)}\n${t.content}`
                    : output;
                if (code === 0) {
                    finish(finalOutput);
                } else {
                    const status = signal || code || "failed";
                    finish(
                        `Error: Command exited with ${status}${finalOutput ? `\n${finalOutput}` : ""}`,
                    );
                }
            });
        });
    },
};

function truncationNote(t: ReturnType<typeof truncateTail>): string {
    return t.truncatedBy === "lines"
        ? `[Truncated: showing last ${t.outputLines} of ${t.totalLines} lines]`
        : `[Truncated: showing last ${formatSize(t.outputBytes)} of ${formatSize(t.totalBytes)}]`;
}
