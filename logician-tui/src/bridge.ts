// ── Bridge communication layer ────────────────────────────────────────────────
// Spawns logician_bridge.py as a child process and speaks its JSON-RPC protocol.
//
// Protocol (matches logician_bridge.py main()):
//   TUI → bridge (one JSON object per line on the child's stdin):
//     {"id": "...", "method": "init"|"chat"|"slash"|"state", "params": {...}}
//     {"method": "cancel"}                       (fire-and-forget notification)
//   bridge → TUI (one JSON object per line on the child's stdout):
//     {"event": "...", ...}                      streamed lifecycle events
//     {"id": "...", "ok": true, "result": {...}} RPC response
//     {"id": "...", "ok": false, "error": "..."} RPC error
//
// Events use a "type" or "event" key; responses carry an "id". We route by
// presence of "id" first, then fall back to treating the line as an event.

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { ParsedBridgeEvent, BridgeEventType } from "./events.ts";

export type EventCallback = (event: ParsedBridgeEvent) => void;
export type ErrorCallback = (err: Error) => void;

interface PendingCall {
    resolve: (value: unknown) => void;
    reject: (err: Error) => void;
}

export interface BridgeOptions {
    program?: string; // default: "python3"
    scriptPath?: string; // default: resolved logician_bridge.py
    configPath?: string;
    cwd?: string;
}

export class Bridge {
    private child: ChildProcessWithoutNullStreams | null = null;
    private stdoutBuffer = "";
    private callbacks: EventCallback[] = [];
    private errorCb: ErrorCallback | null = null;
    private running = false;
    private pending = new Map<string, PendingCall>();
    private nextId = 1;
    private opts: BridgeOptions;

    constructor(opts: BridgeOptions = {}) {
        this.opts = opts;
    }

    // ── Event registration ─────────────────────────────────────────────────

    on(callback: EventCallback): () => void {
        this.callbacks.push(callback);
        return () => {
            this.callbacks = this.callbacks.filter((cb) => cb !== callback);
        };
    }

    onError(callback: ErrorCallback): void {
        this.errorCb = callback;
    }

    private emit(event: ParsedBridgeEvent): void {
        for (const cb of this.callbacks) {
            try {
                cb(event);
            } catch {
                // Don't let a bad handler kill the bridge
            }
        }
    }

    // ── Line routing ───────────────────────────────────────────────────────

    private handleLine(line: string): void {
        line = line.trim();
        if (!line || line === "[DONE]") return;

        let data: Record<string, unknown>;
        try {
            data = JSON.parse(line);
        } catch {
            // Non-JSON line on stdout — surface as an error but keep going.
            this.errorCb?.(
                new Error(`bridge: non-JSON output: ${line.slice(0, 120)}`),
            );
            return;
        }

        // RPC response carries an "id" and an "ok" flag.
        if ("id" in data && "ok" in data) {
            const id = String(data.id);
            const pc = this.pending.get(id);
            if (pc) {
                this.pending.delete(id);
                if (data.ok) pc.resolve(data.result);
                else pc.reject(new Error(String(data.error ?? "bridge error")));
            }
            return;
        }

        // Otherwise it's a streamed event.
        const event = this.normalizeEvent(data);
        if (event) this.emit(event);
    }

    private normalizeEvent(
        raw: Record<string, unknown>,
    ): ParsedBridgeEvent | null {
        // Events use "event" (logician_bridge._emit) or "type" as the discriminator.
        const type = String(raw.event ?? raw.type ?? "") as BridgeEventType;
        if (!type) return null;

        switch (type) {
            case "token":
                return { type: "token", token: String(raw.token || "") };
            case "thinking_token":
                return {
                    type: "thinking_token",
                    token: String(raw.token || ""),
                };
            case "turn_start":
                return {
                    type: "turn_start",
                    turn_id: String(raw.turn_id || ""),
                };
            case "turn_end":
                return {
                    type: "turn_end",
                    turn_id: String(raw.turn_id || ""),
                    message: String(raw.response ?? raw.message ?? ""),
                };
            case "tool_start":
            case "tool_execution_start":
                return {
                    type:
                        type === "tool_start"
                            ? "tool_start"
                            : "tool_execution_start",
                    tool: String(raw.tool ?? raw.name ?? ""),
                    tool_name: String(raw.name ?? raw.tool_name ?? ""),
                    tool_args: (raw.args ?? raw.tool_args) as
                        | Record<string, unknown>
                        | undefined,
                    turn_id: raw.turn_id ? String(raw.turn_id) : undefined,
                    tool_call_id: raw.tool_call_id
                        ? String(raw.tool_call_id)
                        : undefined,
                };
            case "tool_end":
            case "tool_execution_end":
                return {
                    type:
                        type === "tool_end" ? "tool_end" : "tool_execution_end",
                    tool: String(raw.tool ?? raw.name ?? ""),
                    tool_name: String(raw.name ?? raw.tool_name ?? ""),
                    result:
                        raw.result !== undefined
                            ? String(raw.result)
                            : undefined,
                    is_error:
                        (raw.status
                            ? String(raw.status).toLowerCase() === "error"
                            : undefined) ??
                        (raw.is_error as boolean | undefined),
                    turn_id: raw.turn_id ? String(raw.turn_id) : undefined,
                    tool_call_id: raw.tool_call_id
                        ? String(raw.tool_call_id)
                        : undefined,
                };
            case "phase":
                return {
                    type: "phase",
                    state: String(raw.state || ""),
                    note: raw.note ? String(raw.note) : undefined,
                };
            case "decision":
                return {
                    type: "decision",
                    stage: raw.stage ? String(raw.stage) : undefined,
                };
            case "classified":
                return {
                    type: "classified",
                    turn_id: raw.turn_id ? String(raw.turn_id) : undefined,
                    intent: raw.intent ? String(raw.intent) : undefined,
                    domain_groups: raw.domain_groups as string[] | undefined,
                };
            case "image":
                return {
                    type: "image",
                    tool: raw.tool ? String(raw.tool) : undefined,
                    path: raw.path ? String(raw.path) : undefined,
                    source: raw.source ? String(raw.source) : undefined,
                };
            case "guardrail_nudge":
                return {
                    type: "guardrail_nudge",
                    turn_id: raw.turn_id ? String(raw.turn_id) : undefined,
                    guard_name: raw.guard_name
                        ? String(raw.guard_name)
                        : undefined,
                    nudge: raw.nudge ? String(raw.nudge) : undefined,
                };
            case "repair_nudge":
                return {
                    type: "repair_nudge",
                    turn_id: raw.turn_id ? String(raw.turn_id) : undefined,
                    repair_stage: raw.repair_stage
                        ? String(raw.repair_stage)
                        : undefined,
                    attempt: raw.attempt as number | undefined,
                    tool_name: raw.tool_name
                        ? String(raw.tool_name)
                        : undefined,
                    error_type: raw.error_type
                        ? String(raw.error_type)
                        : undefined,
                    message: raw.message ? String(raw.message) : undefined,
                };
            default:
                return null;
        }
    }

    // ── RPC ────────────────────────────────────────────────────────────────

    private writeRaw(obj: Record<string, unknown>): void {
        if (!this.child?.stdin.writable) return;
        try {
            this.child.stdin.write(JSON.stringify(obj) + "\n");
        } catch (err) {
            this.errorCb?.(err instanceof Error ? err : new Error(String(err)));
        }
    }

    call(
        method: string,
        params: Record<string, unknown> = {},
    ): Promise<unknown> {
        const id = String(this.nextId++);
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.writeRaw({ id, method, params });
        });
    }

    /** Fire-and-forget cancel notification. */
    cancel(): void {
        this.writeRaw({ method: "cancel" });
    }

    // ── High-level commands ──────────────────────────────────────────────

    sendMessage(message: string): void {
        // chat streams events then resolves with the final turn payload.
        this.call("chat", { message }).catch((err) => this.errorCb?.(err));
    }

    sendSlash(raw: string): void {
        this.call("slash", { raw }).catch((err) => this.errorCb?.(err));
    }

    setThinkingLevel(level: string): void {
        this.sendSlash(`/thinking ${level}`);
    }

    reset(): void {
        this.sendSlash("/reset");
    }

    // ── Start/Stop ───────────────────────────────────────────────────────

    start(): void {
        if (this.running) return;
        this.running = true;

        const program =
            this.opts.program ?? process.env.LOGICIAN_PYTHON ?? "python3";
        const script =
            this.opts.scriptPath ??
            process.env.LOGICIAN_BRIDGE ??
            new URL("../../logician_bridge.py", import.meta.url).pathname;

        const args = [script];
        this.child = spawn(program, args, {
            cwd: this.opts.cwd ?? process.cwd(),
            stdio: ["pipe", "pipe", "pipe"],
        });

        this.child.stdout.setEncoding("utf-8");
        this.child.stdout.on("data", (chunk: string) => {
            this.stdoutBuffer += chunk;
            let nl: number;
            while ((nl = this.stdoutBuffer.indexOf("\n")) !== -1) {
                const line = this.stdoutBuffer.slice(0, nl);
                this.stdoutBuffer = this.stdoutBuffer.slice(nl + 1);
                this.handleLine(line);
            }
        });

        this.child.stderr.setEncoding("utf-8");
        this.child.stderr.on("data", (chunk: string) => {
            // Bridge logs go to stderr; only surface real errors, not noise.
            const text = String(chunk).trim();
            if (text) this.errorCb?.(new Error(text));
        });

        this.child.on("exit", (code) => {
            this.running = false;
            for (const [, pc] of this.pending) {
                pc.reject(new Error(`bridge exited (code ${code})`));
            }
            this.pending.clear();
        });

        this.child.on("error", (err) => {
            this.errorCb?.(err instanceof Error ? err : new Error(String(err)));
        });

        // Initialize the agent. fast=true keeps startup cheap.
        this.call("init", {
            fast: true,
            ...(this.opts.configPath
                ? { config_path: this.opts.configPath }
                : {}),
        }).catch((err) => this.errorCb?.(err));
    }

    stop(): void {
        if (!this.running) return;
        this.running = false;
        try {
            this.child?.stdin.end();
            this.child?.kill("SIGTERM");
        } catch {
            // ignore
        }
        this.child = null;
    }

    isActive(): boolean {
        return this.running;
    }
}
