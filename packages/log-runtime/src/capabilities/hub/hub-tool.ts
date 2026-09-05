// ── hub tool ─────────────────────────────────────────────────────────────────
// Named process lifecycle management: start, ps, logs, stop, restart, send,
// wait, describe. Each operation is an 'op' parameter.

import type { Tool, ToolContext } from "@logician/log-core";
import type {
	HubLogResult,
	HubProcessManager,
	HubProcessSpec,
	HubProcessState,
} from "./process-manager.ts";

export interface HubToolDeps {
	manager: HubProcessManager;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatState(state: HubProcessState): string {
	const dur = state.startTime
		? ` (${((Date.now() - state.startTime) / 1000).toFixed(1)}s)`
		: "";
	const exit = state.exitCode !== null ? ` exit:${state.exitCode}` : "";
	const sig = state.signal ? ` sig:${state.signal}` : "";
	const restart =
		state.restartCount > 0 ? ` restarts:${state.restartCount}` : "";
	return `- [${state.name}] ${state.status}${exit}${sig}${restart}${dur} | PID: ${state.pid ?? "none"}`;
}

// ── Tool ─────────────────────────────────────────────────────────────────────

export function createHubTool(deps: HubToolDeps): Tool {
	return {
		name: "hub",
		label: "Hub",
		description:
			"Manage named processes: start, ps, logs, stop, restart, send, wait, describe. " +
			"Each 'op' has different parameters — see op-specific docs.",
		promptSnippet:
			"Manage named processes (start, ps, logs, stop, restart, send, wait, describe)",
		promptGuidelines: [
			"Use hub op='start' to launch with readiness detection",
			"Use hub op='ps' to list all processes",
			"Use hub op='logs' with cursor for log pagination",
			"Use hub op='stop' for graceful termination",
			"Use hub op='restart' to reuse the original launch spec",
			"Use hub op='send' to write to process stdin",
			"Use hub op='wait' to block until readiness or exit",
			"Use hub op='describe' for full process state",
		],
		readOnly: false,
		executionMode: "sequential",
		parameters: {
			type: "object",
			properties: {
				op: {
					type: "string",
					enum: [
						"start",
						"ps",
						"logs",
						"stop",
						"restart",
						"send",
						"wait",
						"describe",
					],
					description: "Operation to perform.",
				},
				// start
				name: {
					type: "string",
					description: "Stable process name (≤48 chars).",
				},
				application: {
					type: "string",
					description: "Executable path (e.g. 'bun', 'python3').",
				},
				args: {
					type: "array",
					items: { type: "string" },
					description: "Command-line arguments.",
				},
				cwd: { type: "string", description: "Working directory." },
				pty: {
					type: "boolean",
					description: "Allocate PTY for interactive processes.",
				},
				ready: {
					type: "string",
					description: 'JSON readiness spec: {"log":"pattern","port":3000}',
				},
				restart: {
					type: "string",
					enum: ["no", "on-failure", "always"],
					description: "Restart policy.",
				},
				persist: {
					type: "boolean",
					description: "Persist across agent sessions.",
				},
				detached: { type: "boolean", description: "Survive broker shutdown." },
				// logs
				cursor: {
					type: "integer",
					minimum: 0,
					description: "Log cursor offset.",
				},
				lines: {
					type: "integer",
					minimum: 1,
					maximum: 1000,
					description: "Max lines to return.",
				},
				follow: {
					type: "boolean",
					description: "Wait for new output after cursor.",
				},
				// send
				input: { type: "string", description: "Text to send to stdin." },
				keys: {
					type: "array",
					items: { type: "string" },
					description: "Terminal keys: ENTER, TAB, CTRL_C, etc.",
				},
				signal: {
					type: "string",
					description:
						"Kill signal: SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGKILL.",
				},
				// wait
				for: {
					type: "string",
					enum: ["ready", "exit"],
					description: "Wait for condition.",
				},
				pattern: { type: "string", description: "Pattern to match in output." },
				timeout: {
					type: "integer",
					minimum: 1,
					description: "Seconds to wait.",
				},
				// env
				env: { type: "string", description: 'JSON env map: {"KEY":"value"}.' },
			},
			required: ["op"],
		} as const,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<string> => {
			const op = String(args.op);

			switch (op) {
				case "start": {
					const appName = String(args.application ?? "");
					const name = String(args.name ?? "");
					if (!appName) return "Error: application is required for start.";
					if (!name) return "Error: name is required for start.";

					const rawReady = args.ready;
					const readySpec =
						typeof rawReady === "string" && rawReady.trim()
							? JSON.parse(rawReady)
							: undefined;

					const rawEnv = args.env;
					const envMap: Record<string, string> | undefined =
						typeof rawEnv === "string" && rawEnv.trim()
							? JSON.parse(rawEnv)
							: undefined;

					const spec: HubProcessSpec = {
						name,
						application: appName,
						args: Array.isArray(args.args)
							? (args.args as string[])
							: undefined,
						cwd: typeof args.cwd === "string" ? args.cwd : undefined,
						env: envMap,
						pty: Boolean(args.pty),
						ready: readySpec,
						restart: (args.restart as "no" | "on-failure" | "always") ?? "no",
						persist: Boolean(args.persist),
						detached: Boolean(args.detached),
					};

					try {
						const state = await deps.manager.start(spec);
						return `Started process "${name}" (PID: ${state.pid ?? "none"}, status: ${state.status}).`;
					} catch (err) {
						return `Error starting "${name}": ${err instanceof Error ? err.message : String(err)}`;
					}
				}

				case "ps": {
					const states = deps.manager.ps();
					if (states.length === 0) {
						return "(No managed processes.)";
					}
					return (
						`Managed processes (${states.length}):\n\n` +
						states.map(formatState).join("\n")
					);
				}

				case "describe": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for describe.";
					const state = deps.manager.describe(name);
					if (!state) return `Process "${name}" not found.`;
					const specStr = JSON.stringify(state.spec, null, 2);
					return `Process: ${state.name}\nStatus: ${state.status}\nPID: ${state.pid ?? "none"}\nRestarts: ${state.restartCount}\nSpec:\n${specStr}`;
				}

				case "logs": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for logs.";
					const cursor = Number(args.cursor) || 0;
					const lines = Number(args.lines) || 100;
					const result: HubLogResult | null = deps.manager.logs(name, {
						cursor,
						lines,
						follow: Boolean(args.follow),
					});
					if (!result) return `Process "${name}" not found.`;
					const output = result.lines.join("\n");
					return `[${result.lines.length} lines, cursor: ${result.cursor}/${result.totalLines}]\n${output}`;
				}

				case "stop": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for stop.";
					const res = await deps.manager.stop(name);
					return res.message;
				}

				case "restart": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for restart.";
					const state = await deps.manager.restart(name);
					return `Restarted "${name}" (PID: ${state.pid ?? "none"}, status: ${state.status}).`;
				}

				case "send": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for send.";
					const input = typeof args.input === "string" ? args.input : undefined;
					const keys = Array.isArray(args.keys)
						? (args.keys as string[])
						: undefined;
					const signal =
						typeof args.signal === "string" ? args.signal : undefined;
					const res = deps.manager.send(name, input ?? "", {
						keys,
						enter: true,
						signal,
					});
					return res.message;
				}

				case "wait": {
					const name = String(args.name ?? "");
					if (!name) return "Error: name is required for wait.";
					const forWhat = (args.for as "ready" | "exit") ?? "ready";
					const pattern =
						typeof args.pattern === "string" ? args.pattern : undefined;
					const timeout = Number(args.timeout) || 60;
					const res = await deps.manager.wait(name, {
						for: forWhat,
						pattern,
						timeout,
					});
					return res.message;
				}

				default:
					return `Error: Unknown op "${op}". Supported: start, ps, logs, stop, restart, send, wait, describe.`;
			}
		},
	};
}
