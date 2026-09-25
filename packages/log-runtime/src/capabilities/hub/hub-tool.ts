// ── hub tool ─────────────────────────────────────────────────────────────────
// Unified process + subagent coordination surface. Named processes: start, ps,
// logs, stop, restart, send, wait, describe. When a shared HubMessageBus is
// available (subagent coordination), the same tool also handles peer messaging:
// send/wait to peers, plus jobs and inbox. Each 'op' has different parameters
// — see op-specific docs.
//
// Disambiguation: 'send' and 'wait' route by their targeting parameter.
//   send: name (process stdin) vs to (peer agent, "*" broadcasts)
//   wait: name (process condition) vs handles (peer agent ids)
// Providing both, or neither, is an error.

import type { Tool, ToolContext } from "@logician/log-core";
import type { HubMessageBus } from "../delegation/hub.ts";
import type {
	HubLogResult,
	HubProcessManager,
	HubProcessSpec,
	HubProcessState,
} from "./process-manager.ts";

export interface HubToolDeps {
	manager: HubProcessManager;
	/** When set, peer coordination ops (send/wait to peers, jobs, inbox) work. */
	bus?: HubMessageBus;
	/** This agent's id on the message bus (needed for peer send/inbox). */
	agentId?: string;
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

function formatMessage(m: {
	from: string;
	body: string;
	meta?: unknown;
}): string {
	return `[${m.from}] ${m.body}${
		m.meta !== undefined ? ` (meta: ${JSON.stringify(m.meta)})` : ""
	}`;
}

// ── Tool ─────────────────────────────────────────────────────────────────────

export function createHubTool(deps: HubToolDeps): Tool {
	const { bus, agentId } = deps;

	// Peer coordination ops only exist when a message bus is wired in.
	const processOps = [
		"start",
		"ps",
		"logs",
		"stop",
		"restart",
		"send",
		"wait",
		"describe",
	];
	const opEnum: string[] = bus ? [...processOps, "jobs", "inbox"] : processOps;

	// A single, unified prompt block describing both roles.
	const description =
		"Manage named processes and/or coordinate with spawned subagents. Each 'op' has " +
		"different parameters — see op-specific docs.";

	const promptSnippet = bus
		? "Manage named processes (start, ps, logs, stop, restart, send, wait, describe) " +
			"and subagent peers (send, wait, jobs, inbox)"
		: "Manage named processes (start, ps, logs, stop, restart, send, wait, describe)";

	const promptGuidelines: string[] = [
		"Use hub op='start' to launch with readiness detection",
		"Use hub op='ps' to list all processes",
		"Use hub op='logs' with cursor for log pagination",
		"Use hub op='stop' for graceful termination",
		"Use hub op='restart' to reuse the original launch spec",
		"Use hub op='send' to write to process stdin (op='send', name=<process>) " +
			"or to a peer subagent (op='send', to=<agent id> or '*')",
		"Use hub op='wait' to block until readiness/exit (name) " +
			"or for peer messages (handles)",
		"Use hub op='describe' for full process state",
	];
	if (bus) {
		promptGuidelines.push(
			"Use hub op='jobs' to list registered subagents and their status",
		);
		promptGuidelines.push(
			"Use hub op='inbox' to drain messages addressed to this agent (including broadcasts)",
		);
	}

	return {
		name: "hub",
		label: "Hub",
		description,
		promptSnippet,
		promptGuidelines,
		readOnly: false,
		executionMode: "sequential",
		parameters: {
			type: "object",
			properties: {
				op: {
					type: "string",
					enum: opEnum,
					description: "Operation to perform.",
				},
				// start
				name: {
					type: "string",
					description:
						"Stable process name (≤48 chars). Also the target process for " +
						"send/wait/describe. Omit for peer send (use to).",
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
				input: {
					type: "string",
					description: "Text to send to process stdin.",
				},
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
				// peer send
				to: {
					type: "string",
					description:
						'Recipient peer agent id, or "*" to broadcast. Omit when "name" ' +
						"targets a process.",
				},
				body: {
					type: "string",
					description: "Message content for peer send.",
				},
				meta: {
					type: "object",
					description: "Optional metadata (key-value pairs) for peer send.",
				},
				// peer wait
				handles: {
					type: "array",
					items: { type: "string" },
					description:
						"Peer agent ids to wait for messages from. Omit when " +
						'"name" targets a process.',
				},
				timeout_ms: {
					type: "integer",
					minimum: 100,
					description:
						"Peer wait timeout in milliseconds (default: 30000). Omit for process wait (use timeout).",
				},
				// wait (process)
				for: {
					type: "string",
					enum: ["ready", "exit"],
					description: "Process wait condition.",
				},
				pattern: {
					type: "string",
					description: "Pattern to match in output (process wait).",
				},
				timeout: {
					type: "integer",
					minimum: 1,
					description: "Seconds to wait (process wait).",
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
						return `Started process "${name}" (PID: ${
							state.pid ?? "none"
						}, status: ${state.status}).`;
					} catch (err) {
						return `Error starting "${name}": ${
							err instanceof Error ? err.message : String(err)
						}`;
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
					return (
						`Process: ${state.name}\n` +
						`Status: ${state.status}\n` +
						`PID: ${state.pid ?? "none"}\n` +
						`Restarts: ${state.restartCount}\n` +
						`Spec:\n${specStr}`
					);
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
					return `[${result.lines.length} lines, cursor: ${
						result.cursor
					}/${result.totalLines}]\n${output}`;
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
					return `Restarted "${name}" (PID: ${
						state.pid ?? "none"
					}, status: ${state.status}).`;
				}

				case "send": {
					const name = String(args.name ?? "");
					const to = typeof args.to === "string" ? args.to : "";
					const hasName = name.length > 0;
					const hasTo = to.length > 0;
					if (hasName && hasTo) {
						return "Error: send requires either 'name' (process) or 'to' (peer agent), not both.";
					}
					if (hasName) {
						// Process stdin.
						const input =
							typeof args.input === "string" ? args.input : undefined;
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
					if (hasTo) {
						// Peer message.
						if (!bus) {
							return "Error: peer send (to) requires subagent coordination.";
						}
						if (!agentId) {
							return "Error: peer send requires this agent's id on the bus.";
						}
						const body = typeof args.body === "string" ? args.body : "";
						if (!body) {
							return "Error: peer send requires a 'body'.";
						}
						const meta =
							typeof args.meta === "object" &&
							args.meta !== null &&
							!Array.isArray(args.meta)
								? (args.meta as Record<string, unknown>)
								: undefined;
						bus.send(agentId, to, body, meta);
						return `Message sent to ${to}.`;
					}
					return "Error: send requires either 'name' (process) or 'to' (peer agent, or '*' to broadcast).";
				}

				case "wait": {
					const name = String(args.name ?? "");
					const rawHandles = args.handles;
					const handles =
						typeof rawHandles === "object" && Array.isArray(rawHandles)
							? rawHandles
							: undefined;
					const hasName = name.length > 0;
					const hasHandles = handles !== undefined && handles.length > 0;
					if (hasName && hasHandles) {
						return "Error: wait requires either 'name' (process) or 'handles' (peer agents), not both.";
					}
					if (hasName) {
						// Process condition wait.
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
					if (hasHandles) {
						// Peer message wait.
						if (!bus) {
							return "Error: peer wait (handles) requires subagent coordination.";
						}
						if (!agentId) {
							return "Error: peer wait requires this agent's id on the bus.";
						}
						const timeoutMs =
							typeof args.timeout_ms === "number" && args.timeout_ms > 0
								? args.timeout_ms
								: 30_000;
						const messages = await bus.wait(handles, timeoutMs);
						if (messages.length === 0) {
							return `No messages received from ${handles.join(
								", ",
							)} within ${timeoutMs}ms.`;
						}
						const lines = messages.map(formatMessage);
						return `Received ${messages.length} message(s):\n${lines.join("\n")}`;
					}
					return "Error: wait requires either 'name' (process) or 'handles' (peer agents).";
				}

				case "jobs": {
					if (!bus) {
						return "Error: 'jobs' requires subagent coordination.";
					}
					const jobs = bus.jobs();
					if (jobs.length === 0) {
						return "No active subagents.";
					}
					const lines = jobs.map(
						j =>
							`  ${j.id} (${j.agent}) [${j.status}] task=${
								j.task
							}${j.taskIndex !== undefined ? ` index=${j.taskIndex}` : ""}`,
					);
					return `Active subagents (${jobs.length}):\n${lines.join("\n")}`;
				}

				case "inbox": {
					if (!bus) {
						return "Error: 'inbox' requires subagent coordination.";
					}
					if (!agentId) {
						return "Error: 'inbox' requires this agent's id on the bus.";
					}
					const messages = bus.inbox(agentId);
					if (messages.length === 0) {
						return "No messages in inbox.";
					}
					const lines = messages.map(formatMessage);
					return `Inbox (${messages.length} message(s)):\n${lines.join("\n")}`;
				}

				default:
					return `Error: Unknown op "${op}". Supported: ${opEnum.join(", ")}.`;
			}
		},
	};
}
