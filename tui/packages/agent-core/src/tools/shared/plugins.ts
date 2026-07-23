// ── Plugins barrel ────────────────────────────────────────────────────────────
// CLI entry points and re-exports. All imports that previously used
// `plugins.ts` continue to work unchanged.

import { TsPluginManager, type PluginCommandResult } from "../../compatibility/claude-code/plugin-manager.ts";

// ── CLI functions ─────────────────────────────────────────────────────────────

let pluginRuntimeEnv: NodeJS.ProcessEnv | undefined;

export function configurePluginRuntimeEnv(env: NodeJS.ProcessEnv): void {
	pluginRuntimeEnv = Object.fromEntries(
		Object.entries(env).filter(([, value]) => value !== undefined),
	) as NodeJS.ProcessEnv;
}

export function splitPluginArgs(input: string): string[] {
	const args: string[] = [];
	let current = "";
	let quote: string | null = null;
	let escaped = false;

	for (const ch of input.trim()) {
		if (escaped) {
			current += ch;
			escaped = false;
			continue;
		}
		if (ch === "\\") {
			escaped = true;
			continue;
		}
		if (quote) {
			if (ch === quote) quote = null;
			else current += ch;
			continue;
		}
		if (ch === "'" || ch === "\"") {
			quote = ch;
			continue;
		}
		if (/\s/.test(ch)) {
			if (current) {
				args.push(current);
				current = "";
			}
			continue;
		}
		current += ch;
	}
	if (escaped) current += "\\";
	if (current) args.push(current);
	return args;
}

export async function runPluginBackend(
	action: string,
	args: string[],
): Promise<PluginCommandResult> {
	const manager = new TsPluginManager({ env: pluginRuntimeEnv });
	try {
		switch (action) {
			case "list":
				return {
					...(await manager.listPlugins()),
					session_start_hooks: await manager.sessionStartHookCounts(),
				};
			case "enable":
			case "disable": {
				if (!args[0]) throw new Error(`usage: /plugins ${action} <plugin>`);
				const result = await manager.setEnabled(args[0], action === "enable");
				return {
					...result,
					session_start_hooks: await manager.sessionStartHookCounts(),
				};
			}
			case "install":
				if (!args[0])
					throw new Error("usage: /plugins install <owner/name | path | name>");
				return {
					...(await manager.install(args[0])),
					session_start_hooks: await manager.sessionStartHookCounts(),
				};
			case "remove":
				if (!args[0])
					throw new Error("usage: /plugins remove <plugin> [--keep-checkout]");
				return manager.remove(args[0], args.includes("--keep-checkout"));
			case "update":
				if (!args[0])
					throw new Error("usage: /plugins update <plugin | --all>");
				if (args[0] === "--all") {
					const updates = [];
					for (const pluginId of await manager.allPluginIds()) {
						updates.push(await manager.update(pluginId));
					}
					return {
						status: "ok",
						updates,
						plugins_dir: manager.pluginsDir,
					};
				}
				return manager.update(args[0]);
			case "deps":
				return manager.dependencies(args[0]);
			case "info":
				if (!args[0]) throw new Error("usage: /plugins info <plugin>");
				return {
					...(await manager.info(args[0])),
					session_start_hooks: await manager.sessionStartHookCounts(),
				};
			case "hooks":
				return manager.listHooks(args[0] || "startup");
			case "run-hooks":
			case "session-start":
				return manager.executeSessionStartHooks(
					args[0] || "startup",
					parseJsonArg(args[1]),
				);
			case "hook":
				if (!args[0])
					throw new Error("usage: hook <event-type> [payload-json]");
				return manager.executeHookEvent(
					args[0] as string,
					parseJsonArg(args[1]),
				);
			default:
				throw new Error(
					"usage: /plugins [list|enable|disable|install|remove|update|deps|info|hooks|run-hooks]",
				);
		}
	} catch (error: unknown) {
		return {
			status: "error",
			message: error instanceof Error ? error.message : String(error),
			plugins_dir: manager.pluginsDir,
		};
	}
}

export async function runSessionStartHooks(
	payload: {
		source?: string;
		session_id?: string;
		transcript_path?: string;
		cwd?: string;
	} = {},
): Promise<PluginCommandResult> {
	const source = payload.source || "startup";
	return runPluginBackend("session-start", [
		source,
		JSON.stringify({
			session_id: payload.session_id || "",
			transcript_path: payload.transcript_path || "",
			cwd: payload.cwd || process.cwd(),
		}),
	]);
}

export async function runHookEvent(
	eventType: string,
	payload: Record<string, unknown> = {},
): Promise<PluginCommandResult> {
	return runPluginBackend("hook", [eventType, JSON.stringify(payload)]);
}

function parseJsonArg(raw?: string): Record<string, unknown> {
	if (!raw) return {};
	try {
		const parsed = JSON.parse(raw);
		return parsed && typeof parsed === "object" && !Array.isArray(parsed)
			? parsed : {};
	} catch (e: unknown) {
		return {};
	}
}

// ── Re-exports from sub-modules ──────────────────────────────────────────────

// Manager exports
export { TsPluginManager } from "../../compatibility/claude-code/plugin-manager.ts";
export type { PluginCommandResult, PluginInstall } from "../../compatibility/claude-code/plugin-manager.ts";

// Executor exports
export type {
	HookEventType,
	HookCommand,
	HookDefinition,
	LoadedHook,
	HookContextMessage,
	HookExecutionResult,
} from "../../compatibility/claude-code/plugin-executor.ts";
export {
	loadPluginHooks,
	executeLoadedHook,
	executeCommand,
	parseHookResponse,
	buildHookInput,
	parseHookEventType,
	matcherMatches,
} from "../../compatibility/claude-code/plugin-executor.ts";
