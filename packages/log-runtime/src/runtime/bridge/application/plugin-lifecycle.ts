import type { AgentConfig } from "@logician/log-core";
import type { PluginCommandResult } from "../../../adapters/claude-code/plugin-runtime.ts";
import {
	runHookEvent,
	runPluginBackend,
	runSessionStartHooks,
	splitPluginArgs,
} from "../../../adapters/claude-code/plugin-runtime.ts";
import { formatPluginResult } from "../plugin-result-formatter.ts";
import { RuntimeStartupCoordinator } from "./startup-coordinator.ts";

const EMPTY_HOOK_RESULT: PluginCommandResult = {
	additional_contexts: [],
	context_messages: [],
	initial_user_message: "",
};

export interface PluginLifecycleDependencies {
	config: () => AgentConfig;
	baseSystemPrompt: () => string;
	sessionId: () => string;
	tools: PluginResourceHost;
	injectSubagents: () => Promise<void>;
}

export interface PluginResourceHost {
	getMcpSystemContext(): string | null;
	getSkillsContext(): string | null;
	injectSkillsFromPlugins(): Promise<void>;
	injectPrompts(): Promise<void>;
}

export interface PluginLifecycleStatus {
	pluginCount: number;
	hookResult: PluginCommandResult | null;
}

/** Owns plugin startup, injected context, resource refresh, and shutdown hooks. */
export class PluginLifecycle {
	private readonly dependencies: PluginLifecycleDependencies;
	private readonly startup = new RuntimeStartupCoordinator(source =>
		this.runStartup(source),
	);
	private hookResult: PluginCommandResult | null = null;
	private pluginCount = 0;

	constructor(dependencies: PluginLifecycleDependencies) {
		this.dependencies = dependencies;
	}

	async ensureStarted(source = "startup"): Promise<void> {
		await this.startup.ensure(source);
	}

	reset(options: { clearResult?: boolean } = {}): void {
		this.startup.reset();
		if (options.clearResult) this.hookResult = null;
	}

	refreshContext(): void {
		this.applyContext(this.hookResult ?? EMPTY_HOOK_RESULT);
	}

	applyContext(result: PluginCommandResult): void {
		this.hookResult = result;
		const messageContexts = Array.isArray(result.context_messages)
			? result.context_messages.flatMap(message =>
					message &&
					typeof message === "object" &&
					typeof message.content === "string"
						? [message.content]
						: [],
				)
			: [];
		const contexts = [
			...(result.additional_contexts || []),
			...messageContexts,
			result.initial_user_message || "",
		]
			.map(item => String(item || "").trim())
			.filter(
				(item, index, all) => Boolean(item) && all.indexOf(item) === index,
			);

		const injected: string[] = [];
		if (contexts.length) {
			injected.push(
				`<startup-hook-context>\n${contexts.join("\n\n")}\n</startup-hook-context>`,
			);
		}
		const mcpContext = this.dependencies.tools.getMcpSystemContext();
		const skillsContext = this.dependencies.tools.getSkillsContext();
		if (mcpContext) injected.push(mcpContext);
		if (skillsContext) injected.push(skillsContext);

		this.dependencies.config().systemPrompt = injected.length
			? `${this.dependencies.baseSystemPrompt()}\n\n${injected.join("\n\n")}`
			: this.dependencies.baseSystemPrompt();
	}

	async snapshot(): Promise<PluginCommandResult> {
		return runPluginBackend("list", []);
	}

	async setEnabled(
		pluginId: string,
		enabled: boolean,
	): Promise<PluginCommandResult> {
		const result = await runPluginBackend(enabled ? "enable" : "disable", [
			pluginId,
		]);
		if (result.status !== "error") {
			this.startup.reset();
			await this.ensureStarted();
		}
		return result;
	}

	async runCommand(input: string): Promise<string> {
		const parts = splitPluginArgs(input);
		const action = (parts.shift() || "list").toLowerCase();
		if (action === "help" || action === "-h" || action === "--help") {
			return [
				"# Plugins",
				"Usage: /plugins [list|enable|disable|install|remove|update|deps|info|hooks|run-hooks]",
				"",
				"- /plugins list",
				"- /plugins enable <plugin>",
				"- /plugins disable <plugin>",
				"- /plugins hooks [startup|clear|compact|Stop|PreToolUse|PostToolUse|SessionEnd]",
				"- /plugins run-hooks [startup|clear|compact]",
			].join("\n");
		}

		const backendAction = action === "refresh" ? "run-hooks" : action;
		const result = await runPluginBackend(backendAction, parts);
		if (backendAction === "run-hooks" && result.status !== "error") {
			this.hookResult = result;
			this.applyContext(result);
		}
		return formatPluginResult(backendAction, result);
	}

	async endSession(reason: string): Promise<void> {
		const config = this.dependencies.config();
		if (config.runtimeHooksEnabled === false) return;
		try {
			await runHookEvent("SessionEnd", {
				session_id: this.dependencies.sessionId(),
				transcript_path: config.hookTranscriptPath || "",
				cwd: config.cwd || process.cwd(),
				reason,
			});
		} catch {
			// SessionEnd hooks are best-effort during shutdown/reset.
		}
	}

	status(): PluginLifecycleStatus {
		return { pluginCount: this.pluginCount, hookResult: this.hookResult };
	}

	private async runStartup(source: string): Promise<void> {
		const snapshot = await runPluginBackend("list", []);
		this.pluginCount = (snapshot.plugins || []).filter(
			plugin => plugin.enabled !== false && plugin.on_disk !== false,
		).length;
		const config = this.dependencies.config();
		if (config.runtimeHooksEnabled !== false) {
			const result = await runSessionStartHooks({
				source,
				session_id: this.dependencies.sessionId(),
				transcript_path: config.hookTranscriptPath,
				cwd: config.cwd || process.cwd(),
			});
			this.hookResult = result;
			if (result.status !== "error") this.applyContext(result);
		}
		await this.dependencies.tools.injectSkillsFromPlugins();
		this.refreshContext();
		await this.dependencies.tools.injectPrompts();
		await this.dependencies.injectSubagents();
	}
}
