// ── Claude Code compatibility hook layer ────────────────────────────────────
// Adapts the Claude Code plugin hook protocol (JSON over stdin, Claude event
// names, and Claude hook responses) to Logician's native AgentHooks contract.
// This is a compatibility boundary, not Logician's general hook system.

import { createUserMessage } from "../../core/messages.ts";
import type {
	AgentHooks,
	Message,
	ToolCall,
} from "../../core/types.ts";
import { runHookEvent, type PluginCommandResult } from "../../tools/shared/plugins.ts";

export interface ClaudeCodeHookLayerOptions {
	enabled: boolean;
	sessionId: string;
	transcriptPath: string;
	cwd: string;
	getMatcherValue: (toolName: string) => string;
	runHookEvent?: typeof runHookEvent;
	onHookPermissionDenied?: (toolCall: ToolCall) => void;
}

export interface ClaudeCodeHookLayer {
	hooks: AgentHooks | undefined;
	userPromptMessages(prompt: string): Promise<Message[]>;
	finalStop(): Promise<void>;
	readonly stopObserved: boolean;
}

export function createClaudeCodeHookLayer(
	options: ClaudeCodeHookLayerOptions,
): ClaudeCodeHookLayer {
	return new RuntimeClaudeCodeHookLayer(options);
}

class RuntimeClaudeCodeHookLayer implements ClaudeCodeHookLayer {
	private stopHookContinuationActive = false;
	private _stopObserved = false;
	private readonly preToolContext = new Map<string, string>();

	constructor(private readonly options: ClaudeCodeHookLayerOptions) {}

	get hooks(): AgentHooks | undefined {
		if (!this.options.enabled) return undefined;
		return {
			beforeToolCall: async ({ toolCall, args }) => {
				const result = await this.run("PreToolUse", {
					matcher_value: this.options.getMatcherValue(toolCall.name),
					tool_name: toolCall.name,
					tool_input: args,
				});
				const context = contextText(result);
				if (context) this.preToolContext.set(toolCall.id, context);
				if (result?.permission_decision !== "deny") return undefined;

				this.options.onHookPermissionDenied?.(toolCall);
				this.preToolContext.delete(toolCall.id);
				const reason = result.permission_reason || context || "Blocked by hook.";
				return {
					content: `Permission denied by hook: ${reason}`,
					isError: true,
				};
			},

			afterToolCall: async ({ toolCall, args, result, isError }) => {
				const preToolContext = this.preToolContext.get(toolCall.id) || "";
				this.preToolContext.delete(toolCall.id);
				const hookResult = isError
					? await this.run("PostToolUseFailure", {
							matcher_value: this.options.getMatcherValue(toolCall.name),
							tool_name: toolCall.name,
							tool_input: args,
							tool_error: result,
						})
					: await this.run("PostToolUse", {
							matcher_value: this.options.getMatcherValue(toolCall.name),
							tool_name: toolCall.name,
							tool_input: args,
							tool_response: result,
						});
				const context = [preToolContext, contextText(hookResult)]
					.filter(Boolean)
					.join("\n\n");
				if (!context) return undefined;
				return {
					content: `${result}\n\n<post-tool-use-hook>\n${context}\n</post-tool-use-hook>`,
					isError,
				};
			},

			getFollowUpMessages: async () => this.stopFollowUpMessages(),
		};
	}

	get stopObserved(): boolean {
		return this._stopObserved;
	}

	async userPromptMessages(prompt: string): Promise<Message[]> {
		if (!this.options.enabled) return [];
		const result = await this.run("UserPromptSubmit", {
			prompt,
			timeout_seconds: 30,
		});
		const context = contextText(result);
		return context
			? [
					createUserMessage(
						`<user-prompt-submit-hook>\n${context}\n</user-prompt-submit-hook>`,
					),
				]
			: [];
	}

	async finalStop(): Promise<void> {
		if (!this.options.enabled || this._stopObserved) return;
		await this.run("Stop", { stop_hook_active: false });
	}

	private async stopFollowUpMessages(): Promise<Message[]> {
		this._stopObserved = true;
		const result = await this.run("Stop", {
			stop_hook_active: this.stopHookContinuationActive,
		});
		if (!result) {
			this.stopHookContinuationActive = false;
			return [];
		}

		const context = contextText(result);
		const blocked =
			result.decision === "block" || Boolean(result.reason) || Boolean(context);
		if (!blocked || this.stopHookContinuationActive) {
			this.stopHookContinuationActive = false;
			return [];
		}

		this.stopHookContinuationActive = true;
		return [
			createUserMessage(
				`<stop-hook>\n${context || result.reason || "A stop hook requested that you continue instead of ending now."}\n</stop-hook>\n\nContinue the task using the hook guidance above.`,
			),
		];
	}

	private async run(
		eventType: string,
		payload: Record<string, unknown>,
	): Promise<PluginCommandResult | null> {
		try {
			return await (this.options.runHookEvent ?? runHookEvent)(eventType, {
				session_id: this.options.sessionId,
				transcript_path: this.options.transcriptPath,
				cwd: this.options.cwd,
				...payload,
			});
		} catch (_e: unknown) {
			return null;
		}
	}
}

/** Map Logician-native names to the matcher vocabulary used by Claude plugins. */
export function claudeToolMatcherName(toolName: string): string {
	if (toolName.startsWith("mcp__")) return toolName;
	const aliases: Record<string, string> = {
		bash: "Bash",
		read_file: "Read",
		write_file: "Write",
		edit_file: "Edit",
		grep: "Grep",
		find: "Glob",
		list_files: "Glob",
		todo: "TodoWrite|TaskCreate|TaskUpdate",
		spawn_agent: "Agent",
		ask_user: "AskUserQuestion",
	};
	return aliases[toolName] ?? toolName;
}

/** @deprecated Use ClaudeCodeHookLayerOptions. */
export type PluginHookLayerOptions = ClaudeCodeHookLayerOptions;
/** @deprecated Use ClaudeCodeHookLayer. */
export type PluginHookLayer = ClaudeCodeHookLayer;
/** @deprecated Use createClaudeCodeHookLayer. */
export const createPluginHookLayer = createClaudeCodeHookLayer;

function contextText(result: PluginCommandResult | null): string {
	return (result?.additional_contexts || [])
		.map((item) => String(item || "").trim())
		.filter(Boolean)
		.join("\n\n");
}
