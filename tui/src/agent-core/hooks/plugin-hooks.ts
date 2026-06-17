// ── OpenClaude/Claude plugin hook adapter ────────────────────────────────────
// Converts JSON/stdin plugin hooks into AgentLoopHooks so the core loop only
// deals with one hook contract.

import { createUserMessage } from "../core/messages.ts";
import type {
	AgentLoopHooks,
	Message,
	ToolCall,
} from "../core/types.ts";
import { runHookEvent, type PluginCommandResult } from "../tools/shared/plugins.ts";

export interface PluginHookLayerOptions {
	enabled: boolean;
	sessionId: string;
	transcriptPath: string;
	cwd: string;
	getMatcherValue: (toolName: string) => string;
	onHookPermissionDenied?: (toolCall: ToolCall) => void;
}

export interface PluginHookLayer {
	hooks: AgentLoopHooks | undefined;
	userPromptMessages(prompt: string): Promise<Message[]>;
	finalStop(): Promise<void>;
	readonly stopObserved: boolean;
}

export function createPluginHookLayer(
	options: PluginHookLayerOptions,
): PluginHookLayer {
	return new RuntimePluginHookLayer(options);
}

class RuntimePluginHookLayer implements PluginHookLayer {
	private stopHookContinuationActive = false;
	private _stopObserved = false;

	constructor(private readonly options: PluginHookLayerOptions) {}

	get hooks(): AgentLoopHooks | undefined {
		if (!this.options.enabled) return undefined;
		return {
			beforeToolCall: async ({ toolCall, args }) => {
				const result = await this.run("PreToolUse", {
					matcher_value: this.options.getMatcherValue(toolCall.name),
					tool_name: toolCall.name,
					tool_input: args,
				});
				if (result?.permission_decision !== "deny") return undefined;

				this.options.onHookPermissionDenied?.(toolCall);
				const reason = result.permission_reason || "Blocked by hook.";
				return {
					content: `Permission denied by hook: ${reason}`,
					isError: true,
				};
			},

			afterToolCall: async ({ toolCall, args, result, isError }) => {
				const hookResult = await this.run("PostToolUse", {
					matcher_value: this.options.getMatcherValue(toolCall.name),
					tool_name: toolCall.name,
					tool_input: args,
					tool_response: result,
				});
				const context = contextText(hookResult);
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
			return await runHookEvent(eventType, {
				session_id: this.options.sessionId,
				transcript_path: this.options.transcriptPath,
				cwd: this.options.cwd,
				...payload,
			});
		} catch {
			return null;
		}
	}
}

function contextText(result: PluginCommandResult | null): string {
	return (result?.additional_contexts || [])
		.map((item) => String(item || "").trim())
		.filter(Boolean)
		.join("\n\n");
}
