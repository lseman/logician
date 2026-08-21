// ── Extension/hook runtime composition for AgentHarness ────────────────────
// Builds the effective AgentConfig (tools + hooks) for a turn: merges
// extension tools, builds a fresh HookBus layering builtin safeguards, the
// harness's own queue-draining, extensions, claude-code compat, and
// caller-supplied hooks (in that override order). Mirrors queue-ops.ts's
// Deps pattern — the harness owns the mutable fields and supplies them here.

import type { ExtensionRunner } from "../../../system/extension/runner.ts";
import type { RegisteredTool } from "../../../system/extension/types.ts";
import type { LoopDetector } from "../../../control/guards/loop-detector.ts";
import { buildBuiltinHooks } from "../../hooks/builtin/builtin-hooks.ts";
import { extensionHooks, runControlHooks } from "../../hooks/contracts.ts";
import { HookBus } from "../../hooks/hook-bus.ts";
import type { HarnessInterventionController } from "../../../control/policy/intervention-controller.ts";
import type { AgentRunController } from "../../../control/policy/run-controller.ts";
import type { AgentConfig } from "../../../system/types/types-config.ts";
import type {
	AgentEvent,
	AgentHooks,
	Message,
	Tool,
} from "../../../system/types/types-messages.ts";

export interface ExtensionRuntimeDeps {
	getExtensionRunner: () => ExtensionRunner | undefined;
	getHooksEnabled: () => boolean;
	getSessionId: () => string;
	getTranscriptPath: () => string;
	getCwd: () => string;
	getConfigTools: () => Tool[] | undefined;
	loopDetector: LoopDetector;
	interventions: HarnessInterventionController;
	emit: (event: AgentEvent) => void;
	/** Steering/follow-up messages drained from the harness's own queues. */
	drainHooks: () => AgentHooks;
}

function wrapExtensionTool(
	deps: ExtensionRuntimeDeps,
	tool: RegisteredTool,
): Tool {
	return {
		name: tool.name,
		description: tool.description,
		parameters: tool.parameters as unknown as Record<string, unknown>,
		execute: async (args, ctx) => {
			const result = await tool.execute(
				`extension_${tool.name}_${Date.now()}`,
				args,
				{
					toolCall: {
						id: `extension_${tool.name}`,
						name: tool.name,
						arguments: JSON.stringify(args),
					},
					cwd: ctx.cwd ?? deps.getCwd(),
					sessionId: deps.getSessionId(),
				},
			);
			return { content: result.content, details: result.details };
		},
	};
}

export function resolveRuntimeTools(
	deps: ExtensionRuntimeDeps,
	config: AgentConfig,
): Tool[] {
	const extensionTools =
		deps
			.getExtensionRunner()
			?.getTools()
			.map(tool => wrapExtensionTool(deps, tool)) ?? [];
	return [...(config.tools ?? []), ...extensionTools];
}

export async function runExtensionBeforeAgentStart(
	deps: ExtensionRuntimeDeps,
	promptText: string,
	systemPrompt: string,
	history: Message[],
): Promise<{ messages?: Message[]; systemPrompt?: string } | undefined> {
	const runner = deps.getExtensionRunner();
	if (!runner) return undefined;

	const ctx = {
		sessionId: deps.getSessionId(),
		cwd: deps.getCwd(),
		prompt: promptText,
		systemPrompt,
		messages: [...history],
	};

	let nativeMessages: Message[] | undefined;
	let nativeSystemPrompt: string | undefined;

	// Emit before_agent_start to native extensions.
	if (runner.hasHandlers("before_agent_start")) {
		const result = await runner.emitToAll({
			type: "before_agent_start",
			context: ctx,
		});
		// Native extensions return { messages, systemPrompt } directly
		if (result && typeof result === "object") {
			const value = result as { messages?: Message[]; systemPrompt?: string };
			nativeMessages = Array.isArray(value.messages)
				? value.messages
				: undefined;
			nativeSystemPrompt =
				typeof value.systemPrompt === "string" ? value.systemPrompt : undefined;
		}
	}

	return {
		messages: nativeMessages,
		systemPrompt: nativeSystemPrompt,
	};
}

/** Compose the effective AgentConfig (tools + hooks) for one turn. */
export function withExtensionRuntime(
	deps: ExtensionRuntimeDeps,
	run: AgentRunController,
	config: AgentConfig,
	pluginHookLayer?: { hooks?: AgentHooks },
): AgentConfig {
	const runner = deps.getExtensionRunner();
	const tools = resolveRuntimeTools(deps, config);

	// External interception and internal run control have different lifetimes
	// and authority. Compose them independently through AgentHooks.
	const extensionBus = new HookBus();
	const controlBus = new HookBus();
	const builtinHooks = buildBuiltinHooks({
		config,
		contextWindowTokens: () => config.contextWindowTokens,
		toolDefs: () => tools as unknown as Record<string, unknown>[],
		loopDetector: deps.loopDetector,
		emitEvent: (event: { type: string; [key: string]: unknown }) => {
			deps.emit(event as AgentEvent);
		},
		interventions: deps.interventions,
		progress: run.progress,
		compactionCooldown: run.compaction,
	});
	controlBus.register(runControlHooks(builtinHooks), { source: "builtin" });
	extensionBus.register(extensionHooks(builtinHooks), { source: "builtin" });
	controlBus.register(runControlHooks(deps.drainHooks()), {
		source: "queue-drain",
	});

	const extensionLayer = runner?.getHooks();
	if (extensionLayer) {
		extensionBus.register(extensionLayer, { source: "extensions" });
	}

	if (pluginHookLayer?.hooks) {
		extensionBus.register(extensionHooks(pluginHookLayer.hooks), {
			source: "plugins",
		});
		controlBus.register(runControlHooks(pluginHookLayer.hooks), {
			source: "plugins",
		});
	}

	if (config.hooks) {
		extensionBus.register(extensionHooks(config.hooks), { source: "user" });
		controlBus.register(runControlHooks(config.hooks), { source: "user" });
	}
	const external = extensionBus.toHooks();
	const control = controlBus.toHooks();

	return {
		...config,
		tools,
		hooks: {
			...extensionHooks(external),
			...runControlHooks(control),
		},
	};
}
