// ── Extension/hook runtime composition for AgentHarness ────────────────────
// Builds the effective AgentConfig (tools + hooks) for a turn: merges
// extension tools, builds a fresh HookBus layering builtin safeguards, the
// harness's own queue-draining, extensions, claude-code compat, and
// caller-supplied hooks (in that override order). Mirrors queue-ops.ts's
// Deps pattern — the harness owns the mutable fields and supplies them here.

import type { ExtensionRunner, RegisteredTool } from "../../extension/index.ts";
import { BudgetTracker } from "../../hooks/builtin/budget.ts";
import {
	buildBuiltinHooks,
	COMPACTION_COOLDOWN_TURNS,
} from "../../hooks/builtin/builtin-hooks.ts";
import { HookBus } from "../../hooks/hook-bus.ts";
import {
	type ClaudeCodeHookLayer,
	claudeToolMatcherName,
	createClaudeCodeHookLayer,
} from "../../extension/adapters/claude-code/hook-layer.ts";
import type { HarnessInterventionController } from "../../policy/intervention-controller.ts";
import type { LoopDetector } from "../../../infrastructure/guards/loop-detector.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHooks,
	Message,
	Tool,
} from "../../types/index.ts";

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

/**
 * Owns state that must persist across hook-bus rebuilds within one harness
 * instance: BudgetTracker compares consecutive turns and the compaction
 * cooldown counts turns since the last compaction, so a fresh instance per
 * turn would never trigger either safeguard.
 */
export function createExtensionRuntimeState() {
	let budgetTracker: BudgetTracker | null = null;
	const compactionCooldown = { lastTurn: -COMPACTION_COOLDOWN_TURNS };
	return {
		getBudgetTracker: (): BudgetTracker => {
			budgetTracker ??= new BudgetTracker();
			return budgetTracker;
		},
		compactionCooldown,
	};
}

export type ExtensionRuntimeState = ReturnType<
	typeof createExtensionRuntimeState
>;

export function createClaudeCodeHookLayerFor(
	deps: ExtensionRuntimeDeps,
): ClaudeCodeHookLayer {
	return createClaudeCodeHookLayer({
		enabled: deps.getHooksEnabled(),
		sessionId: deps.getSessionId(),
		transcriptPath: deps.getTranscriptPath(),
		cwd: deps.getCwd(),
		getMatcherValue: toolName => {
			const tool = deps
				.getConfigTools()
				?.find(candidate => candidate.name === toolName);
			return tool?.hookAliases?.join("|") || claudeToolMatcherName(toolName);
		},
	});
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

	// Emit before_agent_start (→ native extensions + Pi's before_agent_start)
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
	state: ExtensionRuntimeState,
	config: AgentConfig,
	pluginHookLayer?: ClaudeCodeHookLayer,
): AgentConfig {
	const runner = deps.getExtensionRunner();
	const extensionTools = runner
		? runner.getTools().map(tool => wrapExtensionTool(deps, tool))
		: [];
	const tools = [...(config.tools ?? []), ...extensionTools];

	// Rebuild HookBus layers each turn (extensions may add/remove tools/hooks).
	// Using a fresh bus avoids stale registrations between turns.
	const hookBus = new HookBus();

	// Layers run in registration order: builtin safeguards, then the
	// harness's own queue-draining, then extensions, then claude-code
	// compat, then caller-supplied hooks last so callers can override.
	const builtinHooks = buildBuiltinHooks({
		config,
		contextWindowTokens: () => config.contextWindowTokens,
		toolDefs: () => tools as unknown as Record<string, unknown>[],
		loopDetector: deps.loopDetector,
		emitEvent: (event: { type: string; [key: string]: unknown }) => {
			deps.emit(event as AgentEvent);
		},
		interventions: deps.interventions,
		budget: state.getBudgetTracker(),
		compactionCooldown: state.compactionCooldown,
	});
	hookBus.register(builtinHooks, { source: "builtin" });
	hookBus.register(deps.drainHooks(), { source: "queue-drain" });

	const extensionHooks = runner?.getHooks();
	if (extensionHooks) {
		hookBus.register(extensionHooks, { source: "extensions" });
	}

	const claudeHooks = (pluginHookLayer ?? createClaudeCodeHookLayerFor(deps))
		.hooks;
	if (claudeHooks) {
		hookBus.register(claudeHooks, { source: "claude-code-compat" });
	}

	if (config.hooks) {
		hookBus.register(config.hooks, { source: "user" });
	}

	return {
		...config,
		tools,
		hooks: hookBus.toHooks(),
	};
}
