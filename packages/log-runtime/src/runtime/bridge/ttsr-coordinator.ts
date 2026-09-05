// ── TTSR Coordinator ─────────────────────────────────────────────────────────
// Bridges TTSRManager (rule matching) with agent-bridge (abort/steer) so that
// when a rule condition matches during streaming, the current turn is aborted,
// the rule content is injected as steering, and the agent retries.
//
// Also persists injected-rule state across compaction via session entries.

import type { TtsrRule, TtsrMatchContext, TtsrBridgeSettings } from "@logician/log-core";
import type { TtsrManager } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";

// ── Coordinator state ─────────────────────────────────────────────────────────

export interface TtsrCoordinatorOptions {
	/** Called to abort the current turn. */
	abort: () => Promise<void>;
	/** Called to inject steering text that will interrupt or queue. */
	steer: (text: string) => void;
	/** Called to record a follow-up message. */
	followUp: (text: string) => void;
	/** Called to emit a runtime event. */
	emit: (event: RuntimeEvent) => void;
	/** Reference to the TtsrManager for injection tracking. */
	manager: TtsrManager;
}

interface ActiveRule {
	rule: TtsrRule;
	injectedAt: number;
}

// ── TtsrCoordinator ───────────────────────────────────────────────────────────

export class TtsrCoordinator {
	readonly #manager: TtsrManager;
	readonly #opts: TtsrCoordinatorOptions;
	readonly #activeRules = new Map<string, ActiveRule>();

	constructor(opts: TtsrCoordinatorOptions) {
		this.#manager = opts.manager;
		this.#opts = opts;
	}

	/** Process a stream event for TTSR matching. */
	processEvent(event: RuntimeEvent): void {
		const delta = this.#extractDelta(event);
		if (delta === null) return;

		const context = this.#buildContext(event);
		const matches = this.#manager.checkDelta(delta, context);
		this.#processMatches(matches, context);
	}

	/** Process an AST snapshot (called externally via checkAstSnapshot). */
	async processAstSnapshot(
		snapshot: string,
		context: TtsrMatchContext,
	): Promise<void> {
		const matches = await this.#manager.checkAstSnapshot(snapshot, context);
		this.#processMatches(matches, context);
	}

	/** Called at the start of each new turn. */
	reset(): void {
		this.#manager.resetBuffer();
		this.#activeRules.clear();
	}

	/** Called at the end of each turn to increment the message counter. */
	incrementMessageCount(): void {
		this.#manager.incrementMessageCount();
	}

	/**
	 * Called during compaction to persist injected rules so they can
	 * be restored after compaction.
	 */
	persistInjected(): string[] {
		const names = this.#manager.getInjectedRuleNames();
		for (const [name] of this.#activeRules) {
			if (!names.includes(name)) {
				names.push(name);
			}
		}
		return names;
	}

	/** Restore injected state after compaction. */
	restoreInjected(names: string[]): void {
		this.#manager.restoreInjected(names);
	}

	/** Get current manager settings. */
	getSettings(): TtsrBridgeSettings {
		return {
			enabled: this.#manager.getSettings().enabled,
			repeatMode: this.#manager.getSettings().repeatMode,
			repeatGap: this.#manager.getSettings().repeatGap,
			contextMode: this.#manager.getSettings().contextMode,
			interruptMode: this.#manager.getSettings().interruptMode,
			builtinRules: this.#manager.getSettings().builtinRules,
			disabledRules: this.#manager.getSettings().disabledRules,
		};
	}

	/** Add a rule to the manager. */
	addRule(rule: TtsrRule): boolean {
		return this.#manager.addRule(rule);
	}

	/** Get all registered rules. */
	getRules(): TtsrRule[] {
		return this.#manager.getRules();
	}

	/** Check if any rules are registered. */
	hasRules(): boolean {
		return this.#manager.hasRules();
	}

	/** Check if any rules have AST conditions. */
	hasAstRules(): boolean {
		return this.#manager.hasAstRules();
	}

	// ── Private helpers ───────────────────────────────────────────────────────

	/** Extract the delta text from a runtime event. */
	#extractDelta(event: RuntimeEvent): string | null {
		switch (event.type) {
			case "token":
				return ("token" in event && typeof event.token === "string")
					? event.token
					: null;
			case "thinking_token":
				return ("token" in event && typeof event.token === "string")
					? event.token
					: null;
			case "tool_call_update":
				return ("delta" in event && typeof event.delta === "string")
					? event.delta
					: null;
			case "tool_execution_update":
				return ("delta" in event && typeof event.delta === "string")
					? event.delta
					: null;
			default:
				return null;
		}
	}

	/** Build a match context from a runtime event. */
	#buildContext(event: RuntimeEvent): TtsrMatchContext {
		let source: TtsrMatchContext["source"] = "text";
		if (event.type === "thinking_token") {
			source = "thinking";
		}

		let toolName: string | undefined;
		let filePaths: string[] | undefined;
		let streamKey: string | undefined;

		if (event.type === "tool_call_update") {
			source = "tool";
			toolName = "toolName" in event && typeof event.toolName === "string" ? event.toolName : undefined;
			if (toolName) streamKey = `tool:${toolName}`;
		} else if (event.type === "tool_execution_update") {
			source = "tool";
			toolName = "toolName" in event ? event.toolName : undefined;
			if (toolName) streamKey = `tool:${toolName}`;
			filePaths = "filePaths" in event ? (event as { filePaths?: string[] }).filePaths : undefined;
		}

		return { source, toolName, filePaths, streamKey };
	}

	/** Process matching rules: abort, inject, and track. */
	#processMatches(rules: TtsrRule[], context: TtsrMatchContext): void {
		if (rules.length === 0) return;

		const seen = new Set<string>();
		const unique = rules.filter(r => {
			if (seen.has(r.name)) return false;
			seen.add(r.name);
			return true;
		});

		if (unique.length === 0) return;

		this.#manager.markInjected(unique);

		for (const rule of unique) {
			this.#activeRules.set(rule.name, {
				rule,
				injectedAt: this.#manager.getSettings().repeatMode === "once" ? 0 : this.#manager.getSettings().repeatGap,
			});

			const injectionText = this.#buildInjectionText(rule, context);

			const interruptMode = this.#manager.getSettings().interruptMode;
			const shouldInterrupt =
				interruptMode === "always" ||
				(interruptMode === "prose-only" && context.source !== "tool") ||
				(interruptMode === "tool-only" && context.source === "tool");

			if (shouldInterrupt) {
				this.#opts.abort();
				this.#opts.emit({
					type: "ttsr_injected",
					ruleName: rule.name,
					ruleContent: rule.content,
				});
				this.#opts.steer(injectionText);
			} else {
				this.#opts.followUp(injectionText);
				this.#opts.emit({
					type: "ttsr_queued",
					ruleName: rule.name,
				});
			}
		}
	}

	/** Build the steering injection text from a rule. */
	#buildInjectionText(rule: TtsrRule, context: TtsrMatchContext): string {
		const header = `⚡ TTSR Rule "${rule.name}" triggered:
`;
		const description = rule.description
			? `Description: ${rule.description}\n`
			: "";
		const body = rule.content
			? `Content: ${rule.content}\n`
			: "";
		const contextHint = context.toolName
			? `Context: tool="${context.toolName}"\n`
			: "";

		return `${header}${description}${body}${contextHint}---`;
	}
}
