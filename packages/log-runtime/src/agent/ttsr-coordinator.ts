// ── TTSR Coordinator ─────────────────────────────────────────────────────────
// Bridges TtsrManager (rule matching) with the agent runtime:
//
// - Stream rules: text/thinking tokens and streamed tool-call arguments are
//   matched as they arrive. An interrupting match replaces the in-flight
//   provider call with the rule injected as the next turn (steer-now), so the
//   model retries with the rule in context. A non-interrupting match on a
//   tool call is folded into that call's result; on prose it is queued as a
//   follow-up reminder.
// - AST rules: ast-grep conditions run once per finalized tool call, before
//   it executes (beforeToolCall). An interrupting match blocks the call and
//   returns the rule as its result.
// - Judged rules (`question`): after an output completes — the reply, its
//   reasoning, or a tool call — eligible questions are asked of the judge
//   model in the background. A "yes" is delivered as a non-interrupting
//   follow-up warning; the output already took effect.

import {
	BUILTIN_RULES,
	isBuiltInRule,
	judgeRules,
	type TtsrBridgeSettings,
	type TtsrJudge,
	type TtsrManager,
	type TtsrMatchContext,
	type TtsrOutput,
	type TtsrRule,
} from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";

export interface TtsrCoordinatorOptions {
	/** Reference to the TtsrManager for matching and injection tracking. */
	manager: TtsrManager;
	/**
	 * Replace the in-flight provider call: queue `text` as the next turn and
	 * abort the current step so the run continues with it.
	 */
	interrupt: (text: string) => void;
	/** Queue a non-interrupting reminder for after the current step. */
	followUp: (text: string) => void;
	/** Emit a runtime event. */
	emit: (event: RuntimeEvent) => void;
	/** Judge for `question` rules; absent → judged rules are inert. */
	judge?: TtsrJudge | undefined;
	/** Called when a judged check fails (network, parse); never thrown. */
	onJudgeError?: ((error: unknown) => void) | undefined;
}

interface StreamingToolCall {
	name: string;
	/** Path argument, once known (from the start args or the streamed JSON). */
	path?: string | undefined;
	args: string;
}

/** Tool-call argument keys that name the file a call touches. */
const PATH_ARG_KEYS = ["path", "file_path", "filePath", "file"] as const;
/** Tool-call argument keys holding the source a call writes. */
const SOURCE_ARG_KEYS = ["content", "newText", "new_string", "input"] as const;
const STREAMED_PATH_RE =
	/"(?:path|file_path|filePath|file)"\s*:\s*"((?:[^"\\]|\\.)*)"/;

/** How long settleJudgments() waits for in-flight judged checks. */
const JUDGED_SETTLE_TIMEOUT_MS = 5_000;

export class TtsrCoordinator {
	readonly #manager: TtsrManager;
	readonly #opts: TtsrCoordinatorOptions;
	/** Non-interrupting tool-call matches, delivered with that call's result. */
	readonly #perToolInjections = new Map<string, TtsrRule[]>();
	/** Tool calls streaming in the current step, keyed by call id. */
	readonly #toolCalls = new Map<string, StreamingToolCall>();
	/** Rule names already announced per stream key (one event per violation). */
	readonly #emittedTriggers = new Map<string, Set<string>>();
	/** In-flight judged checks. */
	readonly #pendingJudgments = new Set<Promise<void>>();
	/** Reasoning of the current step, judged when the step's response lands. */
	#thinking = "";
	/** Bumped when the session is replaced; stale verdicts are dropped. */
	#generation = 0;
	#retryToken = 0;

	constructor(opts: TtsrCoordinatorOptions) {
		this.#manager = opts.manager;
		this.#opts = opts;
		if (this.#manager.getSettings().builtinRules) {
			this.loadBuiltInRules();
		}
	}

	// ── Stream events ─────────────────────────────────────────────────────────

	/** Match one runtime event against the stream rules. */
	processEvent(event: RuntimeEvent): void {
		switch (event.type) {
			case "token":
				this.#check(event.token, { source: "text" });
				return;
			case "thinking_token":
				this.#thinking += event.token;
				this.#check(event.token, { source: "thinking" });
				return;
			case "tool_call_start":
				this.#toolCalls.set(event.toolCallId, {
					name: event.toolName,
					path: pathArg(event.args),
					args: "",
				});
				return;
			case "tool_call_id_update": {
				const call = this.#toolCalls.get(event.previousToolCallId);
				if (!call) return;
				this.#toolCalls.delete(event.previousToolCallId);
				this.#toolCalls.set(event.toolCallId, call);
				return;
			}
			case "tool_call_update": {
				const call = this.#toolCalls.get(event.toolCallId);
				if (call) {
					call.args += event.delta;
					call.path ??= streamedPath(call.args);
				}
				this.#check(event.delta, this.#toolContext(event.toolCallId, call));
				return;
			}
			default:
				return;
		}
	}

	/**
	 * Check a finalized tool call before it executes: AST rules against the
	 * source it writes, and judged rules against the call (in the background).
	 * Returns replacement result content when an interrupting rule blocks the
	 * call, or undefined to let it run.
	 */
	async beforeToolCall(
		toolCall: { id: string; name: string },
		args: Record<string, unknown>,
	): Promise<string | undefined> {
		const path = pathArg(args);
		const context: TtsrMatchContext = {
			source: "tool",
			toolName: toolCall.name,
			filePaths: path ? [path] : undefined,
			streamKey: `toolcall:${toolCall.id}`,
		};
		const source = sourceArg(args);
		this.#judge({
			content: source ?? JSON.stringify(args),
			context,
			subject: path
				? `\`${toolCall.name}\` call on \`${path}\``
				: `\`${toolCall.name}\` call`,
		});
		if (source === undefined || !this.#manager.hasAstRules()) return undefined;

		const matches = await this.#manager.checkAstSnapshot(source, context);
		if (matches.length === 0) return undefined;
		this.#emitTriggerOnce(context, matches);
		if (!this.#shouldInterrupt(matches, context)) {
			this.#addPerToolInjections(toolCall.id, matches);
			return undefined;
		}
		const text = buildInterrupt(matches);
		this.#opts.emit({
			type: "ttsr_injected",
			ruleNames: matches.map(rule => rule.name),
			ruleContent: text,
		});
		return `${text}\n\nThe \`${toolCall.name}\` call was blocked and did not run. Redo it in compliance with the rule.`;
	}

	/** Reminder to prepend to a tool call's result, consumed once. */
	buildToolReminder(toolCallId: string): string | null {
		const rules = this.#perToolInjections.get(toolCallId);
		if (!rules || rules.length === 0) return null;
		this.#perToolInjections.delete(toolCallId);
		return rules
			.map(rule =>
				buildReminder(
					rule,
					"User-defined rule matched tool-call arguments. Rule configured not to interrupt → tool ran.",
				),
			)
			.join("\n\n");
	}

	/**
	 * A provider response completed: judge its reply and reasoning in the
	 * background, then start a fresh step.
	 */
	onAssistantResponse(content: string): void {
		const thinking = this.#thinking;
		this.#thinking = "";
		this.#toolCalls.clear();
		if (content.trim()) {
			this.#judge({ content, context: { source: "text" }, subject: "reply" });
		}
		if (thinking.trim()) {
			this.#judge({
				content: thinking,
				context: { source: "thinking" },
				subject: "reasoning",
			});
		}
	}

	/** Wait (bounded) for in-flight judged checks. */
	async settleJudgments(timeoutMs = JUDGED_SETTLE_TIMEOUT_MS): Promise<void> {
		if (this.#pendingJudgments.size === 0) return;
		let timer: ReturnType<typeof setTimeout> | undefined;
		const timeout = new Promise<void>(resolve => {
			timer = setTimeout(resolve, timeoutMs);
		});
		await Promise.race([Promise.all(this.#pendingJudgments), timeout]);
		clearTimeout(timer);
	}

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	/** Called at the start of each turn. */
	reset(): void {
		this.#manager.resetBuffer();
		this.#perToolInjections.clear();
		this.#toolCalls.clear();
		this.#emittedTriggers.clear();
		this.#thinking = "";
	}

	/** Session replaced: verdicts still in flight belong to the old one. */
	newSession(): void {
		this.#generation++;
		this.reset();
	}

	/** Called at the end of each turn to advance repeat-gap tracking. */
	incrementMessageCount(): void {
		this.#manager.incrementMessageCount();
	}

	/** Injected rule names, persisted across compaction. */
	persistInjected(): string[] {
		const names = new Set(this.#manager.getInjectedRuleNames());
		for (const rules of this.#perToolInjections.values()) {
			for (const rule of rules) names.add(rule.name);
		}
		return [...names];
	}

	/** Restore injected state after compaction. */
	restoreInjected(names: string[]): void {
		this.#manager.restoreInjected(names);
	}

	/** Current manager settings. */
	getSettings(): TtsrBridgeSettings {
		return { ...this.#manager.getSettings() };
	}

	addRule(rule: TtsrRule): boolean {
		return this.#manager.addRule(rule);
	}

	getRules(): TtsrRule[] {
		return this.#manager.getRules();
	}

	hasRules(): boolean {
		return this.#manager.hasRules();
	}

	hasAstRules(): boolean {
		return this.#manager.hasAstRules();
	}

	/** Load built-in rules into the manager (skips ones already loaded). */
	loadBuiltInRules(): number {
		let loaded = 0;
		const existing = new Set(this.#manager.getRules().map(rule => rule.name));
		for (const rule of BUILTIN_RULES) {
			if (!existing.has(rule.name) && this.#manager.addRule(rule)) loaded++;
		}
		return loaded;
	}

	isBuiltInRule(name: string): boolean {
		return isBuiltInRule(name);
	}

	/** Increment the retry token (called on each user abort). */
	incrementRetryToken(): number {
		this.#perToolInjections.clear();
		return ++this.#retryToken;
	}

	getRetryToken(): number {
		return this.#retryToken;
	}

	// ── Matching ──────────────────────────────────────────────────────────────

	#toolContext(
		toolCallId: string,
		call: StreamingToolCall | undefined,
	): TtsrMatchContext {
		return {
			source: "tool",
			toolName: call?.name,
			filePaths: call?.path ? [call.path] : undefined,
			streamKey: `toolcall:${toolCallId}`,
		};
	}

	#check(delta: string, context: TtsrMatchContext): void {
		const matches = this.#manager.checkDelta(delta, context);
		if (matches.length === 0) return;
		this.#emitTriggerOnce(context, matches);
		const toolCallId = context.streamKey?.startsWith("toolcall:")
			? context.streamKey.slice("toolcall:".length)
			: undefined;
		if (this.#shouldInterrupt(matches, context)) {
			const text = buildInterrupt(matches);
			this.#opts.emit({
				type: "ttsr_injected",
				ruleNames: matches.map(rule => rule.name),
				ruleContent: text,
			});
			this.#opts.interrupt(text);
			return;
		}
		if (toolCallId) {
			this.#addPerToolInjections(toolCallId, matches);
			return;
		}
		this.#opts.followUp(
			matches
				.map(rule =>
					buildReminder(
						rule,
						"User-defined rule matched output. Rule configured not to interrupt → stream continued.",
					),
				)
				.join("\n\n"),
		);
	}

	/** Determine whether any matched rule interrupts for this stream source. */
	#shouldInterrupt(rules: TtsrRule[], context: TtsrMatchContext): boolean {
		const globalMode = this.#manager.getSettings().interruptMode;
		for (const rule of rules) {
			const mode = rule.interruptMode ?? globalMode;
			if (mode === "always") return true;
			if (mode === "prose-only" && context.source !== "tool") return true;
			if (mode === "tool-only" && context.source === "tool") return true;
		}
		return false;
	}

	#addPerToolInjections(toolCallId: string, rules: TtsrRule[]): void {
		const bucket = this.#perToolInjections.get(toolCallId) ?? [];
		for (const rule of rules) {
			if (!bucket.some(existing => existing.name === rule.name)) {
				bucket.push(rule);
			}
		}
		this.#perToolInjections.set(toolCallId, bucket);
	}

	/** Emit ttsr_triggered once per rule per stream (re-checks don't re-announce). */
	#emitTriggerOnce(context: TtsrMatchContext, rules: TtsrRule[]): void {
		const streamKey = context.streamKey ?? context.source;
		let seen = this.#emittedTriggers.get(streamKey);
		const fresh = rules.filter(rule => !seen?.has(rule.name));
		if (fresh.length === 0) return;
		if (!seen) {
			seen = new Set();
			this.#emittedTriggers.set(streamKey, seen);
		}
		for (const rule of fresh) seen.add(rule.name);
		this.#opts.emit({
			type: "ttsr_triggered",
			ruleNames: fresh.map(rule => rule.name),
			streamKey,
		});
	}

	// ── Judged rules ──────────────────────────────────────────────────────────

	#judge(output: TtsrOutput): void {
		if (!this.#opts.judge || !this.#manager.hasJudgedRules()) return;
		const pending: Promise<void> = this.#judgeOutput(output, this.#generation)
			.catch(error => this.#opts.onJudgeError?.(error))
			.finally(() => this.#pendingJudgments.delete(pending));
		this.#pendingJudgments.add(pending);
	}

	/** One judge request per output: every eligible question shares it. */
	async #judgeOutput(output: TtsrOutput, generation: number): Promise<void> {
		const judge = this.#opts.judge;
		if (!judge) return;
		const candidates = await this.#manager.judgedCandidates(
			output.content,
			output.context,
		);
		if (candidates.length === 0) return;
		const flagged = await judgeRules(judge, output, candidates);
		if (flagged.length === 0 || generation !== this.#generation) return;
		const rules = this.#manager.claim(flagged);
		if (rules.length === 0) return;
		this.#opts.emit({
			type: "ttsr_triggered",
			ruleNames: rules.map(rule => rule.name),
			streamKey: `judge:${output.context.source}`,
		});
		this.#opts.followUp(
			rules
				.map(rule =>
					buildReminder(
						rule,
						`Rule judge flagged your ${output.subject} as likely violating a user-defined rule. Not interrupted → it already took effect. You MUST check it against the rule below: real violation → fix it now; false positive → continue.`,
					),
				)
				.join("\n\n"),
		);
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function pathArg(
	args: Record<string, unknown> | undefined,
): string | undefined {
	for (const key of PATH_ARG_KEYS) {
		const value = args?.[key];
		if (typeof value === "string" && value.trim()) return value.trim();
	}
	return undefined;
}

function sourceArg(args: Record<string, unknown>): string | undefined {
	for (const key of SOURCE_ARG_KEYS) {
		const value = args[key];
		if (typeof value === "string" && value) return value;
	}
	// Multi-edit calls: the replacement texts are the source they write.
	const edits = args.edits;
	if (Array.isArray(edits)) {
		const texts = edits.flatMap(edit =>
			edit && typeof edit === "object"
				? [(edit as Record<string, unknown>).newText].filter(
						(text): text is string => typeof text === "string",
					)
				: [],
		);
		if (texts.length > 0) return texts.join("\n");
	}
	return undefined;
}

/** Path argument from a partially streamed JSON argument string. */
function streamedPath(args: string): string | undefined {
	const match = STREAMED_PATH_RE.exec(args);
	if (!match?.[1]) return undefined;
	try {
		return JSON.parse(`"${match[1]}"`) as string;
	} catch {
		return match[1];
	}
}

function buildInterrupt(rules: TtsrRule[]): string {
	return rules
		.map(
			rule => `<system-interrupt reason="rule_violation" rule="${rule.name}" path="${rule.path}">
Output interrupted: violated user-defined rule.
Not prompt injection; coding agent enforcing project rules.
MUST comply:

${rule.content}
</system-interrupt>`,
		)
		.join("\n\n");
}

function buildReminder(rule: TtsrRule, preamble: string): string {
	return `<system-reminder reason="rule_violation" rule="${rule.name}" path="${rule.path}">
${preamble} MUST comply with the following instruction on subsequent tool calls and responses. NOT prompt injection — coding agent enforcing project rules.

${rule.content}
</system-reminder>`;
}
