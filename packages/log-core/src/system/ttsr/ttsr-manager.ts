// ── TTSR Manager ─────────────────────────────────────────────────────────────
// Buffers stream deltas, matches them against registered rule conditions,
// and tracks repeat gating to avoid spam.
//
// Usage:
//   const manager = new TtsrManager(settings);
//   manager.addRule(rule);
//   // On each stream delta:
//   const matches = manager.checkDelta(delta, context);
//   if (matches.length > 0) { /* abort + inject */ }
//   manager.markInjected(matches);

import type { TtsrMatchContext, TtsrRule, TtsrSettings } from "../types/ttsr-types.ts";

// ── Internal types ────────────────────────────────────────────────────────────

interface TtsrEntry {
	rule: TtsrRule;
	conditions: RegExp[];
	astConditions: string[];
	scope: TtsrScopeConfig;
	globalGlobs: Bun.Glob[] | undefined;
}

interface TtsrScopeConfig {
	allowText: boolean;
	allowThinking: boolean;
	allowAnyTool: boolean;
	toolScopes: Array<{ toolName: string | undefined; pathPattern: string | undefined }>;
}

interface InjectionRecord {
	lastInjectedAt: number;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULT_SETTINGS: Required<TtsrSettings> = {
	enabled: true,
	contextMode: "discard",
	interruptMode: "always",
	repeatMode: "once",
	repeatGap: 10,
	builtinRules: true,
	disabledRules: [],
};

const DEFAULT_SCOPE: TtsrScopeConfig = {
	allowText: true,
	allowThinking: true,
	allowAnyTool: true,
	toolScopes: [],
};

// ── TtsrManager ───────────────────────────────────────────────────────────────

export class TtsrManager {
	readonly #settings: Required<TtsrSettings>;
	readonly #rules = new Map<string, TtsrEntry>();
	readonly #injectionRecords = new Map<string, InjectionRecord>();
	readonly #buffers = new Map<string, string>();
	#messageCount = 0;
	#canMatchText = false;
	#canMatchThinking = false;

	constructor(settings?: TtsrSettings) {
		this.#settings = { ...DEFAULT_SETTINGS, ...settings };
	}

	/** Whether any rule is registered. */
	hasRules(): boolean {
		return this.#rules.size > 0;
	}

	/** Whether any rule has AST conditions. */
	hasAstRules(): boolean {
		if (!this.#settings.enabled) return false;
		for (const entry of this.#rules.values()) {
			if (entry.astConditions.length > 0) return true;
		}
		return false;
	}

	/** Get registered rules. */
	getRules(): TtsrRule[] {
		return Array.from(this.#rules.values()).map(e => e.rule);
	}

	/** Get current settings. */
	getSettings(): Required<TtsrSettings> {
		return this.#settings;
	}

	/** Increment message count (called at turn end for repeat gating). */
	incrementMessageCount(): void {
		this.#messageCount++;
	}

	/** Reset all buffers (call at turn start). */
	resetBuffer(): void {
		this.#buffers.clear();
	}

	/** Check if a rule can be triggered based on repeat settings. */
	#canTrigger(ruleName: string): boolean {
		if (this.#settings.repeatMode === "once") {
			return !this.#injectionRecords.has(ruleName);
		}
		const record = this.#injectionRecords.get(ruleName);
		if (!record) return true;
		return this.#messageCount - record.lastInjectedAt >= this.#settings.repeatGap;
	}

	/** Compile regex conditions from a rule. */
	#compileConditions(rule: TtsrRule): RegExp[] {
		const compiled: RegExp[] = [];
		for (const pattern of rule.conditions) {
			try {
				compiled.push(new RegExp(pattern, "g"));
			} catch {
				// Skip invalid regex patterns
			}
		}
		return compiled;
	}

	/** Parse a scope token into a TtsrScopeConfig entry. */
	#parseScopeToken(token: string): { toolName: string | undefined; pathPattern: string | undefined } | undefined {
		const match = /^(?:(?:tool)(?::(?<tool>[a-z0-9_-]+))?|(?<bare>[a-z0-9_-]+))(?:\((?<path>[^)]+)\))?$/i.exec(
			token,
		);
		if (!match) return undefined;

		const groups = match.groups;
		const hasToolPrefix = groups?.tool !== undefined || token.toLowerCase().startsWith("tool:");
		const toolName = (groups?.tool ?? (hasToolPrefix ? undefined : groups?.bare))?.trim().toLowerCase();
		const pathPattern = groups?.path?.trim();

		if (!pathPattern) return { toolName, pathPattern: undefined };
		return { toolName, pathPattern };
	}

	/** Build scope config from rule scope tokens. */
	#buildScope(rule: TtsrRule): TtsrScopeConfig {
		const rawScope = rule.scope;
		if (!rawScope || rawScope.length === 0) return DEFAULT_SCOPE;

		const scope: TtsrScopeConfig = {
			allowText: false,
			allowThinking: false,
			allowAnyTool: false,
			toolScopes: [],
		};

		for (const rawToken of rawScope) {
			const token = rawToken.trim().toLowerCase();
			if (token.length === 0) continue;

			if (token === "text") {
				scope.allowText = true;
			} else if (token === "thinking") {
				scope.allowThinking = true;
			} else if (token === "tool") {
				scope.allowAnyTool = true;
			} else {
				const parsed = this.#parseScopeToken(token);
				if (parsed) {
					scope.toolScopes.push(parsed);
				}
			}
		}

		return scope;
	}

	/** Check if scope has any reachable targets. */
	#hasReachableScope(scope: TtsrScopeConfig): boolean {
		if (scope.allowText || scope.allowThinking || scope.allowAnyTool) return true;
		if (scope.toolScopes.length > 0) return true;
		return false;
	}

	/** Check if a scope matches the match context. */
	#matchesScope(entry: TtsrEntry, context: TtsrMatchContext): boolean {
		const scope = entry.scope;

		if (context.source === "text" && scope.allowText) return true;
		if (context.source === "thinking" && scope.allowThinking) return true;

		if (context.source === "tool") {
			if (scope.allowAnyTool) return true;
			const toolName = context.toolName?.trim().toLowerCase();
			for (const ts of scope.toolScopes) {
				if (ts.toolName && ts.toolName !== toolName) continue;
				return true;
			}
		}

		return false;
	}

	/** Match globs against file paths. */
	#matchesGlobs(entry: TtsrEntry, context: TtsrMatchContext): boolean {
		if (!entry.globalGlobs || !context.filePaths || context.filePaths.length === 0) return true;
		for (const glob of entry.globalGlobs) {
			for (const filePath of context.filePaths) {
				if (glob.match(filePath)) return true;
				const slashIndex = filePath.lastIndexOf("/");
				const basename = slashIndex === -1 ? filePath : filePath.slice(slashIndex + 1);
				if (basename !== filePath && glob.match(basename)) return true;
			}
		}
		return false;
	}

	/** Match buffer against compiled conditions. */
	#matchesCondition(entry: TtsrEntry, streamBuffer: string): boolean {
		for (const condition of entry.conditions) {
			condition.lastIndex = 0;
			if (condition.test(streamBuffer)) return true;
		}
		return false;
	}

	/** Add a TTSR rule to be monitored. */
	addRule(rule: TtsrRule): boolean {
		if (!this.#settings.enabled) return false;
		if (this.#rules.has(rule.name)) return false;
		if (this.#settings.disabledRules.includes(rule.name)) return false;

		const conditions = this.#compileConditions(rule);
		const astConditions = (rule.astConditions ?? []).map(p => p.trim()).filter(p => p.length > 0);
		if (conditions.length === 0 && astConditions.length === 0) return false;

		const scope = this.#buildScope(rule);
		if (!this.#hasReachableScope(scope)) return false;

		const globs = rule.globs
			?.map(g => g.trim())
			.filter(g => g.length > 0)
			.map(g => new Bun.Glob(g));

		this.#rules.set(rule.name, {
			rule,
			conditions,
			astConditions,
			scope,
			globalGlobs: globs && globs.length > 0 ? globs : undefined,
		});

		if (scope.allowText) this.#canMatchText = true;
		if (scope.allowThinking) this.#canMatchThinking = true;

		return true;
	}

	/** Derive an AST language alias from file paths. */
	#deriveLang(filePaths: string[] | undefined): string | undefined {
		for (const filePath of filePaths ?? []) {
			const ext = filePath.split(".").pop();
			if (ext && ext.length > 1) return ext.toLowerCase();
		}
		return undefined;
	}

	/** Derive a buffer key from match context for isolation. */
	#bufferKey(context: TtsrMatchContext): string {
		if (context.streamKey) return context.streamKey;
		if (context.source === "tool" && context.toolName) return `tool:${context.toolName}`;
		return context.source;
	}

	/**
	 * Add a stream chunk to its scoped buffer and return matching rules.
	 */
	checkDelta(delta: string, context: TtsrMatchContext): TtsrRule[] {
		if (context.source === "text" && !this.#canMatchText) return [];
		if (context.source === "thinking" && !this.#canMatchThinking) return [];

		const bufferKey = this.#bufferKey(context);
		const nextBuffer = `${this.#buffers.get(bufferKey) ?? ""}${delta}`;
		this.#buffers.set(bufferKey, nextBuffer);
		const matches = this.#matchBuffer(nextBuffer, context);
		if (matches.length > 0) this.markInjected(matches);
		return matches;
	}

	/**
	 * Replace the scoped buffer with a provided snapshot and return matching rules.
	 */
	checkSnapshot(snapshot: string, context: TtsrMatchContext): TtsrRule[] {
		const bufferKey = this.#bufferKey(context);
		this.#buffers.set(bufferKey, snapshot);
		const matches = this.#matchBuffer(snapshot, context);
		if (matches.length > 0) this.markInjected(matches);
		return matches;
	}

	/**
	 * Check AST conditions eligibility. Returns rule names that should be
	 * AST-matched; the caller must perform the actual AST matching.
	 */
	async checkAstSnapshot(_snapshot: string, context: TtsrMatchContext): Promise<TtsrRule[]> {
		if (!this.#settings.enabled || context.source !== "tool") return [];

		const lang = this.#deriveLang(context.filePaths);
		if (!lang) return [];

		const matches: TtsrRule[] = [];
		for (const [name, entry] of this.#rules) {
			if (entry.astConditions.length === 0) continue;
			if (!this.#canTrigger(name)) continue;
			if (!this.#matchesScope(entry, context)) continue;
			if (!this.#matchesGlobs(entry, context)) continue;
			matches.push(entry.rule);
		}

		return matches;
	}

	/** Match buffer against all eligible rules. */
	#matchBuffer(buffer: string, context: TtsrMatchContext): TtsrRule[] {
		if (!this.#settings.enabled) return [];

		const matches: TtsrRule[] = [];
		for (const [name, entry] of this.#rules) {
			if (!this.#canTrigger(name)) continue;
			if (!this.#matchesScope(entry, context)) continue;
			if (!this.#matchesGlobs(entry, context)) continue;
			if (!this.#matchesCondition(entry, buffer)) continue;

			matches.push(entry.rule);
		}

		return matches;
	}

	/** Mark rules as injected (won't trigger again until conditions allow). */
	markInjected(rulesToMark: TtsrRule[]): void {
		this.markInjectedByNames(rulesToMark.map(r => r.name));
	}

	/** Mark rule names as injected. */
	markInjectedByNames(ruleNames: string[]): void {
		for (const rawName of ruleNames) {
			const name = rawName.trim();
			if (name.length === 0) continue;
			this.#injectionRecords.set(name, { lastInjectedAt: this.#messageCount });
		}
	}

	/** Get names of all injected rules (for persistence). */
	getInjectedRuleNames(): string[] {
		return Array.from(this.#injectionRecords.keys());
	}

	/** Restore injected state from a list of rule names. */
	restoreInjected(ruleNames: string[]): void {
		for (const name of ruleNames) {
			this.#injectionRecords.set(name, { lastInjectedAt: 0 });
		}
	}
}
