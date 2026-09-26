// ── TTSR Rule Loader ──────────────────────────────────────────────────────────
// Loads built-in rules, user-defined rules, and discovered external rules
// (Cursor, Cline, Windsurf, GitHub Copilot, .agent/ dirs) into a TtsrManager.
// Handles rule merging, deduplication, and validation.

import type { TtsrRule, TtsrSettings } from "../types/ttsr.ts";
import { BUILTIN_RULES } from "./built-in-rules.ts";
import { discoverRules } from "./discovery.ts";
import type { TtsrManager } from "./ttsr-manager.ts";

// ── Loader state ──────────────────────────────────────────────────────────────

interface LoadedRule {
	rule: TtsrRule;
	source: "builtin" | "user";
}

// ── RuleLoader ────────────────────────────────────────────────────────────────

export class RuleLoader {
	readonly #manager: TtsrManager;
	readonly #loadedRules = new Map<string, LoadedRule>();
	#settings: Required<TtsrSettings>;

	constructor(manager: TtsrManager, settings?: Partial<TtsrSettings>) {
		this.#manager = manager;
		this.#settings = {
			enabled: true,
			contextMode: "discard",
			interruptMode: "always",
			repeatMode: "once",
			repeatGap: 10,
			builtinRules: true,
			disabledRules: [],
			judge: true,
			...settings,
		};
	}

	/** Load all rules (built-in + user + discovered) into the manager. */
	async loadAll(userRules?: TtsrRule[], cwd?: string): Promise<number> {
		let loaded = 0;

		if (this.#settings.builtinRules) {
			loaded += this.loadBuiltIn();
		}

		if (userRules) {
			loaded += this.loadUserRules(userRules);
		}

		// Discover and load external rules from config formats
		if (cwd) {
			const { rules } = await discoverRules(cwd);
			loaded += this.loadUserRules(rules);
		}

		return loaded;
	}

	/** Load built-in rules into the manager. */
	loadBuiltIn(): number {
		let loaded = 0;
		for (const rule of BUILTIN_RULES) {
			if (this.#settings.disabledRules.includes(rule.name)) continue;
			const success = this.#manager.addRule(rule);
			if (success) {
				this.#loadedRules.set(rule.name, { rule, source: "builtin" });
				loaded++;
			}
		}
		return loaded;
	}

	/** Load user-defined rules into the manager. */
	loadUserRules(userRules: TtsrRule[]): number {
		let loaded = 0;
		for (const rule of userRules) {
			if (this.#settings.disabledRules.includes(rule.name)) continue;
			const success = this.#manager.addRule(rule);
			if (success) {
				this.#loadedRules.set(rule.name, { rule, source: "user" });
				loaded++;
			}
		}
		return loaded;
	}

	/** Get all loaded rules with their source. */
	getLoadedRules(): LoadedRule[] {
		return Array.from(this.#loadedRules.values());
	}

	/** Check if a specific rule is loaded. */
	isLoaded(ruleName: string): boolean {
		return this.#loadedRules.has(ruleName);
	}

	/** Get the source of a loaded rule. */
	getSource(ruleName: string): "builtin" | "user" | undefined {
		const entry = this.#loadedRules.get(ruleName);
		return entry?.source;
	}

	/** Remove a user-loaded rule from the manager. */
	unloadUserRule(ruleName: string): boolean {
		const entry = this.#loadedRules.get(ruleName);
		if (!entry || entry.source !== "user") return false;

		this.#loadedRules.delete(ruleName);
		return true;
	}

	/** Clear all user-loaded rules, keeping built-ins. */
	clearUserRules(): void {
		for (const [name, entry] of this.#loadedRules) {
			if (entry.source === "user") {
				this.#loadedRules.delete(name);
			}
		}
	}

	/** Reload all rules (useful after settings change). */
	reload(userRules?: TtsrRule[]): void {
		this.#loadedRules.clear();
		this.loadAll(userRules);
	}

	/** Get the current settings. */
	getSettings(): Required<TtsrSettings> {
		return this.#settings;
	}

	/** Update settings and reload if needed. */
	updateSettings(patch: Partial<TtsrSettings>): void {
		const hadBuiltins = this.#settings.builtinRules;
		this.#settings = { ...this.#settings, ...patch };

		if (
			patch.builtinRules !== undefined &&
			patch.builtinRules !== hadBuiltins
		) {
			// Reload to apply builtin toggle
			this.reload();
		}
	}
}
