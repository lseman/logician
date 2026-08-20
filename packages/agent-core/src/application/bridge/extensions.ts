/** Owns extension discovery, lifecycle, compatibility, and contributed commands. */

import {
	readdir as readdirAsync,
	readFile as readFileAsync,
} from "node:fs/promises";
import path from "node:path";
import {
	type PiRuntimePort,
	piCompatibilityAdapter,
} from "../../adapters/pi/index.ts";
import type { Skill } from "../../capabilities/skills/loader.ts";
import { loadExtensions } from "../../core/extension/loader.ts";
import { ExtensionRunner } from "../../core/extension/runner.ts";
import { parseFrontmatter } from "../../infrastructure/tools/utils/frontmatter.ts";

// ── Options ────────────────────────────────────────────────────────────────────

export interface ExtensionManagerOptions {
	sessionId: string;
	cwd: string;
	extensionDirs?: { user?: string; paths?: string[] };
	projectTrusted: boolean;
	/** Pi runtime callbacks the runner needs at call time. */
	piRuntime: PiRuntimePort;
}

// ── ExtensionManager class ─────────────────────────────────────────────────────

export class ExtensionManager {
	private _runner: ExtensionRunner | null = null;
	private extensionDirs?: { user?: string; paths?: string[] };
	private loadPromise: Promise<void> = Promise.resolve();

	constructor(private readonly opts: ExtensionManagerOptions) {}

	/** Access the internal runner for harness wiring. */
	get runner(): ExtensionRunner | null {
		return this._runner;
	}

	/** Whether the manager has been initialized. */
	isInitialized(): boolean {
		return this._runner !== null;
	}

	/** Create the runner and load extensions. Called once at construction / reload. */
	async initialize(): Promise<void> {
		if (this._runner) return;

		this._runner = new ExtensionRunner({
			sessionId: this.opts.sessionId,
			cwd: this.opts.cwd,
			compatibilityAdapters: [piCompatibilityAdapter],
			compatibilityRuntime: this.opts.piRuntime,
		});

		const extResult = loadExtensions({
			userDir: this.opts.extensionDirs?.user,
			projectDir: this.opts.projectTrusted ? this.opts.cwd : undefined,
			explicitPaths: this.opts.extensionDirs?.paths,
		});

		this.extensionDirs = this.opts.extensionDirs;

		if (extResult.extensions.length > 0) {
			this.loadPromise = this._runner
				.load(extResult.extensions)
				.catch(err => console.error("[logician] extension load error:", err));
		}
	}

	/** Reload extensions (used by /reload). Discards old runner and discovers fresh. */
	async reload(): Promise<void> {
		if (!this._runner) return;

		// Destroy old runner
		this._runner.destroy();
		this._runner = null;
		// Re-initialize
		await this.initialize();
	}

	/** Wait for extensions to finish loading. */
	getLoadPromise(): Promise<void> {
		return this.loadPromise;
	}

	// ── Commands ─────────────────────────────────────────────────────────────

	/** Get registered extension commands. */
	getCommands(): Array<{
		name: string;
		description: string;
		usage?: string;
		acceptsArgs?: boolean;
	}> {
		return (this._runner?.getCommands() ?? []).map(cmd => ({
			name: cmd.name,
			description: cmd.description,
			usage: cmd.usage,
			acceptsArgs: cmd.acceptsArgs,
		}));
	}

	/** Execute a registered extension command. */
	async executeCommand(
		name: string,
		args: string,
	): Promise<string | undefined> {
		return (
			this._runner?.executeCommand(name, args) ?? Promise.resolve(undefined)
		);
	}

	// ── Pi event emission ───────────────────────────────────────────────────

	/**
	 * Emit a Pi input event to all loaded Pi adapters.
	 * @returns action result or null (default: continue).
	 */
	async emitInputEvent(
		text: string,
		images: unknown[] = [],
		source: "interactive" | "rpc" | "extension" = "interactive",
	): Promise<{
		action: "continue" | "transform" | "handled";
		text?: string;
		images?: unknown[];
	} | null> {
		if (!this._runner) return null;
		return this._runner.emitInputEvent(text, images, source);
	}

	/**
	 * Emit a Pi user_bash event to all loaded Pi adapters.
	 * @returns action result or null (default: continue).
	 */
	async emitUserBashEvent(
		command: string,
		excludeFromContext: boolean = false,
	): Promise<{
		action: "continue" | "intercept" | "replace";
		result?: { output: string; exitCode: number; cancelled: boolean };
		operations?: unknown;
	} | null> {
		if (!this._runner) return null;
		return this._runner.emitUserBashEvent(command, excludeFromContext);
	}

	/**
	 * Emit a Pi project_trust event to all loaded Pi adapters.
	 * @returns trust result or null (default: continue).
	 */
	async emitProjectTrustEvent(cwd: string): Promise<{
		trusted: "yes" | "no" | "undecided";
		remember?: boolean;
	} | null> {
		if (!this._runner) return null;
		return this._runner.emitProjectTrustEvent(cwd);
	}
}

// ── Plugin command loading ────────────────────────────────────────────────────

/**
 * Load skill-like command definitions from plugin command directories.
 * Each enabled plugin may have a `commands/` subdirectory with .md files.
 */
export async function loadPluginCommands(
	plugins: Array<{ name: string; installPath: string }>,
): Promise<Skill[]> {
	const out: Skill[] = [];
	for (const { name: pluginName, installPath } of plugins) {
		const dir = path.join(installPath, "commands");
		let entries: string[];
		try {
			entries = await readdirAsync(dir);
		} catch {
			continue;
		}
		for (const entry of entries) {
			if (!entry.endsWith(".md")) continue;
			const filePath = path.join(dir, entry);
			let raw: string;
			try {
				raw = await readFileAsync(filePath, "utf8");
			} catch {
				continue;
			}
			const parsed = parseFrontmatter<Record<string, unknown>>(raw);
			const frontmatter = parsed.ok ? parsed.value.frontmatter : {};
			const body = parsed.ok ? parsed.value.body : raw;
			const cmdName = entry.slice(0, -3);
			const description =
				typeof frontmatter.description === "string" &&
				frontmatter.description.trim()
					? frontmatter.description
					: `Command from the ${pluginName} plugin.`;
			out.push({
				name: `${pluginName}:${cmdName}`,
				displayName: cmdName,
				description,
				content: body,
				filePath,
				baseDir: dir,
				slashName: `${pluginName}:${cmdName}`,
				disableModelInvocation: true,
				aliases: [cmdName],
				source: "path",
			});
		}
	}
	return out;
}
