// ── ToolRouter ────────────────────────────────────────────────────────────────
// Owns "which tools exist at runtime" and the context they contribute to the
// system prompt: default tools, MCP loading, sandbox mode, skill/prompt
// discovery and injection, and the shared default-tools ToolRegistry.
// Extracted from agent-bridge.ts. System-prompt *assembly* (merging this
// router's mcpSystemContext/skillsContext with plugin-hook context) stays on
// the bridge — that merge is cross-cutting, not a tool-management concern.

import {
	readdir as readdirAsync,
	readFile as readFileAsync,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { Tool } from "../core/types/types-messages.ts";
import {
	parseFrontmatter,
	runPluginBackend,
	ToolRegistry,
} from "../infrastructure/tools/index.ts";
import {
	McpManager,
	type McpSnapshotResult,
	type McpToggleResult,
} from "../features/mcp/index.ts";
import { loadPrompts, type Prompt } from "../features/prompts/index.ts";
import type { RuntimeEvent } from "../runtime/events.ts";
import { formatSkillCatalog, loadSkills, type Skill } from "../features/skills/index.ts";
import { ariadne } from "../infrastructure/tools/ariadne.ts";
import { createDefaultTools } from "../infrastructure/tools/default-tools.ts";
import { createReadSkillTool } from "../infrastructure/tools/read-skill.ts";
import {
	getDefaultSandboxProfile,
	type SandboxProfile,
	setDefaultSandboxProfile,
} from "../infrastructure/tools/sandbox.ts";
import { resolveWebSearchConfig } from "./bridge-environment.ts";
import {
	getProjectPromptDirs,
	getProjectSkillDirs,
} from "./resource-directories.ts";

export interface ToolRouterDeps {
	cwd: string;
	projectTrusted: boolean;
	tools?: Tool[];
	extraTools?: Tool[];
	webSearch?: Partial<{ baseUrl: string; maxResults: number }>;
	ariadneEnabled?: boolean;
	fffgrepEnabled?: boolean;
	emit: (event: RuntimeEvent) => void;
	/** Add a tool to the live default set (propagates into config.tools / harness.setTools). */
	onToolAdded: (tool: Tool) => void;
	/** MCP/skills context changed (even with no new tools) — bridge should rebuild the system prompt. */
	onContextChanged: () => void;
	/**
	 * Start MCP discovery in the background the moment this router is
	 * constructed, instead of waiting for the first caller that needs it.
	 * Defaults to true for real sessions — this is a construction-time
	 * side-effect switch only. Set false to keep construction free of network/
	 * subprocess side effects — tests that stub `mcpManager` or
	 * `loadMcpToolsOnce()` after construction need this, since otherwise the
	 * real load can already be in flight (and win the memoized promise) by
	 * the time the stub is installed.
	 */
	autoStartMcp?: boolean;
}

/** Snapshot of MCP/skill state as reported by getState()/init(). */
export interface ToolRouterStatus {
	mcpServerCount: number;
	mcpToolCount: number;
	mcpErrors: string[];
	mcpLoaded: boolean;
	mcpLoading: boolean;
	skillsInjected: boolean;
	skillsVisible: boolean;
	loadedSkills: Skill[];
	enabledPluginRoots: Array<{ name: string; installPath: string }>;
}

export class ToolRouter {
	private readonly cwd: string;
	private readonly projectTrusted: boolean;
	private readonly emit: (event: RuntimeEvent) => void;
	private readonly onToolAdded: (tool: Tool) => void;
	private readonly onContextChanged: () => void;

	private defaultTools: Tool[];
	private readonly mcpManager = new McpManager();
	private mcpLoaded = false;
	private mcpLoadPromise: Promise<void> | null = null;
	private mcpServerCount = 0;
	private mcpErrors: string[] = [];
	private mcpToolNames = new Set<string>();
	private disabledFffTools: Tool[] = [];
	private fffgrepEnabled: boolean;
	private mcpSystemContext = "";

	private skillsInjected = false;
	private skillsContext: string | null = null;
	private loadedSkills: Skill[] = [];
	private enabledPluginRoots: Array<{ name: string; installPath: string }> = [];

	private promptsInjected = false;
	private loadedPrompts: Prompt[] = [];

	private static readonly SANDBOX_CYCLE: SandboxProfile[] = [
		"none",
		"code",
		"full",
	];

	constructor(deps: ToolRouterDeps) {
		this.cwd = deps.cwd;
		this.projectTrusted = deps.projectTrusted;
		this.emit = deps.emit;
		this.onToolAdded = deps.onToolAdded;
		this.onContextChanged = deps.onContextChanged;
		this.fffgrepEnabled = deps.fffgrepEnabled !== false;
		const defaultWebSearch = resolveWebSearchConfig();
		const webSearch = {
			baseUrl: deps.webSearch?.baseUrl || defaultWebSearch.baseUrl,
			maxResults: deps.webSearch?.maxResults ?? defaultWebSearch.maxResults,
		};
		this.defaultTools = deps.tools?.length
			? deps.tools
			: createDefaultTools({ webSearch, ariadneEnabled: deps.ariadneEnabled });
		if (deps.extraTools?.length) {
			this.defaultTools = [
				...this.defaultTools,
				...deps.extraTools.filter(
					tool =>
						!this.defaultTools.some(existing => existing.name === tool.name),
				),
			];
		}

		// Fire-and-forget: start MCP connections as soon as Logician opens,
		// without blocking the first turn. Opt out (autoStartMcp: false) to keep construction
		// free of side effects — see ToolRouterDeps.
		if (deps.autoStartMcp !== false) void this.loadMcpToolsOnce();
	}

	// ── Default tools ────────────────────────────────────────────────────

	getDefaultTools(): Tool[] {
		return this.defaultTools;
	}

	setAriadneEnabled(enabled: boolean): void {
		const hasAriadne = this.defaultTools.some(
			tool => tool.name === ariadne.name,
		);
		if (enabled === hasAriadne) return;
		this.defaultTools = enabled
			? [ariadne, ...this.defaultTools]
			: this.defaultTools.filter(tool => tool.name !== ariadne.name);
		this.onContextChanged();
	}

	setFffgrepEnabled(enabled: boolean): void {
		this.fffgrepEnabled = enabled;
		if (enabled) {
			if (!this.disabledFffTools.length) return;
			this.defaultTools = [...this.defaultTools, ...this.disabledFffTools];
			this.disabledFffTools = [];
		} else {
			const fffTools = this.defaultTools.filter(tool =>
				/^fff(?:__|_)grep$/i.test(tool.name),
			);
			if (!fffTools.length) return;
			this.disabledFffTools = fffTools;
			this.defaultTools = this.defaultTools.filter(
				tool => !fffTools.some(fff => fff.name === tool.name),
			);
		}
		this.onContextChanged();
	}

	/** Append a tool to the router's own set and notify the bridge to propagate it into config/harness/system prompt. */
	private addTool(tool: Tool): void {
		if (this.defaultTools.some(t => t.name === tool.name)) return;
		if (!this.fffgrepEnabled && /^fff(?:__|_)grep$/i.test(tool.name)) {
			this.disabledFffTools = [...this.disabledFffTools, tool];
			return;
		}
		this.defaultTools = [...this.defaultTools, tool];
		this.onToolAdded(tool);
	}

	/** Build a standalone ToolRegistry over the current default tools (used when no harness exists yet). */
	buildRegistry(config: {
		cwd?: string;
		allowedPaths?: string[];
		allowAllPaths?: boolean;
		cacheSize?: number;
		cacheTtlMs?: number;
		maxResultChars?: number;
	}): ToolRegistry {
		const registry = new ToolRegistry(config);
		registry.registerMany(this.defaultTools);
		return registry;
	}

	// ── MCP ──────────────────────────────────────────────────────────────

	isMcpLoaded(): boolean {
		return this.mcpLoaded;
	}

	isMcpLoading(): boolean {
		return this.mcpLoadPromise !== null && !this.mcpLoaded;
	}

	getMcpServerCount(): number {
		return this.mcpServerCount;
	}

	getMcpToolCount(): number {
		return this.mcpToolNames.size;
	}

	getMcpErrors(): string[] {
		return this.mcpErrors;
	}

	getMcpSystemContext(): string {
		return this.mcpSystemContext;
	}

	async loadMcpToolsOnce(): Promise<void> {
		if (this.mcpLoaded || process.env.LOGICIAN_MCP === "0") return;
		if (!this.mcpLoadPromise) {
			this.mcpLoadPromise = (async () => {
				const result = await this.mcpManager.load(
					this.cwd,
					this.defaultTools.map(tool => tool.name),
				);
				this.mcpServerCount = result.servers;
				this.mcpErrors = result.errors;
				this.mcpToolNames = new Set(result.tools.map(tool => tool.name));
				// Tool presence alone doesn't tell the model whether a missing
				// capability was never configured or failed to connect — surface
				// connection failures in the system prompt so it can explain a gap
				// instead of silently working around it or guessing.
				this.mcpSystemContext = result.errors.length
					? `<mcp-status>\n${result.errors.length} MCP server(s) failed to load:\n${result.errors.map(e => `- ${e}`).join("\n")}\n` +
						"Tools from these servers are unavailable this session.\n</mcp-status>"
					: "";
				if (result.tools.length || this.mcpSystemContext) {
					const existing = new Set(this.defaultTools.map(tool => tool.name));
					const newTools = result.tools.filter(
						tool => !existing.has(tool.name),
					);
					for (const tool of newTools) this.addTool(tool);
					// addTool() already triggers onToolAdded (which rebuilds the system
					// prompt); an errors-only load with no new tools still changed
					// mcpSystemContext, so notify explicitly for that case.
					if (!newTools.length) this.onContextChanged();
				}
				this.mcpLoaded = true;
				// Single source of truth for "MCP finished loading," regardless of
				// who triggered it (startup prewarm, a slash command, or the first
				// turn falling back to a lazy load) — the caller doesn't need its
				// own .then() to surface this, and there's no risk of it firing
				// twice for one load since mcpLoadPromise is memoized above.
				this.emit({
					type: "notice",
					level: result.errors.length ? "warn" : "info",
					label: "MCP",
					text: `Loaded ${result.servers} server(s).`,
				});
			})();
		}
		await this.mcpLoadPromise;
	}

	async getMcpSnapshot(): Promise<McpSnapshotResult> {
		return this.mcpManager.getSnapshot(this.cwd);
	}

	async setMcpServerEnabled(
		serverName: string,
		enabled: boolean,
	): Promise<McpToggleResult> {
		return this.mcpManager.setServerEnabled(serverName, enabled, this.cwd);
	}

	async closeMcp(): Promise<void> {
		await this.mcpManager.close();
	}

	// ── Sandbox mode ─────────────────────────────────────────────────────
	// Default profile applied by the sandbox tool when a call omits one.
	// Cycled by the UI (Ctrl+K); "none" is exposed to the user as "off".

	getSandboxMode(): SandboxProfile {
		return getDefaultSandboxProfile();
	}

	setSandboxMode(mode: SandboxProfile): void {
		setDefaultSandboxProfile(mode);
		this.emit({
			type: "notice",
			level: "info",
			label: "Sandbox",
			text: `mode: ${mode === "none" ? "off" : mode}`,
		});
	}

	cycleSandboxMode(): SandboxProfile {
		const cycle = ToolRouter.SANDBOX_CYCLE;
		const currentIndex = cycle.indexOf(this.getSandboxMode());
		const next = cycle[(currentIndex + 1) % cycle.length];
		this.setSandboxMode(next);
		return next;
	}

	// ── Skills ───────────────────────────────────────────────────────────

	isSkillsInjected(): boolean {
		return this.skillsInjected;
	}

	getSkillsContext(): string | null {
		return this.skillsContext;
	}

	getLoadedSkills(): Skill[] {
		return this.loadedSkills;
	}

	getEnabledPluginRoots(): Array<{ name: string; installPath: string }> {
		return this.enabledPluginRoots;
	}

	/**
	 * Discover SKILL.md files from installed plugins and inject them into
	 * the system prompt so the agent can see available skills.
	 * Runs after startup hooks as a fallback when hooks fail to produce context.
	 */
	async injectSkillsFromPlugins(): Promise<void> {
		if (this.skillsInjected) return;
		this.skillsInjected = true;

		const registry = await runPluginBackend("list", []);
		const plugins = registry.plugins || [];

		// Collect skills directories from all enabled, on-disk plugins.
		const skillsDirs: string[] = [];
		const enabledPlugins: Array<{ name: string; installPath: string }> = [];
		for (const plugin of plugins) {
			const enabled = plugin.enabled !== false;
			const onDisk = plugin.on_disk !== false;
			const installPath = String(plugin.install_path || "");
			const pluginName = String(plugin.name || plugin.plugin_id || "");
			if (!enabled || !onDisk || !installPath) continue;
			enabledPlugins.push({ name: pluginName, installPath });
			skillsDirs.push(path.join(installPath, "skills"));
		}
		this.enabledPluginRoots = enabledPlugins;

		// Load user-global skills independently of installed plugins.
		// This is the shared agents convention used by Codex and other harnesses.
		skillsDirs.push(path.join(os.homedir(), ".agents", "skills"));

		// Also discover project-local skills by walking cwd ancestors.
		// Missing directories are skipped silently by loadSkills.
		if (this.projectTrusted) {
			skillsDirs.push(...getProjectSkillDirs(this.cwd));
		}

		if (!skillsDirs.length) return;

		const { skills: rawSkills, diagnostics } = await loadSkills(skillsDirs);

		// Namespace plugin skills as plugin:skill (Claude Code convention); the
		// bare name stays available as an alias when unambiguous.
		const skills = rawSkills.map(skill => {
			const owner = enabledPlugins.find(p =>
				skill.filePath.startsWith(p.installPath + path.sep),
			);
			if (!owner?.name || skill.name.startsWith(`${owner.name}:`)) {
				return skill;
			}
			return {
				...skill,
				name: `${owner.name}:${skill.name}`,
				slashName: `${owner.name}:${skill.slashName}`,
				aliases: [...(skill.aliases ?? []), skill.name],
			};
		});

		// Claude Code plugin commands (commands/*.md) become user-invocable
		// skills: /plugin:command or /command, never advertised to the model.
		skills.push(...(await loadPluginCommands(enabledPlugins)));

		// Log diagnostics to transcript for visibility.
		for (const diag of diagnostics) {
			this.emit({
				type: "token",
				token: `[Skill warning] ${diag.code}: ${diag.message}`,
			});
		}

		// All loaded skills are user-invocable via /<skill-name>; only the ones
		// not flagged disable-model-invocation are advertised to the model.
		this.loadedSkills = skills;
		const visible = skills.filter(s => !s.disableModelInvocation);
		// Only skip catalog injection when there are no skills at all.
		// Plugin commands (disableModelInvocation) still need read_skill.

		// Inject a compact catalog (name + description), not full bodies. The
		// model loads a skill's full instructions on demand via read_skill.
		this.skillsContext = formatSkillCatalog(visible);

		// Register the read_skill tool bound to ALL loaded skills so the model can
		// pull full bodies for any skill, including plugin commands (disableModelInvocation).
		const readSkill = createReadSkillTool(skills);
		if (readSkill) this.addTool(readSkill);
	}

	/** Reset skill/prompt injection flags/state (used by reload(), a full re-init). */
	resetSkillsAndPrompts(): void {
		this.loadedSkills = [];
		this.skillsContext = null;
		this.skillsInjected = false;
		this.loadedPrompts = [];
		this.promptsInjected = false;
	}

	/** Reset just the injected-context flags (used by reset(), a lighter session reset). */
	resetInjectedContext(): void {
		this.skillsContext = null;
		this.skillsInjected = false;
		this.promptsInjected = false;
	}

	// ── Prompts ──────────────────────────────────────────────────────────

	getLoadedPrompts(): Prompt[] {
		return this.loadedPrompts;
	}

	/**
	 * Discover prompts/.logician/prompts markdown files and register them as
	 * direct, user-typed /<name> slash commands. Unlike skills, prompts are
	 * never surfaced to the model — they exist only to be typed by the user.
	 */
	async injectPrompts(): Promise<void> {
		if (this.promptsInjected) return;
		this.promptsInjected = true;

		if (!this.projectTrusted) return;
		const promptDirs = getProjectPromptDirs(this.cwd);
		if (!promptDirs.length) return;
		this.loadedPrompts = await loadPrompts(promptDirs);
	}

	// ── Status snapshot (getState()/init()) ─────────────────────────────

	getStatus(): ToolRouterStatus {
		return {
			mcpServerCount: this.mcpServerCount,
			mcpToolCount: this.mcpToolNames.size,
			mcpErrors: this.mcpErrors,
			mcpLoaded: this.mcpLoaded,
			mcpLoading: this.isMcpLoading(),
			skillsInjected: this.skillsInjected,
			skillsVisible: !!this.skillsContext,
			loadedSkills: this.loadedSkills,
			enabledPluginRoots: this.enabledPluginRoots,
		};
	}
}

/**
 * Claude Code plugin commands (commands/*.md) become user-invocable skills:
 * /plugin:command or /command, never advertised to the model.
 */
async function loadPluginCommands(
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
