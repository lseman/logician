// ── Claude Code plugin manager ───────────────────────────────────────────────
// Installs, lists, updates, and removes plugins. Manages the registry JSON.
// Depends on plugins-executor for hook loading/execution.

import { execFile } from "node:child_process";
import { existsSync, promises as fs, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import {
	buildHookInput,
	childDirNames,
	copyDir,
	emptyHookResult,
	executeLoadedHook,
	findPluginManifest,
	gitHead,
	type HookEventType,
	isDir,
	loadPluginHooks,
	markdownNames,
	matcherMatches,
	mergeHookResult,
	normalizeInstall,
	nowIso,
	parseHookEventType,
	readJson,
	resolvePluginsDir,
	sanitize,
	withHookMetadata,
} from "./plugin-executor.ts";

const execFileAsync = promisify(execFile);

type Scope = "user" | "project" | "local" | "managed";

export interface PluginInstall {
	scope: Scope;
	installPath: string;
	version: string;
	installedAt: string;
	lastUpdated: string;
	gitCommitSha?: string;
	enabled?: boolean;
	projectPath?: string;
	dependencies?: string[];
}

export interface RegistryData {
	version: 2;
	plugins: Record<string, PluginInstall[]>;
}

type PluginSettingsEntry = boolean | { enabled?: boolean };

interface UserSettingsData {
	plugins?: Record<string, PluginSettingsEntry>;
	/** Claude Code's own enablement map (~/.claude/settings.json). */
	enabledPlugins?: Record<string, boolean>;
}

export interface PluginCommandResult {
	status?: string;
	message?: string;
	plugins_dir?: string;
	plugins?: Array<Record<string, unknown>>;
	session_start_hooks?: Record<string, number>;
	hooks?: Array<Record<string, unknown>>;
	updates?: Array<Record<string, unknown>>;
	issues?: Array<Record<string, unknown>>;
	additional_contexts?: string[];
	context_messages?: Array<{
		plugin_id: string;
		plugin_name: string;
		matcher: string;
		content: string;
	}>;
	initial_user_message?: string | null;
	watch_paths?: string[];
	decision?: "block" | "approve";
	reason?: string;
	permission_decision?: "allow" | "deny" | "ask";
	permission_reason?: string;
	errors?: string[];
	hook_count?: number;
	source?: string;
	raw_output?: string;
	event?: string;
	sha?: string;
	name?: string;
	install_path?: string;
	on_disk?: boolean;
	skill_count?: number;
	manifest?: Record<string, unknown>;
	commands?: string[];
	[key: string]: unknown;
}

/** Plugin manager: install, list, update, remove, and manage plugin hooks. */
export class PluginRegistry {
	readonly pluginsDir: string;
	private registryPath: string;
	private readonly env?: NodeJS.ProcessEnv;

	constructor(options: { env?: NodeJS.ProcessEnv } = {}) {
		this.pluginsDir = resolvePluginsDir();
		this.registryPath = path.join(this.pluginsDir, "installed_plugins.json");
		this.env = options.env;
	}

	async listPlugins(): Promise<PluginCommandResult> {
		const installs = await this.allInstalls();
		const rows = await Promise.all(
			installs.map(async ([pluginId, inst]) => {
				const [name, marketplace = ""] = pluginId.split("@");
				const onDisk = await isDir(inst.installPath || "");
				const skillCount = onDisk
					? await childDirNames(path.join(inst.installPath, "skills")).then(
							d => d.length,
						)
					: 0;
				return {
					plugin_id: pluginId,
					name,
					marketplace,
					version: inst.version || "unknown",
					scope: inst.scope || "user",
					enabled: this.isEnabled(inst),
					install_path: inst.installPath || "",
					sha: (inst.gitCommitSha || "").slice(0, 12),
					installed_at: inst.installedAt || "",
					last_updated: inst.lastUpdated || "",
					on_disk: onDisk,
					skill_count: skillCount,
				};
			}),
		);
		return { status: "ok", plugins: rows, plugins_dir: this.pluginsDir };
	}

	async allPluginIds(): Promise<string[]> {
		return Object.keys((await this.loadRegistry()).plugins).sort();
	}

	async allInstalls(): Promise<Array<[string, PluginInstall]>> {
		const registry = await this.loadRegistry();
		const out: Array<[string, PluginInstall]> = [];
		for (const [pluginId, installs] of Object.entries(registry.plugins)) {
			for (const install of installs || []) {
				const normalized = normalizeInstall(
					install as unknown as Record<string, unknown>,
				) as unknown as PluginInstall;
				out.push([
					pluginId,
					{
						...normalized,
						enabled: this.effectiveEnabled(pluginId, normalized),
					},
				]);
			}
		}
		return out;
	}

	async setEnabled(
		name: string,
		enabled: boolean,
	): Promise<PluginCommandResult> {
		const pluginId = await this.resolvePluginId(name);
		if (!pluginId)
			return {
				status: "error",
				message: `Plugin '${name}' not found in registry.`,
				plugins_dir: this.pluginsDir,
			};
		const registry = await this.loadRegistry();
		const records = registry.plugins[pluginId] || [];
		const idx = records.findIndex(r => (r.scope || "user") === "user");
		const targetIdx = idx >= 0 ? idx : 0;
		const inst = normalizeInstall(
			records[targetIdx] as unknown as Record<string, unknown>,
		) as unknown as PluginInstall;
		const alreadyEffective = this.effectiveEnabled(pluginId, inst) === enabled;
		if ((inst.enabled !== false) !== enabled) {
			records[targetIdx] = { ...inst, enabled, lastUpdated: nowIso() };
			registry.plugins[pluginId] = records;
			await this.saveRegistry(registry);
		}
		await this.savePluginSetting(pluginId, enabled);
		if (alreadyEffective) {
			return {
				status: enabled ? "already_enabled" : "already_disabled",
				message: `Plugin '${name}' is already ${enabled ? "enabled" : "disabled"}.`,
				plugin_id: pluginId,
				enabled,
				plugins_dir: this.pluginsDir,
			};
		}
		return {
			status: enabled ? "enabled" : "disabled",
			message: `Plugin '${name}' has been ${enabled ? "enabled" : "disabled"}.`,
			plugin_id: pluginId,
			enabled,
			plugins_dir: this.pluginsDir,
		};
	}

	async install(ref: string): Promise<PluginCommandResult> {
		const localPath = ref.startsWith("file://") ? ref.slice(7) : ref;
		if (await isDir(localPath)) {
			return this.installFromLocal(path.resolve(localPath));
		}

		if (!ref.includes("/")) {
			const marketplace = await this.findMarketplacePlugin(ref);
			if (!marketplace)
				throw new Error(
					`Plugin '${ref}' not found in known marketplaces. Pass owner/name or a local path.`,
				);
			if (marketplace.localPath)
				return this.installFromLocal(marketplace.localPath, marketplace.owner);
			if (marketplace.gitUrl)
				return this.installFromGitUrl(
					marketplace.gitUrl,
					marketplace.owner,
					marketplace.subdir,
					marketplace.ref,
				);
			throw new Error(`Plugin '${ref}' has an unsupported marketplace source.`);
		}

		const [owner, repoVersion] = ref.includes("/")
			? ref.split("/", 2)
			: ["", ref];
		if (!owner)
			throw new Error(
				`Plugin '${ref}' not found in known marketplaces. Pass owner/name or a local path.`,
			);
		const [repo, version = ""] = repoVersion.split("@");
		return this.installFromGitUrl(
			`https://github.com/${owner}/${repo}.git`,
			owner,
			undefined,
			version,
		);
	}

	private async installFromGitUrl(
		gitUrl: string,
		owner: string,
		subdir?: string,
		ref?: string,
	): Promise<PluginCommandResult> {
		const repoName = path.basename(gitUrl.replace(/\.git$/, ""));
		const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "logician_plugin_"));
		const cloneDir = path.join(tmp, repoName);
		try {
			const cmd = ["clone", "--depth=1"];
			if (ref) cmd.push("--branch", ref);
			cmd.push(gitUrl, cloneDir);
			await execFileAsync("git", cmd, { timeout: 180_000, env: this.env });
			return await this.installFromLocal(
				subdir ? path.join(cloneDir, subdir) : cloneDir,
				owner,
			);
		} finally {
			await fs.rm(tmp, { recursive: true, force: true }).catch(() => undefined);
		}
	}

	async remove(name: string, keepCache: boolean): Promise<PluginCommandResult> {
		const pluginId = await this.resolvePluginId(name);
		if (!pluginId)
			return {
				status: "error",
				message: `Plugin '${name}' not found in registry.`,
				plugins_dir: this.pluginsDir,
			};
		const registry = await this.loadRegistry();
		const records = registry.plugins[pluginId] || [];
		const inst = normalizeInstall(
			records[0] as unknown as Record<string, unknown>,
		) as unknown as PluginInstall;
		if (!keepCache && inst.installPath)
			await fs
				.rm(inst.installPath, { recursive: true, force: true })
				.catch(() => undefined);
		delete registry.plugins[pluginId];
		await this.saveRegistry(registry);
		return {
			status: "removed",
			message: `Plugin '${name}' removed from registry.`,
			plugins_dir: this.pluginsDir,
		};
	}

	async update(name: string): Promise<PluginCommandResult> {
		const pluginId = await this.resolvePluginId(name);
		if (!pluginId)
			return {
				status: "error",
				message: `Plugin '${name}' not found in registry.`,
				plugins_dir: this.pluginsDir,
			};
		const [pluginName, owner] = pluginId.split("@");
		if (!owner)
			return {
				status: "error",
				message: `Cannot infer marketplace owner for '${pluginId}'.`,
				plugins_dir: this.pluginsDir,
			};
		const before = (await this.getInstall(pluginId))?.gitCommitSha || "";
		const result = await this.install(`${owner}/${pluginName}`);
		const after = String(result.sha || "");
		if (before && after && before === after) {
			return {
				status: "up_to_date",
				message: `Plugin '${pluginName}' is already at latest commit (${after.slice(0, 12)}).`,
				plugins_dir: this.pluginsDir,
			};
		}
		return {
			...result,
			status: result.status === "installed" ? "updated" : result.status,
		};
	}

	async dependencies(name?: string): Promise<PluginCommandResult> {
		const issues = [];
		const installs = name
			? (await this.resolvePluginId(name))
				? ([
						[
							(await this.resolvePluginId(name)) as string,
							(await this.getInstall(
								(await this.resolvePluginId(name)) as string,
							)) as PluginInstall,
						],
					] as Array<[string, PluginInstall]>)
				: []
			: await this.allInstalls();
		for (const [pluginId, inst] of installs) {
			if (!inst || inst.enabled === false) continue;
			const manifest = await readJson(
				path.join(inst.installPath, ".claude-plugin", "plugin.json"),
			);
			const deps = Array.isArray(manifest.dependencies)
				? manifest.dependencies.map(String)
				: [];
			const missing = [];
			for (const dep of deps) {
				const depId = await this.resolvePluginId(dep);
				if (!depId || !(await this.getInstall(depId))) missing.push(dep);
			}
			if (missing.length)
				issues.push({
					plugin_id: pluginId,
					status: "missing_dependencies",
					missing,
				});
		}
		return {
			status: issues.length ? "issues_found" : "ok",
			issues,
			plugins_dir: this.pluginsDir,
		};
	}

	async info(name: string): Promise<PluginCommandResult> {
		const pluginId = await this.resolvePluginId(name);
		const inst = pluginId ? await this.getInstall(pluginId) : null;
		if (!pluginId || !inst)
			return {
				status: "error",
				message: `Plugin '${name}' not found.`,
				plugins_dir: this.pluginsDir,
			};
		const manifest = await readJson(
			path.join(inst.installPath, ".claude-plugin", "plugin.json"),
		);
		return {
			status: "ok",
			plugin_id: pluginId,
			version: inst.version,
			sha: inst.gitCommitSha || "",
			enabled: this.effectiveEnabled(pluginId, inst),
			install_path: inst.installPath,
			on_disk: await isDir(inst.installPath),
			manifest,
			skills: await childDirNames(path.join(inst.installPath, "skills")),
			commands: await markdownNames(path.join(inst.installPath, "commands")),
			plugins_dir: this.pluginsDir,
		};
	}

	async sessionStartHookCounts(): Promise<Record<string, number>> {
		const installs = await this.allInstalls();
		const counts: Record<string, number> = {};
		await Promise.all(
			installs.map(async ([pluginId, inst]) => {
				if (!this.isEnabled(inst)) {
					counts[pluginId] = 0;
					return;
				}
				counts[pluginId] = (
					await loadPluginHooks(inst.installPath, pluginId)
				).filter(h => h.eventType === "SessionStart").length;
			}),
		);
		return counts;
	}

	async listHooks(source: string): Promise<PluginCommandResult> {
		const eventType = parseHookEventType(source);
		const hooks = eventType
			? await this.getHooks(eventType, "")
			: await this.getHooks("SessionStart", source);
		return {
			status: "ok",
			source: eventType ? "" : source,
			event: eventType || "SessionStart",
			plugins_dir: this.pluginsDir,
			hooks: hooks.map(hook => ({
				plugin_id: hook.pluginId,
				plugin_name: hook.pluginName,
				event: hook.eventType,
				matcher: hook.definition.matcher || "",
				commands: hook.definition.hooks.map(cmd => ({
					type: cmd.type || "command",
					command: cmd.command || cmd.prompt || cmd.agent || cmd.url || "",
				})),
			})),
		};
	}

	async executeSessionStartHooks(
		source: string,
		payload: Record<string, unknown>,
	): Promise<PluginCommandResult> {
		const hooks = await this.getHooks("SessionStart", source);
		const result = emptyHookResult();
		const errors: string[] = [];
		await Promise.all(
			hooks.map(async hook => {
				try {
					mergeHookResult(
						result,
						withHookMetadata(
							hook,
							await executeLoadedHook(
								hook,
								source,
								buildHookInput("SessionStart", {
									...payload,
									source,
								}),
							),
						),
					);
				} catch (error: unknown) {
					errors.push(error instanceof Error ? error.message : String(error));
				}
			}),
		);
		return {
			status: "ok",
			source,
			plugins_dir: this.pluginsDir,
			hook_count: hooks.length,
			additional_contexts: result.additional_contexts,
			context_messages: result.context_messages,
			initial_user_message: result.initial_user_message,
			watch_paths: result.watch_paths,
			errors,
		};
	}

	async executeHookEvent(
		eventType: string,
		payload: Record<string, unknown>,
	): Promise<PluginCommandResult> {
		const matcher = String(payload.matcher_value || "");
		const hooks = await this.getHooks(eventType as typeof eventType, matcher);
		const result = emptyHookResult();
		for (const hook of hooks) {
			mergeHookResult(
				result,
				withHookMetadata(
					hook,
					await executeLoadedHook(
						hook,
						eventType as unknown as HookEventType,
						buildHookInput(eventType as unknown as HookEventType, payload),
					),
				),
			);
		}
		return {
			status: "ok",
			event: eventType,
			plugins_dir: this.pluginsDir,
			additional_contexts: result.additional_contexts,
			context_messages: result.context_messages,
			initial_user_message: result.initial_user_message,
			watch_paths: result.watch_paths,
			decision: result.decision,
			reason: result.reason,
			permission_decision: result.permission_decision,
			permission_reason: result.permission_reason,
			raw_output: result.raw_output,
		};
	}

	private async getHooks(
		eventType: string,
		matcherValue = "",
	): Promise<import("./plugin-executor.ts").LoadedHook[]> {
		const hooks: import("./plugin-executor.ts").LoadedHook[] = [];
		for (const [pluginId, inst] of await this.allInstalls()) {
			if (!this.isEnabled(inst) || !(await isDir(inst.installPath))) continue;
			for (const hook of await loadPluginHooks(inst.installPath, pluginId)) {
				if (
					hook.eventType === eventType &&
					matcherMatches(hook.definition.matcher, matcherValue)
				)
					hooks.push(hook);
			}
		}
		return hooks;
	}

	private async installFromLocal(
		sourceDir: string,
		ownerHint = "local",
	): Promise<PluginCommandResult> {
		const manifest = await findPluginManifest(sourceDir);
		const pluginJson = manifest ? await readJson(manifest) : {};
		const name = String(pluginJson.name || path.basename(sourceDir));
		const owner = sanitize(
			String(pluginJson.marketplace || pluginJson.owner || ownerHint),
		);
		const sha = await gitHead(sourceDir);
		const version = String(
			pluginJson.version || (sha ? sha.slice(0, 12) : `local-${Date.now()}`),
		);
		const pluginId = `${name}@${owner}`;
		const enabled = this.configuredEnabled(pluginId) ?? true;
		const cachePath = path.join(
			this.pluginsDir,
			"cache",
			owner,
			sanitize(name),
			sanitize(version),
		);
		await fs
			.rm(cachePath, { recursive: true, force: true })
			.catch(() => undefined);
		await copyDir(sourceDir, cachePath);
		await this.upsert(pluginId, {
			scope: "user",
			installPath: cachePath,
			version,
			installedAt: nowIso(),
			lastUpdated: nowIso(),
			gitCommitSha: sha,
			enabled,
			dependencies: Array.isArray(pluginJson.dependencies)
				? pluginJson.dependencies.map(String)
				: [],
		} as PluginInstall);
		await this.savePluginSetting(pluginId, enabled);
		return {
			status: "installed",
			message: `Plugin '${name}' v${version} installed to ${cachePath}.`,
			plugin_id: pluginId,
			name,
			version,
			install_path: cachePath,
			sha,
			plugins_dir: this.pluginsDir,
		};
	}

	private async loadRegistry(): Promise<RegistryData> {
		await fs.mkdir(this.pluginsDir, { recursive: true });
		const raw = await readJson(this.registryPath);
		return {
			version: 2,
			plugins:
				raw.plugins && typeof raw.plugins === "object"
					? (raw.plugins as Record<string, PluginInstall[]>)
					: {},
		};
	}

	private async saveRegistry(data: RegistryData): Promise<void> {
		await fs.mkdir(this.pluginsDir, { recursive: true });
		await fs.writeFile(
			this.registryPath,
			JSON.stringify(data, null, 2),
			"utf8",
		);
	}

	private isEnabled(install: PluginInstall): boolean {
		return install.enabled !== false;
	}

	private effectiveEnabled(pluginId: string, install: PluginInstall): boolean {
		return this.configuredEnabled(pluginId) ?? install.enabled !== false;
	}

	private configuredEnabled(pluginId: string): boolean | undefined {
		const configured = this.loadUserSettings().plugins?.[pluginId];
		if (typeof configured === "boolean") return configured;
		if (
			configured &&
			typeof configured === "object" &&
			typeof configured.enabled === "boolean"
		) {
			return configured.enabled;
		}
		// Fall back to Claude Code's own enablement map so plugins toggled in
		// Claude Code behave the same here without re-configuring.
		const claudeCode = this.loadClaudeCodeSettings().enabledPlugins?.[pluginId];
		if (typeof claudeCode === "boolean") return claudeCode;
		return undefined;
	}

	private settingsPath(): string {
		return path.join(os.homedir(), ".logician", "settings.json");
	}

	private loadUserSettings(): UserSettingsData {
		return this.readSettingsFile(this.settingsPath());
	}

	private loadClaudeCodeSettings(): UserSettingsData {
		return this.readSettingsFile(
			path.join(os.homedir(), ".claude", "settings.json"),
		);
	}

	private readSettingsFile(settingsPath: string): UserSettingsData {
		try {
			if (!existsSync(settingsPath)) return {};
			const raw = JSON.parse(readFileSync(settingsPath, "utf8")) as unknown;
			return raw && typeof raw === "object" ? (raw as UserSettingsData) : {};
		} catch (_e: unknown) {
			return {};
		}
	}

	private async savePluginSetting(
		pluginId: string,
		enabled: boolean,
	): Promise<void> {
		const settingsPath = this.settingsPath();
		await fs.mkdir(path.dirname(settingsPath), { recursive: true });
		const raw = this.loadUserSettings() as Record<string, unknown>;
		const plugins =
			raw.plugins &&
			typeof raw.plugins === "object" &&
			!Array.isArray(raw.plugins)
				? { ...(raw.plugins as Record<string, PluginSettingsEntry>) }
				: {};
		plugins[pluginId] = { enabled };
		raw.plugins = plugins;
		await fs.writeFile(
			settingsPath,
			`${JSON.stringify(raw, null, 2)}\n`,
			"utf8",
		);
	}

	private async upsert(
		pluginId: string,
		install: PluginInstall,
	): Promise<void> {
		const registry = await this.loadRegistry();
		const records = registry.plugins[pluginId] || [];
		const idx = records.findIndex(r => (r.scope || "user") === install.scope);
		if (idx >= 0) records[idx] = install;
		else records.push(install);
		registry.plugins[pluginId] = records;
		await this.saveRegistry(registry);
	}

	private async resolvePluginId(name: string): Promise<string | null> {
		const registry = await this.loadRegistry();
		if (registry.plugins[name]) return name;
		const lowered = name.toLowerCase();
		return (
			Object.keys(registry.plugins).find(
				pid => pid.split("@")[0].toLowerCase() === lowered,
			) || null
		);
	}

	private async getInstall(pluginId: string): Promise<PluginInstall | null> {
		const records = (await this.loadRegistry()).plugins[pluginId] || [];
		return records.length
			? (normalizeInstall(
					records[0] as unknown as Record<string, unknown>,
				) as unknown as PluginInstall)
			: null;
	}

	private async findMarketplacePlugin(name: string): Promise<{
		owner: string;
		localPath?: string;
		gitUrl?: string;
		subdir?: string;
		ref?: string;
	} | null> {
		const known = await readJson(
			path.join(this.pluginsDir, "known_marketplaces.json"),
		);
		for (const [marketplaceName, info] of Object.entries(
			typeof known === "object" && known !== null
				? (known as Record<string, unknown>)
				: {},
		)) {
			const installLocation = String(
				(info as Record<string, unknown>).installLocation || "",
			);
			if (!installLocation) continue;
			const manifests = [
				path.join(installLocation, ".claude-plugin", "marketplace.json"),
				path.join(installLocation, ".agents", "plugins", "marketplace.json"),
			];
			for (const manifestPath of manifests) {
				const manifest = await readJson(manifestPath);
				const plugins = Array.isArray(manifest.plugins) ? manifest.plugins : [];
				const found = plugins.find(
					(plugin: unknown) =>
						typeof plugin === "object" &&
						plugin !== null &&
						String(
							(plugin as Record<string, unknown>).name || "",
						).toLowerCase() === name.toLowerCase(),
				);
				if (!found || typeof found !== "object") continue;
				const source = (found as Record<string, unknown>).source;
				if (typeof source === "string") {
					if (source.startsWith("./") || source.startsWith("../")) {
						return {
							owner: String(marketplaceName),
							localPath: path.resolve(path.dirname(manifestPath), source),
						};
					}
					if (source.startsWith("https://") || source.startsWith("git@")) {
						return {
							owner: String(marketplaceName),
							gitUrl: source.endsWith(".git") ? source : `${source}.git`,
						};
					}
				}
				if (typeof source === "object" && source !== null) {
					const url = String((source as Record<string, unknown>).url || "");
					if (
						String((source as Record<string, unknown>).source || "") ===
							"local" &&
						typeof (source as Record<string, unknown>).path === "string"
					) {
						return {
							owner: String(marketplaceName),
							localPath: path.resolve(
								path.dirname(manifestPath),
								String((source as Record<string, unknown>).path),
							),
						};
					}
					if (url) {
						return {
							owner: String(marketplaceName),
							gitUrl: url.endsWith(".git") ? url : `${url}.git`,
							subdir:
								typeof (source as Record<string, unknown>).path === "string"
									? String((source as Record<string, unknown>).path)
									: undefined,
							ref:
								typeof (source as Record<string, unknown>).ref === "string"
									? String((source as Record<string, unknown>).ref)
									: undefined,
						};
					}
				}
			}
		}
		return null;
	}
}
