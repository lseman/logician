import { execFile, spawn } from "node:child_process";
import { existsSync, promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

let pluginRuntimeEnv: NodeJS.ProcessEnv = {};

type Scope = "user" | "project" | "local" | "managed";
type HookEventType =
    | "SessionStart"
    | "SessionEnd"
    | "Setup"
    | "Stop"
    | "Notification"
    | "UserPromptSubmit"
    | "PreToolUse"
    | "PostToolUse"
    | "PreCompact"
    | "PostCompact";

interface PluginInstall {
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

interface RegistryData {
    version: 2;
    plugins: Record<string, PluginInstall[]>;
}

interface HookCommand {
    type: "command" | "prompt" | "agent" | "http";
    command?: string;
    prompt?: string;
    agent?: string;
    url?: string;
    headers?: Record<string, string>;
    timeout?: number;
}

interface HookDefinition {
    matcher?: string;
    hooks: HookCommand[];
}

interface LoadedHook {
    pluginId: string;
    pluginName: string;
    pluginDir: string;
    eventType: HookEventType;
    definition: HookDefinition;
}

interface HookContextMessage {
    plugin_id: string;
    plugin_name: string;
    matcher: string;
    content: string;
}

interface HookExecutionResult {
    additional_contexts: string[];
    context_messages: HookContextMessage[];
    initial_user_message: string | null;
    watch_paths: string[];
    raw_output: string;
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
    context_messages?: HookContextMessage[];
    initial_user_message?: string | null;
    watch_paths?: string[];
    errors?: string[];
    hook_count?: number;
    source?: string;
    raw_output?: string;
    event?: string;
    [key: string]: unknown;
}

export function configurePluginRuntimeEnv(env: NodeJS.ProcessEnv): void {
    pluginRuntimeEnv = Object.fromEntries(
        Object.entries(env).filter(([, value]) => value !== undefined),
    ) as NodeJS.ProcessEnv;
}

export function splitPluginArgs(input: string): string[] {
    const args: string[] = [];
    let current = "";
    let quote: string | null = null;
    let escaped = false;

    for (const ch of input.trim()) {
        if (escaped) {
            current += ch;
            escaped = false;
            continue;
        }
        if (ch === "\\") {
            escaped = true;
            continue;
        }
        if (quote) {
            if (ch === quote) quote = null;
            else current += ch;
            continue;
        }
        if (ch === "'" || ch === "\"") {
            quote = ch;
            continue;
        }
        if (/\s/.test(ch)) {
            if (current) {
                args.push(current);
                current = "";
            }
            continue;
        }
        current += ch;
    }
    if (escaped) current += "\\";
    if (current) args.push(current);
    return args;
}

export async function runPluginBackend(
    action: string,
    args: string[],
): Promise<PluginCommandResult> {
    const manager = new TsPluginManager();
    try {
        switch (action) {
            case "list":
                return {
                    ...(await manager.listPlugins()),
                    session_start_hooks: await manager.sessionStartHookCounts(),
                };
            case "enable":
            case "disable": {
                if (!args[0])
                    throw new Error(`usage: /plugins ${action} <plugin>`);
                const result = await manager.setEnabled(
                    args[0],
                    action === "enable",
                );
                return {
                    ...result,
                    session_start_hooks: await manager.sessionStartHookCounts(),
                };
            }
            case "install":
                if (!args[0])
                    throw new Error(
                        "usage: /plugins install <owner/name | path | name>",
                    );
                return {
                    ...(await manager.install(args[0])),
                    session_start_hooks: await manager.sessionStartHookCounts(),
                };
            case "remove":
                if (!args[0])
                    throw new Error(
                        "usage: /plugins remove <plugin> [--keep-checkout]",
                    );
                return manager.remove(
                    args[0],
                    args.includes("--keep-checkout"),
                );
            case "update":
                if (!args[0])
                    throw new Error("usage: /plugins update <plugin | --all>");
                if (args[0] === "--all") {
                    const updates = [];
                    for (const pluginId of await manager.allPluginIds()) {
                        updates.push(await manager.update(pluginId));
                    }
                    return {
                        status: "ok",
                        updates,
                        plugins_dir: manager.pluginsDir,
                    };
                }
                return manager.update(args[0]);
            case "deps":
                return manager.dependencies(args[0]);
            case "info":
                if (!args[0]) throw new Error("usage: /plugins info <plugin>");
                return {
                    ...(await manager.info(args[0])),
                    session_start_hooks: await manager.sessionStartHookCounts(),
                };
            case "hooks":
                return manager.listHooks(args[0] || "startup");
            case "run-hooks":
            case "session-start":
                return manager.executeSessionStartHooks(
                    args[0] || "startup",
                    parseJsonArg(args[1]),
                );
            case "hook":
                if (!args[0])
                    throw new Error("usage: hook <event-type> [payload-json]");
                return manager.executeHookEvent(
                    args[0] as HookEventType,
                    parseJsonArg(args[1]),
                );
            default:
                throw new Error(
                    "usage: /plugins [list|enable|disable|install|remove|update|deps|info|hooks|run-hooks]",
                );
        }
    } catch (error: unknown) {
        return {
            status: "error",
            message: error instanceof Error ? error.message : String(error),
            plugins_dir: manager.pluginsDir,
        };
    }
}

export async function runSessionStartHooks(
    payload: {
        source?: string;
        session_id?: string;
        transcript_path?: string;
        cwd?: string;
    } = {},
): Promise<PluginCommandResult> {
    const source = payload.source || "startup";
    return runPluginBackend("session-start", [
        source,
        JSON.stringify({
            session_id: payload.session_id || "",
            transcript_path: payload.transcript_path || "",
            cwd: payload.cwd || process.cwd(),
        }),
    ]);
}

export async function runHookEvent(
    eventType: string,
    payload: Record<string, unknown> = {},
): Promise<PluginCommandResult> {
    return runPluginBackend("hook", [eventType, JSON.stringify(payload)]);
}

export function runHookEventBackground(
    eventType: string,
    payload: Record<string, unknown> = {},
): void {
    runHookEvent(eventType, payload).catch(() => undefined);
}

class TsPluginManager {
    readonly pluginsDir: string;
    private registryPath: string;

    constructor() {
        this.pluginsDir = resolvePluginsDir();
        this.registryPath = path.join(
            this.pluginsDir,
            "installed_plugins.json",
        );
    }

    async listPlugins(): Promise<PluginCommandResult> {
        const rows = [];
        for (const [pluginId, inst] of await this.allInstalls()) {
            const [name, marketplace = ""] = pluginId.split("@");
            const onDisk = await isDir(inst.installPath || "");
            const skillCount = onDisk
                ? await childDirNames(path.join(inst.installPath, "skills")).then(
                      (d) => d.length
                  )
                : 0;
            rows.push({
                plugin_id: pluginId,
                name,
                marketplace,
                version: inst.version || "unknown",
                scope: inst.scope || "user",
                enabled: inst.enabled !== false,
                install_path: inst.installPath || "",
                sha: (inst.gitCommitSha || "").slice(0, 12),
                installed_at: inst.installedAt || "",
                last_updated: inst.lastUpdated || "",
                on_disk: onDisk,
                skill_count: skillCount,
            });
        }
        return { status: "ok", plugins: rows, plugins_dir: this.pluginsDir };
    }

    async allPluginIds(): Promise<string[]> {
        return Object.keys((await this.loadRegistry()).plugins).sort();
    }

    async allInstalls(): Promise<Array<[string, PluginInstall]>> {
        const registry = await this.loadRegistry();
        const out: Array<[string, PluginInstall]> = [];
        for (const [pluginId, installs] of Object.entries(registry.plugins)) {
            for (const install of installs || [])
                out.push([pluginId, normalizeInstall(install)]);
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
        const idx = records.findIndex((r) => (r.scope || "user") === "user");
        const targetIdx = idx >= 0 ? idx : 0;
        const inst = normalizeInstall(records[targetIdx]);
        if ((inst.enabled !== false) === enabled) {
            return {
                status: enabled ? "already_enabled" : "already_disabled",
                message: `Plugin '${name}' is already ${enabled ? "enabled" : "disabled"}.`,
                plugin_id: pluginId,
                enabled,
                plugins_dir: this.pluginsDir,
            };
        }
        records[targetIdx] = { ...inst, enabled, lastUpdated: nowIso() };
        registry.plugins[pluginId] = records;
        await this.saveRegistry(registry);
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
                return this.installFromLocal(
                    marketplace.localPath,
                    marketplace.owner,
                );
            if (marketplace.gitUrl)
                return this.installFromGitUrl(
                    marketplace.gitUrl,
                    marketplace.owner,
                    marketplace.subdir,
                    marketplace.ref,
                );
            throw new Error(
                `Plugin '${ref}' has an unsupported marketplace source.`,
            );
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
        const tmp = await fs.mkdtemp(
            path.join(os.tmpdir(), "logician_plugin_"),
        );
        const cloneDir = path.join(tmp, repoName);
        try {
            const cmd = ["clone", "--depth=1"];
            if (ref) cmd.push("--branch", ref);
            cmd.push(gitUrl, cloneDir);
            await execFileAsync("git", cmd, { timeout: 180_000 });
            return await this.installFromLocal(
                subdir ? path.join(cloneDir, subdir) : cloneDir,
                owner,
            );
        } finally {
            await fs
                .rm(tmp, { recursive: true, force: true })
                .catch(() => undefined);
        }
    }

    async remove(
        name: string,
        keepCache: boolean,
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
        const inst = normalizeInstall(records[0]);
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
                if (!depId || !(await this.getInstall(depId)))
                    missing.push(dep);
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
            enabled: inst.enabled !== false,
            install_path: inst.installPath,
            on_disk: await isDir(inst.installPath),
            manifest,
            skills: await childDirNames(path.join(inst.installPath, "skills")),
            commands: await markdownNames(
                path.join(inst.installPath, "commands"),
            ),
            plugins_dir: this.pluginsDir,
        };
    }

    async sessionStartHookCounts(): Promise<Record<string, number>> {
        const counts: Record<string, number> = {};
        for (const [pluginId, inst] of await this.allInstalls()) {
            counts[pluginId] = (
                await loadPluginHooks(inst.installPath, pluginId)
            ).filter((h) => h.eventType === "SessionStart").length;
        }
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
            hooks: hooks.map((hook) => ({
                plugin_id: hook.pluginId,
                plugin_name: hook.pluginName,
                event: hook.eventType,
                matcher: hook.definition.matcher || "",
                commands: hook.definition.hooks.map((cmd) => ({
                    type: cmd.type || "command",
                    command:
                        cmd.command || cmd.prompt || cmd.agent || cmd.url || "",
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
            hooks.map(async (hook) => {
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
                    errors.push(
                        error instanceof Error ? error.message : String(error),
                    );
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
        eventType: HookEventType,
        payload: Record<string, unknown>,
    ): Promise<PluginCommandResult> {
        const matcher = String(payload.matcher_value || "");
        const hooks = await this.getHooks(eventType, matcher);
        const result = emptyHookResult();
        for (const hook of hooks) {
            mergeHookResult(
                result,
                withHookMetadata(
                    hook,
                    await executeLoadedHook(
                        hook,
                        eventType,
                        buildHookInput(eventType, payload),
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
            raw_output: result.raw_output,
        };
    }

    private async getHooks(
        eventType: HookEventType,
        matcherValue = "",
    ): Promise<LoadedHook[]> {
        const hooks: LoadedHook[] = [];
        for (const [pluginId, inst] of await this.allInstalls()) {
            if (inst.enabled === false || !(await isDir(inst.installPath)))
                continue;
            for (const hook of await loadPluginHooks(
                inst.installPath,
                pluginId,
            )) {
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
            pluginJson.version ||
                (sha ? sha.slice(0, 12) : `local-${Date.now()}`),
        );
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
        const pluginId = `${name}@${owner}`;
        await this.upsert(pluginId, {
            scope: "user",
            installPath: cachePath,
            version,
            installedAt: nowIso(),
            lastUpdated: nowIso(),
            gitCommitSha: sha,
            enabled: true,
            dependencies: Array.isArray(pluginJson.dependencies)
                ? pluginJson.dependencies.map(String)
                : [],
        });
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
                    ? raw.plugins
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

    private async upsert(
        pluginId: string,
        install: PluginInstall,
    ): Promise<void> {
        const registry = await this.loadRegistry();
        const records = registry.plugins[pluginId] || [];
        const idx = records.findIndex(
            (r) => (r.scope || "user") === install.scope,
        );
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
                (pid) => pid.split("@")[0].toLowerCase() === lowered,
            ) || null
        );
    }

    private async getInstall(pluginId: string): Promise<PluginInstall | null> {
        const records = (await this.loadRegistry()).plugins[pluginId] || [];
        return records.length ? normalizeInstall(records[0]) : null;
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
            isRecord(known) ? known : {},
        )) {
            const installLocation = String(
                (info as Record<string, unknown>).installLocation || "",
            );
            if (!installLocation) continue;
            const manifests = [
                path.join(
                    installLocation,
                    ".claude-plugin",
                    "marketplace.json",
                ),
                path.join(
                    installLocation,
                    ".agents",
                    "plugins",
                    "marketplace.json",
                ),
            ];
            for (const manifestPath of manifests) {
                const manifest = await readJson(manifestPath);
                const plugins = Array.isArray(manifest.plugins)
                    ? manifest.plugins
                    : [];
                const found = plugins.find(
                    (plugin: unknown) =>
                        isRecord(plugin) &&
                        String(plugin.name || "").toLowerCase() ===
                            name.toLowerCase(),
                );
                if (!isRecord(found)) continue;
                const source = found.source;
                if (typeof source === "string") {
                    if (source.startsWith("./") || source.startsWith("../")) {
                        return {
                            owner: String(marketplaceName),
                            localPath: path.resolve(
                                path.dirname(manifestPath),
                                source,
                            ),
                        };
                    }
                    if (
                        source.startsWith("https://") ||
                        source.startsWith("git@")
                    ) {
                        return {
                            owner: String(marketplaceName),
                            gitUrl: source.endsWith(".git")
                                ? source
                                : `${source}.git`,
                        };
                    }
                }
                if (isRecord(source)) {
                    const url = String(source.url || "");
                    if (
                        String(source.source || "") === "local" &&
                        typeof source.path === "string"
                    ) {
                        return {
                            owner: String(marketplaceName),
                            localPath: path.resolve(
                                path.dirname(manifestPath),
                                source.path,
                            ),
                        };
                    }
                    if (url) {
                        return {
                            owner: String(marketplaceName),
                            gitUrl: url.endsWith(".git") ? url : `${url}.git`,
                            subdir:
                                typeof source.path === "string"
                                    ? source.path
                                    : undefined,
                            ref:
                                typeof source.ref === "string"
                                    ? source.ref
                                    : undefined,
                        };
                    }
                }
            }
        }
        return null;
    }
}

async function loadPluginHooks(
    pluginDir: string,
    pluginId: string,
): Promise<LoadedHook[]> {
    const merged: Record<string, HookDefinition[]> = {};
    const manifest = await readPluginManifest(pluginDir);
    await mergeManifestHooks(merged, pluginDir, manifest.hooks);
    const pluginName = await pluginNameFor(pluginDir, pluginId);
    const out: LoadedHook[] = [];
    for (const [eventType, defs] of Object.entries(merged)) {
        for (const definition of defs) {
            out.push({
                pluginId,
                pluginName,
                pluginDir,
                eventType: eventType as HookEventType,
                definition,
            });
        }
    }
    return out;
}

async function readPluginManifest(
    pluginDir: string,
): Promise<Record<string, unknown>> {
    return readJson(path.join(pluginDir, ".claude-plugin", "plugin.json"));
}

async function mergeManifestHooks(
    merged: Record<string, HookDefinition[]>,
    pluginDir: string,
    hooks: unknown,
): Promise<void> {
    if (typeof hooks === "string") {
        const hookPath = path.resolve(pluginDir, hooks);
        const hookJson = await readJson(hookPath);
        mergeHooks(merged, parseHooksDict(hookJson.hooks || hookJson));
        return;
    }
    mergeHooks(merged, parseHooksDict(hooks));
    const hookJson = await readJson(path.join(pluginDir, "hooks", "hooks.json"));
    mergeHooks(merged, parseHooksDict(hookJson.hooks || hookJson));
}

function parseHooksDict(data: unknown): Record<string, HookDefinition[]> {
    if (!data || typeof data !== "object" || Array.isArray(data)) return {};
    const out: Record<string, HookDefinition[]> = {};
    for (const [eventName, entries] of Object.entries(
        data as Record<string, unknown>,
    )) {
        if (!Array.isArray(entries)) continue;
        const defs = [];
        for (const entry of entries) {
            if (!entry || typeof entry !== "object") continue;
            const raw = entry as Record<string, unknown>;
            const hooks = Array.isArray(raw.hooks)
                ? raw.hooks
                      .filter(
                          (item): item is Record<string, unknown> =>
                              Boolean(item) &&
                              typeof item === "object" &&
                              !Array.isArray(item),
                      )
                      .map((item) => ({
                          type: hookType(String(item.type || "command")),
                          command: stringOrUndefined(item.command),
                          prompt: stringOrUndefined(item.prompt),
                          agent: stringOrUndefined(item.agent),
                          url: stringOrUndefined(item.url),
                          headers: isRecord(item.headers)
                              ? Object.fromEntries(
                                    Object.entries(item.headers).map(
                                        ([k, v]) => [k, String(v)],
                                    ),
                                )
                              : undefined,
                          timeout:
                              typeof item.timeout === "number"
                                  ? item.timeout
                                  : undefined,
                      }))
                : [];
            if (hooks.length)
                defs.push({ matcher: stringOrUndefined(raw.matcher), hooks });
        }
        if (defs.length) out[eventName] = defs;
    }
    return out;
}

async function executeLoadedHook(
    hook: LoadedHook,
    source: string,
    hookInput: string,
): Promise<HookExecutionResult> {
    const aggregate = emptyHookResult();
    for (const command of hook.definition.hooks) {
        const result = await executeCommand(command, hook, source, hookInput);
        if (result) mergeHookResult(aggregate, result);
    }
    return aggregate;
}

async function executeCommand(
    command: HookCommand,
    hook: LoadedHook,
    source: string,
    hookInput: string,
): Promise<HookExecutionResult | null> {
    if (command.type === "prompt") {
        return command.prompt
            ? { ...emptyHookResult(), additional_contexts: [command.prompt] }
            : null;
    }
    if (command.type === "http" && command.url) {
        const controller = new AbortController();
        const timer = setTimeout(
            () => controller.abort(),
            Math.max(1, command.timeout || 30) * 1000,
        );
        try {
            const response = await fetch(command.url, {
                headers: command.headers,
                signal: controller.signal,
            });
            return parseHookResponse(await response.text());
        } finally {
            clearTimeout(timer);
        }
    }
    if (command.type === "agent") return null;
    if (!command.command) return null;
    const timeout = Math.max(
        1,
        command.timeout ||
            (source === "startup"
                ? envNumber("LOGICIAN_STARTUP_HOOK_TIMEOUT_MS", 1200) / 1000
                : 30),
    );
    const { stdout, stderr } = await runShellCommand(command.command, {
        cwd: hook.pluginDir,
        env: {
            ...process.env,
            ...pluginRuntimeEnv,
            CLAUDE_PLUGIN_ROOT: hook.pluginDir,
        },
        input: hookInput,
        timeoutMs: timeout * 1000,
    });
    return parseHookResponse((stdout || stderr || "").trim());
}

function parseHookResponse(rawOutput: string): HookExecutionResult {
    const result = { ...emptyHookResult(), raw_output: rawOutput };
    const trimmedOutput = rawOutput.trim();
    if (!trimmedOutput) return result;
    try {
        const data = JSON.parse(trimmedOutput);
        if (!isRecord(data)) return result;
        applyHookResponseObject(result, data);
        return result;
    } catch {
        // Some hooks emit JSONL control records. Consume those line-by-line so
        // suppressOutput payloads don't leak into the startup transcript.
    }

    const lines = trimmedOutput.split(/\r?\n/);
    const plainLines: string[] = [];
    let parsedJsonLine = false;
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
            plainLines.push(line);
            continue;
        }
        if (!/^[{[]/.test(trimmed)) {
            plainLines.push(line);
            continue;
        }
        try {
            const data = JSON.parse(trimmed);
            if (!isRecord(data)) {
                plainLines.push(line);
                continue;
            }
            parsedJsonLine = true;
            applyHookResponseObject(result, data);
        } catch {
            plainLines.push(line);
        }
    }

    if (parsedJsonLine) {
        const plainText = plainLines.join("\n").trim();
        if (plainText) result.additional_contexts.push(plainText);
        return result;
    }

    result.additional_contexts.push(trimmedOutput);
    return result;
}

function applyHookResponseObject(
    result: HookExecutionResult,
    data: Record<string, unknown>,
): void {
    const hookSpecific = data.hookSpecificOutput;
    if (isRecord(hookSpecific)) {
        const eventName = hookSpecific.hookEventName;
        if (
            (eventName === "SessionStart" ||
                eventName === "UserPromptSubmit") &&
            typeof hookSpecific.additionalContext === "string"
        ) {
            result.additional_contexts.push(hookSpecific.additionalContext);
        }
        if (eventName === "SessionStart") {
            if (typeof hookSpecific.initialUserMessage === "string")
                result.initial_user_message =
                    hookSpecific.initialUserMessage;
            if (Array.isArray(hookSpecific.watchPaths))
                result.watch_paths.push(
                    ...hookSpecific.watchPaths.filter(
                        (p: unknown): p is string => typeof p === "string",
                    ),
                );
        }
        return;
    }

    pushContextValues(result, data.additional_context);
    pushContextValues(result, data.additionalContext);

    if (typeof data.initial_user_message === "string") {
        result.initial_user_message = data.initial_user_message;
    }
    if (typeof data.initialUserMessage === "string") {
        result.initial_user_message = data.initialUserMessage;
    }
    pushWatchPaths(result, data.watch_paths);
    pushWatchPaths(result, data.watchPaths);

    if (data.decision === "block" && typeof data.reason === "string") {
        result.additional_contexts.push(data.reason);
    }
}

function pushContextValues(result: HookExecutionResult, value: unknown): void {
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed) result.additional_contexts.push(trimmed);
        return;
    }
    if (Array.isArray(value)) {
        for (const item of value) pushContextValues(result, item);
    }
}

function pushWatchPaths(result: HookExecutionResult, value: unknown): void {
    if (!Array.isArray(value)) return;
    result.watch_paths.push(
        ...value.filter((item): item is string => typeof item === "string"),
    );
}

function buildHookInput(
    eventType: HookEventType,
    payload: Record<string, unknown>,
): string {
    const data: Record<string, unknown> = {
        session_id: payload.session_id || "",
        transcript_path: payload.transcript_path || "",
        cwd: payload.cwd || process.cwd(),
        hook_event_name: eventType,
    };
    if (eventType === "SessionStart")
        data.source = String(payload.source || "startup").toLowerCase();
    if (eventType === "SessionEnd")
        data.reason = String(payload.reason || "other");
    if (eventType === "UserPromptSubmit" && payload.prompt !== undefined)
        data.prompt = payload.prompt;
    if (
        (eventType === "PreToolUse" || eventType === "PostToolUse") &&
        payload.tool_name !== undefined
    )
        data.tool_name = payload.tool_name;
    if (
        (eventType === "PreToolUse" || eventType === "PostToolUse") &&
        payload.tool_input !== undefined
    )
        data.tool_input = payload.tool_input;
    if (eventType === "PostToolUse" && payload.tool_response !== undefined)
        data.tool_response = payload.tool_response;
    if (eventType === "Stop")
        data.stop_hook_active = Boolean(payload.stop_hook_active);
    return JSON.stringify(data);
}

function parseHookEventType(value: string): HookEventType | null {
    const clean = value.trim().toLowerCase();
    const events: HookEventType[] = [
        "SessionStart",
        "SessionEnd",
        "Setup",
        "Stop",
        "Notification",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
    ];
    return events.find((event) => event.toLowerCase() === clean) || null;
}

function mergeHooks(
    target: Record<string, HookDefinition[]>,
    source: Record<string, HookDefinition[]>,
): void {
    for (const [event, defs] of Object.entries(source))
        target[event] = [...(target[event] || []), ...defs];
}

function mergeHookResult(
    target: HookExecutionResult,
    source: HookExecutionResult | null,
): void {
    if (!source) return;
    target.additional_contexts.push(...source.additional_contexts);
    target.context_messages.push(...source.context_messages);
    if (!target.initial_user_message && source.initial_user_message)
        target.initial_user_message = source.initial_user_message;
    target.watch_paths.push(...source.watch_paths);
    if (source.raw_output) target.raw_output = source.raw_output;
}

function emptyHookResult(): HookExecutionResult {
    return {
        additional_contexts: [],
        context_messages: [],
        initial_user_message: null,
        watch_paths: [],
        raw_output: "",
    };
}

function withHookMetadata(
    hook: LoadedHook,
    result: HookExecutionResult | null,
): HookExecutionResult | null {
    if (!result) return null;
    if (!result.context_messages.length && result.additional_contexts.length) {
        result.context_messages = result.additional_contexts.map((content) => ({
            plugin_id: hook.pluginId,
            plugin_name: hook.pluginName,
            matcher: hook.definition.matcher || "",
            content,
        }));
    }
    return result;
}

function matcherMatches(pattern: string | undefined, source: string): boolean {
    if (!pattern) return true;
    const clean = pattern.trim();
    const sourceClean = source.trim();
    if (clean === "*") return true;
    if (!sourceClean) return true;
    const sourceParts = sourceClean.split("|").map((s) => s.trim());
    try {
        const regex = new RegExp(clean, "i");
        if (regex.test(sourceClean)) return true;
        if (sourceParts.some((part) => regex.test(part))) return true;
    } catch {
        // Fall back to legacy substring matching for non-regex hook matchers.
    }
    const lowerSource = sourceClean.toLowerCase();
    return clean
        .split("|")
        .map((p) => p.trim())
        .some(
            (p) =>
                p === "*" ||
                lowerSource
                    .split("|")
                    .map((s) => s.trim())
                    .includes(p.toLowerCase()) ||
                lowerSource.includes(p.toLowerCase()),
        );
}

function resolvePluginsDir(): string {
    const override = (process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR || "").trim();
    if (override)
        return path.resolve(os.homedir(), override.replace(/^~(?=$|\/)/, ""));
    const openclaude = path.join(
        os.homedir(),
        ".claude",
        "openclaude",
        "plugins",
    );
    if (existsSync(openclaude)) return openclaude;
    return path.join(os.homedir(), ".claude", "plugins");
}

function parseJsonArg(raw?: string): Record<string, unknown> {
    if (!raw) return {};
    try {
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" && !Array.isArray(parsed)
            ? parsed
            : {};
    } catch {
        return {};
    }
}

function normalizeInstall(install: PluginInstall): PluginInstall {
    return {
        scope: install.scope || "user",
        installPath: install.installPath || "",
        version: install.version || "unknown",
        installedAt: install.installedAt || "",
        lastUpdated: install.lastUpdated || "",
        gitCommitSha: install.gitCommitSha || "",
        enabled: install.enabled !== false,
        projectPath: install.projectPath || "",
        dependencies: Array.isArray(install.dependencies)
            ? install.dependencies
            : [],
    };
}

async function readJson(file: string): Promise<any> {
    try {
        return JSON.parse(await fs.readFile(file, "utf8"));
    } catch {
        return {};
    }
}

async function isDir(p: string): Promise<boolean> {
    if (!p) return false;
    try {
        return (await fs.stat(p)).isDirectory();
    } catch {
        return false;
    }
}

async function childDirNames(dir: string): Promise<string[]> {
    try {
        return (await fs.readdir(dir, { withFileTypes: true }))
            .filter((d) => d.isDirectory())
            .map((d) => d.name)
            .sort();
    } catch {
        return [];
    }
}

async function markdownNames(dir: string): Promise<string[]> {
    const out: string[] = [];
    async function walk(base: string): Promise<void> {
        try {
            for (const entry of await fs.readdir(base, {
                withFileTypes: true,
            })) {
                const full = path.join(base, entry.name);
                if (entry.isDirectory()) await walk(full);
                else if (entry.name.endsWith(".md"))
                    out.push(path.relative(dir, full));
            }
        } catch {
            // no-op
        }
    }
    await walk(dir);
    return out.sort();
}

async function findPluginManifest(root: string): Promise<string | null> {
    const direct = path.join(root, ".claude-plugin", "plugin.json");
    if (await fileExists(direct)) return direct;
    try {
        for (const entry of await fs.readdir(root, { withFileTypes: true })) {
            if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
            const nested = path.join(
                root,
                entry.name,
                ".claude-plugin",
                "plugin.json",
            );
            if (await fileExists(nested)) return nested;
        }
    } catch {
        // no-op
    }
    return null;
}

async function fileExists(file: string): Promise<boolean> {
    try {
        await fs.access(file);
        return true;
    } catch {
        return false;
    }
}

async function pluginNameFor(
    pluginDir: string,
    pluginId: string,
): Promise<string> {
    const manifest = await readPluginManifest(pluginDir);
    return typeof manifest.name === "string" && manifest.name
        ? manifest.name
        : pluginId.split("@")[0];
}

async function gitHead(cwd: string): Promise<string> {
    try {
        const { stdout } = await execFileAsync("git", ["rev-parse", "HEAD"], {
            cwd,
            timeout: 10_000,
        });
        const sha = String(stdout).trim();
        return /^[a-f0-9]{40}$/i.test(sha) ? sha : "";
    } catch {
        return "";
    }
}

function runShellCommand(
    command: string,
    options: {
        cwd: string;
        env: NodeJS.ProcessEnv;
        input: string;
        timeoutMs: number;
    },
): Promise<{ stdout: string; stderr: string; code: number | null }> {
    return new Promise((resolve, reject) => {
        const child = spawn(command, {
            cwd: options.cwd,
            env: options.env,
            shell: true,
            stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        const timer = setTimeout(
            () => {
                child.kill("SIGTERM");
                reject(new Error("hook command timed out"));
            },
            Math.max(1, options.timeoutMs),
        );

        child.stdout.setEncoding("utf8");
        child.stderr.setEncoding("utf8");
        child.stdout.on("data", (chunk) => {
            stdout += chunk;
            if (stdout.length > 1024 * 1024)
                stdout = stdout.slice(-1024 * 1024);
        });
        child.stderr.on("data", (chunk) => {
            stderr += chunk;
            if (stderr.length > 1024 * 1024)
                stderr = stderr.slice(-1024 * 1024);
        });
        child.on("error", (error) => {
            clearTimeout(timer);
            reject(error);
        });
        child.on("close", (code) => {
            clearTimeout(timer);
            resolve({ stdout, stderr, code });
        });
        child.stdin.end(options.input);
    });
}

async function copyDir(src: string, dst: string): Promise<void> {
    await fs.mkdir(dst, { recursive: true });
    for (const entry of await fs.readdir(src, { withFileTypes: true })) {
        const from = path.join(src, entry.name);
        const to = path.join(dst, entry.name);
        if (entry.isDirectory()) await copyDir(from, to);
        else if (entry.isSymbolicLink()) {
            const target = await fs.readlink(from);
            if (!path.isAbsolute(target))
                await fs.symlink(target, to).catch(() => undefined);
        } else if (entry.isFile()) await fs.copyFile(from, to);
    }
}

function hookType(value: string): HookCommand["type"] {
    return value === "prompt" || value === "agent" || value === "http"
        ? value
        : "command";
}

function stringOrUndefined(value: unknown): string | undefined {
    return typeof value === "string" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function sanitize(value: string): string {
    return (
        value.replace(/[^a-zA-Z0-9\-_]/g, "-").replace(/^-+|-+$/g, "") ||
        "unknown"
    );
}

function nowIso(): string {
    return new Date().toISOString();
}

function envNumber(name: string, fallback: number): number {
    const raw = Number(process.env[name] || "");
    return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}
