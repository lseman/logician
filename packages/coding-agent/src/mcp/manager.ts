// ── MCP Manager ───────────────────────────────────────────────────────────
// High-level MCP server management: load servers, create tools, manage config.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Tool } from "@logician/agent-core";
import { parseJsonWithComments, runPluginBackend } from "@logician/agent-core";
import { updateConfigFile } from "../configuration/config.ts";
import {
	allocateMcpToolName,
	createMcpClient,
	createMcpTool,
	type McpClient,
	type McpServerConfig,
} from "./client.ts";

export interface McpLoadResult {
	tools: Tool[];
	servers: number;
	errors: string[];
}

export interface McpServerInfo {
	serverName: string;
	server: McpServerConfig;
	enabled: boolean;
	toolCount: number;
	loaded: boolean;
	configPath?: string;
	error?: string;
}

export interface McpSnapshotResult {
	configPath: string;
	servers: McpServerInfo[];
	loadedServers: Record<string, { toolCount: number }>;
	errors: string[];
}

export interface McpToggleResult {
	status: string;
	message: string;
	configPath: string;
	servers: McpServerInfo[];
	loadedServers: Record<string, { toolCount: number }>;
}

interface ResolvedMcpConfigs {
	configs: Record<string, McpServerConfig>;
	configPaths: Record<string, string | undefined>;
	primaryConfigPath: string;
}

function findProjectMcpConfig(cwd: string): string | null {
	const envPath =
		process.env.LOGICIAN_MCP_CONFIG ||
		process.env.MCP_CONFIG ||
		process.env.LOGICIAN_CONFIG;
	if (envPath) {
		const resolved = resolve(
			envPath.replace(/^~(?=$|\/)/, process.env.HOME || ""),
		);
		return existsSync(resolved) ? resolved : null;
	}
	let dir = resolve(cwd);
	while (true) {
		const logicianConfig = join(dir, ".logician.json");
		if (existsSync(logicianConfig)) return logicianConfig;
		const mcpJson = join(dir, ".mcp.json");
		if (existsSync(mcpJson)) return mcpJson;
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return null;
}

function readMcpServerConfigs(
	configPath: string,
): Record<string, McpServerConfig> {
	const raw = parseJsonWithComments<Record<string, unknown>>(
		readFileSync(configPath, "utf8"),
	);
	const configs = raw.mcpServers ?? raw.mcp;
	return configs && typeof configs === "object"
		? (configs as Record<string, McpServerConfig>)
		: {};
}

function fileMcpConfigPaths(cwd: string): string[] {
	const envPath =
		process.env.LOGICIAN_MCP_CONFIG ||
		process.env.MCP_CONFIG ||
		process.env.LOGICIAN_CONFIG;
	const projectPath = findProjectMcpConfig(cwd);
	if (envPath) return projectPath ? [projectPath] : [];

	const paths: string[] = [];
	const home = process.env.HOME;
	if (home) {
		const globalSettings = join(home, ".logician", "settings.json");
		if (existsSync(globalSettings)) paths.push(globalSettings);
		const userMcpJson = join(home, ".logician", "mcp.json");
		if (existsSync(userMcpJson)) paths.push(userMcpJson);
	}
	if (projectPath) paths.push(projectPath);
	return paths;
}

/** Recursively expand ${CLAUDE_PLUGIN_ROOT} in a plugin's server config. */
function expandPluginRoot<T>(value: T, root: string): T {
	if (typeof value === "string") {
		return value.replaceAll("${CLAUDE_PLUGIN_ROOT}", root) as T;
	}
	if (Array.isArray(value)) {
		return value.map(item => expandPluginRoot(item, root)) as T;
	}
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
			out[k] = expandPluginRoot(v, root);
		}
		return out as T;
	}
	return value;
}

/**
 * Collect MCP servers declared by enabled Claude Code plugins — from the
 * plugin's .mcp.json and/or its manifest's mcpServers map. Servers are
 * namespaced plugin_<plugin>_<server> (Claude Code convention) and get
 * CLAUDE_PLUGIN_ROOT in their environment.
 */
async function loadPluginMcpServerConfigs(): Promise<
	Record<string, McpServerConfig>
> {
	const out: Record<string, McpServerConfig> = {};
	try {
		const registry = await runPluginBackend("list", []);
		for (const plugin of registry.plugins || []) {
			if (plugin.enabled === false || plugin.on_disk === false) continue;
			const installPath = String(plugin.install_path || "");
			const pluginName = String(plugin.name || plugin.plugin_id || "");
			if (!installPath || !pluginName) continue;

			const servers: Record<string, unknown> = {};
			try {
				const raw = parseJsonWithComments(
					readFileSync(join(installPath, ".mcp.json"), "utf8"),
				) as Record<string, unknown>;
				const map =
					raw?.mcpServers && typeof raw.mcpServers === "object"
						? raw.mcpServers
						: raw;
				if (map && typeof map === "object") Object.assign(servers, map);
			} catch (_e: unknown) {
				// No .mcp.json — fine.
			}
			try {
				const manifest = parseJsonWithComments(
					readFileSync(
						join(installPath, ".claude-plugin", "plugin.json"),
						"utf8",
					),
				) as Record<string, unknown>;
				if (manifest?.mcpServers && typeof manifest.mcpServers === "object") {
					Object.assign(servers, manifest.mcpServers);
				}
			} catch (_e: unknown) {
				// No manifest — fine.
			}

			for (const [name, config] of Object.entries(servers)) {
				if (!config || typeof config !== "object") continue;
				const expanded = expandPluginRoot(
					config as McpServerConfig,
					installPath,
				);
				expanded.env = {
					CLAUDE_PLUGIN_ROOT: installPath,
					...(expanded.env || {}),
				};
				out[`plugin_${pluginName}_${name}`] = expanded;
			}
		}
	} catch (_e: unknown) {
		// Plugin registry unavailable — plugin servers simply don't load.
	}
	return out;
}

export class McpManager {
	private clients: McpClient[] = [];
	private loaded = false;
	private tools: Tool[] = [];
	private errors: string[] = [];
	private readonly pluginConfigLoader: () => Promise<
		Record<string, McpServerConfig>
	>;

	constructor(
		options: {
			loadPluginConfigs?: () => Promise<Record<string, McpServerConfig>>;
		} = {},
	) {
		this.pluginConfigLoader =
			options.loadPluginConfigs ?? loadPluginMcpServerConfigs;
	}

	private async resolveConfigs(cwd: string): Promise<ResolvedMcpConfigs> {
		const pluginConfigs = await this.pluginConfigLoader();
		const configs = { ...pluginConfigs };
		const configPaths: Record<string, string | undefined> = Object.fromEntries(
			Object.keys(configs).map(name => [name, undefined]),
		);
		const paths = fileMcpConfigPaths(cwd);
		for (const configPath of paths) {
			let fromFile: Record<string, McpServerConfig> = {};
			try {
				fromFile = readMcpServerConfigs(configPath);
			} catch {
				continue;
			}
			for (const [name, config] of Object.entries(fromFile)) {
				configs[name] = config;
				configPaths[name] = configPath;
			}
		}
		return {
			configs,
			configPaths,
			primaryConfigPath: paths.at(-1) || "",
		};
	}

	async load(
		cwd: string,
		reservedToolNames: Iterable<string> = [],
	): Promise<McpLoadResult> {
		if (this.loaded) {
			return {
				tools: this.tools,
				servers: this.clients.length,
				errors: this.errors,
			};
		}
		this.loaded = true;
		const usedToolNames = new Set(reservedToolNames);

		// Project config wins over user and plugin-declared servers on name clash.
		const { configs } = await this.resolveConfigs(cwd);
		for (const [name, config] of Object.entries(configs)) {
			if (config.enabled === false) continue;
			let client: McpClient | null = null;
			try {
				client = createMcpClient(name, config, cwd);
				await client.initialize();
				const defs = await client.listTools();
				for (const def of defs) {
					const exposedName = allocateMcpToolName(
						def.name,
						client.name,
						usedToolNames,
					);
					usedToolNames.add(exposedName);
					this.tools.push(createMcpTool(client, def, exposedName) as Tool);
				}
				this.clients.push(client);
				client = null;
			} catch (error) {
				client?.close();
				const message = error instanceof Error ? error.message : String(error);
				this.errors.push(`${name}: ${message}`);
			}
		}

		return {
			tools: this.tools,
			servers: this.clients.length,
			errors: this.errors,
		};
	}

	close(): void {
		for (const client of this.clients) {
			client.close();
		}
		this.clients = [];
	}

	async getSnapshot(cwd: string): Promise<McpSnapshotResult> {
		const { configs, configPaths, primaryConfigPath } =
			await this.resolveConfigs(cwd);

		const loadedEntries = await Promise.all(
			this.clients.map(
				async (client): Promise<[string, { toolCount: number }]> => {
					try {
						const tools = await client.listTools();
						return [client.name, { toolCount: tools.length }];
					} catch {
						return [client.name, { toolCount: 0 }];
					}
				},
			),
		);
		const loadedServers = Object.fromEntries(loadedEntries);

		const servers: McpServerInfo[] = Object.entries(configs).map(
			([name, server]) => {
				const enabled = server.enabled !== false;
				const loadedInfo = loadedServers[name];
				return {
					serverName: name,
					server,
					enabled,
					toolCount: loadedInfo?.toolCount ?? 0,
					loaded: !!loadedInfo,
					configPath: configPaths[name],
				};
			},
		);

		return {
			configPath: primaryConfigPath,
			servers,
			loadedServers,
			errors: this.errors,
		};
	}

	async setServerEnabled(
		serverName: string,
		enabled: boolean,
		cwd: string,
	): Promise<McpToggleResult> {
		const resolved = await this.resolveConfigs(cwd);
		const configPath = resolved.configPaths[serverName];
		if (!configPath) {
			if (serverName in resolved.configs) {
				throw new Error(
					`MCP server '${serverName}' is managed by a plugin; enable or disable the plugin instead.`,
				);
			}
			throw new Error(`MCP server '${serverName}' not found in config.`);
		}

		const updated = updateConfigFile(configPath, raw => {
			const configKey = raw.mcpServers ? "mcpServers" : "mcp";
			const config = (raw[configKey] as Record<string, McpServerConfig>) || {};
			if (!(serverName in config)) {
				throw new Error(`MCP server '${serverName}' not found in config.`);
			}
			raw[configKey] = {
				...config,
				[serverName]: { ...config[serverName], enabled },
			};
		});
		if (!updated) {
			throw new Error(`Failed to update MCP server '${serverName}' in config.`);
		}

		const loadedServers: Record<string, { toolCount: number }> = {};
		for (const client of this.clients) {
			try {
				const tools = await client.listTools();
				loadedServers[client.name] = { toolCount: tools.length };
			} catch {
				loadedServers[client.name] = { toolCount: 0 };
			}
		}

		const snapshot = await this.getSnapshot(cwd);
		const servers: McpServerInfo[] = snapshot.servers.map(server => ({
			...server,
			toolCount:
				loadedServers[server.serverName]?.toolCount ?? server.toolCount,
			loaded: server.serverName in loadedServers,
		}));

		return {
			status: enabled ? "enabled" : "disabled",
			message: `MCP server '${serverName}' has been ${enabled ? "enabled" : "disabled"}.`,
			configPath,
			servers,
			loadedServers,
		};
	}
}
