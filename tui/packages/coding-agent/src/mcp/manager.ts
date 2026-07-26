// ── MCP Manager ───────────────────────────────────────────────────────────
// High-level MCP server management: load servers, create tools, manage config.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Tool } from "@logician/agent-core";
import {
	createMcpClient,
	createMcpTool,
	type McpClient,
	type McpServerConfig,
} from "./client.ts";
import { parseJsonWithComments } from "@logician/agent-core/tools/shared/json-utils.ts";
import { runPluginBackend } from "@logician/agent-core/tools/shared/plugins.ts";

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

function findMcpConfig(cwd: string): string | null {
	const envPath =
		process.env.LOGICIAN_MCP_CONFIG ||
		process.env.MCP_CONFIG ||
		process.env.LOGICIAN_CONFIG;
	if (envPath && existsSync(envPath)) return envPath;
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

	const home = process.env.HOME;
	if (home) {
		const userMcpJson = join(home, ".logician", "mcp.json");
		if (existsSync(userMcpJson)) return userMcpJson;
		const global = join(home, ".logician", "settings.json");
		if (existsSync(global)) return global;
	}
	return null;
}

/** Recursively expand ${CLAUDE_PLUGIN_ROOT} in a plugin's server config. */
function expandPluginRoot<T>(value: T, root: string): T {
	if (typeof value === "string") {
		return value.replaceAll("${CLAUDE_PLUGIN_ROOT}", root) as T;
	}
	if (Array.isArray(value)) {
		return value.map((item) => expandPluginRoot(item, root)) as T;
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

function loadMcpServerConfigs(cwd: string): Record<string, McpServerConfig> {
	const configPath = findMcpConfig(cwd);
	if (!configPath) return {};
	const raw = JSON.parse(readFileSync(configPath, "utf8")) as Record<
		string,
		unknown
	>;
	const fromMcpJson = raw.mcpServers;
	if (fromMcpJson && typeof fromMcpJson === "object") {
		return fromMcpJson as Record<string, McpServerConfig>;
	}
	const fromAgentConfig = raw.mcp;
	if (fromAgentConfig && typeof fromAgentConfig === "object") {
		return fromAgentConfig as Record<string, McpServerConfig>;
	}
	return {};
}

export class McpManager {
	private clients: McpClient[] = [];
	private loaded = false;
	private tools: Tool[] = [];
	private errors: string[] = [];

	async load(cwd: string): Promise<McpLoadResult> {
		if (this.loaded) {
			return {
				tools: this.tools,
				servers: this.clients.length,
				errors: this.errors,
			};
		}
		this.loaded = true;

		// Project/user config wins over plugin-declared servers on name clash.
		const configs = {
			...(await loadPluginMcpServerConfigs()),
			...loadMcpServerConfigs(cwd),
		};
		for (const [name, config] of Object.entries(configs)) {
			if (config.enabled === false) continue;
			let client: McpClient | null = null;
			try {
				client = createMcpClient(name, config, cwd);
				await client.initialize();
				const defs = await client.listTools();
				for (const def of defs) {
					this.tools.push(createMcpTool(client, def) as Tool);
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
		const configPath = findMcpConfig(cwd);
		if (!configPath) {
			return {
				configPath: "",
				servers: [],
				loadedServers: {},
				errors: [],
			};
		}

		const raw = parseJsonWithComments<Record<string, unknown>>(readFileSync(configPath, "utf8"));
		const config =
			(raw.mcpServers as Record<string, McpServerConfig>) ||
			(raw.mcp as Record<string, McpServerConfig>) ||
			{};

		const loadedServers: Record<string, { toolCount: number }> = {};
		for (const client of this.clients) {
			loadedServers[client.name] = { toolCount: 0 };
		}

		const servers: McpServerInfo[] = Object.entries(config).map(
			([name, server]) => {
				const enabled = server.enabled !== false;
				const loadedInfo = loadedServers[name];
				return {
					serverName: name,
					server,
					enabled,
					toolCount: loadedInfo?.toolCount ?? 0,
					loaded: !!loadedInfo,
				};
			},
		);

		return {
			configPath,
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
		const configPath = findMcpConfig(cwd);
		if (!configPath) {
			throw new Error("No MCP config file found.");
		}

		const raw = parseJsonWithComments<Record<string, unknown>>(readFileSync(configPath, "utf8"));
		const configKey = raw.mcpServers ? "mcpServers" : "mcp";
		const config = (raw[configKey] as Record<string, McpServerConfig>) || {};

		if (!(serverName in config)) {
			throw new Error(`MCP server '${serverName}' not found in config.`);
		}

		config[serverName] = { ...config[serverName], enabled };
		raw[configKey] = config;
		writeFileSync(configPath, JSON.stringify(raw, null, 2), "utf8");

		const loadedServers: Record<string, { toolCount: number }> = {};
		for (const client of this.clients) {
			loadedServers[client.name] = { toolCount: 0 };
		}

		const servers: McpServerInfo[] = Object.entries(config).map(
			([name, server]) => {
				const serverEnabled = server.enabled !== false;
				const loadedInfo = loadedServers[name];
				return {
					serverName: name,
					server,
					enabled: serverEnabled,
					toolCount: loadedInfo?.toolCount ?? 0,
					loaded: !!loadedInfo,
				};
			},
		);

		return {
			status: enabled ? "enabled" : "disabled",
			message: `MCP server '${serverName}' has been ${enabled ? "enabled" : "disabled"}.`,
			configPath,
			servers,
			loadedServers,
		};
	}
}
