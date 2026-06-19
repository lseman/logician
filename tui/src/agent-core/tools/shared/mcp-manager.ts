// ── MCP Manager ───────────────────────────────────────────────────────────
// High-level MCP server management: load servers, create tools, manage config.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { Tool } from "../../core/types.ts";
import {
	createMcpClient,
	createMcpTool,
	parseMcpToolDefinition,
	type McpToolDefinition,
	type McpClient,
} from "./mcp-client.ts";

export interface McpLoadResult {
	tools: Tool[];
	servers: number;
	errors: string[];
}

interface McpServerConfig {
	enabled?: boolean;
	type?: string;
	command?: string;
	args?: string[];
	env?: Record<string, string>;
	cwd?: string;
	url?: string;
	headers?: Record<string, string>;
	timeout?: number;
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

		const configs = loadMcpServerConfigs(cwd);
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

		const raw = JSON.parse(readFileSync(configPath, "utf8")) as Record<
			string,
			unknown
		>;
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

		const raw = JSON.parse(readFileSync(configPath, "utf8")) as Record<
			string,
			unknown
		>;
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
