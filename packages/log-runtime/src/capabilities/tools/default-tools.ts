import type { Tool, WebSearchConfig } from "@logician/log-core";
import type { BrowserManager } from "../browser/browser-manager.ts";
import { createBrowserTool } from "../browser/browser-tool.ts";
import { createCompletionTool } from "../eval/completion-tool.ts";
import { createEvalTool } from "../eval/eval-tool.ts";
import type { KernelManager } from "../eval/kernel-manager.ts";
import { createWaitTool } from "../eval/wait-tool.ts";
import { createWorkpoolTool } from "../eval/workpool-tool.ts";
import { createHubTool } from "../hub/hub-tool.ts";
import { defaultHub } from "../hub/process-manager.ts";
import type { LspClientPool } from "../lsp/lsp-client-pool.ts";
import { createLspTool } from "../lsp/lsp-tool.ts";
import { ast_edit } from "./ast-edit.ts";
import { bash } from "./bash.ts";
import { getBuiltInTools } from "./builtin-blocks.ts";
import { edit_file } from "./edit-file.ts";
import { file_diff } from "./file-diff.ts";
import { find } from "./find.ts";
import { git } from "./git.ts";
import { list_files } from "./list-files.ts";
import { read_file } from "./read-file.ts";
import { OPTIONAL_CAPABILITIES } from "./registry.ts";
import { sandbox } from "./sandbox.ts";
import { grep } from "./search.ts";
import { web_fetch } from "./web-fetch.ts";
import { createWebSearchTool } from "./web-search.ts";
import { write_file } from "./write-file.ts";

// Default SearXNG instance assumed for local development.
export const DEFAULT_SEARXNG_URL = "http://localhost:8090";

export interface DefaultToolsOptions {
	// SearXNG config; defaults to DEFAULT_SEARXNG_URL when omitted.
	webSearch?: WebSearchConfig;
	graphicianEnabled?: boolean;
	// Pre-constructed kernel manager for eval/workpool tools.
	kernelManager?: KernelManager;
	// Pre-constructed browser manager for browser automation.
	browserManager?: BrowserManager;
	// Pre-constructed LSP client pool for language server queries.
	lspPool?: LspClientPool;
	// Whether to include the `todo` tool (default: true).
	todoEnabled?: boolean;
}

// ── Core tools (top-level, always advertised to the provider) ────────────────

/** Tools that are always available as top-level function calls. */
const CORE_TOOL_NAMES = new Set<string>([
	"list_files",
	"find",
	"read_file",
	"grep",
	"edit_file",
	"ast_edit",
	"write_file",
	"bash",
	"todo",
	"ask_user",
	"rag_search",
	"rag_ingest",
	"eval",
	"workpool",
	"completion",
	"wait",
]);

/**
 * Extract only core tools from a tool array.
 */
function filterCoreTools(tools: Tool[]): Tool[] {
	return tools.filter(t => CORE_TOOL_NAMES.has(t.name));
}

// ── Discoverable tools (xd:// devices) ────────────────────────────────────────

/** Tools accessible via `write xd://<name>` — not advertised to the provider. */
const DISCOVERABLE_TOOL_NAMES = new Set<string>([
	"git",
	"sandbox",
	"file_diff",
	"web_fetch",
	"web_search",
	"browser",
	"lsp",
	"hub",
	"graphician",
]);

/**
 * Extract discoverable tools from a tool array.
 */
function filterDiscoverableTools(tools: Tool[]): Tool[] {
	return tools.filter(t => DISCOVERABLE_TOOL_NAMES.has(t.name));
}

// ── Factory functions ─────────────────────────────────────────────────────────

/**
 * Build all default tools. Used by the ToolRouter for dispatch and TUI display.
 */
export function createDefaultTools(opts: DefaultToolsOptions = {}): Tool[] {
	const webSearch = opts.webSearch ?? { baseUrl: DEFAULT_SEARXNG_URL };
	const enabled: Record<string, boolean | undefined> = {
		graphician: opts.graphicianEnabled,
	};
	const optionalTools = OPTIONAL_CAPABILITIES.filter(
		cap => (enabled[cap.id] ?? cap.enabledByDefault) !== false,
	).map(cap => cap.tool);
	const tools: Tool[] = [
		list_files,
		find,
		...optionalTools,
		read_file,
		grep,
		edit_file,
		ast_edit,
		write_file,
		file_diff,
		bash,
		sandbox,
		git,
		...getBuiltInTools(),
		web_fetch,
		...(opts.kernelManager
			? [
					createEvalTool({ kernel: opts.kernelManager }),
					createWorkpoolTool({ kernel: opts.kernelManager }),
					createCompletionTool({ kernel: opts.kernelManager }),
					createWaitTool({ kernel: opts.kernelManager }),
				]
			: []),
		// ── Hub (named process lifecycle) ──────────────────────────────────
		createHubTool({ manager: defaultHub }),
		// ── Web search ──────────────────────────────────────────────────
		createWebSearchTool(webSearch),
		// ── Browser ─────────────────────────────────────────────────────
		...(opts.browserManager
			? [createBrowserTool({ manager: opts.browserManager })]
			: []),
		// ── LSP (language server protocol) ────────────────────────────────
		...(opts.lspPool ? [createLspTool(opts.lspPool)] : []),
	];
	return tools;
}

/**
 * Build only core tools. Used for provider tool list and system prompt.
 * Discoverable tools are NOT advertised to the model as top-level tools.
 */
export function createCoreTools(opts: DefaultToolsOptions = {}): Tool[] {
	const allTools = createDefaultTools(opts);
	return filterCoreTools(allTools);
}

/**
 * Build only discoverable tools. Used for xd:// device mounting.
 */
export function createDiscoverableTools(
	opts: DefaultToolsOptions = {},
): Tool[] {
	const allTools = createDefaultTools(opts);
	return filterDiscoverableTools(allTools);
}
