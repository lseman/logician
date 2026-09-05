import type { KernelManager } from "../eval/kernel-manager.ts";
import { createEvalTool } from "../eval/eval-tool.ts";
import { createWorkpoolTool } from "../eval/workpool-tool.ts";
import { createCompletionTool } from "../eval/completion-tool.ts";
import { createWaitTool } from "../eval/wait-tool.ts";
import type { Tool, WebSearchConfig } from "@logician/log-core";
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
import { ast_edit } from "./ast-edit.ts";
import type { BrowserManager } from "../browser/browser-manager.ts";
import { createBrowserTool } from "../browser/browser-tool.ts";

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
}

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
		createWebSearchTool(webSearch),
		...(opts.browserManager
			? [createBrowserTool({ manager: opts.browserManager })]
			: []),
		...(opts.kernelManager
			? [
					createEvalTool({ kernel: opts.kernelManager }),
					createWorkpoolTool({ kernel: opts.kernelManager }),
					createCompletionTool({ kernel: opts.kernelManager }),
					createWaitTool({ kernel: opts.kernelManager }),
				]
			: []),
	];
	return tools;
}
