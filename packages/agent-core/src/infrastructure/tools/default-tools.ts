import { getBuiltInTools } from "@logician/agent-blocks/tools";
import type { Tool, WebSearchConfig } from "../../core/types/index.ts";
import { ariadne } from "./ariadne.ts";
import { bash } from "./bash.ts";
import { edit_file } from "./edit-file.ts";
import { file_diff } from "./file-diff.ts";
import { find } from "./find.ts";
import { git } from "./git.ts";
import { list_files } from "./list-files.ts";
import { read_file } from "./read-file.ts";
import { sandbox } from "./sandbox.ts";
import { grep } from "./search.ts";
import { web_fetch } from "./web-fetch.ts";
import { createWebSearchTool } from "./web-search.ts";
import { write_file } from "./write-file.ts";
import { write_file_append } from "./write-file-append.ts";

// Default SearXNG instance assumed for local development.
export const DEFAULT_SEARXNG_URL = "http://localhost:8090";

export interface DefaultToolsOptions {
	// SearXNG config; defaults to DEFAULT_SEARXNG_URL when omitted.
	webSearch?: WebSearchConfig;
	ariadneEnabled?: boolean;
}

export function createDefaultTools(opts: DefaultToolsOptions = {}): Tool[] {
	const webSearch = opts.webSearch ?? { baseUrl: DEFAULT_SEARXNG_URL };
	const tools: Tool[] = [
		list_files,
		find,
		...(opts.ariadneEnabled !== false ? [ariadne] : []),
		read_file,
		grep,
		edit_file,
		write_file,
		write_file_append,
		file_diff,
		bash,
		sandbox,
		git,
		...getBuiltInTools(),
		web_fetch,
		createWebSearchTool(webSearch),
	];
	return tools;
}
