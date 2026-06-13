import { ask_user } from "./tools/ask-user.ts";
import { bash } from "./tools/bash.ts";
import { edit_file } from "./tools/edit-file.ts";
import { file_diff } from "./tools/file-diff.ts";
import { find } from "./tools/find.ts";
import { git } from "./tools/git.ts";
import { list_files } from "./tools/list-files.ts";
import { read_file } from "./tools/read-file.ts";
import { grep } from "./tools/search.ts";
import { task_status } from "./tools/task-status.ts";
import { todo_write } from "./tools/todo-write.ts";
import { web_fetch } from "./tools/web-fetch.ts";
import { createWebSearchTool } from "./tools/web-search.ts";
import { write_file } from "./tools/write-file.ts";
import type { Tool, WebSearchConfig } from "./types.ts";

// Default SearXNG instance assumed for local development.
export const DEFAULT_SEARXNG_URL = "http://localhost:8090";

export interface DefaultToolsOptions {
	// SearXNG config; defaults to DEFAULT_SEARXNG_URL when omitted.
	webSearch?: WebSearchConfig;
}

export function createDefaultTools(opts: DefaultToolsOptions = {}): Tool[] {
	const webSearch = opts.webSearch ?? { baseUrl: DEFAULT_SEARXNG_URL };
	const tools: Tool[] = [
		list_files,
		find,
		read_file,
		grep,
		edit_file,
		write_file,
		file_diff,
		bash,
		git,
		todo_write,
		ask_user,
		task_status,
		web_fetch,
		createWebSearchTool(webSearch),
	];
	return tools;
}
