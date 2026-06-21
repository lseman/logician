import { ask_user } from "../skills/ask-user.ts";
import { bash } from "../skills/bash.ts";
import { edit_file } from "../skills/edit-file.ts";
import { file_diff } from "../skills/file-diff.ts";
import { find } from "../skills/find.ts";
import { git } from "../skills/git.ts";
import { list_files } from "../skills/list-files.ts";
import { read_file } from "../skills/read-file.ts";
import { grep } from "../skills/search.ts";
import { task_status } from "../skills/task-status.ts";
import { todo_tool } from "../todos/todo.ts";
import { web_fetch } from "../skills/web-fetch.ts";
import { createWebSearchTool } from "../skills/web-search.ts";
import { write_file } from "../skills/write-file.ts";
import type { Tool, WebSearchConfig } from "../../core/types.ts";

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
		todo_tool,
		ask_user,
		task_status,
		web_fetch,
		createWebSearchTool(webSearch),
	];
	return tools;
}
