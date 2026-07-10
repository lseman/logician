import { ask_user } from "@logician/agent-core/tools/workflow/ask-user.ts";
import { task_status } from "@logician/agent-core/tools/workflow/task-status.ts";
import { todo_tool } from "@logician/agent-core/tools/todos/todo.ts";
import type { Tool, WebSearchConfig } from "@logician/agent-core/core/types.ts";
import { bash } from "./bash.ts";
import { edit_file } from "./edit-file.ts";
import { file_diff } from "./file-diff.ts";
import { find } from "./find.ts";
import { git } from "./git.ts";
import { list_files } from "./list-files.ts";
import { read_file } from "./read-file.ts";
import { grep } from "./search.ts";
import { web_fetch } from "./web-fetch.ts";
import { createWebSearchTool } from "./web-search.ts";
import { write_file } from "./write-file.ts";

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
