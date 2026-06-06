import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Tool } from "./types.ts";

export function buildDefaultSystemPrompt(cwd: string, tools: Tool[]): string {
    const date = new Date().toISOString().slice(0, 10);
    const toolList = tools
        .map((tool) => `- ${tool.name}: ${tool.description}`)
        .join("\n");
    const agentInstructions = loadAgentInstructions(cwd);
    const agentInstructionsBlock = agentInstructions
        ? `\n\nProject instructions from AGENTS.md:\n${agentInstructions}`
        : "";

    return `You are Logician, a coding agent running in a terminal TUI.

You help the user by inspecting the repository, editing files, running commands, and verifying changes. Prefer doing the work with tools over describing what you would do.

Available tools:
${toolList}

Default coding-agent workflow:
- Inspect before editing. Use list_files, find, rg_search, read_file, git status/diff, or bash as needed.
- Use find to locate files by glob pattern (e.g. '**/*.test.ts'); use rg_search to search file contents.
- For multi-step tasks, call todo_write to track the plan. Pass the full list each call, mark exactly one item in_progress while working on it, and complete items as you finish.
- For targeted changes, prefer edit_file with exact unique context.
- For new files or complete rewrites, use write_file.
- After writing or editing, read the changed area or use file_diff to verify the result. Mutation tools already return diffs; use those diffs to explain what changed.
- Run the narrowest useful verification command after risky changes, such as tests, type checks, linters, or a smoke command.
- Keep changes scoped to the user's request. Do not revert unrelated user changes.
- Never use destructive git operations such as reset --hard, checkout --, or deleting files unless the user explicitly asks.
- Be concise in final responses, but include changed files and verification results.
${agentInstructionsBlock}

Current date: ${date}
Current working directory: ${cwd}`;
}

function loadAgentInstructions(cwd: string): string {
    const files = findAgentFiles(cwd);
    const sections: string[] = [];
    for (const file of files) {
        try {
            const content = readFileSync(file, "utf8").trim();
            if (content) {
                sections.push(
                    `<agents-file path="${file}">\n${content}\n</agents-file>`,
                );
            }
        } catch {
            // Ignore unreadable context files; startup should not fail because of one.
        }
    }
    return sections.join("\n\n");
}

function findAgentFiles(cwd: string): string[] {
    const seen = new Set<string>();
    const files: string[] = [];
    const add = (file: string | undefined) => {
        if (!file) return;
        const resolved = resolve(file);
        if (seen.has(resolved) || !existsSync(resolved)) return;
        seen.add(resolved);
        files.push(resolved);
    };

    const explicit = process.env.LOGICIAN_AGENTS_FILE;
    if (explicit) {
        for (const item of explicit.split(":")) {
            if (item.trim()) add(item.trim());
        }
    }

    for (const dir of walkUp(cwd)) {
        add(join(dir, "AGENTS.md"));
        add(join(dir, "AGENTS.MD"));
    }

    add(join(dirname(process.execPath), "AGENTS.md"));

    const packageRoot = findPackageRootFromModule();
    if (packageRoot) add(join(packageRoot, "AGENTS.md"));

    return files;
}

function walkUp(start: string): string[] {
    const dirs: string[] = [];
    let dir = resolve(start);
    while (true) {
        dirs.push(dir);
        const parent = dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    return dirs;
}

function findPackageRootFromModule(): string | null {
    try {
        let dir = dirname(fileURLToPath(import.meta.url));
        while (true) {
            if (existsSync(join(dir, "package.json"))) return dir;
            const parent = dirname(dir);
            if (parent === dir) return null;
            dir = parent;
        }
    } catch {
        return null;
    }
}
