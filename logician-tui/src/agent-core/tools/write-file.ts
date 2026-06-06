// ── write_file tool ───────────────────────────────────────────────────────────────
// Create or overwrite a file and return the resulting diff.

import * as fs from "fs";
import * as path from "path";
import type { Tool } from "../types.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import {
    ensureInsideCwd,
    mutationSummary,
    readUtf8IfExists,
    resolvePath,
} from "./helpers.ts";
import { refreshAfterWrite } from "./read-tracker.ts";

export const write_file: Tool = {
    name: "write_file",
    description:
        "Create or overwrite a complete file. Creates parent directories and returns a unified diff.",
    parameters: {
        type: "object",
        properties: {
            path: { type: "string", description: "File path to write" },
            content: { type: "string", description: "Complete file contents" },
        },
        required: ["path", "content"],
    },
    execute: async (args, ctx): Promise<string> => {
        const filePath = String(args.path);
        const content = String(args.content ?? "");
        const resolved = resolvePath(ctx.cwd, filePath);
        ensureInsideCwd(ctx.cwd, resolved);

        return withFileMutationQueue(resolved, async () => {
            const before = readUtf8IfExists(resolved);
            if (before === content) {
                return `No changes made: ${resolved}`;
            }

            fs.mkdirSync(path.dirname(resolved), { recursive: true });
            fs.writeFileSync(resolved, content, "utf-8");
            refreshAfterWrite(resolved);
            const diff = await mutationSummary(
                ctx.cwd,
                resolved,
                before,
                content,
            );

            return `${before === null ? "Created" : "Wrote"} ${resolved}\n\nDiff:\n${diff}`;
        });
    },
};
