import { readFileSync } from "node:fs";
import path from "node:path";

export function findJbPrompt(cwd: string): string | null {
	for (const candidate of [
		path.join(cwd, "jb.md"),
		path.join(cwd, "tui", "jb.md"),
	]) {
		try {
			return readFileSync(candidate, "utf8");
		} catch (error: unknown) {
			if ((error as { code?: string }).code !== "ENOENT") throw error;
		}
	}
	return null;
}
