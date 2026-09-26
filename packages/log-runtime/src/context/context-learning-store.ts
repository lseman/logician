import { createHash, randomUUID } from "node:crypto";
import {
	existsSync,
	mkdirSync,
	readFileSync,
	renameSync,
	writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import type { AdaptiveContextLearningState } from "@logician/log-core";

function workspaceKey(workspace: string): string {
	return createHash("sha256")
		.update(resolve(workspace))
		.digest("hex")
		.slice(0, 24);
}

/** Durable, workspace-scoped adapter for adaptive context outcomes. */
export class ContextLearningStore {
	readonly path: string;

	constructor(
		workspace: string,
		root = join(homedir(), ".logician", "context-learning"),
	) {
		this.path = join(root, `${workspaceKey(workspace)}.json`);
	}

	load(): AdaptiveContextLearningState | undefined {
		if (!existsSync(this.path)) return undefined;
		try {
			const parsed: unknown = JSON.parse(readFileSync(this.path, "utf8"));
			if (!parsed || typeof parsed !== "object") return undefined;
			const candidate = parsed as Partial<AdaptiveContextLearningState>;
			if (candidate.version !== 1 || !candidate.sources) return undefined;
			return candidate as AdaptiveContextLearningState;
		} catch {
			return undefined;
		}
	}

	save(state: AdaptiveContextLearningState): void {
		const directory = dirname(this.path);
		mkdirSync(directory, { recursive: true, mode: 0o700 });
		const temporary = `${this.path}.${process.pid}.${randomUUID()}.tmp`;
		writeFileSync(temporary, `${JSON.stringify(state, null, 2)}\n`, {
			encoding: "utf8",
			mode: 0o600,
		});
		renameSync(temporary, this.path);
	}
}
