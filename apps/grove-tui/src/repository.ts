import { SessionRegistry } from "@logician/log-core/runtime";
import type { GroveRepository, GroveSession } from "./model.ts";

export class LogicianGroveRepository implements GroveRepository {
	private readonly registry: SessionRegistry;

	constructor(registry = new SessionRegistry()) {
		this.registry = registry;
	}

	list(cwd: string): readonly GroveSession[] {
		return this.registry.listSessionInfos(cwd).map(info => {
			const store = this.registry.getSession(info.id);
			const entries = store?.loadEntries() ?? [];
			const children = new Map<string | undefined, number>();
			for (const entry of entries) {
				children.set(entry.parentId, (children.get(entry.parentId) ?? 0) + 1);
			}
			const branchCount = [...children.values()].reduce(
				(count, childCount) => count + Math.max(0, childCount - 1),
				0,
			);
			return {
				id: info.id,
				name: info.name ?? "Untitled Session",
				preview: info.preview,
				cwd: info.cwd ?? cwd,
				lastActivity: info.lastActivity,
				messageCount: info.messageCount,
				branchCount,
				entries,
			};
		});
	}
}
